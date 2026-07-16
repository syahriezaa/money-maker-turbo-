import sys
import os

# Clean no_proxy to avoid HTTPX IPv6 port parsing crash (e.g. ::1)
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

import re
import torch
import torchaudio as ta

# Mock torch.distributed.ReduceOp if it's missing (happens on some Windows torch builds)
if not hasattr(torch, "distributed") or not hasattr(torch.distributed, "ReduceOp"):
    class MockReduceOp:
        SUM = "SUM"
        PRODUCT = "PRODUCT"
        MIN = "MIN"
        MAX = "MAX"
        BAND = "BAND"
        BOR = "BOR"
        BXOR = "BXOR"
        AVG = "AVG"
    
    if not hasattr(torch, "distributed"):
        import types
        torch.distributed = types.ModuleType("torch.distributed")
        sys.modules["torch.distributed"] = torch.distributed
        
    torch.distributed.ReduceOp = MockReduceOp

# Monkey-patch torch.Tensor.to to convert float64 (double) tensors to float32 on CPU
# before moving them to GPU (MPS/CUDA/ROCm), since many GPU ops expect float32.
_original_to = torch.Tensor.to
def _patched_to(self, *args, **kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    if args:
        if isinstance(args[0], (str, torch.device)):
            device = args[0]
        elif isinstance(args[0], torch.dtype):
            dtype = args[0]
        if len(args) > 1 and isinstance(args[1], torch.dtype):
            dtype = args[1]
            
    is_target_gpu = False
    if device is not None:
        if isinstance(device, str) and ("mps" in device or "cuda" in device):
            is_target_gpu = True
        elif isinstance(device, torch.device) and (device.type == "mps" or device.type == "cuda"):
            is_target_gpu = True
            
    if is_target_gpu and self.dtype == torch.float64:
        self = self.float()
    if is_target_gpu and dtype == torch.float64:
        kwargs["dtype"] = torch.float32
        
    return _original_to(self, *args, **kwargs)
torch.Tensor.to = _patched_to

# Monkey-patch torchaudio.load to use soundfile instead of torchcodec
# since PyTorch 2.9+ rocm on Windows has a hardcoded torchcodec requirement
# which crashes without FFmpeg shared DLLs.
_original_load = ta.load
def _patched_load(filepath, *args, **kwargs):
    import soundfile as sf
    try:
        data, samplerate = sf.read(filepath)
        tensor = torch.from_numpy(data).float()
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)  # [1, time]
        else:
            tensor = tensor.T  # [channels, time]
        return tensor, samplerate
    except Exception as e:
        print(f"[torchaudio.load patch] soundfile read failed for {filepath}: {e}")
        return _original_load(filepath, *args, **kwargs)
ta.load = _patched_load

# Monkey-patch ChatterboxTurboTTS.norm_loudness to cast output to float32
# and avoid float32-to-float64 numpy type promotion which crashes GPU STFT.
try:
    import numpy as np
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    _original_norm_loudness = ChatterboxTurboTTS.norm_loudness
    def _patched_norm_loudness(self, wav, sr, *args, **kwargs):
        out = _original_norm_loudness(self, wav, sr, *args, **kwargs)
        if isinstance(out, np.ndarray):
            return out.astype(np.float32)
        return out
    ChatterboxTurboTTS.norm_loudness = _patched_norm_loudness
except Exception as e:
    print(f"[ChatterboxTurboTTS patch failed] {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_tts_local.py <text> <output_path> [language]")
        sys.exit(1)
        
    text = sys.argv[1]
    if os.path.exists(text) and text.endswith(".txt"):
        print(f"Reading script text from file: {text}")
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
            
    output_path = sys.argv[2]
    language = sys.argv[3].lower() if len(sys.argv) > 3 else "en"
    voice_name = sys.argv[4] if len(sys.argv) > 4 else "default"
    
    device = os.environ.get("TTS_DEVICE", "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"Target language: '{language}'")
    print(f"Voice name selection: '{voice_name}'")
    
    tts_provider = os.environ.get("TTS_PROVIDER", "local-chatterbox")
    print(f"TTS Provider from environment: '{tts_provider}'")
    
    if tts_provider == "local-fishaudio":
        print("Using local Fish Audio S2-Pro...")
        # 1. Download s2-pro checkpoint if not already present
        try:
            from huggingface_hub import snapshot_download
            checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "s2-pro")
            if not os.path.exists(os.path.join(checkpoint_dir, "model.pth")) and not os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
                print("Downloading Fish Audio S2-Pro model weights...")
                snapshot_download(repo_id="fishaudio/s2-pro", local_dir=checkpoint_dir)
            else:
                print(f"Fish Audio S2-Pro weights found in {checkpoint_dir}")
        except Exception as e:
            print(f"Failed to download or check s2-pro weights: {e}")
            sys.exit(1)
            
        # 2. Append fish-speech to sys.path
        fish_speech_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish-speech")
        if fish_speech_path not in sys.path:
            sys.path.append(fish_speech_path)
            
        # 3. Load model using ModelManager
        try:
            from tools.server.model_manager import ModelManager
            from tools.server.inference import inference_wrapper as inference
            from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
            
            print(f"Initializing Fish Speech S2-Pro ModelManager on {device}...")
            half_precision = (device in ["cuda", "mps"])
            model_manager = ModelManager(
                mode="tts",
                device=device,
                half=half_precision,
                compile=False,
                llama_checkpoint_path=checkpoint_dir,
                decoder_checkpoint_path=os.path.join(checkpoint_dir, "codec.pth"),
                decoder_config_name="modded_dac_vq"
            )
            model_sr = model_manager.decoder_model.sample_rate
            print(f"Fish Speech S2-Pro ModelManager initialized successfully! SR: {model_sr}")
        except Exception as e:
            print(f"Failed to initialize Fish Speech S2-Pro model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    elif language == "id":
        print("Loading ChatterboxTTS Indonesian model...")
        try:
            from huggingface_hub import hf_hub_download
            from chatterbox.tts import ChatterboxTTS
            
            repo_id = "grandhigh/Chatterbox-TTS-Indonesian"
            print(f"Downloading model files from HF: {repo_id}...")
            
            local_dir = None
            for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]:
                local_file = hf_hub_download(repo_id=repo_id, filename=fpath)
                local_dir = os.path.dirname(local_file)
                
            # Optionally attempt to download conds.pt
            try:
                hf_hub_download(repo_id=repo_id, filename="conds.pt")
            except Exception:
                pass
                
            print(f"Loading ChatterboxTTS from local folder: {local_dir}")
            model = ChatterboxTTS.from_local(local_dir, device=device)
            model_sr = model.sr
        except Exception as e:
            print(f"Failed to load Indonesian model: {e}")
            sys.exit(1)
    else:
        print("Loading standard ChatterboxTurboTTS model (English)...")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(device=device)
            model_sr = model.sr
        except Exception as e:
            print(f"Failed to load standard ChatterboxTurboTTS model on {device}: {e}")
            if device in ["cuda", "mps"]:
                print("Retrying on CPU...")
                try:
                    model = ChatterboxTurboTTS.from_pretrained(device="cpu")
                    model_sr = model.sr
                except Exception as cpu_e:
                    print(f"Failed to load standard model on CPU: {cpu_e}")
                    sys.exit(1)
            else:
                sys.exit(1)

    # Parse the input text into segments based on double quotes and smart quotes
    raw_segments = []
    # Match content inside double quotes or smart quotes (“” or "")
    pattern = r'["“]([^"”]+)["”]'
    last_idx = 0
    for match in re.finditer(pattern, text):
        start, end = match.span()
        # Narration block before the quote
        narration = text[last_idx:start].strip()
        if narration:
            raw_segments.append((narration, False))
        
        # Dialogue block inside the quote
        dialogue = match.group(1).strip()
        if dialogue:
            raw_segments.append((dialogue, True))
            
        last_idx = end
        
    # Trailing narration block
    trailing = text[last_idx:].strip()
    if trailing:
        raw_segments.append((trailing, False))
        
    if not raw_segments:
        raw_segments.append((text, False))

    # Split each segment into sentences to avoid VRAM overload on long sequences
    sentences_list = []
    for seg_text, is_dialogue in raw_segments:
        raw_sents = re.split(r'(?<=[.!?])\s+', seg_text)
        sents = [s.strip() for s in raw_sents if s.strip()]
        for s in sents:
            sentences_list.append((s, is_dialogue))

    # Pre-download both voice prompt references to enable multi-character voice casting
    ref_wav_ex1 = None
    ref_wav_ex2 = None
    try:
        from huggingface_hub import hf_hub_download
        print("Pre-downloading voice prompt references...")
        ref_wav_ex1 = hf_hub_download(repo_id="grandhigh/Chatterbox-TTS-Indonesian", filename="example1.wav")
        ref_wav_ex2 = hf_hub_download(repo_id="grandhigh/Chatterbox-TTS-Indonesian", filename="example2.wav")
    except Exception as e:
        print(f"HuggingFace prompt downloads failed or skipped (may be running offline): {e}")

    # Set up and pre-download native English reference voice (LJSpeech)
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)
    lj_ref_path = os.path.join(static_dir, "ref_ljspeech.wav")
    if not os.path.exists(lj_ref_path):
        print("Downloading native English reference voice (LJSpeech)...")
        try:
            import urllib.request
            url = "https://github.com/coqui-ai/TTS/raw/main/tests/data/ljspeech/wavs/LJ001-0001.wav"
            with urllib.request.urlopen(url, timeout=10) as response:
                with open(lj_ref_path, "wb") as out_file:
                    out_file.write(response.read())
        except Exception as dl_e:
            print(f"Failed to download LJSpeech reference (will use fallback): {dl_e}")

    # Set up and prepare native English male reference voice (LibriSpeech)
    male_ref_path = os.path.join(static_dir, "ref_male_en.wav")
    if not os.path.exists(male_ref_path):
        print("Downloading and preparing native English male voice (LibriSpeech)...")
        try:
            import urllib.request
            import soundfile as sf
            temp_path = os.path.join(static_dir, "temp_male.flac")
            url = "https://raw.githubusercontent.com/resemble-ai/Resemblyzer/master/audio_data/librispeech_test-other/3005/3005-163389-0000.flac"
            with urllib.request.urlopen(url, timeout=10) as response:
                with open(temp_path, "wb") as out_file:
                    out_file.write(response.read())
            # Load and save as 16-bit PCM WAV (already 8.375 seconds)
            data, sr = sf.read(temp_path)
            sf.write(male_ref_path, data, sr, subtype='PCM_16')
            os.remove(temp_path)
            print("Native English male voice prepared successfully.")
        except Exception as dl_e:
            print(f"Failed to prepare male reference: {dl_e}")

    print(f"Generating TTS for {len(sentences_list)} segments...")
    wav_items = []
    try:
        with torch.no_grad():
            for idx, (content_text, is_dialogue) in enumerate(sentences_list):
                print(f"Generating segment {idx+1}/{len(sentences_list)} (Dialogue={is_dialogue}): '{content_text[:40]}...'")

                if tts_provider == "local-fishaudio":
                    ref_audio_bytes = b""
                    ref_text = ""
                    selected_ref = None
                    
                    if language == "id":
                        if voice_name == "id-chatterbox-female":
                            selected_ref = ref_wav_ex1
                            print("  [CASTING] Selected expressive female model (example1.wav)")
                        elif voice_name == "id-chatterbox-male":
                            selected_ref = ref_wav_ex2
                            print("  [CASTING] Selected deep male narration model (example2.wav)")
                        else:
                            selected_ref = ref_wav_ex1 if is_dialogue else ref_wav_ex2
                            print(f"  [CASTING] Auto-casting voice: {'female (example1.wav)' if is_dialogue else 'male (example2.wav)'}")
                        
                        ref_text = "Bahwa sesungguhnya kemerdekaan itu ialah hak segala bangsa dan oleh sebab itu, maka penjajahan di atas dunia harus dihapuskan, karena tidak sesuai dengan perikemanusiaan dan perikeadilan."
                    else:
                        # English
                        if voice_name == "en-chatterbox-female":
                            selected_ref = lj_ref_path if os.path.exists(lj_ref_path) else ref_wav_ex1
                            print(f"  [CASTING] Selected English female voice ({'ref_ljspeech.wav' if selected_ref == lj_ref_path else 'example1.wav'})")
                        elif voice_name == "en-chatterbox-male":
                            selected_ref = male_ref_path if os.path.exists(male_ref_path) else ref_wav_ex2
                            print(f"  [CASTING] Selected English male narrator (cloned from {'ref_male_en.wav' if selected_ref == male_ref_path else 'example2.wav'})")
                        else:
                            if is_dialogue:
                                selected_ref = lj_ref_path if os.path.exists(lj_ref_path) else ref_wav_ex1
                                print(f"  [CASTING] Auto-casting English dialogue to female voice")
                            else:
                                selected_ref = male_ref_path if os.path.exists(male_ref_path) else ref_wav_ex2
                                print(f"  [CASTING] Auto-casting English narration to male voice (cloned from {'ref_male_en.wav' if selected_ref == male_ref_path else 'example2.wav'})")
                            
                        if not selected_ref or not os.path.exists(selected_ref):
                            selected_ref = ref_wav_ex1 if ref_wav_ex1 and os.path.exists(ref_wav_ex1) else ref_wav_ex2
                            print(f"  [FALLBACK] Fish Audio requires reference; fell back to {selected_ref}")
                            
                        if selected_ref == lj_ref_path:
                            ref_text = "Printing, in the only sense with which we at present have to do with it, is a very modern art,"
                        elif selected_ref in [ref_wav_ex1, ref_wav_ex2]:
                            ref_text = "Bahwa sesungguhnya kemerdekaan itu ialah hak segala bangsa dan oleh sebab itu, maka penjajahan di atas dunia harus dihapuskan, karena tidak sesuai dengan perikemanusiaan dan perikeadilan."
                    
                    if selected_ref and os.path.exists(selected_ref):
                        with open(selected_ref, "rb") as f:
                            ref_audio_bytes = f.read()
                            
                    req = ServeTTSRequest(
                        text=content_text,
                        references=[
                            ServeReferenceAudio(
                                audio=ref_audio_bytes,
                                text=ref_text
                            )
                        ] if ref_audio_bytes else [],
                        reference_id=None,
                        max_new_tokens=1024,
                        chunk_length=200,
                        top_p=0.7,
                        repetition_penalty=1.2,
                        temperature=0.7,
                        format="wav"
                    )
                    
                    fake_audios = next(inference(req, model_manager.tts_inference_engine))
                    # fake_audios is a 1D numpy array of floats. We need a 2D torch tensor (channels, samples)
                    wav_part_tensor = torch.from_numpy(fake_audios).unsqueeze(0).float()
                    wav_items.append(wav_part_tensor)
                else:
                    if language == "id":
                        if voice_name == "id-chatterbox-female":
                            selected_ref = ref_wav_ex1
                            print("  [CASTING] Selected expressive female model (example1.wav)")
                        elif voice_name == "id-chatterbox-male":
                            selected_ref = ref_wav_ex2
                            print("  [CASTING] Selected deep male narration model (example2.wav)")
                        else:
                            selected_ref = ref_wav_ex1 if is_dialogue else ref_wav_ex2
                            print(f"  [CASTING] Auto-casting voice: {'female (example1.wav)' if is_dialogue else 'male (example2.wav)'}")
                        
                        exaggeration = 0.80 if is_dialogue else 0.85
                        temperature = 0.85 if is_dialogue else 0.9
                        cfg_weight = 0.5
                        
                        wav = model.generate(
                            content_text, 
                            audio_prompt_path=selected_ref, 
                            exaggeration=exaggeration, 
                            temperature=temperature,
                            cfg_weight=cfg_weight
                        )
                    else:
                        # English Turbo version
                        selected_ref = None
                        if voice_name == "en-chatterbox-female":
                            selected_ref = lj_ref_path if os.path.exists(lj_ref_path) else ref_wav_ex1
                            print(f"  [CASTING] Selected English female voice ({'ref_ljspeech.wav' if selected_ref == lj_ref_path else 'example1.wav'})")
                        elif voice_name == "en-chatterbox-male":
                            selected_ref = male_ref_path if os.path.exists(male_ref_path) else ref_wav_ex2
                            print(f"  [CASTING] Selected English male narrator (cloned from {'ref_male_en.wav' if selected_ref == male_ref_path else 'example2.wav'})")
                        else:
                            # Auto-cast logic
                            if is_dialogue:
                                selected_ref = lj_ref_path if os.path.exists(lj_ref_path) else ref_wav_ex1
                                print(f"  [CASTING] Auto-casting English dialogue to female voice")
                            else:
                                selected_ref = male_ref_path if os.path.exists(male_ref_path) else ref_wav_ex2
                                print(f"  [CASTING] Auto-casting English narration to male voice (cloned from {'ref_male_en.wav' if selected_ref == male_ref_path else 'example2.wav'})")
                        
                        wav = model.generate(
                            content_text, 
                            audio_prompt_path=selected_ref,
                            temperature=0.85
                        )
                    
                    wav_cpu = wav.cpu()
                    wav_items.append(wav_cpu)
                    del wav
                
                import gc
                gc.collect()
                if device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                elif device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            
        print("Concatenating audio segments with silent pauses...")
        final_wav_parts = []
        for idx, wav_part in enumerate(wav_items):
            final_wav_parts.append(wav_part)
            # Add a 0.6 second silent pause between sentences (except after the last sentence)
            if idx < len(wav_items) - 1:
                silence_len = int(model_sr * 0.6)
                silence = torch.zeros((wav_part.shape[0], silence_len), dtype=wav_part.dtype)
                final_wav_parts.append(silence)
                
        wav_cpu = torch.cat(final_wav_parts, dim=-1)
        
        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        print(f"Saving audio to {output_path}...")
        try:
            import soundfile as sf
            audio_data = wav_cpu.numpy()
            if len(audio_data.shape) > 1 and audio_data.shape[0] == 1:
                audio_data = audio_data[0]
            sf.write(output_path, audio_data, model_sr)
        except Exception as sf_e:
            print(f"soundfile write failed: {sf_e}. Falling back to torchaudio...")
            ta.save(output_path, wav_cpu, model_sr)
        print("TTS generation complete!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
