import sys
import os

# Clean no_proxy to avoid HTTPX IPv6 port parsing crash (e.g. ::1)
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

import re
import torch
import torchaudio as ta

# Monkey-patch torch.Tensor.to to convert float64 (double) tensors to float32 on CPU
# before moving them to MPS, since the Apple Silicon MPS framework does not support float64.
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
            
    is_target_mps = False
    if device is not None:
        if isinstance(device, str) and "mps" in device:
            is_target_mps = True
        elif isinstance(device, torch.device) and device.type == "mps":
            is_target_mps = True
            
    if is_target_mps and self.dtype == torch.float64:
        self = self.float()
    if is_target_mps and dtype == torch.float64:
        kwargs["dtype"] = torch.float32
        
    return _original_to(self, *args, **kwargs)
torch.Tensor.to = _patched_to

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_tts_local.py <text> <output_path> [language]")
        sys.exit(1)
        
    text = sys.argv[1]
    output_path = sys.argv[2]
    language = sys.argv[3].lower() if len(sys.argv) > 3 else "en"
    voice_name = sys.argv[4] if len(sys.argv) > 4 else "default"
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target language: '{language}'")
    print(f"Voice name selection: '{voice_name}'")
    
    if language == "id":
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
        except Exception as e:
            print(f"Failed to load Indonesian model: {e}")
            sys.exit(1)
    else:
        print("Loading standard ChatterboxTurboTTS model (English)...")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(device=device)
        except Exception as e:
            print(f"Failed to load standard ChatterboxTurboTTS model on {device}: {e}")
            if device in ["cuda", "mps"]:
                print("Retrying on CPU...")
                try:
                    model = ChatterboxTurboTTS.from_pretrained(device="cpu")
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

    print(f"Generating TTS for {len(sentences_list)} segments...")
    wav_items = []
    try:
        with torch.no_grad():
            for idx, (content_text, is_dialogue) in enumerate(sentences_list):
                print(f"Generating segment {idx+1}/{len(sentences_list)} (Dialogue={is_dialogue}): '{content_text[:40]}...'")

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
                    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
                    os.makedirs(static_dir, exist_ok=True)
                    lj_ref_path = os.path.join(static_dir, "ref_ljspeech.wav")
                    if not os.path.exists(lj_ref_path):
                        print("Downloading native English reference voice (LJSpeech)...")
                        try:
                            import urllib.request
                            url = "https://github.com/coqui-ai/TTS/raw/main/tests/data/ljspeech/wavs/LJ001-0001.wav"
                            urllib.request.urlretrieve(url, lj_ref_path)
                        except Exception as dl_e:
                            print(f"Failed to download LJSpeech reference: {dl_e}")
                    
                    selected_ref = None
                    if voice_name == "en-chatterbox-female":
                        selected_ref = lj_ref_path if os.path.exists(lj_ref_path) else ref_wav_ex1
                        print(f"  [CASTING] Selected English female voice ({'ref_ljspeech.wav' if selected_ref == lj_ref_path else 'example1.wav'})")
                    elif voice_name == "en-chatterbox-male":
                        selected_ref = None
                        print("  [CASTING] Selected English built-in male narrator")
                    else:
                        # Auto-cast logic
                        if is_dialogue:
                            selected_ref = lj_ref_path if os.path.exists(lj_ref_path) else ref_wav_ex1
                            print(f"  [CASTING] Auto-casting English dialogue to female voice")
                        else:
                            selected_ref = None
                            print("  [CASTING] Auto-casting English narration to built-in male voice")
                    
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
                silence_len = int(model.sr * 0.6)
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
            sf.write(output_path, audio_data, model.sr)
        except Exception as sf_e:
            print(f"soundfile write failed: {sf_e}. Falling back to torchaudio...")
            ta.save(output_path, wav_cpu, model.sr)
        print("TTS generation complete!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
