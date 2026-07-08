import sys
import os

# Clean no_proxy to avoid HTTPX IPv6 port parsing crash (e.g. ::1)
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

import re
import torch
import torchaudio as ta

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_tts_local.py <text> <output_path> [language]")
        sys.exit(1)
        
    text = sys.argv[1]
    output_path = sys.argv[2]
    language = sys.argv[3].lower() if len(sys.argv) > 3 else "en"
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target language: '{language}'")
    
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
        for idx, (content_text, is_dialogue) in enumerate(sentences_list):
            print(f"Generating segment {idx+1}/{len(sentences_list)} (Dialogue={is_dialogue}): '{content_text[:40]}...'")

            if language == "id":
                if is_dialogue:
                    # Younger/emotional voice prompt for characters in dialogue
                    selected_ref = ref_wav_ex1
                    exaggeration = 0.80
                    temperature = 0.85
                    cfg_weight = 0.5
                    print(f"  [CASTING] Selected expressive dialogue model (example1.wav)")
                else:
                    # Deep narrative voice prompt for first-person narration
                    selected_ref = ref_wav_ex2
                    exaggeration = 0.85
                    temperature = 0.9
                    cfg_weight = 0.5
                    print(f"  [CASTING] Selected deep narration model (example2.wav)")

                wav = model.generate(
                    content_text, 
                    audio_prompt_path=selected_ref, 
                    exaggeration=exaggeration, 
                    temperature=temperature,
                    cfg_weight=cfg_weight
                )
            else:
                # English Turbo version: Use native narrator for narration, and native English female for dialogue.
                selected_ref = None
                if is_dialogue:
                    # Download native English reference voice if not present
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
                    
                    if os.path.exists(lj_ref_path):
                        selected_ref = lj_ref_path
                        print(f"  [CASTING] Cloning native English female dialogue voice (ref_ljspeech.wav)")
                    else:
                        selected_ref = ref_wav_ex1
                        print(f"  [CASTING] Fallback to example1.wav")
                else:
                    # Narration uses default native English voice (no clone prompt -> falls back to conds.pt)
                    print(f"  [CASTING] Using built-in native English narrator voice")
                
                wav = model.generate(
                    content_text, 
                    audio_prompt_path=selected_ref,
                    temperature=0.85
                )
            
            wav_cpu = wav.cpu()
            wav_items.append(wav_cpu)
            
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
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
