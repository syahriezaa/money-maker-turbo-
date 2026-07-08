import sys
import os

# Clean no_proxy to avoid HTTPX IPv6 port parsing crash (e.g. ::1)
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

import torch
import scipy.io.wavfile
from transformers import MusicgenForConditionalGeneration, AutoProcessor

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_music_local.py <prompt> <output_path> [duration_seconds]")
        sys.exit(1)
        
    prompt = sys.argv[1]
    output_path = sys.argv[2]
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Prompt: '{prompt}'")
    print(f"Duration: {duration} seconds")
    
    print("Loading MusicGen model (facebook/musicgen-small)...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model_kwargs = {}
    if device == "mps":
        model_kwargs["attn_implementation"] = "eager"
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", **model_kwargs)
    model.to(device)
    
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    # 50 tokens per second of audio
    max_tokens = int(duration * 50)
    print(f"Generating music (max_new_tokens={max_tokens})...")
    
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_tokens)
        
    audio_data = audio_values[0, 0].cpu().numpy()
    sample_rate = model.config.audio_encoder.sampling_rate
    
    print(f"Saving generated audio to: {output_path} (sampling rate: {sample_rate} Hz)")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_data)
    print("Music generation complete!")

if __name__ == "__main__":
    main()
