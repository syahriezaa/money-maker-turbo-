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
    
    # Gunakan TextIteratorStreamer agar bisa tracking token per token
    # Fallback: generate biasa dengan progress manual via print tqdm-style sebelum dan sesudah
    import threading, time

    generated_tokens = [0]
    done_event = threading.Event()

    def progress_reporter():
        """Print progress dalam format tqdm agar backend bisa parse real-time."""
        start = time.time()
        while not done_event.is_set():
            cur = generated_tokens[0]
            if cur > 0 and max_tokens > 0:
                pct = int(cur / max_tokens * 100)
                elapsed = time.time() - start
                speed = cur / elapsed if elapsed > 0 else 0
                remaining = max(0, max_tokens - cur)
                eta = int(remaining / speed) if speed > 0 else 0
                # Format tqdm-compatible: " 45%|████      | 450/1000 [00:15<00:18, 28.50it/s]"
                bar_len = 10
                filled = int(bar_len * cur / max_tokens)
                bar = '█' * filled + '░' * (bar_len - filled)
                sys.stdout.write(
                    f"\r {pct:3d}%|{bar}| {cur}/{max_tokens} "
                    f"[{int(elapsed//60):02d}:{int(elapsed%60):02d}<{eta//60:02d}:{eta%60:02d}, {speed:.2f}it/s]"
                )
                sys.stdout.flush()
            time.sleep(1.0)
        # Print newline setelah selesai
        print()

    # Hook ke generate untuk track token
    class ProgressCallback(torch.nn.Module):
        pass

    reporter_thread = threading.Thread(target=progress_reporter, daemon=True)
    reporter_thread.start()

    try:
        with torch.no_grad():
            # Generate dengan streaming token count
            result = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
            )
            generated_tokens[0] = result.shape[-1]
    finally:
        done_event.set()
        reporter_thread.join(timeout=2)

    audio_data = result[0, 0].cpu().numpy()
    sample_rate = model.config.audio_encoder.sampling_rate
    
    print(f"Saving generated audio to: {output_path} (sampling rate: {sample_rate} Hz)")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_data)
    print("Music generation complete!")

if __name__ == "__main__":
    main()

