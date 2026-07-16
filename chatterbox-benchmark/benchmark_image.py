"""
Standalone Image Generation Benchmark for ROCm / CUDA
Uses runwayml/stable-diffusion-v1-5 via diffusers directly.
"""
import time
import torch
import os, sys

# Redirect HF cache to D: if available
if sys.platform.startswith("win32") and os.path.exists("D:\\"):
    os.environ["HF_HOME"] = r"D:\huggingface_cache"
    os.environ["HF_HUB_CACHE"] = r"D:\huggingface_cache"

# Clean no_proxy to avoid HTTPX IPv6 port parsing crash
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([p for p in os.environ["no_proxy"].split(",") if ":" not in p])

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "runwayml/stable-diffusion-v1-5"
PROMPT      = "A futuristic cyberpunk cityscape at night, neon lights, ultra detailed, cinematic"
NEG_PROMPT  = "low quality, blurry, watermark, text, deformed"
WIDTH, HEIGHT = 512, 512
N_STEPS     = 20   # full quality run
N_WARMUP    = 1    # warmup images (not counted)
N_RUNS      = 5    # benchmark images
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"\n{'='*55}")
    print(f"  Image Generation Benchmark (diffusers + ROCm)")
    print(f"{'='*55}")
    print(f"  Device  : {device}")
    if device == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  Model   : {MODEL_ID}")
    print(f"  Size    : {WIDTH}x{HEIGHT}  |  Steps: {N_STEPS}")
    print(f"{'='*55}\n")

    print("Loading model...")
    t0 = time.time()

    # Download to a flat local directory (no symlinks) to work around Windows
    # HF hub cache symlink limitation that breaks loading config.json in sub-folders
    from huggingface_hub import snapshot_download
    local_model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "stable-diffusion-v1-5")
    local_model_dir = os.path.normpath(local_model_dir)
    if not os.path.exists(os.path.join(local_model_dir, "model_index.json")):
        print(f"  Downloading {MODEL_ID} to {local_model_dir} ...")
        snapshot_download(MODEL_ID, local_dir=local_model_dir)
        print("  Download complete.\n")
    else:
        print(f"  Using cached model at {local_model_dir}\n")

    pipe = StableDiffusionPipeline.from_pretrained(
        local_model_dir,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    # Fast DPM-Solver++ scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()  # reduce VRAM usage
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    generator = torch.Generator(device=device).manual_seed(SEED)

    # Warm-up
    print(f"Warming up ({N_WARMUP} image)...")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _ = pipe(
        PROMPT,
        negative_prompt=NEG_PROMPT,
        width=WIDTH, height=HEIGHT,
        num_inference_steps=N_STEPS,
        generator=generator,
    )
    print("Warm-up done.\n")

    # Reset stats after warmup
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark runs
    print(f"{'Run':<5} | {'Time (s)':<10} | {'it/s':<10}")
    print("-" * 35)

    times = []
    for i in range(N_RUNS):
        generator = torch.Generator(device=device).manual_seed(SEED + i)
        t_start = time.perf_counter()
        result = pipe(
            PROMPT,
            negative_prompt=NEG_PROMPT,
            width=WIDTH, height=HEIGHT,
            num_inference_steps=N_STEPS,
            generator=generator,
        )
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        its = N_STEPS / elapsed
        times.append(elapsed)
        print(f"{i+1:<5} | {elapsed:<10.2f} | {its:<10.2f}")

        # Save last image
        if i == N_RUNS - 1:
            out_path = "benchmark_image_output.png"
            result.images[0].save(out_path)
            print(f"\n  Last image saved to: {out_path}")

    # Summary
    avg_time   = sum(times) / len(times)
    avg_its    = N_STEPS / avg_time
    best_time  = min(times)
    best_its   = N_STEPS / best_time

    print(f"\n{'='*55}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*55}")
    print(f"  Runs            : {N_RUNS} images @ {N_STEPS} steps each")
    print(f"  Resolution      : {WIDTH}x{HEIGHT}")
    print(f"  Avg Time/Image  : {avg_time:.2f}s")
    print(f"  Best Time/Image : {best_time:.2f}s")
    print(f"  Avg it/s        : {avg_its:.2f}")
    print(f"  Best it/s       : {best_its:.2f}")
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  Peak VRAM       : {peak_vram:.1f} MB")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
