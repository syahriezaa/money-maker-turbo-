import sys
import os
import json
import time

# Clean no_proxy to avoid HTTPX IPv6 port parsing crash
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

import torch

# Add backend directory to sys.path to enable relative imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import model_cache as mc

def main():
    print("[Worker] Image worker daemon initialized.", flush=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Read command requests line-by-line from stdin
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            action = req.get("action")
            
            if action == "generate":
                ckpt_name = req.get("ckpt_name")
                style = req.get("style")
                prompt = req.get("prompt")
                neg_prompt = req.get("neg_prompt")
                steps = req.get("steps", 20)
                cfg = req.get("cfg", 7.5)
                seed = req.get("seed", 1337)
                img_h = req.get("img_h", 512)
                img_w = req.get("img_w", 512)
                img_path = req.get("img_path")
                
                # Fetch cached pipeline (loads if cache miss, otherwise immediate return)
                root_dir = os.path.dirname(backend_dir)
                cache = mc.get_cache()
                pipe, lora_scale, autocast_ctx = cache.get_pipeline(
                    ckpt_name=ckpt_name,
                    style=style,
                    device=device,
                    root_dir=root_dir,
                )
                
                # Setup reproducibility seed and latents on CPU first, then transfer to target device
                generator = torch.Generator(device="cpu").manual_seed(seed)
                latent_shape = (1, 4, img_h // 8, img_w // 8)
                latents = torch.randn(latent_shape, generator=generator, device="cpu", dtype=torch.float32)
                if device != "cpu":
                    _dtype = torch.float16 if device == "cuda" else torch.float32
                    latents = latents.to(device, dtype=_dtype)
                
                cross_attn_kwargs = {"scale": lora_scale} if style == "anime" else {}
                
                # Step progress callback forwarding progress to parent process stdout
                def _sd_progress(step, timestep, latents):
                    resp = {
                        "status": "progress",
                        "step": step + 1,
                        "total_steps": steps
                    }
                    print(json.dumps(resp), flush=True)
                
                with autocast_ctx:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        latents=latents,
                        generator=generator,
                        cross_attention_kwargs=cross_attn_kwargs,
                        callback=_sd_progress,
                        callback_steps=1,
                    )
                
                img = result.images[0]
                img.save(img_path)
                
                resp = {"status": "success", "img_path": img_path}
                print(json.dumps(resp), flush=True)
                
            elif action == "ping":
                print(json.dumps({"status": "pong"}), flush=True)
                
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(json.dumps({"status": "error", "error": err_msg}), flush=True)

if __name__ == "__main__":
    main()
