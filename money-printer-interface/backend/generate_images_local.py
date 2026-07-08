import sys
import os
import json
import argparse
import torch

# Optimize CPU thread allocation to prevent thread-switching overhead and CPU contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Redirect all Hugging Face downloads to D: drive on Windows if D: drive is available, otherwise use default
if sys.platform.startswith("win32") and os.path.exists("D:\\"):
    os.environ["HF_HOME"] = r"D:\huggingface_cache"
    os.environ["HF_HUB_CACHE"] = r"D:\huggingface_cache"


# Clean no_proxy to avoid HTTPX IPv6 port parsing crash
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

# Add backend directory to sys.path so we can import comfy_schema_runner
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import comfy_schema_runner

def main():
    parser = argparse.ArgumentParser(description="Generate images locally using ComfyUI workflow via Diffusers.")
    parser.add_argument("prompt", type=str, help="Prompt for image generation.")
    parser.add_argument("output_path", type=str, help="Output path for the generated image.")
    parser.add_argument("--resolution", type=str, default="512x896", help="Resolution in WxH format (default: 512x896).")
    parser.add_argument("--width", type=int, default=None, help="Width override (takes precedence over resolution).")
    parser.add_argument("--height", type=int, default=None, help="Height override (takes precedence over resolution).")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=1337, help="Seed.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cpu, cuda).")

    args = parser.parse_args()

    # Determine width and height
    width, height = 512, 896
    if args.resolution:
        try:
            parts = args.resolution.lower().split("x")
            if len(parts) == 2:
                width = int(parts[0])
                height = int(parts[1])
        except Exception as e:
            print(f"Warning: Failed to parse resolution '{args.resolution}': {e}. Using defaults.")

    if args.width is not None:
        width = args.width
    if args.height is not None:
        height = args.height

    # Determine device (check MPS for Apple Silicon, then CUDA, then CPU)
    device = args.device
    if not device:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Style suffix for positive prompt
    style_suffix = ", high quality anime illustration, masterwork, masterpiece, cell shading, flat colors, clean outlines"
    full_positive_prompt = args.prompt + style_suffix
    negative_prompt = args.negative_prompt if args.negative_prompt else "low quality, worst quality, deformed, bad anatomy, bad hands, blurry, watermark, text, signature"

    # Dynamically construct the ComfyUI workflow JSON schema
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "hassakuAnima_v1.safetensors"
            }
        },
        "4": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "AnimaMythP0rtr4itStyleV1.safetensors",
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": ["1", 0],
                "clip": ["1", 1]
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": full_positive_prompt,
                "clip": ["4", 1]
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            }
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": args.seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["1", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "ComfyUI"
            }
        }
    }

    print("--- Generated Dynamic ComfyUI Workflow JSON ---")
    print(json.dumps(workflow, indent=2))
    print("-----------------------------------------------")

    # Run the workflow
    print(f"Running ComfyUI workflow schema via comfy_schema_runner on device: {device}")
    comfy_schema_runner.run_workflow(
        workflow_data=workflow,
        output_path=args.output_path,
        device=device
    )
    print("Image generation completed successfully!")

if __name__ == "__main__":
    main()
