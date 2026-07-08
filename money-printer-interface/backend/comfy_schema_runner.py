import os
import sys
import json
import argparse
import time
from contextlib import nullcontext
import torch
from PIL import Image

# Redirect all Hugging Face downloads to D: drive on Windows if D: drive is available, otherwise use default
if sys.platform.startswith("win32") and os.path.exists("D:\\"):
    os.environ["HF_HOME"] = r"D:\huggingface_cache"
    os.environ["HF_HUB_CACHE"] = r"D:\huggingface_cache"


# Clean no_proxy to avoid HTTPX IPv6 port parsing crash
if "no_proxy" in os.environ:
    os.environ["no_proxy"] = ",".join([part for part in os.environ["no_proxy"].split(",") if ":" not in part])

import diffusers
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler
)
import diffusers.loaders.single_file_utils as sfu
# Patch diffusers model type detection bug
sfu.DIFFUSERS_TO_LDM_DEFAULT_IMAGE_SIZE_MAP['cosmos-2.0-t2i-2B'] = 512

from transformers import CLIPTextModel

# Monkey patch diffusers.models.lora.text_encoder_attn_modules to support transformers 5.x CLIPTextModel layout
try:
    import diffusers.models.lora as lora
    from torch import nn
    original_text_encoder_attn_modules = lora.text_encoder_attn_modules

    def patched_text_encoder_attn_modules(text_encoder: nn.Module):
        # If the text encoder exposes its encoder directly (transformers 5.x)
        if hasattr(text_encoder, "encoder") and not hasattr(text_encoder, "text_model"):
            attn_modules = []
            for i, layer in enumerate(text_encoder.encoder.layers):
                name = f"text_model.encoder.layers.{i}.self_attn"
                mod = layer.self_attn
                attn_modules.append((name, mod))
            return attn_modules
        try:
            return original_text_encoder_attn_modules(text_encoder)
        except AttributeError:
            if hasattr(text_encoder, "encoder"):
                attn_modules = []
                for i, layer in enumerate(text_encoder.encoder.layers):
                    name = f"text_model.encoder.layers.{i}.self_attn"
                    mod = layer.self_attn
                    attn_modules.append((name, mod))
                return attn_modules
            raise

    lora.text_encoder_attn_modules = patched_text_encoder_attn_modules
    print("[Patch] Patched diffusers.models.lora.text_encoder_attn_modules for transformers 5.x compatibility.")
except Exception as patch_e:
    print(f"[Warning] Failed to apply transformers 5.x compatibility patch: {patch_e}")


def resolve_model_path(name, root_dir):
    """
    Resolves the model name to a path on the local disk.
    Searches in root_dir, common subdirectories, or absolute path.
    """
    if not name:
        return None
    
    # Securely isolate filename to prevent path traversal
    safe_name = os.path.basename(name)
    
    # 1. Absolute path check (only if safe_name equals name to avoid traversal)
    if os.path.isabs(name) and os.path.exists(name) and safe_name == name:
        return name
        
    # 2. Check standard subdirectories with the secure name
    subdirs = ["", "models/checkpoints", "models/loras", "models/vae", "backend"]
    for sub in subdirs:
        path = os.path.normpath(os.path.join(root_dir, sub, safe_name))
        if os.path.exists(path):
            return path
            
    # Fallback path (which might not exist, but is the expected resolved location)
    return os.path.normpath(os.path.join(root_dir, safe_name))


def trace_conditioning(link, nodes):
    """
    Recursively traces conditioning links to extract text prompts from CLIPTextEncode nodes.
    Supports basic combine, average, and set area nodes.
    """
    if not isinstance(link, list) or len(link) < 1:
        return ""
    
    node_id = str(link[0])
    node = nodes.get(node_id)
    if not node:
        return ""
        
    class_type = node.get("class_type")
    
    if class_type == "CLIPTextEncode":
        return node.get("inputs", {}).get("text", "")
        
    elif class_type in ["ConditioningCombine", "ConditioningAverage"]:
        c1 = node.get("inputs", {}).get("conditioning_to") or node.get("inputs", {}).get("conditioning_1")
        c2 = node.get("inputs", {}).get("conditioning_from") or node.get("inputs", {}).get("conditioning_2")
        t1 = trace_conditioning(c1, nodes)
        t2 = trace_conditioning(c2, nodes)
        return f"{t1} {t2}".strip()
        
    elif class_type == "ConditioningSetArea":
        c = node.get("inputs", {}).get("conditioning")
        return trace_conditioning(c, nodes)
        
    # Generic fallback: look for string "text" in inputs
    for key, value in node.get("inputs", {}).items():
        if key == "text" and isinstance(value, str):
            return value
        if isinstance(value, list) and len(value) > 0 and value[0] != node_id:
            if "conditioning" in key or key == "clip":
                t = trace_conditioning(value, nodes)
                if t:
                    return t
                    
    return ""


def map_comfy_sampler_to_diffusers(sampler_name, scheduler_name):
    """
    Maps ComfyUI KSampler's sampler_name and scheduler to a Diffusers scheduler class and parameters.
    """
    sampler_name = sampler_name.lower()
    scheduler_name = scheduler_name.lower()
    
    use_karras = (scheduler_name == "karras")
    
    if sampler_name == "euler":
        return EulerDiscreteScheduler, {"use_karras_sigmas": use_karras}
    elif sampler_name == "euler_ancestral":
        return EulerAncestralDiscreteScheduler, {}
    elif sampler_name == "heun":
        return HeunDiscreteScheduler, {"use_karras_sigmas": use_karras}
    elif sampler_name in ["dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2s_ancestral"]:
        kwargs = {"use_karras_sigmas": use_karras}
        if "sde" in sampler_name:
            kwargs["algorithm_type"] = "sde-dpmsolver++"
        return DPMSolverMultistepScheduler, kwargs
    elif sampler_name == "ddim":
        return DDIMScheduler, {}
    elif sampler_name == "lms":
        return LMSDiscreteScheduler, {"use_karras_sigmas": use_karras}
    elif sampler_name == "uni_pc":
        return UniPCMultistepScheduler, {}
    else:
        print(f"[Scheduler] Unknown sampler '{sampler_name}', falling back to DPMSolverMultistepScheduler.")
        return DPMSolverMultistepScheduler, {"use_karras_sigmas": True}


def parse_workflow(workflow_data):
    """
    Parses a ComfyUI workflow JSON dictionary or path.
    """
    if isinstance(workflow_data, str):
        if os.path.exists(workflow_data):
            try:
                with open(workflow_data, "r", encoding="utf-8") as f:
                    workflow = json.load(f)
            except json.JSONDecodeError as je:
                raise ValueError(f"Invalid JSON file format in '{workflow_data}': {je}")
        else:
            try:
                workflow = json.loads(workflow_data)
            except json.JSONDecodeError as je:
                raise ValueError(f"Invalid JSON string format: {je}")
    elif isinstance(workflow_data, dict):
        workflow = workflow_data
    else:
        raise ValueError("workflow_data must be a dictionary, JSON string, or valid file path")
        
    # Support both API format and Editor format (convert Editor to simple API map)
    nodes = {}
    if "nodes" in workflow and isinstance(workflow["nodes"], list):
        # Editor format mapping
        for node in workflow["nodes"]:
            nid = str(node.get("id"))
            nodes[nid] = {
                "class_type": node.get("type"),
                "inputs": {}
            }
            # Approximate properties
            widgets = node.get("widgets_values", [])
            # Simple fallback
            if widgets:
                nodes[nid]["inputs"]["widgets"] = widgets
    else:
        # Standard API format
        for nid, node in workflow.items():
            if isinstance(node, dict) and "class_type" in node:
                nodes[str(nid)] = node
                
    return nodes


def extract_parameters(nodes):
    """
    Extracts relevant pipeline parameters by parsing standard nodes in the workflow.
    """
    params = {
        "ckpt_name": None,
        "loras": [],
        "positive_prompt": "",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "batch_size": 1,
        "seed": int(time.time()) & 0xFFFFFFFF,
        "steps": 20,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
        "output_prefix": "ComfyUI"
    }
    
    # 1. Find the primary KSampler node (which orchestrates the generation)
    ksampler_nodes = [(nid, node) for nid, node in nodes.items() if node.get("class_type") == "KSampler"]
    
    if not ksampler_nodes:
        # Fallback to searching all nodes individually if KSampler is missing
        print("[Parser] Warning: No KSampler node found. Parsing nodes directly...")
        for nid, node in nodes.items():
            ctype = node.get("class_type")
            inputs = node.get("inputs", {})
            if ctype == "CheckpointLoaderSimple":
                params["ckpt_name"] = inputs.get("ckpt_name")
            elif ctype == "LoraLoader":
                params["loras"].append({
                    "lora_name": inputs.get("lora_name"),
                    "strength_model": float(inputs.get("strength_model", 1.0))
                })
            elif ctype == "EmptyLatentImage":
                params["width"] = int(inputs.get("width", 512))
                params["height"] = int(inputs.get("height", 512))
                params["batch_size"] = int(inputs.get("batch_size", 1))
        return params

    # Use the first KSampler found
    ksampler_id, ksampler = ksampler_nodes[0]
    inputs = ksampler.get("inputs", {})
    
    # Parse KSampler parameters
    params["seed"] = int(inputs.get("seed", params["seed"]))
    params["steps"] = int(inputs.get("steps", params["steps"]))
    params["cfg"] = float(inputs.get("cfg", params["cfg"]))
    params["sampler_name"] = str(inputs.get("sampler_name", params["sampler_name"]))
    params["scheduler"] = str(inputs.get("scheduler", params["scheduler"]))
    params["denoise"] = float(inputs.get("denoise", params["denoise"]))
    
    # Trace positive and negative prompts
    params["positive_prompt"] = trace_conditioning(inputs.get("positive"), nodes)
    params["negative_prompt"] = trace_conditioning(inputs.get("negative"), nodes)
    
    # Trace latent image
    latent_link = inputs.get("latent_image")
    if isinstance(latent_link, list) and len(latent_link) > 0:
        latent_node = nodes.get(str(latent_link[0]))
        if latent_node and latent_node.get("class_type") == "EmptyLatentImage":
            params["width"] = int(latent_node["inputs"].get("width", 512))
            params["height"] = int(latent_node["inputs"].get("height", 512))
            params["batch_size"] = int(latent_node["inputs"].get("batch_size", 1))
            
    # Trace model / checkpoint & lora nodes
    # We trace model inputs recursively to collect LoRAs and find the CheckpointLoaderSimple
    def trace_model(model_link):
        if not isinstance(model_link, list) or len(model_link) < 1:
            return
        node_id = str(model_link[0])
        node = nodes.get(node_id)
        if not node:
            return
            
        ctype = node.get("class_type")
        if ctype == "CheckpointLoaderSimple":
            params["ckpt_name"] = node["inputs"].get("ckpt_name")
        elif ctype == "LoraLoader":
            params["loras"].append({
                "lora_name": node["inputs"].get("lora_name"),
                "strength_model": float(node["inputs"].get("strength_model", 1.0))
            })
            trace_model(node["inputs"].get("model"))
            
    trace_model(inputs.get("model"))
    
    # Fallback to scan if recursive model trace failed to resolve checkpoint
    if not params["ckpt_name"]:
        for nid, node in nodes.items():
            if node.get("class_type") == "CheckpointLoaderSimple":
                params["ckpt_name"] = node["inputs"].get("ckpt_name")
                break
                
    # Parse SaveImage node for output prefix
    save_nodes = [node for node in nodes.values() if node.get("class_type") == "SaveImage"]
    if save_nodes:
        params["output_prefix"] = save_nodes[0].get("inputs", {}).get("filename_prefix", "ComfyUI")
        
    return params


def run_workflow(workflow_data, output_path=None, overrides=None, device=None):
    """
    Runs the parsed ComfyUI workflow using Diffusers and returns the generated image.
    """
    # 1. Parse JSON
    nodes = parse_workflow(workflow_data)
    
    # 2. Extract configuration parameters
    params = extract_parameters(nodes)
    print("\n--- Parsed ComfyUI Parameters ---")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("---------------------------------\n")
    
    # Apply overrides if specified
    if overrides:
        for k, v in overrides.items():
            if k in params:
                print(f"[Override] {k} -> {v}")
                params[k] = v
                
    # Detect device (check MPS for Apple Silicon, then CUDA, then CPU)
    if not device:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"[Device] Running on: {device}")
    
    # Apply CPU-specific thread configuration early to reduce context switching overhead
    if device == "cpu":
        print("[Optimizer] Configuring torch threads for CPU inference...")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
    # Resolve roots
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(backend_dir)
    
    dtype = torch.float32 if device in ["cpu", "mps"] else torch.float16
    
    # Load CLIPTextModel first to solve single checkpoint loading bugs offline/online
    text_encoder = None
    try:
        print("[Loader] Attempting to load CLIPTextModel from local cache first...")
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=dtype,
            local_files_only=True
        )
    except Exception as e:
        print(f"[Loader] Local CLIPTextModel load failed, trying online: {e}")
        try:
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=dtype,
                local_files_only=False
            )
        except Exception as online_e:
            print(f"[Warning] Failed to load CLIPTextModel explicitly: {online_e}")
            
    # Load SD 1.5 single file checkpoint
    ckpt_path = resolve_model_path(params["ckpt_name"], root_dir)
    print(f"[Loader] Loading single-file model checkpoint from: {ckpt_path}")
    
    pipe = None
    if os.path.exists(ckpt_path):
        try:
            load_kwargs = {
                "use_safetensors": True,
                "torch_dtype": dtype,
            }
            if text_encoder is not None:
                load_kwargs["text_encoder"] = text_encoder
                
            config_file = os.path.join(backend_dir, "v1-inference.yaml")
            if os.path.exists(config_file):
                load_kwargs["original_config_file"] = config_file
                load_kwargs["config"] = "runwayml/stable-diffusion-v1-5"
                
            pipe = StableDiffusionPipeline.from_single_file(ckpt_path, **load_kwargs)
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint {ckpt_path}: {e}")
            
        hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        # Check both direct subfolder and hub subfolder formats
        dreamshaper_local = os.path.join(
            hf_home,
            "models--Lykon--dreamshaper-8",
            "snapshots",
            "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917"
        )
        if not os.path.exists(dreamshaper_local):
            dreamshaper_local = os.path.join(
                hf_home,
                "hub",
                "models--Lykon--dreamshaper-8",
                "snapshots",
                "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917"
            )
        if os.path.exists(dreamshaper_local):
            print(f"[Loader] Detected fully cached local Dreamshaper model at {dreamshaper_local}. Loading offline...")
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    dreamshaper_local,
                    torch_dtype=dtype,
                    safety_checker=None,
                    local_files_only=True
                )
            except Exception as e:
                print(f"[Warning] Failed to load Dreamshaper from local path: {e}")
                
    if pipe is None:
        print("[Loader] Falling back to standard public model runwayml/stable-diffusion-v1-5...")
        try:
            print("[Loader] Attempting to load runwayml/stable-diffusion-v1-5 from local cache...")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,
                local_files_only=True
            )
        except Exception as cache_e:
            print(f"[Loader] Local runwayml/stable-diffusion-v1-5 load failed, trying online: {cache_e}")
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=dtype,
                    safety_checker=None,
                    local_files_only=False
                )
            except Exception as fallback_e:
                print(f"[Error] Fallback loading failed: {fallback_e}")
                sys.exit(1)
            
    # Load LoRAs if specified
    lora_scale = 1.0
    for lora in params["loras"]:
        lora_path = resolve_model_path(lora["lora_name"], root_dir)
        if os.path.exists(lora_path):
            print(f"[Loader] Loading LoRA weights: {lora_path}")
            try:
                pipe.load_lora_weights(lora_path)
                # Keep scale of first/main LoRA
                lora_scale = lora["strength_model"]
            except Exception as e:
                print(f"[Warning] Failed to load LoRA weights: {e}")
        else:
            print(f"[Loader] Warning: LoRA weights file not found at: {lora_path}")
            
    # Map and set scheduler
    scheduler_class, scheduler_kwargs = map_comfy_sampler_to_diffusers(
        params["sampler_name"], params["scheduler"]
    )
    print(f"[Scheduler] Instantiating {scheduler_class.__name__} with config {scheduler_kwargs}")
    
    # Clean config keys that might conflict with the target scheduler class
    clean_config = dict(pipe.scheduler.config)
    if scheduler_class.__name__ == "DPMSolverMultistepScheduler":
        algo = clean_config.get("algorithm_type")
        if algo not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            clean_config.pop("algorithm_type", None)
            clean_config.pop("final_sigmas_type", None)
    pipe.scheduler = scheduler_class.from_config(clean_config, **scheduler_kwargs)
    
    # Send pipeline to target device
    pipe = pipe.to(device)
    
    # ------------------ Apply CPU & iGPU Optimizations ------------------
    if device == "cpu":
        print("[Optimizer] Applying memory layout channels_last for speedups on CPU...")
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        
    print("[Optimizer] Enabling attention slicing ('auto')...")
    pipe.enable_attention_slicing("auto")
    
    # Determine autocast capability
    # Check if CPU/Platform supports bfloat16 autocast safely (avoid float16 on CPU as it is emulated and slow)
    is_bf16_supported = False
    if device == "cpu":
        try:
            test_tensor = torch.zeros((1, 1), dtype=torch.bfloat16)
            _ = torch.matmul(test_tensor, test_tensor)
            is_bf16_supported = True
            print("[Optimizer] CPU supports bfloat16. Applying bfloat16 autocast context.")
        except Exception:
            print("[Optimizer] CPU bfloat16 execution is unsupported or unsafe. Running in float32.")
            
    # Autocast context setup
    if device == "cpu" and is_bf16_supported:
        autocast_ctx = torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)
    elif device == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
    elif device == "mps":
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = nullcontext()
        
    # Generate random latents on CPU (ensuring pin_memory=False to save CPU memory allocations)
    print(f"[Latents] Generating random latents on CPU with seed {params['seed']} (pin_memory=False)...")
    generator = torch.Generator(device="cpu").manual_seed(params["seed"])
    
    # Latent shape for SD 1.5: batch_size, 4, height//8, width//8
    latent_shape = (params["batch_size"], 4, params["height"] // 8, params["width"] // 8)
    latents = torch.randn(
        latent_shape, 
        generator=generator, 
        device="cpu", 
        dtype=torch.float32, 
        pin_memory=False
    )
    
    # Move latents to device if not CPU
    if device != "cpu":
        latents = latents.to(device, dtype=dtype)
        
    # Setup cross attention scale for LoRA
    cross_attention_kwargs = {}
    if params["loras"]:
        cross_attention_kwargs["scale"] = lora_scale
        print(f"[Inference] Injecting LoRA scale: {lora_scale}")
        
    # Run pipeline inside the optimized autocast context
    print(f"[Inference] Executing pipeline. Steps: {params['steps']}, Guidance: {params['cfg']}")
    start_time = time.time()
    with autocast_ctx:
        pipeline_output = pipe(
            prompt=params["positive_prompt"],
            negative_prompt=params["negative_prompt"],
            num_inference_steps=params["steps"],
            guidance_scale=params["cfg"],
            latents=latents,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs
        )
    elapsed = time.time() - start_time
    print(f"[Inference] Finished in {elapsed:.2f} seconds.")
    
    image = pipeline_output.images[0]
    
    # Save image securely
    try:
        if output_path:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            print(f"[Output] Saving resulting image to: {output_path}")
            image.save(output_path)
        else:
            default_out = f"{params['output_prefix']}_output.png"
            print(f"[Output] No output path specified, saving to default: {default_out}")
            image.save(default_out)
    except Exception as e:
        print(f"[Error] Failed to write generated image to disk: {e}")
        
    return image


def main():
    parser = argparse.ArgumentParser(description="Run standard ComfyUI API workflows using Hugging Face Diffusers.")
    parser.add_argument("--workflow", type=str, required=True, help="Path to ComfyUI API JSON workflow file.")
    parser.add_argument("--output", type=str, default=None, help="Output image file path.")
    parser.add_argument("--prompt", type=str, default=None, help="Override positive prompt.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Override negative prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--steps", type=int, default=None, help="Override steps.")
    parser.add_argument("--cfg", type=float, default=None, help="Override cfg.")
    parser.add_argument("--width", type=int, default=None, help="Override width.")
    parser.add_argument("--height", type=int, default=None, help="Override height.")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu, cuda, mps).")
    
    args = parser.parse_args()
    
    overrides = {}
    if args.prompt is not None:
        overrides["positive_prompt"] = args.prompt
    if args.negative_prompt is not None:
        overrides["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.cfg is not None:
        overrides["cfg"] = args.cfg
    if args.width is not None:
        overrides["width"] = args.width
    if args.height is not None:
        overrides["height"] = args.height
        
    run_workflow(
        args.workflow, 
        output_path=args.output, 
        overrides=overrides, 
        device=args.device
    )


if __name__ == "__main__":
    main()
