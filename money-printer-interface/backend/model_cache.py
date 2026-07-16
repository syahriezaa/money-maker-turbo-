"""
model_cache.py - Persistent In-Process Image Pipeline Cache Manager

Keeps the StableDiffusionPipeline resident in VRAM/RAM across scene generations
within a single video task so the model is only loaded from disk once.

Strategy: Lazy Load + Idle Timeout Release
  - Pipeline is NOT loaded at server startup.
  - Loaded on first image generation request.
  - Stays resident for all subsequent generations (same checkpoint + style).
  - An idle watcher thread releases VRAM after IDLE_TIMEOUT_SECONDS of inactivity.
"""
import gc
import os
import time
import threading
from contextlib import nullcontext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IDLE_TIMEOUT_SECONDS = 300   # 5 minutes
WATCHER_POLL_SECONDS = 30    # check every 30 s


# ---------------------------------------------------------------------------
# Singleton implementation
# ---------------------------------------------------------------------------
class ModelCacheManager:
    """
    Thread-safe singleton that manages a single StableDiffusionPipeline instance.
    """

    _instance = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Singleton access                                                     #
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #
    def __init__(self):
        self._pipe = None
        self._lora_scale = 1.0
        self._autocast_ctx = None
        self._loaded_key = None          # "<ckpt_name>|<style>|<device>"
        self._last_used = 0.0
        self._pipeline_lock = threading.Lock()

        # Start idle watcher daemon thread
        t = threading.Thread(target=self._idle_watcher, daemon=True, name="ModelCacheWatcher")
        t.start()
        print("[ModelCache] Idle watcher thread started (timeout=%ds)." % IDLE_TIMEOUT_SECONDS)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def get_pipeline(self, ckpt_name, style, device, root_dir):
        """
        Return a resident StableDiffusionPipeline.

        If the requested checkpoint/style/device matches the cached pipeline it
        is returned immediately (sub-millisecond).  Otherwise the old pipeline
        is offloaded and a new one is loaded from disk.

        Returns:
            tuple: (pipeline, lora_scale: float, autocast_ctx)
        """
        cache_key = "%s|%s|%s" % (ckpt_name, style, device)

        with self._pipeline_lock:
            if self._pipe is not None and self._loaded_key == cache_key:
                self._touch()
                print("[ModelCache] Cache HIT - returning resident pipeline (%s)" % cache_key)
                return self._pipe, self._lora_scale, self._autocast_ctx

            # Need to (re-)load
            if self._pipe is not None:
                print("[ModelCache] Cache SWAP - offloading '%s' -> loading '%s'" % (self._loaded_key, cache_key))
                self._release_locked()
            else:
                print("[ModelCache] Cache MISS - loading pipeline: %s" % cache_key)

            pipe, lora_scale, autocast_ctx = self._load_pipeline(ckpt_name, style, device, root_dir)
            self._pipe = pipe
            self._lora_scale = lora_scale
            self._autocast_ctx = autocast_ctx
            self._loaded_key = cache_key
            self._touch()
            return pipe, lora_scale, autocast_ctx

    def release(self):
        """Explicitly release the pipeline and free VRAM."""
        with self._pipeline_lock:
            self._release_locked()

    @property
    def is_loaded(self):
        return self._pipe is not None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _touch(self):
        self._last_used = time.monotonic()

    def _release_locked(self):
        """Must be called while holding _pipeline_lock."""
        if self._pipe is None:
            return
        try:
            import torch
            try:
                self._pipe.to("cpu")
            except Exception:
                pass
            del self._pipe
            self._pipe = None
            self._loaded_key = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
                print("[ModelCache] VRAM cache cleared (torch.cuda.empty_cache).")
            except Exception:
                pass
            print("[ModelCache] Pipeline offloaded and memory freed.")
        except Exception as e:
            print("[ModelCache] Warning during release: %s" % e)
            self._pipe = None
            self._loaded_key = None

    def _idle_watcher(self):
        """Background daemon: offload pipeline after IDLE_TIMEOUT_SECONDS of inactivity."""
        while True:
            time.sleep(WATCHER_POLL_SECONDS)
            try:
                with self._pipeline_lock:
                    if self._pipe is None:
                        continue
                    idle_secs = time.monotonic() - self._last_used
                    if idle_secs >= IDLE_TIMEOUT_SECONDS:
                        print(
                            "[ModelCache] Idle timeout (%.0fs >= %ds). Releasing pipeline to free VRAM..."
                            % (idle_secs, IDLE_TIMEOUT_SECONDS)
                        )
                        self._release_locked()
            except Exception as e:
                print("[ModelCache] Watcher error: %s" % e)

    # ------------------------------------------------------------------ #
    # Pipeline loader                                                      #
    # ------------------------------------------------------------------ #
    def _load_pipeline(self, ckpt_name, style, device, root_dir):
        """
        Load a StableDiffusionPipeline from disk.
        LoRA (AnimaMythP0rtr4itStyleV1) is ONLY applied for anime style to prevent
        the portrait bias from bleeding into story-relevant realistic/gtav scenes.

        Returns:
            tuple: (pipe, lora_scale, autocast_ctx)
        """
        import torch
        from diffusers import (
            StableDiffusionPipeline,
            DPMSolverMultistepScheduler,
        )
        from transformers import CLIPTextModel

        backend_dir_abs = os.path.join(root_dir, "backend")
        dtype = torch.float32 if device in ["cpu", "mps"] else torch.float16

        # ------------------------------------------------------------------
        # 1. CLIPTextModel
        # ------------------------------------------------------------------
        text_encoder = None
        try:
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14", torch_dtype=dtype, local_files_only=True
            )
        except Exception:
            try:
                text_encoder = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14", torch_dtype=dtype, local_files_only=False
                )
            except Exception as e:
                print("[ModelCache] Warning: CLIPTextModel load failed: %s" % e)

        # ------------------------------------------------------------------
        # 2. Checkpoint path
        # ------------------------------------------------------------------
        ckpt_path = self._resolve_model_path(ckpt_name, root_dir)
        print("[ModelCache] Checkpoint path: %s" % ckpt_path)

        # ------------------------------------------------------------------
        # 3. Load pipeline
        # ------------------------------------------------------------------
        pipe = None
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                load_kwargs = {"use_safetensors": True, "torch_dtype": dtype}
                if text_encoder is not None:
                    load_kwargs["text_encoder"] = text_encoder
                config_file = os.path.join(backend_dir_abs, "v1-inference.yaml")
                if os.path.exists(config_file):
                    load_kwargs["original_config_file"] = config_file
                    load_kwargs["config"] = "runwayml/stable-diffusion-v1-5"
                pipe = StableDiffusionPipeline.from_single_file(ckpt_path, **load_kwargs)
                print("[ModelCache] Loaded single-file checkpoint.")
            except Exception as e:
                print("[ModelCache] Single-file load failed: %s. Trying HF pretrained." % e)

        if pipe is None:
            hf_model = self._hf_model_for_ckpt(ckpt_name)
            print("[ModelCache] Loading HF pretrained: %s" % hf_model)
            load_kwargs = {"torch_dtype": dtype, "safety_checker": None}
            if text_encoder is not None:
                load_kwargs["text_encoder"] = text_encoder
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    hf_model, local_files_only=True, **load_kwargs
                )
            except Exception:
                pipe = StableDiffusionPipeline.from_pretrained(
                    hf_model, local_files_only=False, **load_kwargs
                )

        # ------------------------------------------------------------------
        # 4. LoRA - ONLY for anime style
        # ------------------------------------------------------------------
        lora_scale = 1.0
        if style == "anime":
            lora_name = "AnimaMythP0rtr4itStyleV1.safetensors"
            lora_path = self._resolve_model_path(lora_name, root_dir)
            if lora_path and os.path.exists(lora_path):
                try:
                    pipe.load_lora_weights(lora_path)
                    print("[ModelCache] Loaded anime LoRA: %s" % lora_path)
                except Exception as e:
                    print("[ModelCache] Warning: LoRA load failed: %s" % e)
            else:
                print("[ModelCache] LoRA file not found at %s - skipping." % lora_path)
        else:
            print("[ModelCache] Style '%s': portrait LoRA skipped (anime-only)." % style)

        # ------------------------------------------------------------------
        # 5. Scheduler: DPM++ 2M Karras
        # ------------------------------------------------------------------
        clean_config = dict(pipe.scheduler.config)
        algo = clean_config.get("algorithm_type", "")
        if algo not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            clean_config.pop("algorithm_type", None)
            clean_config.pop("final_sigmas_type", None)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            clean_config, use_karras_sigmas=True
        )

        # ------------------------------------------------------------------
        # 6. Device + optimizations
        # ------------------------------------------------------------------
        pipe = pipe.to(device)
        pipe.enable_attention_slicing("auto")
        if device == "cpu":
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.vae.to(memory_format=torch.channels_last)
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        # ------------------------------------------------------------------
        # 7. Autocast context
        # ------------------------------------------------------------------
        is_bf16 = False
        if device == "cpu":
            try:
                test_t = torch.zeros((1, 1), dtype=torch.bfloat16)
                _ = torch.matmul(test_t, test_t)
                is_bf16 = True
            except Exception:
                pass

        if device == "cpu" and is_bf16:
            autocast_ctx = torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16)
        elif device == "cuda":
            autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        else:
            autocast_ctx = nullcontext()

        print("[ModelCache] Pipeline ready on %s." % device)
        return pipe, lora_scale, autocast_ctx

    # ------------------------------------------------------------------ #
    # Static helpers                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_model_path(name, root_dir):
        if not name:
            return None
        safe_name = os.path.basename(name)
        if os.path.isabs(name) and os.path.exists(name):
            return name
        subdirs = ["", "models/checkpoints", "models/loras", "models/vae", "backend"]
        for sub in subdirs:
            path = os.path.normpath(os.path.join(root_dir, sub, safe_name))
            if os.path.exists(path):
                return path
        return os.path.normpath(os.path.join(root_dir, safe_name))

    @staticmethod
    def _hf_model_for_ckpt(ckpt_name):
        lower = (ckpt_name or "").lower()
        if "anything" in lower:
            return "stablediffusionapi/anything-v5"
        if "dreamshaper" in lower:
            return "Lykon/DreamShaper"
        if "disney-pixar" in lower or "pixar" in lower:
            return "stablediffusionapi/disney-pixar-cartoon"
        return "ItsJayQz/GTA5_Artwork_Diffusion"


# ---------------------------------------------------------------------------
# Module-level convenience accessor
# ---------------------------------------------------------------------------
def get_cache():
    """Return the global ModelCacheManager singleton."""
    return ModelCacheManager.instance()
