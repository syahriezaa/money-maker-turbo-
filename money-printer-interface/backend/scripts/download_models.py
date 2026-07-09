import os
import sys
import time
import torch
from diffusers import StableDiffusionPipeline

MODELS = {
    "GTA V Cartoon (Rotoscoped)": "ItsJayQz/GTA5_Artwork_Diffusion",
    "Flat Anime 2D": "stablediffusionapi/anything-v5",
    "Cinematic Realistic": "Lykon/DreamShaper",
    "3D Pixar Cartoon": "stablediffusionapi/disney-pixar-cartoon"
}

def main():
    print("=" * 60)
    print("      MONEYPRINTERTURBO - MODEL PRE-DOWNLOAD UTILITY      ")
    print("=" * 60)
    print("Script ini akan mengunduh model visual yang diperlukan agar generator")
    print("dapat berjalan 100% offline secara optimal.")
    print("-" * 60)
    
    total_models = len(MODELS)
    for index, (name, repo_id) in enumerate(MODELS.items(), 1):
        print(f"\n[{index}/{total_models}] Memulai pengunduhan model: {name}")
        print(f"Hugging Face Repo ID: {repo_id}")
        print("Mengunduh ke cache lokal...")
        
        start_time = time.time()
        try:
            # Load pipeline on CPU to trigger download and caching
            pipe = StableDiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float32,
                safety_checker=None,
                local_files_only=False
            )
            elapsed = time.time() - start_time
            print(f"✓ Sukses mengunduh {name} dalam {elapsed:.2f} detik!")
            
            # Free memory immediately
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"✗ Gagal mengunduh {name}: {e}")
            print("Silakan periksa koneksi internet Anda atau coba lagi nanti.")
            
    print("\n" + "=" * 60)
    print(" PROSES UNDUH SELESAI!")
    print("Semua model visual sekarang tersimpan di cache lokal Anda.")
    print("=" * 60)

if __name__ == "__main__":
    main()
