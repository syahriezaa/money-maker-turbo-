import time
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# List of natural sentences of varying lengths to cycle through
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we work, live, and communicate with each other.",
    "Chatterbox Turbo is a state-of-the-art text-to-speech model optimized for low-latency real-time voice agents.",
    "Deep learning has made incredible progress in speech synthesis over the last few years.",
    "To measure the speed of audio generation, we run multiple sentences and calculate the real-time factor.",
    "Every sentence is processed individually and then concatenated to form the final audio track.",
    "It is important to split long texts into smaller chunks because the model has a limited context window.",
    "We can use paralinguistic tags like [laugh] or [chuckle] to make the generated voice sound more natural and expressive.",
    "The RTX thirty fifty laptop GPU has four gigabytes of video memory, which is sufficient for running this model.",
    "Python is the programming language of choice for machine learning and artificial intelligence applications.",
    "When evaluating text-to-speech systems, we look at both the quality of the audio and the speed of synthesis.",
    "A lower real-time factor means the model can generate audio much faster than the time it takes to speak it.",
    "In this benchmark, we aim to generate exactly ten minutes of high-fidelity audio.",
    "CUDA acceleration allows us to run neural network inference in parallel, dramatically reducing the computation time.",
    "The model converts input text into phonemes, then to speech tokens, then to mel spectrograms, and finally to audio waveforms.",
    "We are using PyTorch version two point six point zero with CUDA twelve point four support on Windows.",
    "Natural language processing helps the system understand the structure and meaning of sentences before generating speech.",
    "Each sentence should be pronounced clearly with proper intonation, stress, and rhythm.",
    "Let's see how long it takes to generate this entire audio stream from start to finish.",
    "The results of this benchmark will help us determine if the model is suitable for production use cases."
]

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Chatterbox Turbo 10-Minute Audio Benchmark")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"==================================================")

    # 1. Load the model
    print("Loading model...")
    t_start_load = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    load_duration = time.time() - t_start_load
    print(f"Model loaded in {load_duration:.2f} seconds.")

    # Reset peak memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 2. Warm-up
    print("\nPerforming warm-up run...")
    warmup_text = "Warm-up run to compile shaders and load CUDA kernels."
    _ = model.generate(warmup_text)
    print("Warm-up complete.")

    # 3. Main generation loop
    target_seconds = 600.0  # 10 minutes
    total_audio_duration = 0.0
    total_generation_time = 0.0
    
    generated_wavs = []
    chunk_index = 0
    sentence_index = 0
    
    print(f"\nStarting benchmark to generate at least {target_seconds} seconds (10 minutes) of audio...")
    print(f"{'Chunk':<6} | {'Length (chars)':<15} | {'Audio Dur (s)':<13} | {'Gen Time (s)':<12} | {'RTF':<8}")
    print("-" * 65)

    benchmark_start_time = time.time()

    while total_audio_duration < target_seconds:
        text = SENTENCES[sentence_index]
        sentence_index = (sentence_index + 1) % len(SENTENCES)
        chunk_index += 1
        
        t0 = time.perf_counter()
        wav = model.generate(text)
        t_gen = time.perf_counter() - t0
        
        # Move to CPU immediately to free VRAM
        wav_cpu = wav.cpu()
        generated_wavs.append(wav_cpu)
        
        dur = wav_cpu.shape[-1] / model.sr
        rtf = t_gen / dur
        
        total_audio_duration += dur
        total_generation_time += t_gen
        
        print(f"{chunk_index:<6} | {len(text):<15} | {dur:<13.2f} | {t_gen:<12.2f} | {rtf:<8.3f}")
        
        # Empty GPU cache to prevent memory accumulation
        if device == "cuda":
            torch.cuda.empty_cache()

    total_benchmark_wall_time = time.time() - benchmark_start_time
    
    print("\n" + "=" * 50)
    print("Benchmark Completed!")
    print("=" * 50)
    
    # 4. Concatenate and save the audio
    print("Concatenating audio chunks...")
    final_wav = torch.cat(generated_wavs, dim=-1)
    
    output_filename = "chatterbox_turbo_10min.wav"
    print(f"Saving final audio to {output_filename}...")
    try:
        import soundfile as sf
        audio_data = final_wav.numpy()
        if len(audio_data.shape) > 1 and audio_data.shape[0] == 1:
            audio_data = audio_data[0]
        sf.write(output_filename, audio_data, model.sr)
    except Exception as sf_e:
        print(f"soundfile write failed: {sf_e}. Falling back to torchaudio...")
        ta.save(output_filename, final_wav, model.sr)
    
    # Calculate stats
    avg_rtf = total_generation_time / total_audio_duration
    speed_factor = 1.0 / avg_rtf
    
    print("\nMetrics:")
    print(f"  - Total Generated Audio: {total_audio_duration:.2f} seconds ({total_audio_duration/60.0:.2f} minutes)")
    print(f"  - Total Pure Inference Time: {total_generation_time:.2f} seconds ({total_generation_time/60.0:.2f} minutes)")
    print(f"  - Total Wall-Clock Benchmark Time (including disk save, overhead): {total_benchmark_wall_time:.2f} seconds")
    print(f"  - Average Real-Time Factor (RTF): {avg_rtf:.4f}")
    print(f"  - Generation Speed: {speed_factor:.2f}x real-time (1 second of compute generates {speed_factor:.2f} seconds of audio)")
    
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  - Peak GPU VRAM allocated: {peak_vram:.2f} MB")
        
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()
