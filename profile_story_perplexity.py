import time
import csv
import math
import os
import numpy as np
import moonshine_onnx as moonshine
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from refiner import RealLocalLLM
from pydub import AudioSegment
from pydub.silence import split_on_silence

print("[Setup] Loading Qwen2.5 4-bit LLM for semantic refinement...")
llm_engine = RealLocalLLM(model_path="./models/qwen2.5-1.5b-instruct-q5_k_m.gguf")

print("[Setup] Initializing Stable Edge Perplexity Gate (Public Tiny-GPT2 Engine)...")
MODEL_ID = "sshleifer/tiny-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
ppl_model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
ppl_model.eval()

@torch.no_grad()
def evaluate_perplexity_gate(raw_text):
    text_clean = raw_text.strip()
    words = text_clean.split()
    
    if len(words) == 0:
        return False, 0.0
    if len(words) <= 2:
        return True, 999.0
        
    inputs = tokenizer(text_clean, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = ppl_model(**inputs, labels=input_ids)
    loss = outputs.loss.item()
    perplexity = math.exp(loss)
    
    PERPLEXITY_THRESHOLD = 300.0
    should_refine = perplexity > PERPLEXITY_THRESHOLD
    return should_refine, perplexity

# ========================================================
# AUTOMATED NARRATIVE EVALUATION EXPERIMENT
# ========================================================
def evaluate_story_pipeline(audio_path):
    csv_file = "perplexity_gate_results_compare.csv"
    fields = ["Sentence_Index", "ASR_Text", "Refined_Text", "Perplexity_Score", ...]
    
    if not os.path.exists(audio_path):
        print(f"[Error] Audio asset missing at target path: {audio_path}")
        return
        
    print(f"\n[Processing Engine] Ingesting narrative track: {audio_path}")
    sound = AudioSegment.from_wav(audio_path)
    
    print("[Processing Engine] Slicing track into segments using silence boundaries...")
    chunks = split_on_silence(sound, min_silence_len=600, silence_thresh=sound.dBFS-14, keep_silence=200)
    
    if not chunks:
        print("[Error] No valid audio segments could be parsed from the file.")
        return
        
    print(f" -> Successfully segmented long story track into {len(chunks)} short processing units.")
    print("\n[Running Simulation] Commencing Perplexity Filter vs Always-On Benchmark...")
    print("--------------------------------------------------------------------------------")
    
    # PERFORMANCE TRACKING METRICS
    total_saved_cpu_time = 0.0
    total_gated_triggers = 0
    total_asr_time = 0.0          # Tracks pure Moonshine execution time
    total_pipeline_time = 0.0     # Tracks actual total time under our Gated system
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        
        for idx, chunk in enumerate(chunks, 1):
            chunk = chunk.set_frame_rate(16000).set_channels(1)
            audio_samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
            
            if len(audio_samples) < 1600:
                continue
                
            # 1. PROFILE FAST PATH (Acoustic ASR Layer)
            t_start = time.time()
            segments = moonshine.transcribe(audio_samples, "moonshine/tiny")
            asr_text = segments[0].strip() if segments else ""
            t_asr = time.time() - t_start
            
            if not asr_text:
                continue
                
            # Accumulate standalone ASR processing time
            total_asr_time += t_asr

            # --- NEW INTERMEDIATE PRINT ---
            print(f"\nSegment #{idx:02d}")
            print(f"  [Immediate Transcript]: \"{asr_text}\"")
            
            # 2. EXECUTE THE OPEN-DOMAIN LINGUISTIC GATE
            t_gate_start = time.time()
            should_refine, ppl_score = evaluate_perplexity_gate(asr_text)
            t_gate = time.time() - t_gate_start
            
            # Profile Cascade Gated Pathway Compute Cost
            if should_refine:
                total_gated_triggers += 1
                t_llm_start = time.time()
                _ = llm_engine.generate(f"Fix ASR: {asr_text}")
                t_llm = time.time() - t_llm_start
                gated_compute = t_asr + t_gate + t_llm
            else:
                gated_compute = t_asr + t_gate  # Heavy LLM completely bypassed!
                
            # Accumulate the actual time our gating framework took to process
            total_pipeline_time += gated_compute
                
            # 3. PROFILE ALWAYS-ON LLM BASELINE (For computing savings delta)
            t_always_start = time.time()
            refined_text = llm_engine.generate(f"Fix ASR errors and return clean text: {asr_text}")
            refined_text = refined_text.strip() if refined_text else asr_text
            always_on_compute = t_asr + (time.time() - t_always_start)
            
            # Compute Absolute Saved Time
            compute_saved = always_on_compute - gated_compute
            total_saved_cpu_time += compute_saved
            
            # Log metrics to CSV
            writer.writerow([idx, asr_text, refined_text, f"{ppl_score:.2f}", ...])
            
            # SHOW TRANSCRIPT LIVE IN TERMINAL WITH RAW ASR TIME INCLUDED

            if should_refine:
                print(f"  [Refined Transcript] : \"{refined_text}\"")
            else:
                print(f"  [Refined Transcript] : (Skipped - Text accepted as clean)")
            print(f"  [Metrics Summary]    : ASR Time: {t_asr:.3f}s | Gate PPL: {ppl_score:.1f} | Gated Triggered: {should_refine} | Time Saved: {compute_saved:.3f}s")
            print("-" * 80)

    # Calculate what the total time would look like if the LLM was always on
    projected_always_on_time = total_pipeline_time + total_saved_cpu_time

    print(f"\n=======================================================")
    print(f"EXPERIMENT COMPLETE: Matrix exported to '{csv_file}'")
    print(f"Total Sentences Decoded: {len(chunks)}")
    print(f"Number of times heavy LLM was triggered: {total_gated_triggers}")
    print(f"-------------------------------------------------------")
    print(f"[Performance Profile] Pure Moonshine Tiny ASR : {total_asr_time:.2f} seconds.")
    print(f"[Performance Profile] Our Cascade Gated System: {total_pipeline_time:.2f} seconds.")
    print(f"[Performance Profile] Baseline Always-On LLM  : {projected_always_on_time:.2f} seconds.")
    print(f"-------------------------------------------------------")
    print(f"Total CPU processing time saved by Gate: {total_saved_cpu_time:.2f} seconds!")
    print(f"=======================================================")

if __name__ == "__main__":
    evaluate_story_pipeline("audio/woodchuck.wav")

    if 'llm_engine' in locals() or 'llm_engine' in globals():
        try:
            if hasattr(llm_engine, 'llm'):
                del llm_engine.llm
        except Exception:
            pass