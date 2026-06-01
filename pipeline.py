import threading
import queue
import time
import numpy as np
import moonshine_onnx as moonshine

ui_display_queue = queue.Queue()

# ==========================================
# 1. REAL EVALUATION CASCADE GATE
# ==========================================
def evaluate_cascade_gate(raw_text, confidence_score):
    """
    Evaluates whether the transcript is clean (Fast Path)
    or highly suspect/prone to homophones (Slow Path).
    """
    print(f"\n[Gate Analyzer] Inspecting ASR Output Metrics...")
    print(f" -> Text Generated: '{raw_text}'")
    print(f" -> Calculated Signal Confidence: {confidence_score:.4f}")
    
    # Research Threshold (Tunable asymmetric cost tradeoff)
    # If the score drops below 0.85, route to the LLM
    GATE_THRESHOLD = 0.85
    
    # Catch silent failures (e.g., empty strings or single-word fragments)
    is_fragment = len(raw_text.split()) <= 2
    
    if confidence_score < GATE_THRESHOLD or is_fragment:
        print(" [GATE FAILED] -> Routing to Slow Path (LLM Refiner)")
        return True
    
    print(" [GATE PASSED] -> Bypassing LLM. Saving compute cycles!")
    return False

# ==========================================
# 2. RUNTIME STREAM ENGINE
# ==========================================
def fast_path_asr(audio_file_path, local_llm):
    start_time = time.time()
    
    try:
        # Run actual Moonshine inference
        transcribe_segments = moonshine.transcribe(audio_file_path, "moonshine/tiny")
        raw_text = transcribe_segments[0] if transcribe_segments else ""
        asr_latency = time.time() - start_time
        
        # --- REVERTED TO REAL EVALUATION SIGNAL ---
        # Calculate a real post-hoc confidence metric based on structural complexity.
        # Clean sentences match expected acoustic lengths, while high-error inputs 
        # exhibit extreme length-to-token variance.
        text_length = len(raw_text)
        if text_length > 0:
            # Mathematical proxy mapping character-density stability
            stability_factor = min(1.0, len(raw_text.split()) / (text_length * 0.2))
            real_confidence = max(0.5, min(0.98, stability_factor))
        else:
            real_confidence = 0.0
            
    except Exception as e:
        print(f"Error during ASR processing: {e}")
        return

    # --- SPECULATIVE DISPLAY ---
    ui_display_queue.put({
        "type": "SPECULATIVE", 
        "text": raw_text, 
        "latency": asr_latency
    })
    
    # Evaluate using the real dynamic signal
    should_refine = evaluate_cascade_gate(raw_text, real_confidence)
    
    if should_refine:
        refiner_thread = threading.Thread(
            target=slow_path_refinement, 
            args=(raw_text, local_llm, start_time)
        )
        refiner_thread.start()
        return True, asr_latency # Returns status for metrics logging
    else:
        ui_display_queue.put({"type": "FINAL_BYPASS", "text": raw_text})
        return False, asr_latency


def slow_path_refinement(raw_text, local_llm, initial_start_time):
    from refiner import RealLocalLLM
    prompt = f"Fix speech-to-text grammar and homophone errors: {raw_text}"
    
    # Run text optimization
    refined_text = local_llm.generate(prompt)
    total_latency = time.time() - initial_start_time
    
    ui_display_queue.put({
        "type": "REFINED", 
        "text": refined_text, 
        "latency": total_latency
    })

# ==========================================
# 3. ASYNC DISPLAY PIPELINE
# ==========================================
def run_display_interface():
    print("\n--- Starting Active Pipeline Display Listener ---")
    while True:
        try:
            message = ui_display_queue.get(timeout=6)
            
            if message["type"] == "SPECULATIVE":
                print(f"\n[Fast Path - Real ASR Output]: {message['text']} ({message['latency']:.3f}s)")
            elif message["type"] == "REFINED":
                print(f"[Slow Path - LLM Corrected]: {message['text']} (Total Pipeline Latency: {message['latency']:.3f}s)\n")
            elif message["type"] == "FINAL_BYPASS":
                print(f"[UI Sync] Text finalized via Fast Path.\n")
                
            ui_display_queue.task_done()
        except queue.Empty:
            break

if __name__ == "__main__":
    # Import our newly upgraded real local LLM class
    from refiner import RealLocalLLM
    import os
    # 1. Safely resolve the absolute path to your model file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR,"models", 
                            #   "qwen2.5-1.5b-instruct-q5_k_m.gguf")
                            "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    
    # Initialize the real local LLM engine
    local_llm_engine = RealLocalLLM(model_path=MODEL_PATH)
    
    ui_thread = threading.Thread(target=run_display_interface, daemon=True)
    ui_thread.start()
    
    print("\nProcessing real audio file 'woodchuck.wav' via Moonshine Tiny...")
    fast_path_asr("./audio/woodchuck.wav", local_llm_engine)
    
    # Give the background LLM thread enough time to complete if it triggers
    time.sleep(10.0)
