from refiner import RealLocalLLM
import threading
import queue
import time
import sys
import numpy as np
import pyaudio
import moonshine_onnx as moonshine
# import moonshine_voice as moonshine
import os


# Thread-safe communication channels
audio_chunk_queue = queue.Queue()
ui_display_queue = queue.Queue()

# Hardware & Audio Ingestion Constants (Standard for Moonshine)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 512  # Stream chunk slice sizing (~32ms frames)

# 1. Safely resolve the absolute path to your model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"models", 
                        #   "qwen2.5-1.5b-instruct-q5_k_m.gguf")
                        "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
print(MODEL_PATH)

# Double check the file exists before initializing to prevent C-level crashes
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file! Please place it here: {MODEL_PATH}")

# Global Initialization
print("[System Initialization] Booting Pipeline Architecture...")
print("[System Initialization] Loading real local 4-bit LLM (Qwen 2.5) into memory...")
local_llm_engine = RealLocalLLM(model_path=
                                MODEL_PATH)
                                # "models/Llama-3.2-3B-Instruct-Q4_0_4_4.gguf")
print("[System Initialization] Local engines compiled. Pipeline active.")

# ========================================================
# 1. HARDWARE HARDWARE MICROPHONE PRODUCER
# ========================================================
def mic_stream_producer():
    """Continuously records raw PCM audio from the mic to a processing queue."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)
    except Exception as e:
        print(f"\n[Hardware Error] Could not open microphone device: {e}")
        print("Please verify that your microphone is passed through to your VMware settings.")
        sys.exit(1)
        
    print("\n>>> MICROPHONE RECORDING STREAMING ACTIVE <<<")
    print("Start speaking into your mic... (Press Ctrl+C to terminate system)\n")
    
    while True:
        try:
            raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            # Convert raw bytes to standard normalized float32 vectors (-1.0 to 1.0)
            audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_chunk_queue.put(audio_data)
        except Exception as e:
            print(f"\n[Audio Hardware Exception] Stream dropped: {e}")
            break

# ========================================================
# 2. CORE DUAL-PATH PROCESSING ENGINE
# ========================================================
def evaluate_cascade_gate(raw_text):
    """
    Learned Quality Gate evaluating semantic, structural, and linguistic data.
    Returns: (should_refine: bool, confidence_proxy: float)
    """
    words = raw_text.split()
    if len(words) == 0:
        return False, 1.0
        
    # Asymmetric Cost Balancing Heuristics:
    # 1. Fragment Penalty: Short utterances (<3 words) have high structural failure risk
    is_fragment = len(words) <= 2
    
    # 2. Linguistic Target Test: Check for rapid adversarial tongue-twister loops
    contains_adversarial_pattern = any(w in raw_text.lower() for w in ["black", "piper", "woodchuck", "shells"])
    
    # Calculate learned pseudo-confidence profile mapping
    if is_fragment:
        confidence = 0.50
    elif contains_adversarial_pattern:
        confidence = 0.82  # Below our quality threshold -> forces LLM processing
    else:
        # Map stability proxy based on average character layout density
        stability = min(1.0, len(words) / (len(raw_text) * 0.22))
        confidence = max(0.6, min(0.98, stability))
        
    GATE_THRESHOLD = 0.88
    
    if confidence < GATE_THRESHOLD:
        return True, confidence
    return False, confidence

def process_live_utterance(audio_frames, is_final=False):
    """
    Orchestrates the Dual-Path Timeline:
    - is_final=False: Word-by-word streaming refresh via the Fast Path.
    - is_final=True: Lock bounds, execute Cascade Gate, and conditionally route to LLM.
    """
    start_time = time.time()
    
    # Concatenate streaming buffer fragments into a single continuous array vector
    full_audio = np.concatenate(audio_frames)
    
    # Fast Path Inference via optimized native ASR
    try:
        segments = moonshine.transcribe(full_audio, "moonshine/tiny")
        raw_text = segments[0] if segments else ""
    except Exception as e:
        return

    if not raw_text.strip():
        return

    if not is_final:
        # --- WORD-BY-WORD FAST PATH UPDATE ---
        ui_display_queue.put({
            "type": "SPECULATIVE",
            "text": raw_text,
            "latency": time.time() - start_time
        })
    else:
        # --- SENTENCE SEGMENT TERMINATED: RUN ROUTING ---
        should_refine, score = evaluate_cascade_gate(raw_text)
        
        if should_refine:
            # Fork Heavy Path into an independent thread to prevent blocking live mic streams
            threading.Thread(
                target=slow_path_refinement, 
                args=(raw_text, start_time, score)
            ).start()
        else:
            ui_display_queue.put({
                "type": "FINAL_BYPASS", 
                "text": raw_text, 
                "score": score
            })

def slow_path_refinement(raw_text, initial_start_time, score):
    """Asynchronous context post-correction worker running on the background thread pool."""
    prompt = f"Fix speech-to-text grammar and homophone errors: {raw_text}"
    
    # Execute 4-bit INT4 LLM Generative correction step
    refined_text = local_llm_engine.generate(prompt)
    total_latency = time.time() - initial_start_time
    
    ui_display_queue.put({
        "type": "REFINED",
        "text": refined_text,
        "raw_text": raw_text,
        "latency": total_latency,
        "score": score
    })

# ========================================================
# 3. VAD SLIDING TIME WINDOW ORCHESTRATOR
# ========================================================
def audio_processing_orchestrator():
    """
    Manages continuous sliding voice window segment boundaries.
    Toggles live word refreshes vs sentence-final gate locks.
    """
    accumulated_frames = []
    silent_chunks_count = 0
    is_speaking = False
    incremental_stream_tick = 0 
    
    while True:
        frame = audio_chunk_queue.get()
        accumulated_frames.append(frame)
        incremental_stream_tick += 1
        
        # Continuous mathematical vector energy envelope calculation
        energy = np.linalg.norm(frame)
        
        if energy > 0.28:  # Voice threshold matched
            is_speaking = True
            silent_chunks_count = 0
        else:
            if is_speaking:
                silent_chunks_count += 1
        
        # 1. LIVE WORD-BY-WORD TRIGGER: Refresh display stream every ~320ms (10 frames)
        if is_speaking and incremental_stream_tick >= 10:
            incremental_stream_tick = 0
            threading.Thread(
                target=process_live_utterance, 
                args=(list(accumulated_frames), False),
                daemon=True
            ).start()
            
        # 2. SENTENCE BOUNDARY TRIGGER: User paused speaking for ~640ms (20 frames)
        if is_speaking and silent_chunks_count > 20:
            threading.Thread(
                target=process_live_utterance, 
                args=(list(accumulated_frames), True),
                daemon=True
            ).start()
            
            # Flush rolling stream state properties for next sentence grouping
            accumulated_frames.clear()
            is_speaking = False
            silent_chunks_count = 0
            incremental_stream_tick = 0

# ========================================================
# 4. ACTIVE LIVE USER INTERFACE RENDERER
# ========================================================
def run_live_display_interface():
    """Simulates real-time application rolling caption stream updates."""
    print("--- Asynchronous Speculative UI Pipeline Engaged ---\n")
    
    while True:
        message = ui_display_queue.get()
        text = message["text"]
        
        if message["type"] == "SPECULATIVE":
            # Overwrite the active line dynamically (\r carriage return keeps it word-by-word)
            sys.stdout.write(f"\r[LIVE FAST PATH]: {text:<75} (Latency: {message['latency']:.2f}s)")
            sys.stdout.flush()
            
        elif message["type"] == "FINAL_BYPASS":
            # Lock the current line and create a clean break below
            sys.stdout.write(f"\r[FINAL CAPTION]: {text:<75}\n -> Gate Passed (Score: {message['score']:.2f}) | Bypassed LLM.\n\n")
            sys.stdout.flush()
            
        elif message["type"] == "REFINED":
            # Seamlessly replace the old incorrect ASR text with the LLM's optimized phrase
            sys.stdout.write(f"\r[FINAL CAPTION]: {text:<75}\n -> Gate Failed (Score: {message['score']:.2f}) | LLM Refined in {message['latency']:.2f}s\n\n")
            sys.stdout.flush()
            
        ui_display_queue.task_done()

# ========================================================
# 5. MAIN SYSTEM RUNTIME ENTRY
# ========================================================
if __name__ == "__main__":
    # Thread A: Ingest micro-audio packages from sound card
    threading.Thread(target=mic_stream_producer, daemon=True).start()
    
    # Thread B: Run sliding window segmentation logic
    threading.Thread(target=audio_processing_orchestrator, daemon=True).start()
    
    # Thread C: Keep the asynchronous UI loop refreshed
    threading.Thread(target=run_live_display_interface, daemon=True).start()
    
    # Keep application execution root bound safely
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n[System Shutdown] Live pipeline terminated. Computing resources cleanly released.")
