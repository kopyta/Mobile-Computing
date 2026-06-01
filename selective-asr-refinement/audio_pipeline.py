"""
audio_pipeline.py
-----------------
Audio file transcription pipeline:
  - Feeds audio file in chunks to moonshine_voice.Transcriber (TINY_STREAMING)
  - Built-in VAD cuts the stream into speech segments (on_line_completed events)
  - Each completed segment is refined by LLMRefiner
  - Results written to results/results_<timestamp>.csv

moonshine_voice.Transcriber handles VAD and segmentation internally —
no external VAD library needed.

Usage:
    python audio_pipeline.py path/to/audio.wav
    python audio_pipeline.py path/to/audio.wav --model models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from moonshine_voice import (
    ModelArch,
    Transcriber,
    TranscriptEventListener,
    get_model_for_language,
)

from llm_refiner import LLMRefiner, RefineResult

RESULTS_DIR = Path("results")
CHUNK_DURATION_S = 0.1   # feed audio in 100 ms chunks to simulate streaming
SAMPLE_RATE = 16_000     # Moonshine expects 16 kHz
# Estimated average power draw for ASR inference (Watts) — adjust to your hardware
ASR_POWER_W = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_audio(path: str | Path) -> np.ndarray:
    """Load audio file as mono float32 at 16 kHz."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Simple resample via scipy
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


def open_csv(path: Path) -> tuple[csv.DictWriter, object]:
    """Open a CSV file for writing and return (writer, file_handle)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=[
        "segment",
        "raw_text",
        "refined_text",
        "asr_latency_s",
        "llm_latency_s",
        "asr_energy_j",
        "llm_energy_j",
    ])
    writer.writeheader()
    return writer, fh


# ---------------------------------------------------------------------------
# Listener — fired by Transcriber when a VAD segment completes
# ---------------------------------------------------------------------------

class PipelineListener(TranscriptEventListener):
    """
    Called by the Transcriber on each completed speech line.

    event.line attributes used:
      .text        — final transcription for this segment
      .audio_data  — raw audio list for this segment (from VAD segmenter)
      .latency_ms  — time from end-of-speech to text being ready (ms)
    """

    def __init__(self, refiner: LLMRefiner, csv_writer: csv.DictWriter) -> None:
        self._refiner = refiner
        self._writer = csv_writer
        self._segment_idx = 0

    def on_line_completed(self, event) -> None:
        self._segment_idx += 1
        raw_text: str = event.line.text.strip()
        asr_latency_s: float = getattr(event.line, "latency_ms", 0.0) / 1000.0

        # Estimate ASR energy from the segment's audio duration
        audio_data = getattr(event.line, "audio_data", [])
        audio_duration_s = len(audio_data) / SAMPLE_RATE if audio_data else 0.0
        asr_energy_j = ASR_POWER_W * asr_latency_s

        print(f"\n[Segment {self._segment_idx}] RAW: {raw_text!r}")

        # LLM refinement
        refine: RefineResult = self._refiner.refine(raw_text)
        print(f"[Segment {self._segment_idx}] REFINED: {refine.refined_text!r} "
              f"({refine.latency_s*1000:.0f} ms)")

        self._writer.writerow({
            "segment": self._segment_idx,
            "raw_text": raw_text,
            "refined_text": refine.refined_text,
            "asr_latency_s": round(asr_latency_s, 4),
            "llm_latency_s": round(refine.latency_s, 4),
            "asr_energy_j": round(asr_energy_j, 6),
            "llm_energy_j": round(refine.energy_j, 6),
        })


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(audio_path: str | Path, model_path: str | Path) -> Path:
    """
    Main pipeline: load audio → stream to Transcriber → refine → save CSV.

    Returns the path to the results CSV.
    """
    audio_path = Path(audio_path)
    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    duration_s = len(audio) / SAMPLE_RATE
    print(f"Duration: {duration_s:.1f}s  Samples: {len(audio):,}")

    # Output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"results_{timestamp}.csv"
    csv_writer, csv_fh = open_csv(csv_path)

    # LLM refiner
    refiner = LLMRefiner(model_path)
    print("LLM loaded.")

    # Moonshine Transcriber — TINY_STREAMING with built-in VAD
    moonshine_path, moonshine_arch = get_model_for_language("en", ModelArch.TINY_STREAMING)
    transcriber = Transcriber(model_path=moonshine_path, model_arch=moonshine_arch)

    listener = PipelineListener(refiner, csv_writer)
    transcriber.add_listener(listener)
    transcriber.start()

    # Feed audio in chunks (simulates streaming from file)
    chunk_size = int(CHUNK_DURATION_S * SAMPLE_RATE)
    print(f"Streaming {len(audio)//chunk_size} chunks ({CHUNK_DURATION_S*1000:.0f} ms each)…")

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size].tolist()
        transcriber.add_audio(chunk, SAMPLE_RATE)

    # Let the transcriber finish processing remaining audio
    transcriber.stop()
    transcriber.close()

    csv_fh.flush()
    csv_fh.close()

    print(f"\nDone. {listener._segment_idx} segment(s) written to {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # option for running from terminal
    # parser = argparse.ArgumentParser(
    #     description="Transcribe an audio file with Moonshine + LLM refinement."
    # )
    # parser.add_argument("audio", help="Path to input audio file (WAV, FLAC, …)")
    # parser.add_argument(
    #     "--model",
    #     default="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    #     help="Path to Llama GGUF model file.",
    # )
    # args = parser.parse_args()
    # run_pipeline(args.audio, args.model)

    # Running straight in code 
    audio_path = "audio/spontaneous-speech-en-11.wav"
    # "audio/seashells_slow.wav"
    model_path = "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    # "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    
    # # Convert mp3 to wav
    # from pydub import AudioSegment
    # sound = AudioSegment.from_mp3(audio_path)
    # audio_path, _ = os.path.splitext(audio_path)
    # audio_path = audio_path + ".wav"
    # sound.export(audio_path, format="wav") 
    
    run_pipeline(audio_path, model_path)


if __name__ == "__main__":
    main()
