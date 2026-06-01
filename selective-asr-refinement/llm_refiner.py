"""
llm_refiner.py
--------------
ASR transcription refinement using llama-cpp-python.

Usage:
    from llm_refiner import LLMRefiner
    refiner = LLMRefiner("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    result = refiner.refine("she sells sea shells on the sea shore")
    print(result.refined_text)
"""

import time
from dataclasses import dataclass
from pathlib import Path

from llama_cpp import Llama


SYSTEM_PROMPT = (
    """You are an ASR transcription corrector, work like an input and output machine.

    Step 1 - Detect: Does the text contain ASR errors (wrong words, missing words, 
            spelling from mishearing)? If NO errors, return the text unchanged.
    Step 2 - Correct: Fix only clear ASR errors. Do NOT rephrase, summarize, 
            or add content not present in the original.
    Step 3 - Verify: Check your correction didn't introduce new words 
            absent from the original meaning.

    Return ONLY the corrected text. Do not add explanations."""
)


@dataclass
class RefineResult:
    original: str
    refined_text: str
    latency_s: float
    energy_j: float  # estimated: avg_power_W * latency_s

    @property
    def changed(self) -> bool:
        return self.original.strip() != self.refined_text.strip()


class LLMRefiner:
    """
    Wraps a llama-cpp Llama model to post-correct ASR transcriptions.

    Parameters
    ----------
    model_path : str | Path
    n_threads : int
        CPU threads for inference.
    n_gpu_layers : int
        Layers to offload to GPU (0 = CPU only).
    avg_power_w : float
        Estimated average CPU/GPU power draw in Watts, used to compute
        energy = power * latency. Defaults to 15 W (typical laptop CPU).
    """

    # Rough average power draw used for energy estimation (Watts)
    DEFAULT_POWER_W = 15.0

    def __init__(
        self,
        model_path: str | Path,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        avg_power_w: float = DEFAULT_POWER_W,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download Llama-3.2-3B-Instruct-Q4_K_M.gguf and place it in models/"
            )
        self._avg_power_w = avg_power_w
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def refine(self, text: str) -> RefineResult:
        """
        Refine a raw ASR transcription.

        Parameters
        ----------
        text : str   Raw ASR output.

        Returns
        -------
        RefineResult
        """
        if not text.strip():
            return RefineResult(
                original=text, refined_text=text, latency_s=0.0, energy_j=0.0
            )

        t0 = time.perf_counter()
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        latency = time.perf_counter() - t0

        refined = response["choices"][0]["message"]["content"].strip()

        # Sanity check: if LLM returns something absurdly long, fall back
        if len(refined) > len(text) * 3:
            refined = text

        return RefineResult(
            original=text,
            refined_text=refined,
            latency_s=latency,
            energy_j=self._avg_power_w * latency,
        )
