"""Smoke test XTTS-v2 - uruchom PRZED odpaleniem fazy 4 pipeline'u."""
from pathlib import Path
import time
import torch
from TTS.api import TTS

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEAKER_WAV = REPO_ROOT / "lektor.wav"
OUT_WAV = REPO_ROOT / "smoke_test.wav"

assert SPEAKER_WAV.is_file(), f"Brak próbki głosu: {SPEAKER_WAV}"
assert torch.cuda.is_available(), "CUDA niedostępna! Sprawdź instalację PyTorch."

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM przed: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

t0 = time.monotonic()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print(f"Model załadowany w {time.monotonic() - t0:.1f}s")
print(f"VRAM po load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

t1 = time.monotonic()
tts.tts_to_file(
    text="Cześć! To jest test polskiej syntezy mowy z modelem XTTS dwa.",
    speaker_wav=str(SPEAKER_WAV),
    language="pl",
    file_path=str(OUT_WAV),
    split_sentences=True,
)
print(f"Synteza w {time.monotonic() - t1:.1f}s -> {OUT_WAV}")