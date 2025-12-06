#!/usr/bin/env python3
"""Generate baseline and advanced TTS samples for the ML Workshop 2025 assignment."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import importlib
import librosa
import numpy as np
import torch
import torch.serialization

# Ensure backwards compatibility with the import path change in transformers>=4.57.
try:  # pragma: no cover - compatibility shim
    from transformers import BeamSearchScorer as _  # type: ignore
except ImportError:  # pragma: no cover
    import transformers

    from transformers.generation.beam_search import BeamSearchScorer as _BeamSearchScorer

    setattr(transformers, "BeamSearchScorer", _BeamSearchScorer)

from TTS.api import TTS

# Allow list XTTS config classes for torch.load with weights_only=True (PyTorch >=2.6).
try:  # pragma: no cover
    xtts_config_module = importlib.import_module("TTS.tts.configs.xtts_config")
    xtts_model_module = importlib.import_module("TTS.tts.models.xtts")
    base_config_module = importlib.import_module("TTS.config.shared_configs")

    XttsConfig = getattr(xtts_config_module, "XttsConfig", None)
    XttsAudioConfig = getattr(xtts_config_module, "XttsAudioConfig", None)
    XttsArgs = getattr(xtts_model_module, "XttsArgs", None)
    BaseDatasetConfig = getattr(base_config_module, "BaseDatasetConfig", None)

    safe_classes = [cls for cls in (XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig) if cls is not None]
    if safe_classes:
        torch.serialization.add_safe_globals(safe_classes)
except Exception:
    pass

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("COQUI_TTS_TOS_AGREED", "1")

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

EVALUATION_TEXT = (
    "He found himself standing in a landscape that looked exactly like a giant chessboard. "
    "On every black square there was a monster: there were two-tongued snakes and lions "
    "with three rows of teeth, and four-headed dogs and five-headed demon kings and so on. "
    "He was, so to speak, looking out through the eyes of the young hero of the story. "
    "It was like being in the passenger seat of an automobile: all he had to do was watch, "
    "while the hero dispatched one monster after another and advanced up the chessboard "
    "towards the white stone tower at the end."
)


@dataclass
class SampleMetrics:
    """Container for synthesised audio diagnostics."""

    clip_id: str
    model_name: str
    preset: str
    char_count: int
    inference_time_s: float
    duration_s: float
    real_time_factor: float
    mean_rms: float
    spectral_centroid_mean: float
    pitch_std: float
    speaking_rate_chars_per_s: float

    def to_serialisable(self) -> Dict[str, Any]:
        data = asdict(self)
        serialised: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, np.floating):
                serialised[key] = float(value)
            elif isinstance(value, np.integer):  # type: ignore[attr-defined]
                serialised[key] = int(value)
            else:
                serialised[key] = value
        return serialised


def analyse_audio(file_path: Path) -> Dict[str, float]:
    """Compute simple signal statistics for objective comparison."""
    y, sr = librosa.load(file_path, sr=None)
    duration = len(y) / sr
    rms = float(librosa.feature.rms(y=y).mean())
    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    voiced = pitches[magnitudes > np.percentile(magnitudes, 90)]
    pitch_std = float(np.std(voiced)) if voiced.size else 0.0

    return {
        "duration_s": duration,
        "mean_rms": rms,
        "spectral_centroid_mean": spectral_centroid,
        "pitch_std": pitch_std,
    }


def synthesize_sample(
    model: TTS,
    text: str,
    clip_id: str,
    file_path: Path,
    *,
    preset: str,
    char_count: Optional[int] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> SampleMetrics:
    """Generate speech with timing and diagnostics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if extra_kwargs is None:
        extra_kwargs = {}

    start = time.perf_counter()
    model.tts_to_file(text=text, file_path=file_path, **extra_kwargs)
    inference_time = time.perf_counter() - start

    signal_stats = analyse_audio(file_path)
    duration = signal_stats["duration_s"]
    char_count = char_count or len(text)
    speaking_rate = char_count / duration if duration > 0 else 0.0

    metrics = SampleMetrics(
        clip_id=clip_id,
        model_name=getattr(model, "model_name", "unknown"),
        preset=preset,
        char_count=char_count,
        inference_time_s=inference_time,
        duration_s=duration,
        real_time_factor=inference_time / duration if duration else float("inf"),
        mean_rms=signal_stats["mean_rms"],
        spectral_centroid_mean=signal_stats["spectral_centroid_mean"],
        pitch_std=signal_stats["pitch_std"],
        speaking_rate_chars_per_s=speaking_rate,
    )
    return metrics


def load_existing_metrics() -> Dict[str, Any]:
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def persist_metrics(metrics: Dict[str, Any]) -> None:
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate workshop TTS audio artefacts.")
    parser.add_argument("--force", action="store_true", help="Re-generate files even if they exist.")
    parser.add_argument(
        "--device",
        default=None,
        help="Target device for inference (e.g. 'cuda', 'cpu'). Defaults to library preference.",
    )
    args = parser.parse_args()

    metrics = load_existing_metrics()

    tacotron_path = OUTPUT_DIR / "tacotron.wav"
    xtts_natural_path = OUTPUT_DIR / "xtts_v2_natural.wav"
    xtts_happy_path = OUTPUT_DIR / "xtts_v2_happy.wav"
    xtts_sad_path = OUTPUT_DIR / "xtts_v2_sad.wav"

    if args.force and tacotron_path.exists():
        tacotron_path.unlink()
    if args.force:
        for path in (xtts_natural_path, xtts_happy_path, xtts_sad_path):
            if path.exists():
                path.unlink()

    # Baseline Tacotron (Coqui TTS)
    if args.force or "tacotron" not in metrics or not tacotron_path.exists():
        tacotron = TTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            progress_bar=True,
            gpu=False if args.device == "cpu" else None,
        )
        if args.device:
            tacotron.to(args.device)

        tacotron_metrics = synthesize_sample(
            tacotron,
            EVALUATION_TEXT,
            clip_id="tacotron",
            file_path=tacotron_path,
            preset="Tacotron2-DDC baseline (English)",
        )
        metrics["tacotron"] = tacotron_metrics.to_serialisable()

    # Advanced model: XTTS v2
    xtts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
        gpu=False if args.device == "cpu" else None,
    )
    if args.device:
        xtts.to(args.device)

    speaker_manager = getattr(xtts.synthesizer, "speaker_manager", None)
    available_speakers: list[str] = []
    if speaker_manager is not None:
        speakers_attr = getattr(speaker_manager, "speakers", None)
        if isinstance(speakers_attr, dict):
            available_speakers = list(speakers_attr.keys())
    if not available_speakers:
        model_speakers = getattr(xtts, "speakers", None)
        if isinstance(model_speakers, (list, tuple)):
            available_speakers = list(model_speakers)
    default_speaker = available_speakers[0] if available_speakers else "Claribel Dervla"

    emotion_variants = [
        ("xtts_v2_natural", xtts_natural_path, "XTTS v2 default neutral", {}),
        ("xtts_v2_happy", xtts_happy_path, "XTTS v2 emotion=Happy", {"emotion": "Happy"}),
        ("xtts_v2_sad", xtts_sad_path, "XTTS v2 emotion=Sad", {"emotion": "Sad"}),
    ]

    for clip_id, file_path, preset, extra in emotion_variants:
        if not args.force and clip_id in metrics and file_path.exists():
            continue

        kwargs = {"language": "en", **extra}
        if default_speaker:
            kwargs.setdefault("speaker", default_speaker)
        if available_speakers and kwargs.get("speaker") not in available_speakers:
            kwargs["speaker"] = available_speakers[0]

        try:
            sample_metrics = synthesize_sample(
                xtts,
                EVALUATION_TEXT,
                clip_id=clip_id,
                file_path=file_path,
                preset=preset,
                extra_kwargs=kwargs,
            )
        except TypeError:
            # Some library versions do not accept emotion; fall back to prompt engineering.
            fallback_kwargs = {"speaker": "en_female_3", "language": "en"}
            prompt = (
                f"Deliver the following narration in a {extra.get('emotion', 'expressive')} tone: "
                f"{EVALUATION_TEXT}"
            )
            sample_metrics = synthesize_sample(
                xtts,
                prompt,
                clip_id=clip_id,
                file_path=file_path,
                preset=f"{preset} (prompted)",
                char_count=len(EVALUATION_TEXT),
                extra_kwargs=fallback_kwargs,
            )

        metrics[clip_id] = sample_metrics.to_serialisable()

    persist_metrics(metrics)


if __name__ == "__main__":
    main()

