"""
Preprocessing pipeline: VAD -> segmentation -> DeepFilterNet enhancement.
"""

import subprocess
import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm

from config import (
    NOVA_VAD_PATH,
    TARGET_SR,
    MIN_SPEECH_DURATION,
    VAD_TIMEOUT,
    DF_ENHANCE_KWARGS,
)

logger = logging.getLogger(__name__)

# Global model cache
_df_model = None

def get_deepfilternet_model():
    """Lazy load DeepFilterNet3 model."""
    global _df_model
    if _df_model is None:
        try:
            from df import DeepFilterNet3, df
            _df_model = DeepFilterNet3()
            logger.info("DeepFilterNet3 model loaded.")
            # We also need to expose the df module for later use
            global _df_module
            _df_module = df
        except ImportError as e:
            logger.error(f"DeepFilterNet not installed: {e}")
            raise
    return _df_model, _df_module

def get_speech_timestamps(audio_path: Path) -> Optional[List[Tuple[float, float]]]:
    """
    Run NOVA-VAD and return list of (start, end) in seconds.
    Returns None if VAD says "SPEECH" but no timestamps (fallback to full file).
    Returns empty list if no speech detected.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "src.explainer", str(audio_path)],
            capture_output=True,
            text=True,
            cwd=str(NOVA_VAD_PATH),
            timeout=VAD_TIMEOUT,
        )
        output = result.stdout + result.stderr

        # Attempt to parse timestamps
        segments = []
        # Patterns: (0.12, 2.34) or 0.12-2.34
        pattern = r'\(?(\d+\.?\d*)\s*[,–\-]\s*(\d+\.?\d*)\)?'
        matches = re.findall(pattern, output)
        for match in matches:
            start = float(match[0])
            end = float(match[1])
            if end > start and (end - start) >= MIN_SPEECH_DURATION:
                segments.append((start, end))

        # If we found segments, return them
        if segments:
            return segments

        # If no segments but "SPEECH" in output, signal fallback
        if "SPEECH" in output:
            logger.warning(f"NOVA-VAD said SPEECH but no timestamps for {audio_path}. Using full file.")
            return None  # fallback to full file

        return []  # no speech

    except subprocess.TimeoutExpired:
        logger.error(f"NOVA-VAD timeout for {audio_path}")
        return []
    except Exception as e:
        logger.error(f"NOVA-VAD error for {audio_path}: {e}")
        return []
    
def enhance_segment(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply DeepFilterNet3 enhancement to a single audio segment."""
    model, df_module = get_deepfilternet_model()
    return df_module.enhance(audio, sr, model, **DF_ENHANCE_KWARGS)

def preprocess_file(input_path: Path, output_path: Path, mode: str = "concatenate") -> bool:
    """
    Preprocess a single audio file:
    1. VAD → speech timestamps.
    2. Extract, enhance, and optionally concatenate speech segments.
    3. Write output.

    Returns True on success, False if skipped/failed.
    """
    try:
        # 1. Get speech segments
        segments = get_speech_timestamps(input_path)

        # Fallback: if VAD says speech but no timestamps, process whole file
        if segments is None:
            audio, sr = librosa.load(input_path, sr=TARGET_SR)
            if len(audio) < int(MIN_SPEECH_DURATION * sr):
                return False
            enhanced = enhance_segment(audio, sr)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, enhanced, sr)
            return True

        if not segments:
            logger.debug(f"No speech detected: {input_path}")
            return False

        # 2. Load full audio
        audio, sr = librosa.load(input_path, sr=TARGET_SR)

        # 3. Process each segment
        enhanced_segments = []
        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = audio[start_sample:end_sample]
            if len(segment) < int(MIN_SPEECH_DURATION * sr):
                continue
            enhanced = enhance_segment(segment, sr)
            enhanced_segments.append(enhanced)

        if not enhanced_segments:
            logger.debug(f"No valid segments after enhancement: {input_path}")
            return False

        # 4. Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "concatenate":
            final_audio = np.concatenate(enhanced_segments)
            sf.write(output_path, final_audio, sr)
        else:  # separate
            for idx, seg in enumerate(enhanced_segments):
                seg_path = output_path.parent / f"{output_path.stem}_seg{idx:03d}.wav"
                sf.write(seg_path, seg, sr)

        return True

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False