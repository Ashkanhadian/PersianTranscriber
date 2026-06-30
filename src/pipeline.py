import os
import sys
import subprocess
import json
import re
from pathlib import Path
import logging
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

from deepfilternet import DeepFilterNet3, df

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_INPUT = BASE_DIR / "Data" / "asterisk_monitor"
DATA_OUTPUT = BASE_DIR / "Data" / "asterisk_processed"
NOVA_VAD_PATH = BASE_DIR / "Models" / "nova-vad"

df_model = None

def get_deepfilternet():
    """Lazy load DeepFilterNet3 model (loaded once)."""
    global df_model
    if df_model is None:
        df_model = DeepFilterNet3()
        logger.info("DeepFilterNet3 model loaded.")
    return df_model

# ---------------------------------------------------
# VAD: NOVA-VAD to get speech timestamps
# ---------------------------------------------------
def get_speech_timestamps(audio_path: Path) -> list:
    """
    Run NOVA-VAD and return a list of (start, end) speech segments in seconds.
    Returns empty list if no speech is detected or on error.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "src.explainer", str(audio_path)],
            capture_output=True,
            text=True,
            cwd=str(NOVA_VAD_PATH),
            timeout=120,
        )

        # Parse the output to extract timestamps
        # NOVA-VAD's explainer outputs something like:
        # "Speech segments: (0.12, 2.34), (4.56, 7.89)"
        # or individual frames with timestamps
        output = result.stdout + result.stderr

        # Try to parse timestamps from the output
        segments = []
        # Look for patterns like (0.12, 2.34) or 0.12-2.34
        pattern = r'\(?(\d+\.?\d*)\s*[,–\-]\s*(\d+\.?\d*)\)?'
        matches = re.findall(pattern, output)
        for match in matches:
            start = float(match[0])
            end = float(match[1])
            if end > start and (end - start) >= 0.2:  # Minimum 200ms speech
                segments.append((start, end))

        # If no segments found, check if "SPEECH" is in output
        if not segments and "SPEECH" in output:
            # If the explainer only says SPEECH but no timestamps,
            # we fall back to the entire file (or we can log a warning)
            # For now, we'll treat it as speech from 0 to total duration
            # (but we'll handle this in the caller)
            logger.warning(f"NOVA-VAD returned SPEECH but no timestamps for {audio_path}. Using full file.")
            return None  # Signal that VAD passed but no timestamps

        return segments

    except Exception as e:
        logger.error(f"NOVA-VAD error for {audio_path}: {e}")
        return [] 

# ------------------------------
# Main preprocessing function
# ------------------------------
def preprocess_file(input_path: Path, output_path: Path, mode: str = "concatenate") -> bool:
    """
    Preprocess a single audio file:
    1. VAD to get speech segments.
    2. Load audio.
    3. For each segment, extract audio, enhance with DeepFilterNet3.
    4. Either concatenate all segments or save separately.

    Args:
        input_path: Path to input audio file.
        output_path: Path to output audio file.
        mode: "concatenate" (one file) or "separate" (multiple files).

    Returns:
        True on success, False if skipped or failed.
    """
    try:
        # 1. Get speech segments from VAD
        segments = get_speech_timestamps(input_path)

        # If no segments found, skip the file
        if segments is None:
            # VAD detected speech but no timestamps – fallback to full file
            logger.debug(f"Fallback to full file: {input_path}")
            audio, sr = librosa.load(input_path, sr=16000)
            if len(audio) < 4800:  # < 0.3s at 16kHz
                return False
            model = get_deepfilternet()

            enhanced = df.enhance(audio, sr, model)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, enhanced, sr)
            return True

        if not segments:
            logger.debug(f"No speech detected: {input_path}")
            return False

        # 2. Load the full audio
        audio, sr = librosa.load(input_path, sr=16000)

        # 3. Get DeepFilterNet model
        model = get_deepfilternet()

        # 4. Process each segment
        enhanced_segments = []
        for idx, (start, end) in enumerate(segments):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < 4800:  # < 0.3s
                continue

            # Enhance segment
            enhanced_segment = df.enhance(segment_audio, sr, model)
            enhanced_segments.append(enhanced_segment)

        if not enhanced_segments:
            logger.debug(f"No valid segments after enhancement: {input_path}")
            return False

        # 5. Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "concatenate":
            # Concatenate all enhanced segments into one file
            final_audio = np.concatenate(enhanced_segments)
            sf.write(output_path, final_audio, sr)
            logger.debug(f"Concatenated {len(enhanced_segments)} segments: {output_path}")
        else:
            # Save each segment as a separate file
            for idx, segment in enumerate(enhanced_segments):
                seg_path = output_path.parent / f"{output_path.stem}_seg{idx:03d}.wav"
                sf.write(seg_path, segment, sr)
            logger.debug(f"Saved {len(enhanced_segments)} segments: {output_path.stem}")

        return True

    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False

def process_all_files(input_root: Path, output_root: Path, mode: str = "concatenate") -> dict:
    """
    Process all .wav files from input_root to output_root.
    mode: "concatenate" or "separate"
    Returns stats dictionary.
    """
    # Find all WAV files
    wav_files = list(input_root.rglob("*.wav"))
    total = len(wav_files)
    logger.info(f"Found {total} WAV files to process.")

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    # Process with progress bar
    for input_path in tqdm(wav_files, desc="Processing audio"):
        rel_path = input_path.relative_to(input_root)
        output_path = output_root / rel_path

        if preprocess_file(input_path, output_path, mode):
            stats["processed"] += 1
        else:
            if output_path.exists():
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

    return stats

def find_wav_files(root_dir: Path) -> list:
    if not root_dir.exists():
        logger.error(f"Directory does not exist: {root_dir}")
        return []

    wav_files = []
    for path in root_dir.rglob("*.wav"):
        wav_files.append(path)

    return wav_files

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess audio files with VAD and enhancement.")
    parser.add_argument("--mode", choices=["concatenate", "separate"], default="concatenate",
                        help="Save segments concatenated or separately")
    parser.add_argument("--input", type=Path, default=DATA_INPUT,
                        help="Input directory")
    parser.add_argument("--output", type=Path, default=DATA_OUTPUT,
                        help="Output directory")
    args = parser.parse_args()

    logger.info("=== Starting Audio Preprocessing Pipeline ===")
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Mode:   {args.mode}")

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    stats = process_all_files(args.input, args.output, args.mode)

    logger.info("=== Preprocessing Complete ===")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped:   {stats['skipped']}")
    logger.info(f"Failed:    {stats['failed']}")


if __name__ == "__main__":
    main()