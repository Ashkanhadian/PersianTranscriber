"""
Speaker diarization and separation using pyannote.audio.
This should run after preprocessing (VAD + DeepFilterNet3) in a separate environment.
"""

import os
import sys
import torch
import librosa
import scipy.io.wavfile
from pathlib import Path
import logging
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "asterisk_processed"
SEPARATED_DIR = BASE_DIR / "Data" / "asterisk_separated"

load_dotenv(BASE_DIR / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN environment variable not set.\n"
        "Please create a .env file with: HF_TOKEN=hf_xxxxxxxxxxxx"
    )

_separation_pipeline = None

def get_pipeline():
    global _separation_pipeline
    if _separation_pipeline is None:
        from pyannote.audio import Pipeline
        logger.info("Loading speaker separation pipeline...")
        _separation_pipeline = Pipeline.from_pretrained(
            "pyannote/speech-separation-ami-1.0",
            token=HF_TOKEN,
        )
        if torch.cuda.is_available():
            _separation_pipeline.to(torch.device("cuda"))
            logger.info("Moved to GPU.")
        else:
            logger.info("Running on CPU.")
    return _separation_pipeline

def separate_speakers(audio_path: Path, output_dir: Path) -> List[Path]:
    pipeline = get_pipeline()
    logger.info(f"Processing: {audio_path}")

    # Load audio with librosa (resamples to 16kHz automatically)
    audio, sr = librosa.load(str(audio_path), sr=16000)
    # Convert to torch tensor (shape: channels x time)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
    inputs = {"waveform": waveform, "sample_rate": sr}

    # Run pipeline
    diarization, sources = pipeline(inputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    speaker_ids = list(diarization.labels())

    if not speaker_ids:
        logger.warning(f"No speakers in {audio_path}")
        return saved

    for idx, speaker in enumerate(speaker_ids):
        # sources is a SlidingWindowFeature with shape (n_frames, n_speakers)
        speaker_audio = sources[:, idx]  # numpy array, float32 in [-1, 1]
        # Convert to int16 for WAV
        audio_int16 = (speaker_audio * 32767).astype('int16')
        out_path = output_dir / f"{audio_path.stem}_speaker_{speaker}.wav"
        scipy.io.wavfile.write(str(out_path), 16000, audio_int16)
        saved.append(out_path)
        logger.info(f"Saved: {out_path}")

    return saved

def main():
    # Find all cleaned WAVs
    wav_files = list(PROCESSED_DIR.rglob("*.wav"))
    logger.info(f"Found {len(wav_files)} cleaned audio files.")

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    for wav_path in tqdm(wav_files, desc="Diarizing"):
        rel_path = wav_path.relative_to(PROCESSED_DIR)
        output_dir = SEPARATED_DIR / rel_path.parent / wav_path.stem

        try:
            out_files = separate_speakers(wav_path, output_dir)
            if out_files:
                stats["processed"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            logger.error(f"Failed: {wav_path} - {e}")
            stats["failed"] += 1

    logger.info(f"Done: Processed {stats['processed']}, Skipped {stats['skipped']}, Failed {stats['failed']}")

if __name__ == "__main__":
    main()