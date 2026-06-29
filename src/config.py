"""
Configuration file for PersianTranscriber pipeline.
All paths, constants, and settings live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "Data"
RAW_AUDIO_DIR = DATA_DIR / "asterisk_monitor"
PROCESSED_AUDIO_DIR = DATA_DIR / "asterisk_processed"   # After VAD+DeepFilterNet
SEPARATED_AUDIO_DIR = DATA_DIR / "asterisk_separated"   # After diarization+separation

# Models directories
MODELS_DIR = BASE_DIR / "Models"
WAV2VEC2_PATH = MODELS_DIR / "wav2vec2-large-xlsr-53-persian"
WHISPER_PATH = MODELS_DIR / "whisper-persian-v4"
NOVA_VAD_PATH = MODELS_DIR / "nova-vad"

# Hugging Face token (loaded from environment)
HF_TOKEN = os.getenv("HF_TOKEN")

# Processing Settings
TARGET_SR = 16000               # Target sample rate (Hz)
MIN_SPEECH_DURATION = 0.3       # Minimum speech segment duration (seconds)
VAD_TIMEOUT = 90                # Timeout for VAD subprocess (seconds)

DF_ENHANCE_KWARGS = {}          # Additional kwargs for df.enhance for DeepFilterNet settings

# Output modes
OUTPUT_MODE_CONCATENATE = "concatenate"
OUTPUT_MODE_SEPARATE = "separate"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "logs" / "pipeline.log"