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