from enum import Enum
from dataclasses import dataclass

class AblationMode(Enum):
    FULL_FUSION = "full_fusion"
    AUDIO_ONLY = "audio_only"
    VISION_ONLY = "vision_only"
    BASELINE = "baseline"


@dataclass
class AblationConfig:
    mode: AblationMode