import os
from platformdirs import user_cache_dir

from .utils import write_outputs

BASE_PATH = os.path.dirname(__file__)

CACHE_DIR = user_cache_dir("whisper")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_model(model_identifier="large-v3",
               **model_kwargs):
    
    from .backend.tensorrt.model import WhisperModelTRT as WhisperModel
        
    return WhisperModel(model_identifier, **model_kwargs)
        