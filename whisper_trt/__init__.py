import os
from platformdirs import user_cache_dir

from .utils import write_outputs

BASE_PATH = os.path.dirname(__file__)
ASSETS_PATH = os.path.join(BASE_PATH, "assets")
CACHE_DIR = os.path.join(BASE_PATH, "model")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_model(model_identifier="large-v3",
               **model_kwargs):
    
    if model_identifier in ['large-v3']:
        model_kwargs['n_mels'] = 128
    
    from .backend.tensorrt.model import WhisperModelTRT as WhisperModel
        
    return WhisperModel(model_identifier, **model_kwargs)
        