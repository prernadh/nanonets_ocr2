import logging
from huggingface_hub import snapshot_download
from .zoo import NanoNetsOCR

logger = logging.getLogger(__name__)


def download_model(model_name, model_path):
    """Downloads the NanoNets-OCR model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name=None, model_path=None, **kwargs):
    """Load a NanoNets-OCR model for use with FiftyOne.
    
    Args:
        model_name: Model name (unused, for compatibility)
        model_path: HuggingFace model ID or path to model files
        **kwargs: Additional config parameters (resolution_mode, operation, custom_prompt, etc.)
        
    Returns:
        NanoNetsOCR: Initialized model ready for inference
    """
    if model_path is None:
        model_path = "nanonets/Nanonets-OCR2-3B"
    
    return NanoNetsOCR(model_path=model_path, **kwargs)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    pass
