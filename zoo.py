import logging
import sys
import io
import warnings
from contextlib import contextmanager
from typing import Union

from PIL import Image
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

@contextmanager
def suppress_output():
    """Suppress stdout, stderr, warnings, and transformers logging."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Suppress transformers logging
    transformers_logger = logging.getLogger("transformers")
    old_transformers_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        transformers_logger.setLevel(old_transformers_level)



def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NanoNetsOCR(Model, SamplesMixin):
    """FiftyOne model for NanoNets-OCR vision-language tasks.
    
    Simple OCR model that extracts text from documents using vision-language processing.
    
    Automatically selects optimal dtype based on hardware:
    - bfloat16 for CUDA devices with compute capability 8.0+ (Ampere and newer)
    - float16 for older CUDA devices
    - float32 for CPU/MPS devices
    
    Args:
        model_path: HuggingFace model ID or local path (default: "nanonets/Nanonets-OCR2-3B")
        custom_prompt: Custom prompt for OCR task (optional)
        max_new_tokens: Maximum tokens to generate (default: 15000)
        torch_dtype: Override automatic dtype selection
    """
    
    def __init__(
        self,
        model_path: str = "nanonets/Nanonets-OCR2-3B",
        custom_prompt: str = None,
        max_new_tokens: int = 15000,
        torch_dtype: torch.dtype = None,
        **kwargs
    ):
        SamplesMixin.__init__(self) 
        self.model_path = model_path
        self._custom_prompt = custom_prompt
        self.max_new_tokens = max_new_tokens
        
        # Device setup
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Dtype selection
        if torch_dtype is not None:
            self.dtype = torch_dtype
        elif self.device == "cuda":
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            logger.info(f"Using {self.dtype} dtype (compute capability {capability[0]}.{capability[1]})")
        else:
            self.dtype = torch.float32
            logger.info(f"Using float32 dtype for {self.device}")
        
        # Load model, tokenizer, and processor
        logger.info(f"Loading NanoNets-OCR from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        self.model = self.model.eval()
        
        logger.info("NanoNets-OCR model loaded successfully")
    
    @property
    def media_type(self):
        """The media type processed by this model."""
        return "image"
    
    def _predict(self, image: Image.Image, sample) -> str:
        """Process image through NanoNets-OCR.
        
        Args:
            image: PIL Image to process
            sample: FiftyOne sample (has filepath attribute)
        
        Returns:
            str: Extracted text from the document
        """
        # Use custom prompt if provided, otherwise use default
        prompt = self._custom_prompt if self._custom_prompt else DEFAULT_PROMPT
        
        # Get the image path from the sample
        image_path = sample.filepath if sample else "temp_image.jpg"
        
        # Prepare messages in the chat format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ]},
        ]
        
        # Run inference with suppressed output
        with suppress_output():
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate output
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                do_sample=False
            )
            
            # Decode only the generated tokens (excluding input)
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            # Decode to text
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            result = output_text[0]
        
        return result
    
    def predict(self, image, sample=None):
        """Process an image with NanoNets-OCR.
        
        Args:
            image: PIL Image or numpy array to process
            sample: FiftyOne sample containing the image filepath
        
        Returns:
            str: Extracted text from the document
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)
