
from typing import Literal, Optional, List

class DPOConfig:
    model_names: Optional[List[str]] = None # might need to specify at some point this is the hf models
    tokenizer_names: Optional[List[str]] = None
    batch_size: Optional[int] = None
    output_dir: Optional[str] = None
    loss_method: Optional[str] = None






