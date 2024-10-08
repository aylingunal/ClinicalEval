
from typing import Literal, Optional, List

class DPOConfig:
    model_names: Optional[List[str]] = None # might need to specify at some point this is the hf models
    tokenizer_names: Optional[List[str]] = None
    tokenizer_max_len: Optional[int] = None
    batch_size: Optional[int] = None
    output_dir: Optional[str] = None
    loss_method: Optional[str] = None
    num_epochs: Optional[int] = None
    lr: Optional[float] = None






