# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)


# Need to call this before importing transformers.
from tsgpt.train.qwen_flash_attn_monkey_patch import (
    replace_qwen_attn_with_flash_attn,
)

replace_qwen_attn_with_flash_attn()

import torch.distributed as dist
from datetime import timedelta


from tsgpt.train.train_st_lightning import train

if __name__ == "__main__":
    train()
