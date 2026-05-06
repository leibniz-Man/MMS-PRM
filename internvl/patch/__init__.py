# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings

# Always available utilities
from .internvit_liger_monkey_patch import apply_liger_kernel_to_internvit
from .llama_rmsnorm_monkey_patch import replace_llama_rmsnorm_with_fused_rmsnorm
from .pad_data_collator import (
    concat_pad_data_collator,
    dpo_concat_pad_data_collator,
    pad_data_collator,
)
from .train_dataloader_patch import replace_train_dataloader
from .train_sampler_patch import replace_train_sampler

# Conditionally import flash-attn dependent patches to avoid crashing when flash_attn is unavailable
_FLASH_ATTN_AVAILABLE = True
try:
    import flash_attn  # noqa: F401
except Exception:
    _FLASH_ATTN_AVAILABLE = False
    warnings.warn(
        'flash_attn is not installed or unavailable. Flash-Attn based patches will be skipped.'
    )

# Initialize optional symbols to None; fill them when flash_attn is present
replace_llama_attn_with_flash_attn = None
replace_llama2_attn_with_flash_attn = None
replace_llama_attention_class = None
replace_internlm2_attention_class = None
replace_qwen2_attention_class = None
replace_phi3_attention_class = None

if _FLASH_ATTN_AVAILABLE:
    from .llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    from .llama2_flash_attn_monkey_patch import replace_llama2_attn_with_flash_attn
    from .llama_packed_training_patch import replace_llama_attention_class
    from .internlm2_packed_training_patch import replace_internlm2_attention_class
    from .qwen2_packed_training_patch import replace_qwen2_attention_class
    from .phi3_packed_training_patch import replace_phi3_attention_class

__all__ = [
    'apply_liger_kernel_to_internvit',
    'replace_llama_rmsnorm_with_fused_rmsnorm',
    'replace_train_sampler',
    'replace_train_dataloader',
    'pad_data_collator',
    'dpo_concat_pad_data_collator',
    'concat_pad_data_collator',
]

if replace_llama_attn_with_flash_attn is not None:
    __all__.append('replace_llama_attn_with_flash_attn')
if replace_llama2_attn_with_flash_attn is not None:
    __all__.append('replace_llama2_attn_with_flash_attn')
if replace_llama_attention_class is not None:
    __all__.append('replace_llama_attention_class')
if replace_internlm2_attention_class is not None:
    __all__.append('replace_internlm2_attention_class')
if replace_qwen2_attention_class is not None:
    __all__.append('replace_qwen2_attention_class')
if replace_phi3_attention_class is not None:
    __all__.append('replace_phi3_attention_class')
