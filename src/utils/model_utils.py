"""Model and tokenizer loading utilities.

Provides factory functions that encapsulate the patterns used across
SFT training, inference, and reward model evaluation (device detection,
dtype selection, LoRA adapter application, tokenizer padding setup).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect the best available device (CUDA > MPS > CPU).

    Returns:
        A torch.device object for the preferred compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_compute_dtype(device: torch.device) -> torch.dtype:
    """Select the recommended floating-point dtype for a device.

    Args:
        device: The torch device that will be used for computation.

    Returns:
        ``torch.bfloat16`` when CUDA supports it, ``torch.float16`` for
        CUDA without bf16, and ``torch.float32`` otherwise (MPS/CPU).
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_tokenizer(
    model_name: str,
    padding_side: str = "right",
    use_fast: bool = True,
) -> AutoTokenizer:
    """Load a HuggingFace tokenizer and ensure a pad token is set.

    Mistral tokenizers do not ship with a pad token; this function sets
    ``pad_token = eos_token`` as a sensible default.

    Args:
        model_name: HuggingFace model identifier or local path.
        padding_side: ``"left"`` for generation, ``"right"`` for training.
        use_fast: Whether to use the fast (Rust) tokenizer implementation.

    Returns:
        The configured tokenizer.
    """
    tok = AutoTokenizer.from_pretrained(
        model_name,
        padding_side=padding_side,
        use_fast=use_fast,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    model_name: str,
    device: torch.device,
    quantization_mode: str = "none",
) -> AutoModelForCausalLM:
    """Load the base Mistral model with optional quantization.

    Args:
        model_name: HuggingFace model identifier.
        device: Target compute device.
        quantization_mode: One of ``"none"``, ``"8bit"``, or ``"4bit"``.

    Returns:
        The loaded base causal LM.

    Raises:
        ImportError: If ``bitsandbytes`` is required but not installed.
    """
    load_kwargs: dict = {"trust_remote_code": True}

    if device.type == "cuda":
        load_kwargs["device_map"] = "auto"
        if quantization_mode == "4bit":
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization_mode == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            load_kwargs["torch_dtype"] = get_compute_dtype(device)
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if device.type == "mps":
        model.to(device)

    return model


def load_model_with_adapters(
    base_model_name: str,
    adapter_path: str | Path,
    device: torch.device,
    for_inference: bool = True,
) -> Tuple[PeftModel, AutoTokenizer]:
    """Load a base model with LoRA adapters for inference.

    Args:
        base_model_name: HuggingFace model identifier.
        adapter_path: Directory containing ``adapter_config.json`` and
            ``adapter_model.safetensors``.
        device: Target device for inference.
        for_inference: If ``True``, set ``is_trainable=False`` and
            ``model.eval()``.

    Returns:
        A tuple of ``(peft_model, tokenizer)`` ready for generation.
    """
    tokenizer = load_tokenizer(base_model_name, padding_side="left")
    dtype = get_compute_dtype(device)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if device.type == "mps" and base.device.type != "mps":
        base.to(device)

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=not for_inference,
    )

    if for_inference:
        model.eval()

    return model, tokenizer


def format_chat_prompt(
    question: str,
    tokenizer: AutoTokenizer,
) -> str:
    """Apply Mistral's chat template to a user question.

    Args:
        question: The user's natural-language query.
        tokenizer: The tokenizer whose ``chat_template`` will be used.

    Returns:
        The formatted prompt string with a generation prompt appended.
    """
    messages = [{"role": "user", "content": question.strip()}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
