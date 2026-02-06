"""
Shared Utility Functions for HuggingFace Tutorial Notebooks

This module contains common helper functions used across multiple notebooks
to avoid code duplication.

Usage in notebooks:
    from shared_utils import load_image_from_url, setup_device, print_device_info
"""

import requests
from io import BytesIO
from PIL import Image
import torch


def load_image_from_url(url):
    """
    Load an image from a URL.

    Args:
        url (str): URL of the image to load

    Returns:
        PIL.Image: Loaded image

    Example:
        >>> img = load_image_from_url("https://example.com/image.jpg")
        >>> img.size
        (800, 600)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {e}")


def setup_device(prefer_gpu=True):
    """
    Determine the best available device (GPU or CPU).

    Args:
        prefer_gpu (bool): Whether to prefer GPU if available (default: True)

    Returns:
        str: Device string ("cuda" or "cpu")

    Example:
        >>> device = setup_device()
        >>> print(f"Using: {device}")
        Using: cuda
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_device_info():
    """
    Print PyTorch version and CUDA availability information.

    This is commonly used at the start of notebooks to show the environment.

    Example output:
        PyTorch version: 2.0.1
        CUDA available: True
        GPU: NVIDIA GeForce RTX 4080
    """
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


def get_device_string(use_gpu=True):
    """
    Get device identifier for HuggingFace pipeline.

    Args:
        use_gpu (bool): Whether to use GPU if available

    Returns:
        int: Device ID (0 for GPU, -1 for CPU)

    Example:
        >>> device_id = get_device_string(use_gpu=True)
        >>> pipe = pipeline("task", model="model-name", device=device_id)
    """
    if use_gpu and torch.cuda.is_available():
        return 0  # GPU device 0
    return -1  # CPU


def format_size_mb(size_bytes):
    """
    Format size in bytes to megabytes.

    Args:
        size_bytes (int): Size in bytes

    Returns:
        str: Formatted size string

    Example:
        >>> format_size_mb(500_000_000)
        '476.84 MB'
    """
    size_mb = size_bytes / (1024 ** 2)
    return f"{size_mb:.2f} MB"


def format_size_gb(size_bytes):
    """
    Format size in bytes to gigabytes.

    Args:
        size_bytes (int): Size in bytes

    Returns:
        str: Formatted size string

    Example:
        >>> format_size_gb(2_000_000_000)
        '1.86 GB'
    """
    size_gb = size_bytes / (1024 ** 3)
    return f"{size_gb:.2f} GB"
