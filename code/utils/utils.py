#!/usr/bin/env python3
"""
Utility module for SEAM benchmark.
Contains helper functions for image processing, file operations, and result analysis.
"""

# Standard library imports
import base64
import hashlib
import io
import os
from io import BytesIO
from typing import Optional, List, Union

# Third-party imports  
from PIL import Image


# ============================================================================
# Image Encoding and Processing Functions
# ============================================================================

def encode_image(image_path: str, max_size: tuple = (1280, 1280)) -> str:
    """
    Encode and optionally resize an image to base64.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimensions as (width, height) tuple
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        img = Image.open(image_file)
        img.load()
    
    img.thumbnail(max_size, Image.LANCZOS)
    
    # Convert RGBA to RGB if necessary
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = rgb_img
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_encoding(pil_image: Image.Image) -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    # Convert PIL Image to bytes using BytesIO buffer
    buffered = BytesIO()
    # Save image to buffer in PNG format
    pil_image.save(buffered, format="PNG")
    # Get the byte value from the buffer
    img_bytes = buffered.getvalue()
    # Encode to base64 and convert to string
    return base64.b64encode(img_bytes).decode("utf-8")


# ============================================================================
# Domain-Specific Image Loading Functions
# ============================================================================

def read_image_chess(cfg, fen: str, directory: str) -> Image.Image:
    """
    Load chess board image from FEN notation.
    
    Args:
        cfg: Configuration object with benchmark_root
        fen: FEN notation string
        directory: Image directory name
        
    Returns:
        PIL Image object
    """
    filename = fen.split(" ")[0].replace("/", "_") + ".png"
    img_path = os.path.join(cfg.benchmark_root, directory, filename)
    return Image.open(img_path).convert("RGB")


def read_image_chem(cfg, smiles: str, directory: str) -> Image.Image:
    """
    Load chemistry molecule image from SMILES notation.
    
    Args:
        cfg: Configuration object with benchmark_root
        smiles: SMILES notation string
        directory: Image directory name
        
    Returns:
        PIL Image object
    """
    filename = hash_str(smiles) + ".png"
    img_path = os.path.join(cfg.benchmark_root, directory, filename)
    return Image.open(img_path).convert("RGB")


def read_image_music(cfg, index: Union[int, str], directory: str) -> Image.Image:
    """
    Load music notation image by index.
    
    Args:
        cfg: Configuration object with benchmark_root
        index: Image index
        directory: Image directory name
        
    Returns:
        PIL Image object
    """
    filename = f"{index}.png"
    img_path = os.path.join(cfg.benchmark_root, directory, filename)
    return Image.open(img_path).convert("RGB")


def read_image_graph(cfg, idx: Union[int, str], directory: str) -> Image.Image:
    """
    Load graph visualization image by index.
    
    Args:
        cfg: Configuration object with benchmark_root
        idx: Image index
        directory: Image directory name
        
    Returns:
        PIL Image object
    """
    filename = f"{idx}.png"
    img_path = os.path.join(cfg.benchmark_root, directory, filename)
    return Image.open(img_path).convert("RGB")


# ============================================================================
# String and File Utilities
# ============================================================================

def hash_str(input_string: str) -> str:
    """
    Generate MD5 hash of input string.
    
    Args:
        input_string: String to hash
        
    Returns:
        MD5 hash as hexadecimal string
    """
    return hashlib.md5(input_string.encode()).hexdigest()


def get_model_safe_name(model_name: str) -> str:
    """
    Convert model name to filesystem-safe name.
    
    Args:
        model_name: Original model name (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')
        
    Returns:
        Filesystem-safe version (e.g., 'Qwen_Qwen2.5-VL-7B-Instruct')
    """
    return model_name.replace('/', '_').replace(':', '_')


# ============================================================================
# Result Analysis Functions
# ============================================================================

def show_results_exact_match(results: List[int]) -> None:
    """
    Display exact match results statistics.
    
    Args:
        results: List of result codes (0=correct, 1=wrong, 2=invalid)
    """
    if not results:
        print("No results to analyze", flush=True)
        return
    
    correct = 0
    wrong = 0
    invalid = 0
    
    for result in results:
        if result == 0:
            correct += 1
        elif result == 1:
            wrong += 1
        elif result == 2:
            invalid += 1
    
    total = len(results)
    correct_rate = round(correct / total, 3)
    wrong_rate = round(wrong / total, 3)
    invalid_rate = round(invalid / total, 3)
    
    print(f"Correct: {correct_rate}, Wrong: {wrong_rate}, Invalid Rate: {invalid_rate}, Total: {total}", flush=True)