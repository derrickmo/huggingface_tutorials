#!/usr/bin/env python3
"""
OCR (Optical Character Recognition) CLI Tool

Extract text from images using TrOCR models.

Examples:
    python ocr.py document.jpg
    python ocr.py receipt.png --model small
    python ocr.py image.jpg --output extracted_text.txt

Example Output:
    $ python ocr.py receipt.jpg

    Loading model: microsoft/trocr-small-printed...
    Using device: CPU
    Loading image: receipt.jpg
    Image size: (800, 600)

    Extracting text from image...

    ======================================================================
    EXTRACTED TEXT
    ======================================================================

    Invoice #12345
    Date: 2024-01-15
    Total: $127.50

    ======================================================================

    $ python ocr.py document.png --output text.txt

    Loading model: microsoft/trocr-small-printed...
    Extracting text from image...

    ======================================================================
    EXTRACTED TEXT
    ======================================================================

    This is a sample document with printed text.
    The model extracts text line by line.

    ======================================================================

    Text saved to: text.txt

Note:
    The 'large' model option refers to PaddleOCR, which requires separate
    installation and is not included in this CLI tool. PaddleOCR offers:
    - 80+ languages support
    - Better accuracy for complex layouts
    - Table and form detection

    To use PaddleOCR, install separately:
        pip install paddlepaddle paddleocr

    See Notebook 06 for PaddleOCR usage examples.
"""

import argparse
import sys
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def extract_text_from_image(image_path, model_size='small'):
    """
    Extract text from an image using OCR.

    Args:
        image_path: Path to image file
        model_size: 'small' (trocr-small-printed) or 'large' (note about PaddleOCR)

    Returns:
        Extracted text string
    """
    # Select model (2+1 pattern: small model, large option notes PaddleOCR)
    if model_size == 'large':
        print("=" * 70)
        print("LARGE MODEL OPTION: PaddleOCR")
        print("=" * 70)
        print("\nThe 'large' option refers to PaddleOCR, which offers:")
        print("  - 80+ languages support")
        print("  - Superior accuracy for complex layouts")
        print("  - Table and form detection")
        print("  - Production-grade OCR")
        print("\nTo use PaddleOCR, install separately:")
        print("  pip install paddlepaddle paddleocr")
        print("\nSee Notebook 06 (notebooks/02_computer_vision/06_cv_ocr.ipynb)")
        print("for complete PaddleOCR usage examples.")
        print("=" * 70)
        sys.exit(0)

    model_name = "microsoft/trocr-small-printed"  # 558MB, English printed text

    print(f"Loading model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'GPU' if device == 'cuda' else 'CPU'}")

    # Load processor and model
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model = model.to(device)

    # Load image
    print(f"Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image size: {image.size}, Mode: {image.mode}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    print(f"\nExtracting text from image...")

    # Process image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from images using TrOCR models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.jpg
  %(prog)s receipt.png --model small
  %(prog)s image.jpg --output extracted_text.txt

Supported formats: JPG, PNG, TIFF, BMP

Model Options:
  small: microsoft/trocr-small-printed (558MB, English printed text)
  large: PaddleOCR (requires separate installation - see help message)

Note:
  - Works best with single lines of printed text
  - For multi-line documents, process line by line
  - Higher resolution images = better accuracy
  - Clean backgrounds improve results
        """
    )

    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (trocr-small-printed, 558MB) or large (PaddleOCR info)')
    parser.add_argument('--output', '-o', type=str,
                        help='Save extracted text to file')

    args = parser.parse_args()

    try:
        # Extract text
        text = extract_text_from_image(
            args.image,
            model_size=args.model
        )

        # Print results
        print("\n" + "=" * 70)
        print("EXTRACTED TEXT")
        print("=" * 70)
        print(f"\n{text}\n")
        print("=" * 70)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text + '\n')
            print(f"\nText saved to: {args.output}")

    except FileNotFoundError:
        print(f"Error: Image file '{args.image}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOCR cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
