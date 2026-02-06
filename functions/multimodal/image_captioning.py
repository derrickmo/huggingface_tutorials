#!/usr/bin/env python3
"""
Image Captioning CLI Tool

Generate text descriptions of images using multimodal models.

Examples:
    python image_captioning.py photo.jpg
    python image_captioning.py image.png --model large
    python image_captioning.py cat.jpg --num-captions 3

Example Output:
    $ python image_captioning.py sunset.jpg

    Loading model: Salesforce/blip-image-captioning-base...
    Using device: CPU
    Loading image: sunset.jpg
    Image size: (1920, 1080)

    Generating 1 caption(s)...

    ======================================================================
    IMAGE CAPTIONS
    ======================================================================

    a beautiful sunset over the ocean with orange and pink clouds

    ======================================================================

    $ python image_captioning.py family_photo.jpg --num-captions 3

    Loading model: Salesforce/blip-image-captioning-base...
    Generating 3 caption(s)...

    ======================================================================
    IMAGE CAPTIONS
    ======================================================================

    Generated captions:

    1. a group of people posing for a photo in front of a house
    2. a family of four standing together outside
    3. people smiling at the camera on a sunny day

    ======================================================================

    $ python image_captioning.py scene.jpg --model large --max-length 100

    Using large model for higher quality captions...

    a bustling city street with tall buildings, people walking on the sidewalk,
    cars driving by, and street vendors selling food under colorful umbrellas
    on a warm summer afternoon
"""

import argparse
import sys
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def caption_image(image_path, model_size='small', max_length=50, num_captions=1):
    """
    Generate captions for an image.

    Args:
        image_path: Path to image file
        model_size: 'small' (blip-base) or 'large' (blip-large)
        max_length: Maximum caption length
        num_captions: Number of different captions to generate

    Returns:
        List of captions
    """
    # Select model
    models = {
        'small': 'Salesforce/blip-image-captioning-base',
        'large': 'Salesforce/blip-image-captioning-large'
    }
    model_name = models.get(model_size, models['small'])

    print(f"Loading model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'GPU' if device == 'cuda' else 'CPU'}")

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # Load image
    print(f"Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Image size: {image.size}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Generate captions
    print(f"\nGenerating {num_captions} caption(s)...")
    captions = []

    for i in range(num_captions):
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate
        if num_captions > 1:
            # Use sampling for variety
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                do_sample=True,
                temperature=0.7
            )
        else:
            # Deterministic for single caption
            generated_ids = model.generate(**inputs, max_length=max_length)

        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        captions.append(caption)

    return captions


def main():
    parser = argparse.ArgumentParser(
        description='Generate text captions for images using multimodal models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg
  %(prog)s image.png --model large
  %(prog)s cat.jpg --num-captions 3
        """
    )

    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (blip-base, 990MB) or large (blip-large, 1.9GB)')
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum caption length (default: 50)')
    parser.add_argument('--num-captions', type=int, default=1,
                        help='Number of different captions to generate (default: 1)')

    args = parser.parse_args()

    # Validate arguments
    if args.num_captions < 1:
        print("Error: num-captions must be at least 1", file=sys.stderr)
        sys.exit(1)

    try:
        # Generate captions
        captions = caption_image(
            args.image,
            model_size=args.model,
            max_length=args.max_length,
            num_captions=args.num_captions
        )

        # Print results
        print("\n" + "="*70)
        print("IMAGE CAPTIONS")
        print("="*70)

        if len(captions) == 1:
            print(f"\n{captions[0]}")
        else:
            print("\nGenerated captions:\n")
            for i, caption in enumerate(captions, 1):
                print(f"{i}. {caption}")

        print("\n" + "="*70)

    except FileNotFoundError:
        print(f"Error: Image file '{args.image}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCaptioning cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
