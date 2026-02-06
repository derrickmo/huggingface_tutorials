#!/usr/bin/env python3
"""
Image Classification CLI Tool

Classify images using Vision Transformer models.

Examples:
    python image_classification.py image.jpg
    python image_classification.py photo.png --model large --top-k 10
    python image_classification.py cat.jpg --threshold 0.5

Example Output:
    $ python image_classification.py dog.jpg

    Loading model: google/vit-base-patch16-224...
    Using device: CPU
    Loading image: dog.jpg
    Image size: (800, 600)

    Classifying image...

    ======================================================================
    IMAGE CLASSIFICATION RESULTS
    ======================================================================

    Image: dog.jpg

    Top 5 predictions:
    1. golden_retriever          95.42%  ██████████████████████████████
    2. Labrador_retriever        2.18%   ██
    3. kuvasz                    0.89%   █
    4. English_setter            0.34%
    5. clumber                   0.21%

    ======================================================================

    $ python image_classification.py scene.jpg --top-k 3 --threshold 0.1

    Top 3 predictions:
    1. park_bench                34.27%  ██████████████████
    2. lakeside                  28.91%  ███████████████
    3. palace                    12.43%  ███████

    (Only showing predictions >= 10.0%)
"""

import argparse
import sys
import torch
from transformers import pipeline
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def classify_image(image_path, model_size='small', top_k=5):
    """
    Classify an image.

    Args:
        image_path: Path to image file
        model_size: 'small' (vit-base) or 'large' (vit-large)
        top_k: Number of top predictions to return

    Returns:
        Classification results
    """
    # Select model
    models = {
        'small': 'google/vit-base-patch16-224',
        'large': 'google/vit-large-patch16-224'
    }
    model_name = models.get(model_size, models['small'])

    print(f"Loading model: {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Create pipeline
    classifier = pipeline(
        "image-classification",
        model=model_name,
        device=device
    )

    # Load image
    print(f"Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"Image size: {image.size}, Mode: {image.mode}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Classify
    print(f"\nClassifying image...")
    results = classifier(image, top_k=top_k)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Classify images using Vision Transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg
  %(prog)s photo.png --model large --top-k 10
  %(prog)s cat.jpg --threshold 0.5
        """
    )

    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (vit-base, 346MB) or large (vit-large, 1.2GB)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to show (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Only show predictions with score >= threshold (default: 0.0)')

    args = parser.parse_args()

    # Validate arguments
    if args.top_k < 1:
        print("Error: top-k must be at least 1", file=sys.stderr)
        sys.exit(1)

    if args.threshold < 0 or args.threshold > 1:
        print("Error: threshold must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    try:
        # Classify image
        results = classify_image(
            args.image,
            model_size=args.model,
            top_k=args.top_k
        )

        # Filter by threshold
        results = [r for r in results if r['score'] >= args.threshold]

        # Print results
        print("\n" + "="*70)
        print("CLASSIFICATION RESULTS")
        print("="*70)

        if not results:
            print(f"\nNo predictions found with confidence >= {args.threshold}")
        else:
            print(f"\nTop {len(results)} prediction(s):\n")
            for i, result in enumerate(results, 1):
                bar_length = int(result['score'] * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"{i:2d}. {result['label']:30s} {bar} {result['score']:.4f}")

        print("\n" + "="*70)

    except FileNotFoundError:
        print(f"Error: Image file '{args.image}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nClassification cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
