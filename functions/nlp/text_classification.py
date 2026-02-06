#!/usr/bin/env python3
"""
Text Classification CLI Tool

Classify text sentiment using HuggingFace transformer models.

Examples:
    python text_classification.py "I love this product!"
    python text_classification.py "This is terrible" --model large
    python text_classification.py --file reviews.txt

Example Output:
    $ python text_classification.py "This movie was absolutely amazing!"

    Loading model: distilbert-base-uncased-finetuned-sst-2-english...
    Using device: CPU

    Classifying 1 text(s)...

    ======================================================================
    CLASSIFICATION RESULTS
    ======================================================================

    😊 POSITIVE (confidence: 0.9998)
    Text: This movie was absolutely amazing!

    ======================================================================

    $ python text_classification.py --file reviews.txt

    Loading model: distilbert-base-uncased-finetuned-sst-2-english...
    Classifying 3 text(s)...

    ======================================================================
    CLASSIFICATION RESULTS
    ======================================================================

    😊 POSITIVE (confidence: 0.9995)
    Text: Great product! Highly recommend.

    😞 NEGATIVE (confidence: 0.9987)
    Text: Terrible service. Very disappointed.

    😊 POSITIVE (confidence: 0.8342)
    Text: It's okay, nothing special.

    ======================================================================

    Summary: 2/3 positive (66.7%)
"""

import argparse
import sys
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


def classify_text(texts, model_size='small'):
    """
    Classify text sentiment.

    Args:
        texts: Single text string or list of texts
        model_size: 'small' (distilbert) or 'large' (bert-base)

    Returns:
        Classification results
    """
    # Select model
    models = {
        'small': 'distilbert-base-uncased-finetuned-sst-2-english',
        'large': 'bert-base-uncased'
    }
    model_name = models.get(model_size, models['small'])

    print(f"Loading model: {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Create pipeline
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device
    )

    # Ensure texts is a list
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]

    print(f"\nClassifying {len(texts)} text(s)...")
    results = classifier(texts)

    return results[0] if is_single else results


def main():
    parser = argparse.ArgumentParser(
        description='Classify text sentiment using transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "I love this product!"
  %(prog)s "This is terrible" --model large
  %(prog)s --file reviews.txt
  %(prog)s --file data.txt --threshold 0.8
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('text', nargs='?', type=str, help='Text to classify')
    group.add_argument('--file', type=str, help='File containing texts (one per line)')

    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (distilbert, 268MB) or large (bert-base, 440MB)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Only show predictions with confidence >= threshold (default: 0.0)')

    args = parser.parse_args()

    try:
        # Load texts
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            if not texts:
                print("Error: File is empty or contains no valid text", file=sys.stderr)
                sys.exit(1)
        else:
            texts = args.text

        # Classify
        results = classify_text(texts, model_size=args.model)

        # Ensure results is a list
        if isinstance(results, dict):
            results = [results]
            texts = [texts]

        # Print results
        print("\n" + "="*70)
        print("CLASSIFICATION RESULTS")
        print("="*70)

        for i, (text, result) in enumerate(zip(texts if isinstance(texts, list) else [texts], results), 1):
            if result['score'] >= args.threshold:
                emoji = "😊" if result['label'] == 'POSITIVE' else "😞"
                print(f"\n{emoji} {result['label']} (confidence: {result['score']:.4f})")
                print(f"Text: {text}")

        print("\n" + "="*70)

        # Summary statistics
        if isinstance(results, list) and len(results) > 1:
            positive = sum(1 for r in results if r['label'] == 'POSITIVE' and r['score'] >= args.threshold)
            total = len([r for r in results if r['score'] >= args.threshold])
            if total > 0:
                print(f"\nSummary: {positive}/{total} positive ({positive/total*100:.1f}%)")

    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nClassification cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
