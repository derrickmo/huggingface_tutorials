#!/usr/bin/env python3
"""
Text Summarization CLI Tool

Summarize text documents using transformer models.

Examples:
    python text_summarization.py --file article.txt
    python text_summarization.py --file document.txt --model large --max-length 150
    python text_summarization.py --text "Long text here..." --output summary.txt

Example Output:
    $ python text_summarization.py --file article.txt

    Loading model: sshleifer/distilbart-cnn-12-6...
    Using device: CPU

    Summarizing text (1247 characters)...

    ======================================================================
    SUMMARY
    ======================================================================

    Scientists have discovered a new species of butterfly in the Amazon
    rainforest. The butterfly, named Morpho cypris amazonensis, has
    distinctive bright blue wings and is believed to be endemic to a
    small region near the Peruvian border. Researchers estimate there
    may be fewer than 500 individuals remaining in the wild.

    ======================================================================

    Original length: 1247 characters
    Summary length: 312 characters
    Compression ratio: 75.0%

    $ python text_summarization.py --text "Your long text..." --output summary.txt

    Summary saved to: summary.txt
"""

import argparse
import sys
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


def summarize_text(text, model_size='small', max_length=130, min_length=30):
    """
    Summarize text.

    Args:
        text: Input text to summarize
        model_size: 'small' (distilbart) or 'large' (bart-large)
        max_length: Maximum summary length
        min_length: Minimum summary length

    Returns:
        Summary text
    """
    # Select model
    models = {
        'small': 'sshleifer/distilbart-cnn-12-6',
        'large': 'facebook/bart-large-cnn'
    }
    model_name = models.get(model_size, models['small'])

    print(f"Loading model: {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Create pipeline
    summarizer = pipeline(
        "summarization",
        model=model_name,
        device=device
    )

    # Count words
    word_count = len(text.split())
    print(f"Input text: {word_count} words")

    # Summarize
    print(f"Generating summary...")
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )

    return result[0]['summary_text']


def main():
    parser = argparse.ArgumentParser(
        description='Summarize text documents using transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file article.txt
  %(prog)s --file document.txt --model large --max-length 150
  %(prog)s --text "Long text..." --output summary.txt
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='File containing text to summarize')
    group.add_argument('--text', type=str, help='Text to summarize (use quotes)')

    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (distilbart, 1.2GB) or large (bart-large, 1.6GB)')
    parser.add_argument('--max-length', type=int, default=130,
                        help='Maximum summary length (default: 130)')
    parser.add_argument('--min-length', type=int, default=30,
                        help='Minimum summary length (default: 30)')
    parser.add_argument('--output', '-o', type=str,
                        help='Save summary to file')

    args = parser.parse_args()

    # Validate arguments
    if args.min_length >= args.max_length:
        print("Error: min-length must be less than max-length", file=sys.stderr)
        sys.exit(1)

    try:
        # Load text
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                print("Error: File is empty", file=sys.stderr)
                sys.exit(1)
        else:
            text = args.text

        # Check text length
        if len(text.split()) < 50:
            print("Warning: Text is very short. Summarization may not be meaningful.", file=sys.stderr)

        # Summarize
        summary = summarize_text(
            text,
            model_size=args.model,
            max_length=args.max_length,
            min_length=args.min_length
        )

        # Print results
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n{summary}")
        print("\n" + "="*70)

        summary_words = len(summary.split())
        original_words = len(text.split())
        compression = (1 - summary_words/original_words) * 100
        print(f"\nSummary: {summary_words} words (compressed by {compression:.1f}%)")

        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary + '\n')
            print(f"Summary saved to: {args.output}")

    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSummarization cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
