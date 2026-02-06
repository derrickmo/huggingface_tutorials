#!/usr/bin/env python3
"""
Text Generation CLI Tool

Generate text completions using HuggingFace transformer models.

Examples:
    python text_generation.py "Once upon a time"
    python text_generation.py "The future of AI is" --model large --max-length 100
    python text_generation.py "Hello" --temperature 0.9 --num-sequences 3

Example Output:
    $ python text_generation.py "The quick brown fox"

    Loading model: distilgpt2...
    Using device: CPU

    Generating 1 completion(s)...

    ======================================================================
    GENERATED TEXT
    ======================================================================
    The quick brown fox and the brown fox are both very similar in appearance,
    but the brown fox is much more aggressive and can be much more aggressive
    than the brown fox.
    ======================================================================

    $ python text_generation.py "Once upon a time" --num-sequences 2 --max-length 60

    ======================================================================
    GENERATED TEXT
    ======================================================================

    --- Completion 1 ---
    Once upon a time, I was a very good student. I was a very good student,
    and I was a very good student. I was a very good student, and I was a
    very good student.

    --- Completion 2 ---
    Once upon a time, the world was a very different place. The world was a
    very different place. The world was a very different place. The world
    was a very different place.
    ======================================================================
"""

import argparse
import sys
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


def generate_text(prompt, model_size='small', max_length=50, temperature=0.7,
                  num_sequences=1, top_k=50, top_p=0.95):
    """
    Generate text using a language model.

    Args:
        prompt: Input text to complete
        model_size: 'small' (distilgpt2) or 'large' (gpt2-medium)
        max_length: Maximum length of generated text
        temperature: Sampling temperature (0.1-2.0)
        num_sequences: Number of different completions
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter

    Returns:
        List of generated texts
    """
    # Select model
    models = {
        'small': 'distilgpt2',
        'large': 'gpt2-medium'
    }
    model_name = models.get(model_size, 'distilgpt2')

    print(f"Loading model: {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Create pipeline
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=device
    )

    # Generate
    print(f"\nGenerating {num_sequences} completion(s)...")
    results = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_sequences,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=50256  # GPT-2 EOS token
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate text completions using transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Once upon a time"
  %(prog)s "The future of AI is" --model large --max-length 100
  %(prog)s "Hello world" --temperature 0.9 --num-sequences 3
        """
    )

    parser.add_argument('prompt', type=str, help='Input text prompt to complete')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (distilgpt2, 82MB) or large (gpt2-medium, 1.5GB)')
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum length of generated text (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature: lower=conservative, higher=creative (default: 0.7)')
    parser.add_argument('--num-sequences', type=int, default=1,
                        help='Number of different completions to generate (default: 1)')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling parameter (default: 50)')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Nucleus sampling parameter (default: 0.95)')

    args = parser.parse_args()

    # Validate arguments
    if args.temperature <= 0:
        print("Error: temperature must be positive", file=sys.stderr)
        sys.exit(1)

    if args.num_sequences < 1:
        print("Error: num-sequences must be at least 1", file=sys.stderr)
        sys.exit(1)

    try:
        # Generate text
        results = generate_text(
            args.prompt,
            model_size=args.model,
            max_length=args.max_length,
            temperature=args.temperature,
            num_sequences=args.num_sequences,
            top_k=args.top_k,
            top_p=args.top_p
        )

        # Print results
        print("\n" + "="*70)
        print("GENERATED TEXT")
        print("="*70)

        for i, result in enumerate(results, 1):
            if args.num_sequences > 1:
                print(f"\n--- Completion {i} ---")
            print(result['generated_text'])

        print("\n" + "="*70)

    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
