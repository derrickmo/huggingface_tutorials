#!/usr/bin/env python3
"""
Speech Recognition CLI Tool

Transcribe audio files using Whisper models.

Examples:
    python speech_recognition.py audio.wav
    python speech_recognition.py speech.mp3 --model large --timestamps
    python speech_recognition.py recording.flac --output transcript.txt

Example Output:
    $ python speech_recognition.py interview.mp3

    Loading model: openai/whisper-tiny...
    Using device: CPU
    Loading audio: interview.mp3

    Transcribing audio...

    ======================================================================
    TRANSCRIPTION
    ======================================================================

    Hello and welcome to today's podcast. In this episode, we'll be
    discussing the latest developments in artificial intelligence and
    machine learning. Our guest today is Dr. Sarah Johnson, who has been
    working in this field for over 15 years.

    ======================================================================

    $ python speech_recognition.py lecture.wav --timestamps

    ======================================================================
    TRANSCRIPTION (with timestamps)
    ======================================================================

    [00:00:00.000 -> 00:00:04.280]  Good morning everyone and thank you for joining us today.
    [00:00:04.280 -> 00:00:08.960]  We'll be covering three main topics in this lecture.
    [00:00:08.960 -> 00:00:14.120]  First, we'll discuss the fundamentals of neural networks.
    [00:00:14.120 -> 00:00:18.440]  Second, we'll explore convolutional architectures.
    [00:00:18.440 -> 00:00:23.200]  And finally, we'll look at some practical applications.

    ======================================================================

    $ python speech_recognition.py audio.wav --model large --output transcript.txt

    Transcription saved to: transcript.txt
"""

import argparse
import sys
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


def transcribe_audio(audio_path, model_size='small', return_timestamps=False):
    """
    Transcribe audio to text.

    Args:
        audio_path: Path to audio file
        model_size: 'small' (fast) or 'large' (accurate)
        return_timestamps: Whether to include timestamps

    Returns:
        Transcription result
    """
    # Select model (2+1 pattern: small + large)
    models = {
        'small': 'openai/whisper-tiny',    # Small model (72MB, CPU-friendly)
        'large': 'openai/whisper-small'    # Large model (483MB, better accuracy)
    }
    model_name = models.get(model_size, models['small'])

    print(f"Loading model: {model_name}...")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Create pipeline
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device
    )

    print(f"Transcribing: {audio_path}")
    result = asr(audio_path, return_timestamps=return_timestamps)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe audio files using Whisper models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s speech.mp3 --model large --timestamps
  %(prog)s recording.flac --output transcript.txt

Supported formats: WAV, MP3, FLAC, OGG, M4A
        """
    )

    parser.add_argument('audio', type=str, help='Path to audio file')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (72MB, fast), large (483MB, better accuracy)')
    parser.add_argument('--timestamps', action='store_true',
                        help='Include timestamps in output')
    parser.add_argument('--output', '-o', type=str,
                        help='Save transcription to file')

    args = parser.parse_args()

    try:
        # Transcribe
        result = transcribe_audio(
            args.audio,
            model_size=args.model,
            return_timestamps=args.timestamps
        )

        # Format output
        print("\n" + "="*70)
        print("TRANSCRIPTION")
        print("="*70)

        if args.timestamps and 'chunks' in result:
            print("\nWith timestamps:\n")
            for chunk in result['chunks']:
                start, end = chunk['timestamp']
                text = chunk['text']
                print(f"[{start:.2f}s - {end:.2f}s]: {text}")
            print(f"\nFull text:\n{result['text']}")
        else:
            print(f"\n{result['text']}")

        print("\n" + "="*70)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.timestamps and 'chunks' in result:
                    for chunk in result['chunks']:
                        start, end = chunk['timestamp']
                        text = chunk['text']
                        f.write(f"[{start:.2f}s - {end:.2f}s]: {text}\n")
                    f.write(f"\nFull text:\n{result['text']}\n")
                else:
                    f.write(result['text'] + '\n')
            print(f"Transcription saved to: {args.output}")

    except FileNotFoundError:
        print(f"Error: Audio file '{args.audio}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTranscription cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
