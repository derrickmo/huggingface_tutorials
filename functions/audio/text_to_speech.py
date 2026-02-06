#!/usr/bin/env python3
"""
Text-to-Speech (TTS) CLI Tool

Convert text to natural-sounding speech using SpeechT5 models.

Examples:
    python text_to_speech.py "Hello world" --output hello.wav
    python text_to_speech.py --file script.txt --output speech.wav
    python text_to_speech.py "Welcome to the tutorial" --output welcome.wav --voice 2

Example Output:
    $ python text_to_speech.py "Hello, welcome to HuggingFace!" --output hello.wav

    Loading TTS model: microsoft/speecht5_tts...
    Loading vocoder: microsoft/speecht5_hifigan...
    Using device: CPU
    Loading speaker embeddings...

    Generating speech from text...
    Text: "Hello, welcome to HuggingFace!"

    ======================================================================
    SPEECH GENERATION COMPLETE
    ======================================================================

    Audio saved to: hello.wav
    Duration: ~2.5 seconds
    Sample rate: 16000 Hz
    File size: 78 KB

    ======================================================================

    $ python text_to_speech.py --file script.txt --output narration.wav

    Loading TTS model...
    Reading text from: script.txt
    Text length: 156 characters

    Generating speech from text...

    Audio saved to: narration.wav
    Duration: ~8.3 seconds
    Sample rate: 16000 Hz

    $ python text_to_speech.py "Testing different voices" --output test.wav --voice 5

    Loading speaker embeddings (voice 5)...
    Generating speech with voice 5...
    Audio saved to: test.wav

Note:
    Only one model is available for this tool (SpeechT5). This is because
    SpeechT5 offers the best balance of quality and ease of use for
    educational purposes.

    For advanced TTS needs, consider:
    - XTTS v2: Voice cloning, multilingual (requires separate installation)
    - Bark: Emotional speech, music, sound effects (requires separate installation)
    - VITS: Faster inference (language-specific models)

    See Notebook 08 for more details on advanced TTS models.
"""

import argparse
import sys
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')


def text_to_speech(text, output_file, voice_id=0):
    """
    Convert text to speech and save to file.

    Args:
        text: Text to convert to speech
        output_file: Output WAV file path
        voice_id: Speaker voice ID (0-7306, default: 0)

    Returns:
        None (saves audio file)
    """
    MODEL_NAME = "microsoft/speecht5_tts"
    VOCODER_NAME = "microsoft/speecht5_hifigan"

    print(f"Loading TTS model: {MODEL_NAME}...")
    print(f"Loading vocoder: {VOCODER_NAME}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'GPU' if device == 'cuda' else 'CPU'}")

    # Load model components
    processor = SpeechT5Processor.from_pretrained(MODEL_NAME)
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_NAME)
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_NAME)

    model = model.to(device)
    vocoder = vocoder.to(device)

    # Load speaker embeddings
    print(f"Loading speaker embeddings (voice {voice_id})...")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    # Validate voice_id
    if voice_id < 0 or voice_id >= len(embeddings_dataset):
        print(f"Warning: voice_id {voice_id} out of range (0-{len(embeddings_dataset)-1}). Using default voice 0.")
        voice_id = 0

    speaker_embeddings = torch.tensor(embeddings_dataset[voice_id]["xvector"]).unsqueeze(0).to(device)

    print(f"\nGenerating speech from text...")
    print(f'Text: "{text[:100]}{"..." if len(text) > 100 else ""}"')

    # Process text
    inputs = processor(text=text, return_tensors="pt").to(device)

    # Generate speech
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=vocoder
        )

    # Save audio
    speech_np = speech.cpu().numpy()
    sf.write(output_file, speech_np, samplerate=16000)

    # Calculate duration
    duration = len(speech_np) / 16000

    return speech_np, duration


def main():
    parser = argparse.ArgumentParser(
        description='Convert text to speech using SpeechT5 TTS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world" --output hello.wav
  %(prog)s --file script.txt --output speech.wav
  %(prog)s "Test" --output test.wav --voice 5

Model Information:
  This tool uses SpeechT5, which provides high-quality English TTS.
  Only one model is available as it offers the best balance for learning.

  For advanced features, see Notebook 08:
  - XTTS v2: Voice cloning, multilingual
  - Bark: Emotional speech, music, sound effects
  - VITS: Faster inference

Voice Selection:
  Use --voice to select different speaker characteristics (0-7306)
  Default is voice 0 (balanced, neutral)
  Try different values to hear variations in voice quality

Tips for Best Results:
  - Use proper punctuation (affects pauses and intonation)
  - Spell out numbers: "twenty-three" instead of "23"
  - Expand abbreviations: "Doctor" instead of "Dr."
  - Split very long text (>200 words) into multiple generations
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('text', nargs='?', type=str, help='Text to convert to speech')
    group.add_argument('--file', type=str, help='File containing text to convert')

    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output WAV file path (required)')
    parser.add_argument('--voice', type=int, default=0,
                        help='Speaker voice ID (0-7306, default: 0)')

    args = parser.parse_args()

    try:
        # Load text
        if args.file:
            print(f"Reading text from: {args.file}")
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                print("Error: File is empty or contains no text", file=sys.stderr)
                sys.exit(1)
            print(f"Text length: {len(text)} characters")
        else:
            text = args.text

        # Validate text length
        if len(text) > 1000:
            print("\nWarning: Text is very long (>1000 characters).")
            print("Generation may take a while. Consider splitting into smaller chunks.")

        # Generate speech
        speech_np, duration = text_to_speech(text, args.output, voice_id=args.voice)

        # Print results
        print("\n" + "=" * 70)
        print("SPEECH GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nAudio saved to: {args.output}")
        print(f"Duration: ~{duration:.1f} seconds")
        print(f"Sample rate: 16000 Hz")
        print(f"Samples: {len(speech_np)}")
        print("\n" + "=" * 70)

        print(f"\nYou can play the audio file with any media player or in Jupyter:")
        print(f"  from IPython.display import Audio")
        print(f"  Audio('{args.output}')")

    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSpeech generation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
