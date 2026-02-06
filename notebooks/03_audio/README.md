# Audio Processing

This folder contains 2 notebooks covering speech recognition and text-to-speech synthesis.

## Notebooks

### Notebook 07: Speech Recognition (ASR)
**Concepts**: Automatic Speech Recognition, Whisper architecture
**Models**: openai/whisper-tiny (72MB), openai/whisper-small (483MB)
**Demo**: Transcribe speech from audio files

**Quick Demo:**
```python
from transformers import pipeline

asr = pipeline('automatic-speech-recognition', model='openai/whisper-tiny')
result = asr('audio.wav')
print(result['text'])
```

**Expected Output:**
```
Hello and welcome to today's podcast. In this episode, we'll be discussing
the latest developments in artificial intelligence and machine learning.
```

**With Timestamps:**
```python
result = asr('audio.wav', return_timestamps=True)
for chunk in result['chunks']:
    start, end = chunk['timestamp']
    text = chunk['text']
    print(f"[{start:.2f}s - {end:.2f}s]: {text}")
```

**Expected Output:**
```
[0.00s - 4.28s]: Hello and welcome to today's podcast.
[4.28s - 8.96s]: In this episode, we'll be discussing
[8.96s - 14.12s]: the latest developments in artificial intelligence.
```

**Supported Formats**: WAV, MP3, FLAC, OGG, M4A

---

### Notebook 08: Text-to-Speech (TTS)
**Concepts**: Speech synthesis, vocoder, prosody
**Models**: microsoft/speecht5_tts (118MB)
**Demo**: Generate natural-sounding speech from text

**Quick Demo:**
```python
from transformers import pipeline
import soundfile as sf

tts = pipeline('text-to-speech', model='microsoft/speecht5_tts')
text = "Hello, welcome to the HuggingFace tutorial."
result = tts(text)

# Save audio
sf.write('output.wav', result['audio'][0], samplerate=result['sampling_rate'])
print("Audio saved to output.wav")
```

**Voice Control:**
- Use speaker embeddings to change voice characteristics
- Adjust speaking rate and pitch
- Generate expressive speech

**Use Cases:**
- Audiobook narration
- Voice assistants
- Accessibility tools
- Language learning

---

## Hardware Requirements

| Notebook | Minimum | Recommended | Performance |
|----------|---------|-------------|-------------|
| 07 (ASR) | 8GB RAM (CPU) | 8GB VRAM (GPU) | Real-time on GPU |
| 08 (TTS) | 8GB RAM (CPU) | 8GB VRAM (GPU) | Fast synthesis on GPU |

**Performance Expectations:**
- **Whisper-tiny (CPU)**: 2-5 seconds per minute of audio
- **Whisper-small (GPU)**: Near real-time transcription
- **SpeechT5 (CPU)**: 1-2 seconds per sentence
- **SpeechT5 (GPU)**: <0.5 seconds per sentence

## Audio File Requirements

### For Speech Recognition (Notebook 07):
- **Format**: WAV, MP3, FLAC (16kHz recommended)
- **Quality**: Clear speech, minimal background noise
- **Length**: Any (models work on 30-second chunks)
- **Languages**: English (default), 100+ languages with Whisper

### For Text-to-Speech (Notebook 08):
- **Input**: Plain text strings
- **Output**: 16kHz WAV audio
- **Text Length**: Any (split into sentences automatically)

## Running the Demos

1. **Prepare audio files** (for ASR): Place `.wav` or `.mp3` files in `sample_data/`
2. **Install audio dependencies**:
   ```bash
   pip install soundfile librosa
   # macOS: brew install ffmpeg
   # Linux: sudo apt-get install libsndfile1
   ```
3. **Activate environment**: `source venv/bin/activate`
4. **Launch Jupyter**: `jupyter notebook`

## CLI Tools

Corresponding CLI tools in `functions/audio/`:
- `speech_recognition.py` - Transcribe audio from terminal
- `text_to_speech.py` - Generate speech from text

Examples:
```bash
# Transcribe audio
python functions/audio/speech_recognition.py audio.wav --model small --timestamps

# Generate speech
python functions/audio/text_to_speech.py "Hello world" --output greeting.wav
```

## Common Issues

### Speech Recognition:
- **Issue**: "Audio file not found"
  - **Solution**: Use absolute paths or place files in project root
- **Issue**: Poor transcription accuracy
  - **Solution**: Use `--model large` or ensure clear audio quality
- **Issue**: FFmpeg errors
  - **Solution**: Install FFmpeg separately (see getting_started.md)

### Text-to-Speech:
- **Issue**: Robotic/unnatural voice
  - **Solution**: Use speaker embeddings for better prosody
- **Issue**: Slow generation on CPU
  - **Solution**: Use GPU or generate shorter text segments

## Practical Applications

### Speech Recognition:
- Meeting transcription
- Podcast subtitles
- Voice commands
- Call center analytics

### Text-to-Speech:
- Accessibility (screen readers)
- Voice assistants
- Audiobook generation
- Language learning apps

## Next Steps

After completing this section:
- Try **Multimodal** (04_multimodal/) for vision-language tasks
- Explore **Best Practices** (05_best_practices/) for optimization
- Build **Agentic Workflows** (06_agentic_workflows/) to create voice-controlled agents
