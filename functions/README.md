# HuggingFace CLI Tools

This folder contains standalone Python scripts that provide command-line interfaces to HuggingFace models. These tools allow you to use the models directly from your terminal without Jupyter notebooks.

## 📁 Folder Structure

The CLI tools are organized by domain to match the notebook structure:

```
functions/
├── README.md                      # This file
│
├── nlp/                          # Natural Language Processing
│   ├── text_generation.py
│   ├── text_classification.py
│   └── text_summarization.py
│
├── cv/                           # Computer Vision
│   ├── image_classification.py
│   ├── object_detection.py
│   └── ocr.py
│
├── audio/                        # Audio Processing
│   ├── speech_recognition.py
│   └── text_to_speech.py
│
└── multimodal/                   # Multimodal AI
    └── image_captioning.py
```

---

## Prerequisites

1. **Activate your virtual environment**:
   ```bash
   cd HuggingFace_Tutorial
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔤 Natural Language Processing (NLP) Tools

### 1. Text Generation (`nlp/text_generation.py`)

Generate text completions from a prompt.

**Basic Usage:**
```bash
python functions/nlp/text_generation.py "Once upon a time"
```

**Advanced Examples:**
```bash
# Use larger model with custom parameters
python functions/nlp/text_generation.py "The future of AI is" --model large --max-length 100

# Generate multiple variations
python functions/nlp/text_generation.py "Hello world" --temperature 0.9 --num-sequences 3

# Very creative output
python functions/nlp/text_generation.py "Write a poem about" --temperature 1.5
```

**Parameters:**
- `--model`: `small` (distilgpt2, 82MB) or `large` (gpt2-medium, 1.5GB)
- `--max-length`: Maximum length of generated text (default: 50)
- `--temperature`: Sampling temperature 0.1-2.0 (default: 0.7)
- `--num-sequences`: Number of different completions (default: 1)
- `--top-k`: Top-k sampling (default: 50)
- `--top-p`: Nucleus sampling (default: 0.95)

---

### 2. Text Classification (`nlp/text_classification.py`)

Classify text sentiment (positive/negative).

**Basic Usage:**
```bash
python functions/nlp/text_classification.py "I love this product!"
```

**Advanced Examples:**
```bash
# Classify from file
python functions/nlp/text_classification.py --file reviews.txt

# Use larger model with confidence threshold
python functions/nlp/text_classification.py --file data.txt --model large --threshold 0.8

# Single text with threshold
python functions/nlp/text_classification.py "This is okay" --threshold 0.7
```

**Parameters:**
- `--model`: `small` (distilbert, 268MB) or `large` (bert-base, 440MB)
- `--file`: File containing texts (one per line)
- `--threshold`: Only show predictions >= threshold (default: 0.0)

---

### 3. Text Summarization (`nlp/text_summarization.py`)

Summarize long documents into concise summaries.

**Basic Usage:**
```bash
python functions/nlp/text_summarization.py --file article.txt
```

**Advanced Examples:**
```bash
# Longer summary with large model
python functions/nlp/text_summarization.py --file document.txt --model large --max-length 200

# Summarize text directly and save output
python functions/nlp/text_summarization.py --text "Long text here..." --output summary.txt

# Short summary
python functions/nlp/text_summarization.py --file report.txt --max-length 50 --min-length 20
```

**Parameters:**
- `--model`: `small` (distilbart, 1.2GB) or `large` (bart-large, 1.6GB)
- `--file`: File containing text to summarize
- `--text`: Direct text input (use quotes)
- `--max-length`: Maximum summary length (default: 130)
- `--min-length`: Minimum summary length (default: 30)
- `--output`: Save summary to file

---

## 🖼️ Computer Vision (CV) Tools

### 4. Image Classification (`cv/image_classification.py`)

Classify objects in images (1000 ImageNet categories).

**Basic Usage:**
```bash
python functions/cv/image_classification.py photo.jpg
```

**Advanced Examples:**
```bash
# Use large model with more predictions
python functions/cv/image_classification.py image.png --model large --top-k 10

# Only show confident predictions
python functions/cv/image_classification.py cat.jpg --threshold 0.5

# Show top 3 predictions
python functions/cv/image_classification.py dog.jpg --top-k 3
```

**Parameters:**
- `--model`: `small` (vit-base, 346MB) or `large` (vit-large, 1.2GB)
- `--top-k`: Number of top predictions (default: 5)
- `--threshold`: Only show predictions >= threshold (default: 0.0)

**Supported formats:** JPG, PNG, JPEG, WEBP

---

### 5. Object Detection (`cv/object_detection.py`) ⭐ NEW

Detect and localize multiple objects in images.

**Basic Usage:**
```bash
python functions/cv/object_detection.py photo.jpg
```

**Advanced Examples:**
```bash
# Use large model with custom threshold
python functions/cv/object_detection.py image.png --model large --threshold 0.8

# Save image with bounding boxes drawn
python functions/cv/object_detection.py street.jpg --save-image output.jpg

# High confidence detections only
python functions/cv/object_detection.py scene.jpg --threshold 0.9
```

**Parameters:**
- `--model`: `small` (detr-resnet-50, 159MB) or `large` (detr-resnet-101, 232MB)
- `--threshold`: Confidence threshold (default: 0.7)
- `--save-image`: Save annotated image with bounding boxes

**Supported formats:** JPG, PNG, JPEG, WEBP
**Detects:** 80 COCO categories (person, car, dog, etc.)

---

### 6. OCR (Optical Character Recognition) (`cv/ocr.py`) ⭐ NEW

Extract text from images using TrOCR.

**Basic Usage:**
```bash
python functions/cv/ocr.py document.jpg
```

**Advanced Examples:**
```bash
# Save extracted text to file
python functions/cv/ocr.py receipt.png --output extracted.txt

# Info about advanced OCR (PaddleOCR)
python functions/cv/ocr.py image.jpg --model large
```

**Parameters:**
- `--model`: `small` (trocr-small-printed, 558MB) or `large` (shows PaddleOCR info)
- `--output`: Save extracted text to file

**Supported formats:** JPG, PNG, JPEG, WEBP
**Best for:** Single lines of printed English text

---

## 🎵 Audio Processing Tools

### 7. Speech Recognition (`audio/speech_recognition.py`)

Transcribe audio files to text using Whisper.

**Basic Usage:**
```bash
python functions/audio/speech_recognition.py audio.wav
```

**Advanced Examples:**
```bash
# Use better model with timestamps
python functions/audio/speech_recognition.py speech.mp3 --model small --timestamps

# Save transcription to file
python functions/audio/speech_recognition.py recording.flac --output transcript.txt
```

**Parameters:**
- `--model`: `tiny` (72MB, fast) or `small` (483MB, better accuracy)
- `--timestamps`: Include timestamps in output
- `--output`: Save transcription to file

**Supported formats:** WAV, MP3, FLAC, OGG, M4A
**Languages:** 99+ languages (auto-detected)

---

### 8. Text-to-Speech (`audio/text_to_speech.py`) ⭐ NEW

Convert text to natural-sounding speech.

**Basic Usage:**
```bash
python functions/audio/text_to_speech.py "Hello world" --output hello.wav
```

**Advanced Examples:**
```bash
# Generate from file
python functions/audio/text_to_speech.py --file script.txt --output speech.wav

# Use different voice
python functions/audio/text_to_speech.py "Test" --output test.wav --voice 5
```

**Parameters:**
- `--text`: Direct text input (use quotes)
- `--file`: Read text from file
- `--output`: **Required** - output WAV file path
- `--voice`: Speaker ID 0-7306 (default: 7306)

**Output format:** 16kHz WAV files
**Model:** SpeechT5 (179MB)

---

## 🤝 Multimodal Tools

### 9. Image Captioning (`multimodal/image_captioning.py`)

Generate text descriptions of images.

**Basic Usage:**
```bash
python functions/multimodal/image_captioning.py photo.jpg
```

**Advanced Examples:**
```bash
# Use large model for better captions
python functions/multimodal/image_captioning.py image.png --model large

# Generate multiple caption variations
python functions/multimodal/image_captioning.py sunset.jpg --num-captions 5

# Longer captions
python functions/multimodal/image_captioning.py scene.jpg --max-length 100
```

**Parameters:**
- `--model`: `small` (blip-base, 990MB) or `large` (blip-large, 1.9GB)
- `--max-length`: Maximum caption length (default: 50)
- `--num-captions`: Number of different captions (default: 1)

**Supported formats:** JPG, PNG, JPEG, WEBP

---

## General Tips

### Getting Help

All scripts have built-in help. Use `--help` or `-h`:

```bash
python functions/nlp/text_generation.py --help
python functions/cv/image_classification.py -h
python functions/audio/speech_recognition.py --help
```

### Model Selection (2+1 Pattern)

All tools follow the **2+1 model pattern**:
- **Small models**: Fast, work on CPU, good for learning (72MB-558MB)
- **Large models**: Better quality, need GPU for reasonable speed (483MB-1.9GB)

### Performance

- **First run**: Downloads models (can take time depending on size)
- **Subsequent runs**: Uses cached models (much faster)
- **GPU vs CPU**: GPU is 5-20x faster for most tasks
- **Model cache**: Located in `~/.cache/huggingface/hub/`

### Common Patterns

**Process multiple files:**
```bash
# Classify all images in a directory
for img in images/*.jpg; do
    python functions/cv/image_classification.py "$img"
done

# Transcribe all audio files
for audio in audio/*.mp3; do
    python functions/audio/speech_recognition.py "$audio" --output "${audio%.mp3}.txt"
done

# Extract text from all documents
for doc in documents/*.png; do
    python functions/cv/ocr.py "$doc" --output "${doc%.png}.txt"
done
```

**Batch processing text:**
```bash
# Create file with one text per line
echo "I love this!" > texts.txt
echo "This is terrible" >> texts.txt
echo "Pretty good" >> texts.txt

# Classify all at once
python functions/nlp/text_classification.py --file texts.txt
```

### Error Handling

If you get errors:

1. **"No module named '...'"**: Activate virtual environment
2. **"File not found"**: Check file path is correct
3. **"CUDA out of memory"**: Use `--model small` or run on CPU
4. **Connection errors**: Check internet for first-time model download
5. **Image format errors**: Ensure image is a valid format (JPG, PNG, etc.)

### Output Redirection

Save output to files:

```bash
# Save complete output
python functions/nlp/text_generation.py "Hello" > output.txt

# Save only the generated text (stderr contains logs)
python functions/nlp/text_generation.py "Hello" 2>/dev/null

# Append to file
python functions/cv/image_classification.py photo.jpg >> results.txt
```

---

## Integration Examples

### Chaining Tools

```bash
# Generate text, then analyze sentiment
python functions/nlp/text_generation.py "Today I feel" --max-length 30 > generated.txt
python functions/nlp/text_classification.py --file generated.txt

# Summarize document, then check sentiment
python functions/nlp/text_summarization.py --file article.txt --output summary.txt
python functions/nlp/text_classification.py --file summary.txt

# Transcribe audio, then summarize
python functions/audio/speech_recognition.py lecture.mp3 --output transcript.txt
python functions/nlp/text_summarization.py --file transcript.txt

# Extract text from image, then analyze
python functions/cv/ocr.py document.jpg --output extracted.txt
python functions/nlp/text_classification.py --file extracted.txt
```

### Shell Scripts

Create a bash script for batch processing:

```bash
#!/bin/bash
# process_images.sh - Classify and caption all images

for img in "$1"/*.jpg; do
    echo "Processing: $img"
    echo "Classification:"
    python functions/cv/image_classification.py "$img" --top-k 3
    echo "Caption:"
    python functions/multimodal/image_captioning.py "$img"
    echo "---"
done
```

Usage: `bash process_images.sh ./photos`

**OCR and Speech Pipeline:**
```bash
#!/bin/bash
# ocr_pipeline.sh - Extract text from images and generate audio

python functions/cv/ocr.py "$1" --output extracted.txt
python functions/audio/text_to_speech.py --file extracted.txt --output speech.wav
echo "Text extracted and converted to speech!"
```

---

## Tool Comparison

| Tool | Input | Output | Models | Size Range |
|------|-------|--------|--------|------------|
| **Text Generation** | Text prompt | Generated text | distilgpt2, gpt2-medium | 82MB - 1.5GB |
| **Text Classification** | Text | Sentiment label | distilbert, bert-base | 268MB - 440MB |
| **Text Summarization** | Long text | Summary | distilbart, bart-large | 1.2GB - 1.6GB |
| **Image Classification** | Image | Object labels | vit-base, vit-large | 346MB - 1.2GB |
| **Object Detection** | Image | Objects + boxes | detr-resnet-50/101 | 159MB - 232MB |
| **OCR** | Image | Extracted text | trocr-small | 558MB |
| **Speech Recognition** | Audio | Transcription | whisper-tiny, small | 72MB - 483MB |
| **Text-to-Speech** | Text | Audio (WAV) | speecht5_tts | 179MB |
| **Image Captioning** | Image | Text description | blip-base, large | 990MB - 1.9GB |

---

## Troubleshooting

### Virtual Environment Not Activated

If you see "No module named 'transformers'":

```bash
# Activate venv first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Model Download Issues

If downloads are slow or fail:
- Check internet connection
- HuggingFace Hub may be slow; retry later
- Models cache in `~/.cache/huggingface/`
- For large models (>1GB), downloads may take 10-20 minutes

### Memory Issues

If you run out of memory:
- Use `--model small` instead of `large`
- Close other applications
- Reduce batch size (process fewer texts at once)
- For object detection, increase `--threshold` to reduce detections
- Restart your terminal/computer

### Audio Issues

For text-to-speech or speech recognition:
- Ensure audio libraries are installed: `pip install soundfile librosa`
- TTS requires output file: use `--output speech.wav`
- Supported formats vary by tool - check tool help

---

## Additional Resources

- [HuggingFace Documentation](https://huggingface.co/docs)
- [Transformers Pipeline Docs](https://huggingface.co/docs/transformers/main_classes/pipelines)
- Main README: `../README.md`
- Getting Started Guide: `../getting_started.md`
- Notebooks: `../notebooks/` (corresponding Jupyter notebooks for each tool)

---

## Quick Command Reference

```bash
# NLP
python functions/nlp/text_generation.py "prompt" --model small
python functions/nlp/text_classification.py "text" --threshold 0.8
python functions/nlp/text_summarization.py --file doc.txt --output summary.txt

# Computer Vision
python functions/cv/image_classification.py image.jpg --model large --top-k 5
python functions/cv/object_detection.py photo.jpg --threshold 0.7 --save-image out.jpg
python functions/cv/ocr.py document.jpg --output text.txt

# Audio
python functions/audio/speech_recognition.py audio.mp3 --model small --timestamps
python functions/audio/text_to_speech.py "text" --output speech.wav

# Multimodal
python functions/multimodal/image_captioning.py image.jpg --num-captions 3
```

---

**Total Tools**: 9 CLI tools covering NLP, Computer Vision, Audio, and Multimodal tasks
**Total Models**: 18 model options across all tools
**Organization**: Domain-based folders matching notebook structure
