# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **educational HuggingFace tutorial** repository designed to introduce students to the HuggingFace ecosystem through 17 hands-on Jupyter notebooks. The project is optimized for students with limited hardware resources while providing GPU-optimized alternatives for those with access to better equipment (RTX 4080 or similar).

**Target Audience**: Students learning machine learning with varying hardware capabilities (CPU-only to consumer-grade GPUs)

**Key Design Principle**: Each notebook offers dual model options:
- Small models that run on CPU (4-8GB RAM)
- Medium models optimized for RTX 4080-class GPUs (6-12GB VRAM)

**New in 2025**: Advanced topics notebooks (11-13) covering performance optimization, responsible AI, and fine-tuning techniques. Agentic workflows notebooks (14-17) teaching MCP-based tool-using agents, RAG (Retrieval-Augmented Generation), and local Ollama models.

## Project Architecture

### Dual Interface Design

The project provides **two ways to use HuggingFace models**:

1. **Jupyter Notebooks** (`notebooks/`): Interactive, educational, step-by-step learning
   - Best for understanding concepts
   - Includes explanations, examples, and exercises
   - Organized by domain (NLP, CV, Audio, Multimodal, Tools, Advanced)
   - See Notebook Structure Pattern below

2. **CLI Tools** (`functions/`): Standalone Python scripts for direct command-line usage
   - Best for quick tasks and automation
   - Fully self-contained with `if __name__ == "__main__"`
   - Use argparse for parameter handling
   - Include example outputs in docstrings (visible with `--help`)
   - Can be integrated into shell scripts or workflows
   - Organized by domain (nlp/, cv/, audio/, multimodal/, agentic/)
   - **12 total CLI tools** matching core notebook functionality

### Notebook Organization

Notebooks are organized into **6 domain-based folders**:

```
notebooks/
├── 01_nlp/                    # Natural Language Processing (5 notebooks)
│   ├── 01-03: Core NLP tasks (generation, classification, summarization)
│   └── 04-05: Fine-tuning (Unsloth GPU-optimized, LoRA CPU-compatible)
│
├── 02_computer_vision/        # Computer Vision (3 notebooks)
├── 03_audio/                  # Audio Processing (2 notebooks)
├── 04_multimodal/             # Multimodal AI (1 notebook)
├── 05_best_practices/         # Best Practices & Production (3 notebooks)
│   ├── Ollama integration (local LLM deployment)
│   ├── Performance & caching
│   └── Model cards & responsible AI
│
└── 06_agentic_workflows/      # Agentic AI with MCP (4 notebooks)
    ├── 14: MCP Basics (tool-using agents)
    ├── 15: MCP Servers (reusable tool servers)
    ├── 16: Multi-Tool Agents (ReAct, Plan-and-Execute, Reflection patterns)
    └── 17: RAG with Local LLMs (vector databases, semantic search)
```

This structure:
- Aligns with HuggingFace task taxonomy
- Makes it easy to focus on specific domains
- Supports learning paths (beginners can start with NLP, intermediate can explore CV, etc.)
- Preserves numeric ordering within each domain

### CLI Tools Organization

CLI tools in `functions/` mirror the notebook structure:

```
functions/
├── nlp/             # 3 tools: text_generation, text_classification, text_summarization
├── cv/              # 3 tools: image_classification, object_detection, ocr
├── audio/           # 2 tools: speech_recognition, text_to_speech
├── multimodal/      # 1 tool: image_captioning
└── agentic/         # 3 tools: mcp_agent, mcp_server, multi_agent
```

**All CLI tools:**
- Follow the 2+1 model pattern (small/large options)
- Include comprehensive help via `--help` flag
- Support batch processing and output redirection
- Can be chained together for pipelines
- Mirror notebook functionality for production use

### Notebook Structure Pattern

**Notebooks 01-10** (Core tutorials) follow a **consistent template** with these sections in order:
1. **Title & Learning Objectives** - Clear educational goals
2. **Prerequisites Table** - Hardware requirements showing CPU and GPU model options side-by-side
3. **Expected Behaviors** - What students should observe when running the notebook
4. **Overview** - Conceptual explanation of the task
5. **Setup and Installation** - Import statements and version checks
6. **Model Selection** - Commented code showing both CPU and GPU model options
7. **Method 1: Pipeline API** - Simplest approach for beginners
8. **Method 2: Manual Model Loading** - Advanced approach with more control
9. **Practical Applications** - 2-3 real-world examples
10. **Performance Benchmarking** - Timing and device information
11. **Exercises** - Hands-on challenges for students
12. **Key Takeaways** - Bullet-point summary
13. **Next Steps & Resources** - Links to documentation and next notebook

**Notebooks 04-05 (NLP Fine-tuning)** have specialized structures:
- **Notebook 04 (Unsloth)**: GPU-only, production-focused, 2-5x speedup, Llama 3.2/3.1, TinyStories dataset
- **Notebook 05 (LoRA)**: CPU-compatible, educational, GPT-2, standard LoRA approach

**Notebooks 10-12 (Best Practices)** have specialized structures:
- **Notebook 10**: Integration-focused with Ollama setup and usage patterns
- **Notebook 11**: Performance analysis with visualization-heavy code cells
- **Notebook 12**: Concept-focused with minimal code, emphasis on critical thinking

### Model Selection Pattern (2+1 Rule)

**All notebooks follow a standardized 2+1 model pattern:**

- **Small Model**: Trainable/runnable with local GPU (RTX 4080-class, 8-12GB VRAM) OR CPU-friendly for inference
- **Large Model**: Inference-only, can be SOTA if it's the best available option
- **SOTA Model**: Only if significantly different from Large model (optional)

Each notebook includes a model selection cell like:
```python
# Option 1: Small Model (CPU-friendly, recommended for beginners)
MODEL_NAME = "small-model-name"  # Size in MB, description

# Option 2: Large Model/SOTA (GPU-optimized, production-grade)
# MODEL_NAME = "larger-model-name"  # Size in MB, description
```

**Examples of 2+1 implementation:**
- **Notebook 01**: distilgpt2 (small) + gpt2-medium (large)
- **Notebook 04 (Unsloth)**: Llama 3.2-1B (small) + Llama 3.2-3B/3.1-8B (large)
- **Notebook 05 (LoRA)**: GPT-2 (base model, CPU-compatible)
- **Notebook 06**: trocr-small-printed (small) + PaddleOCR (large/SOTA combined)
- **Notebook 07**: whisper-tiny (small) + whisper-small (large)

Students uncomment the GPU line if they have appropriate hardware.

### Built-in Dataset Integration

**Key notebooks now include built-in dataset demonstrations:**

- **Notebook 02 (Text Classification)**: SST-2 dataset (7MB, sentiment analysis)
- **Notebook 04 (Unsloth Fine-tuning)**: TinyStories (HuggingFace dataset, 5000 examples for training)
- **Notebook 04 (Image Classification)**: CIFAR-10 (170MB, 10 classes)
- **Notebook 06 (OCR)**: MNIST (12MB, handwritten digits)
- **Notebook 07 (Speech Recognition)**: LibriSpeech ASR Dummy (small test set)

All dataset examples:
- Appear in a new "Using Built-in Datasets" section before "Exercises"
- Include error handling and helpful installation messages
- Show comparison between predictions and ground truth
- Are documented with dataset size and characteristics

### Notebook Categories

**NLP (01_nlp/ - 5 notebooks)**:
- **Notebooks 01-03**: Core NLP tasks (text generation, classification, summarization)
- **Notebook 04**: Fine-tuning with Unsloth (2-5x faster, GPU-only, Llama 3.2/3.1, TinyStories dataset)
- **Notebook 05**: Fine-tuning with LoRA (CPU-compatible, GPT-2, standard LoRA approach)

**Computer Vision (02_computer_vision/ - 3 notebooks)**:
- **Notebook 04**: Image Classification (ViT)
- **Notebook 05**: Object Detection (DETR, YOLO)
- **Notebook 06**: OCR (TrOCR, PaddleOCR)

**Audio (03_audio/ - 2 notebooks)**:
- **Notebook 07**: Speech Recognition (Whisper)
- **Notebook 08**: Text-to-Speech (SpeechT5)

**Multimodal (04_multimodal/ - 1 notebook)**:
- **Notebook 09**: Image Captioning (BLIP)

**Best Practices (05_best_practices/ - 3 notebooks)**:
- **Notebook 10**: Ollama Integration (local LLM deployment)
- **Notebook 11**: Performance, Caching, and Cost Analysis
- **Notebook 12**: Model Cards and Responsible AI

**Agentic Workflows (06_agentic_workflows/ - 4 notebooks)**:
- **Notebook 14**: MCP Basics (Model Context Protocol fundamentals, tool calling, agent loop)
- **Notebook 15**: MCP Servers (reusable tool servers, file system operations, data analysis)
- **Notebook 16**: Multi-Tool Agents (ReAct, Plan-and-Execute, Reflection patterns)
- **Notebook 17**: RAG with Local LLMs (vector databases, embeddings, semantic search, FAISS, ChromaDB)
- **Key Technologies**: MCP, Ollama, Llama 3.2 (1b/3b), Llama 3.1 (8b), RAG, sentence-transformers, tool chaining

### Visualization Enhancements

**See `VISUALIZATIONS.md`** for comprehensive visualization code snippets.

The project includes rich visualizations to enhance learning:

**Notebook 11 - Performance Visualizations**:
- Cache size bar charts
- Latency comparison charts (model size vs speed)
- Throughput vs batch size plots
- Memory usage over time
- Cost estimation comparisons

**Notebook 12 - Responsible AI Visualizations**:
- Bias comparison heatmaps
- Gender bias score charts
- Model card comparison tables (styled dataframes)

**Notebook 13 - Training Visualizations**:
- Training/validation loss curves
- Learning rate schedules
- Before/after text generation comparison displays
- Model size comparison charts (full model vs LoRA adapter)

**CV Notebooks (04, 05, 06) - Visual Outputs**:
- Input image display with prediction overlays
- Bounding boxes for object detection (Notebook 05)
- Highlighted text regions for OCR (Notebook 06)
- Top-K predictions as bar charts

**Visualization Libraries**:
- `matplotlib`: Primary plotting library
- `seaborn`: Statistical visualizations and color palettes
- `plotly`: Interactive plots (optional)

All visualization code is documented in `VISUALIZATIONS.md` with:
- Exact code to copy/paste
- Where to insert in notebooks (after which cell)
- Customization options
- Sample outputs

## Development Commands

### Environment Setup

**Note:** Full setup instructions are in `getting_started.md` for students. This section is for quick reference only.

```bash
# Quick setup (see getting_started.md for details)
python -m venv venv
source venv/bin/activate  # macOS/Linux or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Running Notebooks
```bash
# Launch Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Running CLI Tools
```bash
# All CLI tools must be run with venv activated
# General pattern: python functions/<tool>.py [arguments]

# Text generation
python functions/text_generation.py "Once upon a time"
python functions/text_generation.py "Hello" --model large --max-length 100

# Text classification
python functions/text_classification.py "I love this!"
python functions/text_classification.py --file reviews.txt --threshold 0.8

# Image classification
python functions/image_classification.py photo.jpg
python functions/image_classification.py image.png --model large --top-k 10

# Speech recognition
python functions/speech_recognition.py audio.wav
python functions/speech_recognition.py speech.mp3 --model small --timestamps

# Text summarization
python functions/text_summarization.py --file article.txt
python functions/text_summarization.py --file doc.txt --max-length 150 --output summary.txt

# Image captioning
python functions/image_captioning.py photo.jpg
python functions/image_captioning.py sunset.jpg --model large --num-captions 3

# Get help for any tool
python functions/text_generation.py --help
```

### HuggingFace Authentication

**Note:** See `getting_started.md` for complete authentication instructions.

Quick reference: `huggingface-cli login` (paste token from `hf_credential` file)

### Testing Individual Notebooks
```bash
# Convert notebook to Python script for testing
jupyter nbconvert --to script notebooks/01_nlp_text_generation.ipynb

# Run specific notebook programmatically
jupyter nbconvert --execute --to notebook notebooks/01_nlp_text_generation.ipynb
```

### Ollama Setup (for Notebook 10)
```bash
# Install Ollama from ollama.com, then:
ollama pull tinyllama

# List available models
ollama list

# Test Ollama
ollama run tinyllama "Hello, world!"
```

## Important Files

### `hf_credential`
Contains the HuggingFace access token. **Never commit this file to public repositories**. See `getting_started.md` for setup details.

### `sample_data/`
Directory for test images and audio files. Notebooks work with URLs when local files don't exist.

### `notebooks/shared_utils.py`
Common helper functions shared across notebooks (e.g., `load_image_from_url`, `setup_device`). Import in notebooks to avoid code duplication.

## Notebook Interdependencies

**Independence**: Each notebook is self-contained and can run independently. There is no required order, though README.md suggests a learning path:
- **Beginners**: 01, 02, 04
- **Intermediate**: 03, 05, 06, 07
- **Advanced**: 08, 09, 10
- **Production & Best Practices**: 11, 12, 13

**Shared Patterns**: All notebooks use similar helper functions (e.g., `load_image_from_url()`) but re-define them locally rather than importing from a shared module. This makes each notebook standalone for educational clarity.

**Advanced Notebooks**: Notebooks 11-13 may reference concepts from earlier notebooks but can be completed independently:
- Notebook 11 uses models from earlier notebooks for performance comparison
- Notebook 12 examines model cards of models used in notebooks 01-10
- Notebook 13 is fully standalone (fine-tunes GPT-2 from scratch)

## Model Caching

HuggingFace models are automatically cached in `~/.cache/huggingface/hub/`. Key implications:

- First run of each notebook downloads models (can be slow)
- Subsequent runs use cached models (much faster)
- Cache can grow to 10-20GB with all notebooks
- Students on shared systems may benefit from `HF_HOME` environment variable

## Adding New Notebooks

When creating additional notebooks:

1. **Follow the template structure** shown in existing notebooks
2. **Include Prerequisites table** with both CPU and GPU options
3. **Provide dual model options** with clear size and hardware requirements
4. **Use shared utilities**: Import from `shared_utils.py` instead of redefining common functions
   ```python
   from shared_utils import load_image_from_url, setup_device, print_device_info
   ```
5. **Add to README.md table** with model sizes and hardware specs
6. **Update learning path** if it fits a specific difficulty level
7. **Test on both CPU and GPU** if possible
8. **Use commented model selection** pattern for easy switching

## Common Student Issues

### Memory Issues
- Students hitting OOM errors should use the CPU (smaller) model option
- Suggest reducing batch size in code examples
- Recommend kernel restart between notebooks

### Download Failures
- Models download from HuggingFace Hub; requires internet
- Corporate firewalls may block; suggest using mobile hotspot or VPN
- Some models require authentication via `hf_credential`

### Missing Dependencies
- Audio notebooks need `soundfile` and `librosa`
- Vision notebooks need `Pillow`
- Advanced notebooks need `matplotlib`, `seaborn`, `plotly` (visualization)
- Notebook 13 needs `peft` library for LoRA fine-tuning
- Notebook 11 needs `psutil` for memory profiling
- Some students may need to separately install `ffmpeg` for audio
- Windows users may need Visual C++ redistributables for some packages

### Ollama Integration (Notebook 10)
- Requires separate Ollama installation from ollama.com
- Model must be pulled before running notebook (`ollama pull tinyllama`)
- Ollama daemon must be running in background
- Not available on all platforms (check Ollama docs)

## Model Size Reference

Quick reference for planning storage/bandwidth:

**Tiny (<100MB)**: whisper-tiny (72MB), distilgpt2 (82MB)
**Small (100-500MB)**: distilbert (268MB), vit-base (346MB), detr-resnet-50 (159MB), gpt2 (500MB for notebook 13)
**Medium (500MB-1.5GB)**: trocr-small (558MB), blip-base (990MB), distilbart (1.2GB), gpt2-medium (1.5GB)
**Large (1.5GB+)**: bart-large (1.6GB), blip-large (1.9GB)

**Dataset Sizes** (Notebook 13):
- TinyStories dataset: ~300MB for 5000 examples

**Total Storage Requirements**:
- Core notebooks (01-10) CPU models: ~8-10GB
- Core notebooks (01-10) GPU models: ~12-15GB
- Advanced notebooks (11-13): ~2-3GB (additional models + datasets)
- **Overall**: Plan for 15-20GB total cache storage

## Validation Process

To validate notebooks work correctly:

1. Run each notebook top to bottom in a fresh kernel
2. Verify model loads successfully
3. Check outputs are reasonable (not gibberish)
4. Confirm timing information appears
5. Ensure exercises section is present
6. Test with both CPU and GPU model options (if available)

No automated tests exist - this is intentional for educational simplicity. The notebooks themselves are the "tests" that students run.
