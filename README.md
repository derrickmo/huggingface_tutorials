# HuggingFace Tutorial: Comprehensive Guide to Modern AI Models

**Learn HuggingFace transformers with 38 hands-on Jupyter notebooks covering Transformer Fundamentals, NLP, Computer Vision, Audio, Multimodal AI, Best Practices, and Agentic Workflows.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet.svg)](https://claude.ai/code)

> **A passion project by [Derrick Mo](https://www.linkedin.com/in/derrickmo/).**
>
> I believe the best way to learn is by actually doing. This tutorial was built from the ground up with [Claude Code](https://claude.ai/code) as a hands-on resource for anyone looking to get started with HuggingFace and modern AI. Whether you're a student, a hobbyist, or a professional exploring new tools, I hope these notebooks help you learn by building real things. Feedback, ideas, and collaboration are always welcome -- feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/derrickmo/)!

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Notebooks](#notebooks)
- [CLI Tools](#cli-tools)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Learning Paths](#learning-paths)
- [Resources](#resources)

---

## Overview

This tutorial provides a complete introduction to the HuggingFace ecosystem with **38 hands-on Jupyter notebooks** across 7 sections. Each notebook is a self-contained, complete walkthrough -- no fill-in-the-blank exercises, just runnable code with explanations. Every notebook is designed to run on modest hardware with options to scale up for better performance.

### What You'll Learn

| Domain | Topics |
|--------|--------|
| **Fundamentals** | Tokenization, embeddings, transformer architecture, HuggingFace ecosystem tour |
| **NLP** | Text generation, classification, summarization, NER, QA, translation, fine-tuning (LoRA, Unsloth) |
| **Computer Vision** | Image classification (ViT), object detection (DETR, YOLOv8), OCR (TrOCR), image segmentation (SegFormer) |
| **Audio** | Speech-to-text (Whisper), text-to-speech (SpeechT5), audio classification (AST) |
| **Multimodal** | Image captioning (BLIP), VQA (ViLT), text-to-image (Stable Diffusion), image editing, document understanding |
| **Best Practices** | Performance, caching, responsible AI, datasets, Gradio, quantization (basic + GPTQ/AWQ), Trainer API |
| **Agentic AI** | MCP agents, MCP servers, multi-tool patterns, RAG, structured output & function calling |

### Key Features

- **Self-contained**: Each notebook runs independently from top to bottom
- **Dual model options**: Small CPU-friendly models + larger GPU-optimized models
- **Complete walkthroughs**: All code is fully implemented, no placeholders
- **Reproducible**: All notebooks use seed 1103 for consistent results
- **Well-documented**: Google-style docstrings, type hints, transition explanations throughout
- **Production-ready**: Learn optimization, deployment, and ethical AI practices

---

## Quick Start

```bash
# 1. Clone or download this repository
cd HuggingFace_Tutorial

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook

# 5. Open notebooks/00_fundamentals/00_01_tokenization_embeddings.ipynb
```

**For detailed setup, GPU configuration, and troubleshooting: [getting_started.md](getting_started.md)**

---

## Notebooks

All notebooks are in `notebooks/` organized by domain (7 sections, 38 notebooks). Each can be run independently in any order.

### Transformer Fundamentals (`00_fundamentals/`)

| # | Notebook | Models/Tools | Description | Difficulty |
|---|----------|-------------|-------------|------------|
| 00_01 | **Tokenization & Embeddings** | AutoTokenizer | BPE, WordPiece, vocabulary, special tokens, embeddings | Beginner |
| 00_02 | **Transformer Architecture** | AutoModel | Self-attention, multi-head attention, encoder vs decoder | Beginner |
| 00_03 | **HuggingFace Ecosystem** | Pipeline, HfApi | Hub navigation, AutoClasses, pipelines, model cards | Beginner |
| 00_04 | **Preprocessors & Feature Extractors** | AutoProcessor | Unified preprocessing, padding/truncation, multimodal processors | Beginner |
| 00_05 | **Model Configuration & Customization** | AutoConfig | config.json, model surgery, freezing, memory estimation | Beginner |

### NLP -- Natural Language Processing (`01_nlp/`)

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 01_01 | **Text Generation** | GPT-2, DistilGPT2 | Causal LM, greedy/beam/sampling decoding | Beginner |
| 01_02 | **Text Classification** | DistilBERT, BERT | Sentiment analysis and topic classification | Beginner |
| 01_03 | **Text Summarization** | DistilBART, BART | Abstractive summarization, seq2seq | Intermediate |
| 01_04 | **Named Entity Recognition** | DistilBERT-NER | Token classification, BIO tags, span extraction | Intermediate |
| 01_05 | **Question Answering** | DistilBERT-QA | Extractive QA, context + question, answer span | Intermediate |
| 01_06 | **Translation** | MarianMT | Machine translation, multilingual, beam search | Intermediate |
| 01_07 | **Fine-tuning (Unsloth)** | Llama 3.2/3.1 | 2-5x faster LLM fine-tuning (GPU-only) | Advanced |
| 01_08 | **Fine-tuning (LoRA)** | GPT-2 | Adapter training with LoRA (CPU-compatible) | Advanced |

### Computer Vision (`02_computer_vision/`)

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 02_01 | **Image Classification** | ViT (Vision Transformer) | Identify objects in images | Beginner |
| 02_02 | **Object Detection** | DETR, YOLOv8 | Detect and localize multiple objects | Intermediate |
| 02_03 | **OCR** | TrOCR, PaddleOCR | Extract text from images (80+ languages) | Intermediate |
| 02_04 | **Image Segmentation** | SegFormer | Semantic, instance, and panoptic segmentation | Intermediate |

### Audio Processing (`03_audio/`)

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 03_01 | **Speech Recognition** | Whisper (tiny/small) | Transcribe audio to text | Intermediate |
| 03_02 | **Text-to-Speech** | SpeechT5 | Generate natural-sounding speech | Intermediate |
| 03_03 | **Audio Classification** | Audio Spectrogram Transformer | Environmental sounds, music genre classification | Intermediate |

### Multimodal (`04_multimodal/`)

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 04_01 | **Image-to-Text** | BLIP | Generate captions from images | Intermediate |
| 04_02 | **Visual Question Answering** | ViLT | Ask questions about images, VQA datasets | Intermediate |
| 04_03 | **Text-to-Image Generation** | Stable Diffusion | Prompt engineering, negative prompts, schedulers | Intermediate |
| 04_04 | **Image Editing & Inpainting** | SD Inpainting, InstructPix2Pix | Inpainting, img2img, instruction-based editing | Advanced |
| 04_05 | **Document Understanding** | LayoutLM, Donut | Form parsing, receipt extraction, document QA | Advanced |

### Best Practices & Production (`05_best_practices/`)

| # | Notebook | Focus | Description | Difficulty |
|---|----------|-------|-------------|------------|
| 05_01 | **Ollama Integration** | Local LLMs | Use Ollama models with HuggingFace tools | Intermediate |
| 05_02 | **Performance & Caching** | Optimization | Measure latency, optimize throughput, estimate costs | Intermediate |
| 05_03 | **Model Cards & AI Ethics** | Responsible AI | Understand bias, fairness, and documentation | Intermediate |
| 05_04 | **HuggingFace Datasets** | Data management | Load, filter, preprocess, and create custom datasets | Beginner |
| 05_05 | **Gradio & Spaces** | Interactive demos | Build web UIs and deploy to HuggingFace Spaces | Intermediate |
| 05_06 | **Quantization & Compression** | Model optimization | INT8 quantization, ONNX export, size/speed benchmarking | Intermediate |
| 05_07 | **Quantization Deep Dive** | Advanced quantization | GPTQ, AWQ, bitsandbytes 4-bit/8-bit, perplexity evaluation | Advanced |
| 05_08 | **Training Best Practices** | Trainer API | LR schedules, gradient accumulation, mixed precision, early stopping | Advanced |

### Agentic Workflows (`06_agentic_workflows/`)

| # | Notebook | Focus | Description | Difficulty |
|---|----------|-------|-------------|------------|
| 06_01 | **MCP Basics** | Tool-using agents | Model Context Protocol fundamentals, tool calling | Intermediate |
| 06_02 | **MCP Servers** | Reusable tools | Build file system and data analysis servers | Advanced |
| 06_03 | **Multi-Tool Agents** | Agent patterns | ReAct, Plan-and-Execute, Reflection patterns | Advanced |
| 06_04 | **RAG with Local LLMs** | Retrieval-Augmented Generation | Vector databases, semantic search, context injection | Advanced |
| 06_05 | **Structured Output** | Function calling | JSON mode, Pydantic validation, tool use patterns | Advanced |

---

## CLI Tools

The `functions/` directory provides **12 standalone CLI tools** that mirror notebook functionality for command-line use. Each tool supports `--help`, batch processing, and model selection (small/large).

```bash
# Examples
python functions/nlp/text_generation.py "Once upon a time"
python functions/nlp/text_classification.py "I love this!"
python functions/cv/image_classification.py photo.jpg
python functions/audio/speech_recognition.py audio.wav

# See all options
python functions/nlp/text_generation.py --help
```

See [functions/README.md](functions/README.md) for full CLI documentation.

---

## System Requirements

### Minimum (CPU-only)

- **Python**: 3.10 or 3.11 (recommended), 3.9 also supported
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 15GB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 20.04+

### Recommended (with GPU)

- **Python**: 3.10 or 3.11
- **RAM**: 16GB+
- **GPU**: NVIDIA with 8GB+ VRAM (RTX 3060/4070/4080/4090)
- **CUDA**: 11.8 or 12.1
- **Storage**: 25GB free space

### What Runs Where?

| Hardware | Compatibility |
|----------|---------------|
| **CPU (8GB RAM)** | All notebooks except Unsloth (01_07), text-to-image/editing (04_03-04_04). LoRA (01_08) will be slow. Agentic works with small Ollama models. |
| **GPU (8GB VRAM)** | All notebooks. Unsloth with Llama 3.2-1B. Stable Diffusion for image generation. YOLO, TrOCR. |
| **GPU (16GB+ VRAM)** | Everything including Unsloth with Llama 3.2-3B/3.1-8B, bitsandbytes/GPTQ/AWQ quantization, RAG with Llama 3.1:8b. |

---

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install all dependencies
pip install -r requirements.txt

# (Optional) GPU support
# Visit https://pytorch.org/get-started/locally/ for CUDA-enabled PyTorch
```

**For step-by-step instructions, GPU setup, and troubleshooting: [getting_started.md](getting_started.md)**

### Core Dependencies

| Category | Packages |
|----------|----------|
| HuggingFace | `transformers`, `datasets`, `huggingface-hub`, `accelerate` |
| Deep Learning | `torch`, `torchvision`, `torchaudio` |
| Media | `Pillow`, `soundfile`, `librosa` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Fine-tuning | `peft`, `trl` |

### Optional Dependencies

| Feature | Packages | Used In |
|---------|----------|---------|
| YOLO detection | `ultralytics` | 02_02 Object Detection |
| PaddleOCR | `paddleocr`, `paddlepaddle` | 02_03 OCR |
| Unsloth | `unsloth` (special install) | 01_07 Fine-tuning |
| Diffusion models | `diffusers` | 04_03 Text-to-Image, 04_04 Image Editing |
| Gradio | `gradio` | 05_05 Gradio & Spaces |
| ONNX quantization | `onnx`, `onnxruntime`, `optimum` | 05_06 Quantization Intro |
| Advanced quantization | `bitsandbytes`, `auto-gptq`, `autoawq` | 05_07 Quantization Deep Dive |
| Training | `evaluate` | 05_08 Training Best Practices |
| Local LLMs | `ollama` | 05_01, 06_01-06_05 |
| Agentic tools | `mcp` | 06_01-06_03 |
| RAG | `sentence-transformers`, `faiss-cpu`, `chromadb` | 06_04 RAG |
| Structured output | `pydantic`, `instructor` | 06_05 Structured Output |

---

## Project Structure

```
HuggingFace_Tutorial/
├── README.md                           # This file
├── CLAUDE.md                           # Claude Code conventions and rules
├── SYLLABUS.md                         # Complete 38-notebook syllabus
├── getting_started.md                  # Detailed setup instructions
├── requirements.txt                    # Python dependencies
│
├── notebooks/                          # 38 Jupyter notebooks
│   ├── shared_utils.py                 # Common helper functions
│   ├── 00_fundamentals/                # Transformer Fundamentals (5 notebooks)
│   ├── 01_nlp/                         # NLP (8 notebooks)
│   ├── 02_computer_vision/             # CV (4 notebooks)
│   ├── 03_audio/                       # Audio (3 notebooks)
│   ├── 04_multimodal/                  # Multimodal (5 notebooks)
│   ├── 05_best_practices/              # Best Practices (8 notebooks)
│   └── 06_agentic_workflows/           # Agentic AI (5 notebooks)
│
├── functions/                          # 12 standalone CLI tools
│   ├── nlp/                            # text_generation, text_classification, text_summarization
│   ├── cv/                             # image_classification, object_detection, ocr
│   ├── audio/                          # speech_recognition, text_to_speech
│   ├── multimodal/                     # image_captioning
│   └── agentic/                        # mcp_agent, mcp_server, multi_agent
│
├── rules/                              # Code quality and style rules
│   ├── RULES_CORE.md                   # Foundational conventions
│   ├── RULES_STRUCTURE.md              # Notebook structure template
│   ├── RULES_STYLE.md                  # Code quality, docstrings, type hints
│   └── RULES_QUALITY.md               # Review checklist, error handling
│
└── sample_data/                        # Test images and audio files
```

---

## Learning Paths

### Path 1: Complete Beginner

**Goal**: Learn ML fundamentals with hands-on examples

1. **00_01-00_03** (Fundamentals) -- Tokenization, Transformers, HuggingFace Ecosystem
2. **01_01** (NLP) -- Text Generation
3. **01_02** (NLP) -- Text Classification
4. **02_01** (CV) -- Image Classification
5. **03_01** (Audio) -- Speech Recognition

**Time**: 10-12 hours | **Hardware**: CPU with 8GB RAM

### Path 2: NLP Specialist

**Goal**: Master natural language processing

1. **00_01-00_03** -- Fundamentals
2. **01_01-01_06** (NLP) -- All core NLP tasks
3. **01_07-01_08** (NLP) -- Fine-tuning (Unsloth + LoRA)
4. **05_01** -- Ollama Integration
5. **05_03** -- Model Cards & Responsible AI

**Time**: 20-25 hours | **Hardware**: GPU with 8GB+ VRAM recommended

### Path 3: Computer Vision & Multimodal

**Goal**: Master vision and multimodal tasks

1. **00_01-00_03** -- Fundamentals
2. **02_01-02_04** (CV) -- Classification, Detection, OCR, Segmentation
3. **04_01-04_05** (Multimodal) -- Captioning, VQA, Text-to-Image, Editing, Documents
4. **05_02** -- Performance Optimization

**Time**: 15-18 hours | **Hardware**: GPU with 8GB+ VRAM recommended

### Path 4: Production ML Engineer

**Goal**: Deploy and optimize models

1. **01_01-01_03** -- Core NLP understanding
2. **01_07-01_08** -- Fine-tuning techniques
3. **05_02** -- Performance, Caching, Costs
4. **05_03** -- Model Cards & Responsible AI
5. **05_04** -- HuggingFace Datasets
6. **05_05** -- Gradio & Spaces (demo deployment)
7. **05_06-05_07** -- Quantization (Intro + Deep Dive)
8. **05_08** -- Training Best Practices

**Time**: 25-30 hours | **Hardware**: GPU with 16GB+ VRAM recommended

### Path 5: Agentic AI Developer

**Goal**: Build autonomous agents that use tools

1. **01_01** (NLP) -- Text Generation (LLM basics)
2. **05_01** -- Ollama Integration (local models)
3. **06_01** -- MCP Basics (tool-using agents)
4. **06_02** -- MCP Servers (reusable tool servers)
5. **06_03** -- Multi-Tool Agents (ReAct, Plan-and-Execute)
6. **06_04** -- RAG with Local LLMs (vector databases)
7. **06_05** -- Structured Output & Function Calling

**Time**: 18-22 hours | **Hardware**: GPU with 8GB+ VRAM recommended

### Path 6: Complete Coverage

All 38 notebooks by section. **Time**: 80-100 hours | **Hardware**: GPU with 12GB+ VRAM recommended

---

## Resources

### Official Documentation

- [HuggingFace Transformers](https://huggingface.co/docs/transformers) -- API reference
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) -- Dataset library
- [HuggingFace Hub](https://huggingface.co/docs/hub) -- Model hub
- [PyTorch Documentation](https://pytorch.org/docs/) -- Deep learning framework

### Learning Resources

- [HuggingFace Course](https://huggingface.co/course) -- Free comprehensive course
- [HuggingFace Forums](https://discuss.huggingface.co/) -- Community help
- [Papers with Code](https://paperswithcode.com/) -- Research and benchmarks

### Model Repositories

- [HuggingFace Model Hub](https://huggingface.co/models) -- 400,000+ models
- [HuggingFace Spaces](https://huggingface.co/spaces) -- ML demos and applications

---

## Contributing

Found an issue or want to add more examples?

- Open an issue to report bugs or suggest improvements
- Submit a pull request with your enhancements
- Share your fine-tuned models on HuggingFace Hub

---

## License

This project is for educational purposes. Individual models and datasets may have their own licenses.

**MIT License** for the tutorial code and notebooks.

---

## Acknowledgments

**Built with**:
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Claude Code](https://claude.ai/code)
- [Jupyter](https://jupyter.org/)

**Inspired by**:
- [HuggingFace Course](https://huggingface.co/course)
- [Fast.ai](https://www.fast.ai/)
- The open-source ML community

---

**Setup Guide**: [getting_started.md](getting_started.md) | **Dependencies**: [requirements.txt](requirements.txt) | **CLI Tools**: [functions/README.md](functions/README.md)
