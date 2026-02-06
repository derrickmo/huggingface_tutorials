# HuggingFace Tutorial: Comprehensive Guide to Modern AI Models

**Learn HuggingFace transformers with 17 hands-on Jupyter notebooks covering NLP, Computer Vision, Audio, Multimodal AI, and Agentic Workflows.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet.svg)](https://claude.ai/code)

> **A passion project by [Derrick Mo](https://www.linkedin.com/in/derrickmo/).**
>
> I believe the best way to learn is by actually doing. This tutorial was built from the ground up with [Claude Code](https://claude.ai/code) as a hands-on resource for anyone looking to get started with HuggingFace and modern AI. Whether you're a student, a hobbyist, or a professional exploring new tools, I hope these notebooks help you learn by building real things. Feedback, ideas, and collaboration are always welcome -- feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/derrickmo/)!

---

## 📚 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Notebooks](#notebooks)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Learning Paths](#learning-paths)
- [Resources](#resources)

---

## 🌟 Overview

This tutorial provides a complete introduction to the HuggingFace ecosystem with **17 hands-on Jupyter notebooks**. Each notebook is designed to run on modest hardware with options to scale up for better performance.

### What You'll Learn

**🔤 Natural Language Processing (NLP)**
- Text generation with GPT-2
- Sentiment analysis and classification
- Document summarization

**🖼️ Computer Vision (CV)**
- Image classification with Vision Transformers
- Object detection (DETR + YOLOv8)
- Optical Character Recognition (TrOCR + PaddleOCR)

**🎵 Audio Processing**
- Speech-to-text with Whisper
- Text-to-speech with SpeechT5

**🤝 Multimodal AI**
- Image captioning with BLIP
- Vision-language models

**⚡ Production & Best Practices**
- Performance optimization and caching
- Model cards and responsible AI
- Fine-tuning with LoRA (+ Unsloth speedup)
- Integration with Ollama for local LLMs

**🤖 Agentic Workflows**
- Model Context Protocol (MCP) fundamentals
- Building reusable MCP servers
- Multi-tool agent patterns (ReAct, Plan-and-Execute, Reflection)
- Retrieval-Augmented Generation (RAG) with vector databases

### Key Features

✅ **Beginner-friendly**: Start with small CPU-compatible models
✅ **Progressively complex**: Scale to GPU-optimized and state-of-the-art models
✅ **Three-tier approach**: Basic → Advanced → State-of-the-Art
✅ **Self-contained**: Each notebook is independent
✅ **Reproducible**: All notebooks use seed 1103 for consistent results
✅ **Production-ready**: Learn optimization and deployment techniques
✅ **Ethical AI**: Understand bias, fairness, and responsible practices

---

## 🚀 Quick Start

**Get up and running in 5 minutes:**

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

# 5. Open notebooks/01_nlp/01_nlp_text_generation.ipynb
```

**👉 For detailed setup instructions, see [getting_started.md](getting_started.md)**

---

## 📖 Notebooks

All notebooks are in the `notebooks/` folder and can be run independently in any order.

### 🔤 NLP - Natural Language Processing

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 01 | **Text Generation** | GPT-2, DistilGPT2 | Generate coherent text continuations | ⭐ Beginner |
| 02 | **Text Classification** | DistilBERT, BERT | Sentiment analysis and topic classification | ⭐ Beginner |
| 03 | **Text Summarization** | DistilBART, BART | Condense long documents into summaries | ⭐⭐ Intermediate |
| 04 | **Fine-tuning with Unsloth** ⚡ | Llama 3.2/3.1 | 2-5x faster LLM fine-tuning with Unsloth | ⭐⭐⭐ Advanced |
| 05 | **Fine-tuning with LoRA** | GPT-2 | Adapter training with LoRA (CPU-compatible) | ⭐⭐⭐ Advanced |

**SOTA Models Referenced**: Llama 2/3, Mistral, Qwen, Gemma, RoBERTa, DeBERTa, PEGASUS, T5

### 🖼️ Computer Vision

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 04 | **Image Classification** | ViT (Vision Transformer) | Identify objects in images | ⭐ Beginner |
| 05 | **Object Detection** | DETR, **YOLOv8** ⚡ | Detect and localize multiple objects | ⭐⭐ Intermediate |
| 06 | **OCR** | TrOCR, **PaddleOCR** 🌟 | Extract text from images (80+ languages) | ⭐⭐ Intermediate |

**SOTA Models Referenced**: ConvNeXt, EfficientNet, Swin Transformer, DINO, SAM, Grounding DINO, GOT-OCR

### 🎵 Audio Processing

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 07 | **Speech Recognition** | Whisper (tiny/small) | Transcribe audio to text | ⭐⭐ Intermediate |
| 08 | **Text-to-Speech** | SpeechT5 | Generate natural-sounding speech | ⭐⭐ Intermediate |

**SOTA Models Referenced**: Whisper Large V3, Wav2Vec 2.0, HuBERT, XTTS, Bark, VITS

### 🤝 Multimodal

| # | Notebook | Models | Description | Difficulty |
|---|----------|--------|-------------|------------|
| 09 | **Image-to-Text** | BLIP | Generate captions from images | ⭐⭐ Intermediate |

**SOTA Models Referenced**: LLaVA, BLIP-2, InstructBLIP, Qwen-VL, CogVLM

### 🛠️ Best Practices & Production

| # | Notebook | Focus | Description | Difficulty |
|---|----------|-------|-------------|------------|
| 10 | **Ollama Integration** | Local LLMs | Use Ollama models with HuggingFace tools | ⭐⭐ Intermediate |
| 11 | **Performance & Caching** | Optimization | Measure latency, optimize throughput, estimate costs | ⭐⭐ Intermediate |
| 12 | **Model Cards & AI Ethics** | Responsible AI | Understand bias, fairness, and documentation | ⭐⭐ Intermediate |

### 🤖 Agentic Workflows

| # | Notebook | Focus | Description | Difficulty |
|---|----------|-------|-------------|------------|
| 14 | **MCP Basics** | Tool-using agents | Model Context Protocol fundamentals, tool calling | ⭐⭐ Intermediate |
| 15 | **MCP Servers** | Reusable tools | Build file system and data analysis servers | ⭐⭐⭐ Advanced |
| 16 | **Multi-Tool Agents** | Agent patterns | ReAct, Plan-and-Execute, Reflection patterns | ⭐⭐⭐ Advanced |
| 17 | **RAG with Local LLMs** | Retrieval-Augmented Generation | Vector databases, semantic search, context injection | ⭐⭐⭐ Advanced |

**Key Technologies**: MCP, Ollama, Llama 3.2/3.1, RAG, FAISS, ChromaDB, Sentence Transformers

---

## 💻 System Requirements

### Minimum (CPU-only)
**✅ Good for**: Notebooks 01-12 (basic methods), 14-17 (agentic workflows), learning fundamentals

- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 15GB free space
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended (with GPU)
**✅ Good for**: All notebooks, SOTA models, fine-tuning

- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
  - RTX 3060 (12GB): Good
  - RTX 4070/4080 (12-16GB): Excellent
  - RTX 4090 (24GB): Can run larger SOTA models
- **CUDA**: 11.8 or 12.1
- **Storage**: 25GB free space

### What Runs Where?

| Hardware | Can Run |
|----------|---------|
| **CPU (8GB RAM)** | ✅ Notebooks 01-12 with basic models<br>⚠️ Notebook 05 NLP (LoRA, very slow)<br>❌ Notebook 04 NLP (Unsloth, GPU-only)<br>✅ Notebooks 14-17 (agentic, with small Ollama models) |
| **GPU (8GB VRAM)** | ✅ All notebooks<br>✅ YOLO, TrOCR<br>✅ Unsloth with Llama 3.2-1B<br>⚠️ PaddleOCR (tight fit)<br>✅ Agentic workflows with Llama 3.2 |
| **GPU (16GB+ VRAM)** | ✅ Everything including:<br>✅ PaddleOCR, Large YOLO models<br>✅ Unsloth with Llama 3.2-3B/3.1-8B<br>✅ RAG with Llama 3.1:8b |

**👉 See [getting_started.md](getting_started.md) for detailed hardware recommendations**

---

## 📦 Installation

### Standard Installation (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 2. Install all dependencies
pip install -r requirements.txt

# 3. (Optional) GPU support
# Visit https://pytorch.org/get-started/locally/ for CUDA-enabled PyTorch
```

### What Gets Installed

**Core dependencies** (~2-3 GB):
- `transformers`, `datasets`, `huggingface-hub` - HuggingFace ecosystem
- `torch`, `torchvision`, `torchaudio` - PyTorch framework
- `jupyter`, `ipywidgets` - Notebook environment
- `Pillow`, `soundfile`, `librosa` - Media processing
- `matplotlib`, `seaborn`, `plotly` - Visualizations

**Advanced features** (installed by default):
- `ultralytics` - YOLOv8 for object detection (Notebook 05)
- `paddleocr`, `paddlepaddle` - PaddleOCR for advanced OCR (Notebook 06)
- `peft`, `trl` - LoRA fine-tuning (Notebooks 04-05 NLP)

**Optional** (commented out in requirements.txt):
- `unsloth` - 2-5x faster LoRA training (requires special installation)
- `ollama` - Local LLM integration (Notebook 10)

**👉 See [getting_started.md](getting_started.md) for:**
- Step-by-step installation guide
- GPU setup instructions
- Troubleshooting common issues
- Platform-specific notes (Windows/macOS/Linux)

---

## 📁 Project Structure

```
HuggingFace_Tutorial/
├── README.md                           # This file - project overview
├── getting_started.md                  # Detailed setup instructions ⭐
├── requirements.txt                    # Python dependencies (organized by category)
│
├── notebooks/                          # 17 Jupyter notebooks (organized by domain)
│   ├── shared_utils.py                 # Common helper functions
│   │
│   ├── 01_nlp/                         # Natural Language Processing (5 notebooks)
│   │   ├── 01_nlp_text_generation.ipynb
│   │   ├── 02_nlp_text_classification.ipynb
│   │   ├── 03_nlp_text_summarization.ipynb
│   │   ├── 04_nlp_fine_tuning_unsloth.ipynb   # 2-5x faster fine-tuning
│   │   └── 05_nlp_fine_tuning_lora.ipynb      # Standard LoRA (CPU-compatible)
│   │
│   ├── 02_computer_vision/             # Computer Vision (3 notebooks)
│   │   ├── 04_cv_image_classification.ipynb
│   │   ├── 05_cv_object_detection.ipynb    # Includes YOLO
│   │   └── 06_cv_ocr.ipynb                 # TrOCR + PaddleOCR
│   │
│   ├── 03_audio/                       # Audio Processing (2 notebooks)
│   │   ├── 07_audio_speech_recognition.ipynb
│   │   └── 08_audio_text_to_speech.ipynb
│   │
│   ├── 04_multimodal/                  # Multimodal AI (1 notebook)
│   │   └── 09_multimodal_image_to_text.ipynb
│   │
│   ├── 05_best_practices/              # Best Practices & Production (3 notebooks)
│   │   ├── 10_ollama_integration.ipynb
│   │   ├── 11_performance_caching_costs.ipynb
│   │   └── 12_model_cards_responsible_ai.ipynb
│   │
│   └── 06_agentic_workflows/           # Agentic Workflows with MCP (4 notebooks)
│       ├── 14_mcp_basics.ipynb
│       ├── 15_mcp_servers.ipynb
│       ├── 16_multi_tool_agents.ipynb
│       └── 17_rag_local_llms.ipynb
│
├── functions/                          # Standalone CLI tools (12 total)
│   ├── README.md                       # CLI tool documentation
│   │
│   ├── nlp/                           # NLP CLI tools
│   │   ├── text_generation.py
│   │   ├── text_classification.py
│   │   └── text_summarization.py
│   │
│   ├── cv/                            # Computer Vision CLI tools
│   │   ├── image_classification.py
│   │   ├── object_detection.py
│   │   └── ocr.py
│   │
│   ├── audio/                         # Audio CLI tools
│   │   ├── speech_recognition.py
│   │   └── text_to_speech.py
│   │
│   ├── multimodal/                    # Multimodal CLI tools
│   │   └── image_captioning.py
│   │
│   └── agentic/                       # Agentic Workflow CLI tools
│       ├── mcp_agent.py
│       ├── mcp_server.py
│       └── multi_agent.py
│
└── sample_data/                        # Test images and audio files
```

---

## 📦 Dependencies

### Core Libraries (Required)

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| HuggingFace | `transformers` | ≥4.30.0 | Model library |
| HuggingFace | `datasets` | ≥2.14.0 | Dataset loading |
| HuggingFace | `huggingface-hub` | ≥0.16.0 | Model hub interaction |
| Deep Learning | `torch` | ≥2.0.0 | PyTorch framework |
| Deep Learning | `torchvision` | ≥0.15.0 | Computer vision utilities |
| Deep Learning | `torchaudio` | ≥2.0.0 | Audio utilities |
| Image | `Pillow` | ≥9.5.0 | Image processing |
| Audio | `soundfile` | ≥0.12.0 | Audio file I/O |
| Audio | `librosa` | ≥0.10.0 | Audio analysis |
| Utilities | `numpy` | ≥1.24.0 | Numerical computing |
| Notebooks | `jupyter` | ≥1.0.0 | Notebook environment |
| Visualization | `matplotlib` | ≥3.7.0 | Plotting |
| Visualization | `seaborn` | ≥0.12.0 | Statistical plots |
| Visualization | `plotly` | ≥5.14.0 | Interactive plots |

### Advanced Features (Installed by Default)

| Feature | Package | Version | Used In | Size |
|---------|---------|---------|---------|------|
| Object Detection | `ultralytics` | ≥8.0.0 | Notebook 05 (YOLO) | ~50MB |
| Advanced OCR | `paddleocr` | ≥2.7.0 | Notebook 06 (PaddleOCR) | ~3.5GB |
| Advanced OCR | `paddlepaddle` | ≥2.5.0 | Backend for PaddleOCR | Included |
| Advanced OCR | `opencv-python` | ≥4.8.0 | Image processing for OCR | ~100MB |
| Fine-tuning | `peft` | ≥0.5.0 | Notebooks 04-05 NLP (LoRA) | ~10MB |
| Fine-tuning | `trl` | Latest | Notebook 04 NLP (SFTTrainer) | ~5MB |
| Memory Profiling | `psutil` | ≥5.9.0 | Notebook 11 | ~1MB |

### Optional Advanced Features

| Feature | Package | Installation | Used In |
|---------|---------|--------------|---------|
| Fast LoRA | `unsloth` | Special (see below) | Notebook 04 NLP (Unsloth) |
| Local LLMs | `ollama` | `pip install ollama` | Notebooks 10, 14-17 |

**Installing Unsloth** (2-5x faster LoRA training):
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Requires**: CUDA GPU with 8GB+ VRAM

### GPU Support

**Default installation uses CPU-only PyTorch.**

For GPU acceleration:

```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version (choose your CUDA version)
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**👉 Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to generate the correct command for your system**

### Dependency Summary by Notebook

| Notebooks | Dependencies |
|-----------|-------------|
| 01-03 (NLP basics) | Core only |
| 04 (Unsloth Fine-tuning) | Core + `peft` + optional `unsloth` (GPU-only) |
| 05 (LoRA Fine-tuning) | Core + `peft` + `trl` |
| 04 (Image Classification) | Core only |
| 05 (Object Detection) | Core + `ultralytics` (YOLO) |
| 06 (OCR) | Core + `paddleocr` + `paddlepaddle` + `opencv-python` |
| 07-09 (Audio, Multimodal) | Core only |
| 10 (Ollama) | Core + optional `ollama` |
| 11 (Performance) | Core + `psutil` + visualization libs |
| 12 (Model Cards) | Core only (no model downloads) |
| 14-16 (MCP Agentic) | Core + `ollama` + `mcp` |
| 17 (RAG) | Core + `ollama` + `sentence-transformers` + `faiss-cpu` + `chromadb` |

**Full details in [requirements.txt](requirements.txt) and [getting_started.md](getting_started.md)**

---

## 🎓 Learning Paths

### Path 1: Complete Beginner (Start Here!)

**Goal**: Learn ML fundamentals with hands-on examples

1. ✅ Read [getting_started.md](getting_started.md) and install dependencies
2. ✅ **Notebook 01** (NLP) - Text Generation (understand transformers)
3. ✅ **Notebook 02** (NLP) - Text Classification (learn about fine-tuned models)
4. ✅ **Notebook 04** (CV) - Image Classification (computer vision basics)
5. ✅ **Notebook 07** (Audio) - Speech Recognition (audio processing)

**Time**: 8-10 hours | **Hardware**: CPU with 8GB RAM

### Path 2: NLP Specialist

**Goal**: Master natural language processing

1. ✅ **Notebook 01** (NLP) - Text Generation
2. ✅ **Notebook 02** (NLP) - Text Classification
3. ✅ **Notebook 03** (NLP) - Text Summarization
4. ✅ **Notebook 04** (NLP) - Unsloth Fine-tuning (GPU-only, 2-5x faster)
5. ✅ **Notebook 05** (NLP) - LoRA Fine-tuning (CPU-compatible)
6. ✅ **Notebook 10** (Best Practices) - Ollama Integration
7. ✅ **Notebook 12** (Best Practices) - Model Cards & Responsible AI

**Time**: 15-20 hours | **Hardware**: GPU with 8GB+ VRAM recommended for Notebook 04

### Path 3: Computer Vision Specialist

**Goal**: Master vision tasks

1. ✅ **Notebook 04** (CV) - Image Classification
2. ✅ **Notebook 05** (CV) - Object Detection (include YOLO)
3. ✅ **Notebook 06** (CV) - OCR (include PaddleOCR)
4. ✅ **Notebook 09** (Multimodal) - Image-to-Text
5. ✅ **Notebook 11** (Best Practices) - Performance Optimization

**Time**: 10-12 hours | **Hardware**: GPU with 8GB+ VRAM recommended

### Path 4: Production ML Engineer

**Goal**: Deploy models in production

1. ✅ **Notebooks 01-03** (NLP basics) - Core understanding
2. ✅ **Notebook 04** (NLP) - Unsloth Fine-tuning (2-5x faster) ⭐
3. ✅ **Notebook 05** (NLP) - LoRA Fine-tuning ⭐
4. ✅ **Notebook 11** (Best Practices) - Performance, Caching, Costs ⭐
5. ✅ **Notebook 12** (Best Practices) - Model Cards & Responsible AI ⭐
6. ✅ **Notebooks 05-06** (CV) - SOTA models (YOLO, PaddleOCR)

**Time**: 15-20 hours | **Hardware**: GPU with 16GB+ VRAM recommended

### Path 5: Agentic AI Developer

**Goal**: Build autonomous agents that use tools and APIs

1. ✅ **Notebook 01** (NLP) - Text Generation (LLM basics)
2. ✅ **Notebook 10** (Best Practices) - Ollama Integration (local models)
3. ✅ **Notebook 14** (Agentic) - MCP Basics (tool-using agents)
4. ✅ **Notebook 15** (Agentic) - MCP Servers (reusable tool servers)
5. ✅ **Notebook 16** (Agentic) - Multi-Tool Agents (advanced patterns)
6. ✅ **Notebook 17** (Agentic) - RAG with Local LLMs (vector databases)

**Time**: 15-18 hours | **Hardware**: GPU with 8GB+ VRAM recommended for llama3.1:8b

### Path 6: All-in-One (Complete Coverage)

**Goal**: Comprehensive understanding of HuggingFace ecosystem

Complete all 17 notebooks by domain:
- **01_nlp/**: Notebooks 01-05 (5 notebooks)
- **02_computer_vision/**: Notebooks 04-06 (3 notebooks)
- **03_audio/**: Notebooks 07-08 (2 notebooks)
- **04_multimodal/**: Notebook 09 (1 notebook)
- **05_best_practices/**: Notebooks 10-12 (3 notebooks)
- **06_agentic_workflows/**: Notebooks 14-17 (4 notebooks)

**Time**: 45-50 hours | **Hardware**: GPU with 12GB+ VRAM recommended

---

## 📚 Resources

### Official Documentation

- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Complete API reference
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) - Dataset library docs
- [HuggingFace Hub](https://huggingface.co/docs/hub) - Model hub documentation
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework

### Learning Resources

- [HuggingFace Course](https://huggingface.co/course) - Free comprehensive course
- [HuggingFace Forums](https://discuss.huggingface.co/) - Community help and discussions
- [Papers with Code](https://paperswithcode.com/) - Latest research and benchmarks

### Model Repositories

- [HuggingFace Model Hub](https://huggingface.co/models) - 400,000+ models
- [HuggingFace Spaces](https://huggingface.co/spaces) - ML demos and applications

### Leaderboards & Benchmarks

- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Language models
- [GLUE Benchmark](https://gluebenchmark.com/) - NLP tasks
- [ImageNet](https://image-net.org/) - Computer vision
- [Papers with Code Leaderboards](https://paperswithcode.com/sota) - State-of-the-art across tasks

### Community & Support

- [HuggingFace Discord](https://discord.gg/hugging-face) - Real-time community chat
- [HuggingFace Twitter](https://twitter.com/huggingface) - Latest updates
- [GitHub Issues](https://github.com/huggingface/transformers/issues) - Bug reports and feature requests

---

## 🤝 Contributing

Found an issue or want to add more examples?

- Open an issue to report bugs or suggest improvements
- Submit a pull request with your enhancements
- Share your fine-tuned models on HuggingFace Hub

---

## 📄 License

This project is for educational purposes. Individual models and datasets may have their own licenses.

**MIT License** for the tutorial code and notebooks.

---

## 🙏 Acknowledgments

**Built with**:
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - The amazing ML library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Jupyter](https://jupyter.org/) - Interactive computing

**Inspired by**:
- [HuggingFace Course](https://huggingface.co/course)
- [Fast.ai](https://www.fast.ai/)
- The open-source ML community

**Special thanks** to the HuggingFace team for democratizing AI! 🤗

---

## 🎯 Quick Links

- **📖 Setup Guide**: [getting_started.md](getting_started.md)
- **📦 Dependencies**: [requirements.txt](requirements.txt)
- **🛠️ CLI Tools**: [functions/README.md](functions/README.md)

---

## ⭐ Star This Repository!

If you find this tutorial helpful, please give it a star! It helps others discover this resource.

**Happy Learning! 🚀**
