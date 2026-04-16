# Getting Started with HuggingFace Tutorial

**Complete setup guide to run all 36 notebooks successfully.**

## Table of Contents

- [Quick Start (TL;DR)](#quick-start-tldr)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
  - [Step 1: Python Setup](#step-1-python-setup)
  - [Step 2: Virtual Environment](#step-2-create-virtual-environment)
  - [Step 3: Core Dependencies](#step-3-install-core-dependencies)
  - [Step 4: GPU Support (Optional)](#step-4-gpu-support-optional)
  - [Step 5: Advanced Features (Optional)](#step-5-advanced-features-optional)
- [HuggingFace Token](#huggingface-token-configuration)
- [Verification](#verifying-your-setup)
- [Running Notebooks](#running-jupyter-notebooks)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (TL;DR)

For experienced users who want to get started immediately:

```bash
# 1. Navigate to project directory
cd HuggingFace_Tutorial

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) GPU support - install CUDA-enabled PyTorch
# Visit: https://pytorch.org/get-started/locally/

# 5. Launch Jupyter
jupyter notebook
```

**First notebook to try**: `notebooks/00_fundamentals/00_01_tokenization_embeddings.ipynb`

---

## System Requirements

### Minimum Specifications (CPU-only)

**Good for**: Learning fundamentals, most notebooks in sections 00-05

- **Python**: 3.10 or 3.11 (recommended), 3.9 also supported
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 15GB free space (5GB dependencies + 5-10GB model cache)
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 20.04+

### Recommended Specifications (with GPU)

**Good for**: All 36 notebooks including fine-tuning, image generation, and advanced quantization

- **Python**: 3.10 or 3.11
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM
  - RTX 3060 (12GB VRAM): Good for most notebooks
  - RTX 4070/4080 (12-16GB VRAM): Excellent for all notebooks including Unsloth
  - RTX 4090 (24GB VRAM): Can run larger SOTA models and larger Llama models
- **CUDA**: Version 11.8 or 12.1 (matches PyTorch version)
- **Storage**: 25GB free space (5GB dependencies + 10-15GB model cache + 5-10GB training outputs)

### What Works on What Hardware?

| Notebooks | CPU (8GB RAM) | GPU (8GB VRAM) | GPU (16GB+ VRAM) |
|-----------|---------------|----------------|------------------|
| 00_01-00_03 (Fundamentals) | All methods | All methods | All methods |
| 01_01-01_06 (NLP core) | All methods | All methods | All methods |
| 01_07 (Unsloth) | GPU-only | Llama 3.2-1B | Llama 3.2-3B/3.1-8B |
| 01_08 (LoRA) | Very slow | Works well | Works well |
| 02_01-02_04 (CV) | Small models | All methods | All methods |
| 03_01-03_03 (Audio) | Small models | All methods | All methods |
| 04_01-04_02 (Captioning, VQA) | Works | Works | Works |
| 04_03-04_04 (Image Gen/Edit) | Fallback mode | Stable Diffusion | All methods |
| 04_05 (Document Understanding) | Works | Works | Works |
| 05_01-05_06 (Best Practices) | Works | Works | Works |
| 05_07 (Quantization Deep Dive) | INT8 only | All methods | All methods |
| 05_08 (Training) | Slow | Works well | Works well |
| 06_01-06_05 (Agentic) | Small Ollama | Llama 3.2 | Llama 3.1:8b |

---

## Installation Guide

### Step 1: Python Setup

#### Check if Python is installed

```bash
python --version
# or
python3 --version
```

**Expected output**: `Python 3.10.x` or `Python 3.11.x`

#### Install Python (if needed)

**Windows**:
1. Download from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify: Open Command Prompt and run `python --version`

**macOS**:
```bash
# Using Homebrew (recommended)
brew install python@3.11
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

---

### Step 2: Create Virtual Environment

**Why use a virtual environment?**
- Isolates project dependencies from other Python projects
- Prevents version conflicts
- Easy to delete and recreate if something breaks

#### Navigate to project directory

```bash
cd path/to/HuggingFace_Tutorial
```

#### Create venv

**Windows**:
```bash
python -m venv venv
```

**macOS/Linux**:
```bash
python3 -m venv venv
```

#### Activate the virtual environment

**Windows (Command Prompt)**:
```bash
venv\Scripts\activate
```

**Windows (PowerShell)** -- if you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**macOS/Linux**:
```bash
source venv/bin/activate
```

**Success indicator**: Your prompt should now show `(venv)` at the beginning:
```
(venv) C:\Users\YourName\HuggingFace_Tutorial>
```

**To deactivate later** (when you're done):
```bash
deactivate
```

---

### Step 3: Install Core Dependencies

**With your virtual environment activated**, install the core packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What gets installed** (~2-3 GB):

| Category | Packages |
|----------|----------|
| HuggingFace | `transformers`, `datasets`, `huggingface-hub`, `accelerate` |
| Deep Learning | `torch`, `torchvision`, `torchaudio` |
| Media | `Pillow`, `soundfile`, `librosa` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Notebooks | `jupyter`, `ipywidgets` |
| Fine-tuning | `peft`, `trl` |
| Utilities | `numpy`, `tqdm`, `psutil` |

**Installation time**: 5-15 minutes depending on internet speed.

---

### Step 4: GPU Support (Optional)

**Skip this if** you don't have an NVIDIA GPU or are happy using CPU.

#### Check if CUDA is already working

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

- `True`: You're all set!
- `False`: Continue below

#### Install CUDA-enabled PyTorch

**Step 1**: Check your CUDA version:
```bash
nvidia-smi
```
Look for "CUDA Version" in the output (e.g., 12.1, 11.8).

**Step 2**: Uninstall CPU-only PyTorch:
```bash
pip uninstall torch torchvision torchaudio
```

**Step 3**: Install CUDA-enabled PyTorch from [pytorch.org/get-started](https://pytorch.org/get-started/locally/):

```bash
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 4**: Verify GPU is detected:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### Step 5: Advanced Features (Optional)

These unlock additional methods in specific notebooks. All are optional.

#### YOLOv8 for Object Detection (Notebook 05 CV)

Installed by default via `requirements.txt`. Used in Notebook 05 - Method 3.

#### PaddleOCR for Advanced OCR (Notebook 06 CV)

Installed by default (CPU version) via `requirements.txt`.

**For GPU version** (faster):
```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

#### Unsloth for 2-5x Faster Fine-Tuning (Notebook 04 NLP)

**Special installation** (not in requirements.txt):

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Requirements**: CUDA GPU with 8GB+ VRAM, Python 3.10+.

#### Diffusion Models for Image Generation (Notebooks 04_03, 04_04)

```bash
pip install diffusers safetensors
```

**Requirements**: GPU with 6GB+ VRAM recommended. CPU fallback available but very slow.

#### Advanced Quantization (Notebook 05_07)

```bash
pip install bitsandbytes auto-gptq autoawq
```

**Requirements**: CUDA GPU on Linux. Windows/CPU users can still follow the conceptual content.

#### Training Metrics (Notebook 05_08)

```bash
pip install evaluate
```

#### Ollama for Local LLMs (Notebooks 05_01, 06_01-06_05)

```bash
pip install ollama
```

Then install Ollama itself from [ollama.com](https://ollama.com/) and pull models:

```bash
ollama pull llama3.2:1b   # Small model (1.3GB, CPU-friendly)
ollama pull llama3.1:8b   # Large model (4.7GB, GPU-optimized)
```

#### Gradio (Notebook 14)

```bash
pip install gradio
```

#### ONNX Optimization (Notebook 15)

```bash
pip install onnx onnxruntime optimum[onnxruntime]
```

#### Structured Output (Notebook 06_05)

```bash
pip install pydantic instructor
```

#### MCP for Agentic Workflows (Notebooks 06_01-06_03)

```bash
pip install mcp
```

Requires Ollama installed with Llama models pulled.

#### RAG Dependencies (Notebook 06_04)

```bash
pip install sentence-transformers faiss-cpu chromadb
```

For GPU-accelerated search: `pip install faiss-gpu` instead of `faiss-cpu`.

---

## HuggingFace Token Configuration

### Do I need a token?

**For this tutorial**: No. All default models across the 36 notebooks work without authentication.

**When you need a token**:
- Accessing gated models (Llama 2, Mistral Instruct, etc.)
- Uploading models to HuggingFace Hub
- Using private models

### Getting a Token

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to Settings -> Access Tokens
3. Click "New token" -> Name it -> Select "Read" permissions -> Generate
4. Copy the token (you won't see it again)

### Using Your Token

**Method 1: CLI Login (Recommended)**
```bash
huggingface-cli login
```
Paste your token when prompted. Saved for all future use.

**Method 2: In Notebooks**
```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

**Method 3: Credential File**

Create `hf_credential` in the project root containing just the token string, then:
```python
from huggingface_hub import login
with open("../hf_credential", "r") as f:
    login(token=f.read().strip())
```

---

## Verifying Your Setup

### Quick Verification

Run this in your terminal with venv activated:

```bash
python -c "
import sys, transformers, torch
print(f'Python: {sys.version.split()[0]}')
print(f'Transformers: {transformers.__version__}')
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Running on CPU')
print('Setup OK!')
"
```

### Test a Model

```bash
python -c "
from transformers import pipeline
print('Loading sentiment analyzer...')
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
result = classifier('I love learning about AI!')
print(f'Result: {result}')
print('Setup verified!')
"
```

Expected: `[{'label': 'POSITIVE', 'score': 0.9998}]`

---

## Running Jupyter Notebooks

### Important: Launch from Virtual Environment

You **must** activate your virtual environment before launching Jupyter:

```bash
# 1. Navigate to project directory
cd path/to/HuggingFace_Tutorial

# 2. Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 3. Launch Jupyter
jupyter notebook          # Classic interface
# or
jupyter lab               # Modern interface
```

Your browser should open automatically to `http://localhost:8888`.

### Navigate to Notebooks

1. Click `notebooks/`
2. Choose a section folder:
   - `00_fundamentals/` -- Transformer Fundamentals (3 notebooks)
   - `01_nlp/` -- Natural Language Processing (8 notebooks)
   - `02_computer_vision/` -- Computer Vision (4 notebooks)
   - `03_audio/` -- Audio Processing (3 notebooks)
   - `04_multimodal/` -- Multimodal AI (5 notebooks)
   - `05_best_practices/` -- Best Practices & Production (8 notebooks)
   - `06_agentic_workflows/` -- Agentic AI with MCP (5 notebooks)
3. **Recommended start**: `00_fundamentals/00_01_tokenization_embeddings.ipynb`

### Verify Environment in Notebook

In the first cell of any notebook, run:

```python
import sys
print("Python executable:", sys.executable)
# Should contain 'venv' in the path
```

If you don't see `venv` in the path: exit Jupyter (Ctrl+C), activate venv, relaunch.

### Alternative: Register as Jupyter Kernel

```bash
# With venv activated
pip install ipykernel
python -m ipykernel install --user --name=hf_tutorial --display-name "HuggingFace Tutorial"
```

Then select the kernel in any notebook: Kernel -> Change Kernel -> "HuggingFace Tutorial".

---

## Troubleshooting

### General Issues

**`python: command not found`**
- Windows: Reinstall Python with "Add to PATH" checked
- macOS/Linux: Try `python3` instead of `python`

**`pip: command not found`**
```bash
python -m pip install --upgrade pip    # Windows
python3 -m pip install --upgrade pip   # macOS/Linux
```

**`No module named 'transformers'`**
1. Check venv is activated: `which python` should show venv path
2. Reinstall: `pip install transformers`
3. Restart Jupyter kernel

### Installation Issues

**"Microsoft Visual C++ 14.0 is required" (Windows)**

Download [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**PaddleOCR installation fails**
```bash
pip install paddlepaddle    # Install backend first
pip install paddleocr opencv-python
```

**Audio libraries fail (librosa, soundfile)**

| OS | Fix |
|----|-----|
| Windows | Install Visual C++ Redistributable; `pip install soundfile --no-cache-dir` |
| macOS | `brew install libsndfile && pip install soundfile` |
| Linux | `sudo apt-get install libsndfile1 && pip install soundfile` |

### GPU/CUDA Issues

**`torch.cuda.is_available()` returns `False`**
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall CUDA-enabled PyTorch (see Step 4 above)
3. Verify CUDA version matches PyTorch CUDA version

**"CUDA out of memory"**
1. Switch to the smaller model option in the notebook
2. Reduce batch size: `batch_size=1`
3. Restart Jupyter kernel to clear GPU memory
4. Close other GPU-using programs

**"cudnn error" or "cublas error"**

Update GPU drivers from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx).

### Model Download Issues

**Download is very slow**

First run downloads models (100MB-3GB). Subsequent runs use cached models from `~/.cache/huggingface/hub/`.

**"Access denied" or "401 Unauthorized"**
1. The model requires authentication -- get a HuggingFace token (see above)
2. Visit the model page and accept terms of use
3. Run: `huggingface-cli login`

### Jupyter Issues

**Kernel dies or crashes**
1. Out of memory: use smaller models or reduce batch size
2. Corrupt environment: `pip install --upgrade jupyter ipykernel`
3. Restart: Kernel -> Restart Kernel

**Wrong Python version**
```bash
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=hf_tutorial
```
Then: Kernel -> Change Kernel -> "hf_tutorial"

### Platform-Specific Issues

**macOS Apple Silicon (M1/M2/M3)**
- PyTorch has native MPS support
- Some notebooks may use `device="mps"` instead of `device="cuda"`
- Some dependencies may require Rosetta 2

**Windows Subsystem for Linux (WSL)**
- Use WSL 2 (not WSL 1)
- Install NVIDIA drivers for WSL
- Follow the [WSL CUDA setup guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

---

## Next Steps

### Setup Complete!

**1. Start with fundamentals** (Section 00):
- Tokenization, transformer architecture, HuggingFace ecosystem overview

**2. Learn core tasks** (Sections 01-03):
- NLP (generation, classification, summarization, NER, QA, translation), CV, Audio

**3. Explore multimodal AI** (Section 04):
- Image captioning, VQA, text-to-image, image editing, document understanding

**4. Best practices** (Section 05):
- Ollama, performance, responsible AI, datasets, Gradio, quantization, Trainer API

**5. Agentic workflows** (Section 06):
- MCP agents, MCP servers, multi-tool agents, RAG, structured output

### Additional Resources

- [HuggingFace Course](https://huggingface.co/course) -- Free comprehensive course
- [HuggingFace Forums](https://discuss.huggingface.co/) -- Community help
- [Transformers Docs](https://huggingface.co/docs/transformers) -- Official documentation
- [PyTorch Forums](https://discuss.pytorch.org/) -- GPU/CUDA issues

---

## Getting Help

1. Check this troubleshooting guide
2. Read error messages carefully (they often tell you what's wrong)
3. Search [HuggingFace Forums](https://discuss.huggingface.co/)
4. Check [PyTorch Forums](https://discuss.pytorch.org/) for GPU/CUDA issues
5. Ask your instructor or TA

**Debugging commands**:
```bash
python --version                        # Python version
pip list                                # Installed packages
nvidia-smi                              # GPU info
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```
