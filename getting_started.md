# Getting Started with HuggingFace Tutorial

**Complete setup guide to run all 17 notebooks successfully.**

## Table of Contents
- [Quick Start (TL;DR)](#quick-start-tldr)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
  - [Step 1: Python Setup](#step-1-python-setup)
  - [Step 2: Virtual Environment](#step-2-create-virtual-environment-recommended)
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

**First notebook to try**: `notebooks/01_nlp/01_nlp_text_generation.ipynb`

---

## System Requirements

### Minimum Specifications (CPU-only)

✅ **Recommended for**: Learning fundamentals, notebooks 01-12 and 14-17

- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 15GB free space
  - 5GB for dependencies
  - 5-10GB for model cache
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Specifications (with GPU)

✅ **Recommended for**: Advanced methods, notebooks 05-06 (SOTA models), notebooks 04-05 (fine-tuning)

- **Python**: 3.8, 3.9, 3.10, or 3.11 (3.10+ recommended for Unsloth)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM
  - RTX 3060 (12GB VRAM): Good for most notebooks
  - RTX 4070/4080 (12-16GB VRAM): Excellent for all notebooks including Unsloth
  - RTX 4090 (24GB VRAM): Can run larger SOTA models and larger Llama models
- **CUDA**: Version 11.8 or 12.1 (matches PyTorch version)
- **Storage**: 25GB free space
  - 5GB for dependencies
  - 10-15GB for model cache (SOTA models are larger)
  - 5-10GB for training outputs and Llama models

### What Works on What Hardware?

| Notebooks | CPU (8GB RAM) | GPU (8GB VRAM) | GPU (16GB+ VRAM) |
|-----------|---------------|----------------|------------------|
| 01-03 (NLP basics) | ✅ All methods | ✅ All methods | ✅ All methods |
| 04 (Unsloth Fine-tuning) | ❌ GPU-only | ✅ Llama 3.2-1B | ✅ Llama 3.2-3B/3.1-8B |
| 05 (LoRA Fine-tuning) | ⚠️ Very slow | ✅ Works well | ✅ Works well |
| 04 (Image Classification) | ✅ All methods | ✅ All methods | ✅ All methods |
| 05 (Object Detection) | ✅ DETR only | ✅ DETR + YOLO | ✅ All + larger YOLO |
| 06 (OCR) | ✅ TrOCR only | ✅ TrOCR + PaddleOCR | ✅ All |
| 07-09 (Audio, Multimodal) | ✅ Small models | ✅ All methods | ✅ All methods |
| 10-12 (Best Practices) | ✅ Works | ✅ Works | ✅ Works |
| 14-16 (MCP Agentic) | ✅ With small Ollama models | ✅ With Llama 3.2 | ✅ All models |
| 17 (RAG) | ✅ With small models | ✅ With Llama 3.2 | ✅ With Llama 3.1:8b |

---

## Installation Guide

### Step 1: Python Setup

#### Check if Python is installed

```bash
python --version
# or
python3 --version
```

**Expected output**: `Python 3.8.x`, `Python 3.9.x`, `Python 3.10.x`, or `Python 3.11.x`

#### Install Python (if needed)

**Windows**:
1. Download from [python.org](https://www.python.org/downloads/)
2. ✅ **Important**: Check "Add Python to PATH" during installation
3. Verify: Open Command Prompt and run `python --version`

**macOS**:
```bash
# Using Homebrew (recommended)
brew install python@3.10
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

---

### Step 2: Create Virtual Environment (Recommended)

**Why use a virtual environment?**
- Isolates project dependencies
- Prevents conflicts with other Python projects
- Easy to delete and recreate if something breaks

#### Create the virtual environment

**Navigate to project directory first**:
```bash
cd path/to/HuggingFace_Tutorial
```

**Create venv**:

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

**Windows (PowerShell)** - if you get an execution policy error:
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
pip install --upgrade pip  # Upgrade pip first
pip install -r requirements.txt
```

**What gets installed:**

**Core Libraries** (~2-3 GB):
- `transformers` - HuggingFace's model library
- `torch`, `torchvision`, `torchaudio` - PyTorch framework
- `datasets` - Dataset loading
- `jupyter` - Notebook environment
- `Pillow`, `soundfile`, `librosa` - Media processing
- `matplotlib`, `seaborn`, `plotly` - Visualizations
- `peft`, `trl` - Fine-tuning utilities

**Installation time**: 5-15 minutes depending on internet speed

**Expected output** (last lines):
```
Successfully installed transformers-4.3x.x torch-2.x.x ...
```

---

### Step 4: GPU Support (Optional)

**Skip this section if**:
- You don't have an NVIDIA GPU
- You're happy using CPU for learning

#### Check if you have CUDA already

```python
python -c "import torch; print(torch.cuda.is_available())"
```

- `True`: ✅ You're all set!
- `False`: Continue below to install CUDA-enabled PyTorch

#### Install CUDA-enabled PyTorch

**Step 1: Check your CUDA version**

```bash
nvidia-smi
```

Look for "CUDA Version" in the output (e.g., 12.1, 11.8)

**Step 2: Uninstall CPU-only PyTorch**

```bash
pip uninstall torch torchvision torchaudio
```

**Step 3: Install CUDA-enabled PyTorch**

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select:
- PyTorch Build: Stable
- Your OS: Windows/Linux/Mac
- Package: Pip
- Language: Python
- Compute Platform: CUDA 11.8 or CUDA 12.1 (match your version)

**Example for CUDA 11.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Example for CUDA 12.1**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 4: Verify GPU is detected**

```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output**:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4080
```

---

### Step 5: Advanced Features (Optional)

These are **NOT required** for the core tutorial but unlock additional methods in specific notebooks.

#### YOLOv8 for Real-Time Object Detection (Notebook 05)

**Already installed by default** in requirements.txt

If you want to skip it:
```bash
pip uninstall ultralytics
```

**Usage**: Notebook 05 - Method 3

#### PaddleOCR for Advanced OCR (Notebook 06)

**Already installed by default** in requirements.txt (CPU version)

**For GPU version** (faster OCR):
```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu  # Requires CUDA
pip install paddleocr opencv-python
```

**Usage**: Notebook 06 - Method 2

#### Unsloth for 2-5x Faster Fine-Tuning (Notebook 04)

**Special installation** (not in requirements.txt by default):

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Note**: May take 5-10 minutes to compile

**Requirements**:
- CUDA GPU (does not work on CPU)
- 8GB+ VRAM (16GB+ recommended for larger models)
- Python 3.10+ recommended

**Usage**: Notebook 04 - Unsloth Fine-tuning (dedicated tutorial)

#### Ollama Integration (Notebooks 10, 14-16)

**Optional** - only if you want to run local LLMs:

```bash
pip install ollama
```

Then install Ollama itself from [ollama.ai](https://ollama.ai/)

**Pull models for agentic workflows**:
```bash
ollama pull llama3.2:1b   # Small model (1.3GB, CPU-friendly)
ollama pull llama3.1:8b   # Large model (4.7GB, GPU-optimized)
```

**Usage**: Notebook 10 (local LLM deployment), Notebooks 14-16 (agentic workflows)

#### MCP for Agentic Workflows (Notebooks 14-16)

**Optional** - only if you want to build tool-using agents:

```bash
pip install mcp
```

**What is MCP?**
Model Context Protocol is Anthropic's open protocol for connecting AI assistants to external tools and data sources.

**Requirements**:
- Ollama installed (see above)
- Llama 3.2 or 3.1 models pulled
- Python 3.10+ recommended

**Usage**:
- Notebook 14: MCP Basics (tool calling fundamentals)
- Notebook 15: MCP Servers (reusable tool servers)
- Notebook 16: Multi-Tool Agents (advanced patterns)

#### RAG Dependencies (Notebook 17)

**Optional** - only if you want to build RAG (Retrieval-Augmented Generation) systems:

```bash
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu if you have CUDA
pip install chromadb
```

**What is RAG?**
RAG combines information retrieval with text generation, allowing LLMs to answer questions using external knowledge bases.

**What you'll need**:
- Embedding models for semantic search (downloaded automatically)
- Vector database (FAISS or ChromaDB)
- Ollama with Llama models (see above)

**Components**:
- `sentence-transformers`: Creates embeddings from text
- `faiss-cpu`: Fast similarity search (Facebook AI)
- `chromadb`: Vector database with persistence

**Usage**: Notebook 17 - RAG with Local LLMs

---

## HuggingFace Token Configuration

### Do I need a token?

**For this tutorial**: No, not required for notebooks 01-12 and 14-17 with default models.

**When you DO need a token**:
- Accessing gated models (Llama 2, Mistral Instruct, etc.)
- Uploading models to HuggingFace Hub
- Using private models

### Getting a HuggingFace Token

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Click "New token"
4. Name it (e.g., "Tutorial Token")
5. Select "Read" permissions
6. Click "Generate token"
7. **Copy the token** (you won't see it again!)

### Using Your Token

**Method 1: CLI Login (Recommended)**

```bash
huggingface-cli login
```

Paste your token when prompted. It will be saved for all future use.

**Method 2: In Notebooks**

```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

**Method 3: Save to File**

Create a file named `hf_credential` in the project root:
```
hf_your_token_here_without_quotes
```

Then in notebooks:
```python
from huggingface_hub import login
with open('../hf_credential', 'r') as f:
    token = f.read().strip()
login(token=token)
```

---

## Verifying Your Setup

### Quick Verification Script

Run this in your terminal with venv activated:

```python
python -c "
import transformers
import torch
import PIL
import soundfile
import datasets

print('✅ All core packages imported successfully!')
print(f'Transformers version: {transformers.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Running on CPU')
"
```

**Expected output (CPU)**:
```
✅ All core packages imported successfully!
Transformers version: 4.3x.x
PyTorch version: 2.x.x
CUDA available: False
Running on CPU
```

**Expected output (GPU)**:
```
✅ All core packages imported successfully!
Transformers version: 4.3x.x
PyTorch version: 2.x.x+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 4080
GPU Memory: 16.0 GB
```

### Test a Simple Model

```python
python -c "
from transformers import pipeline
print('Loading sentiment analyzer...')
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
result = classifier('I love learning about AI!')
print(f'Result: {result}')
print('✅ Setup verified! You are ready to go!')
"
```

**Expected output**:
```
Loading sentiment analyzer...
Result: [{'label': 'POSITIVE', 'score': 0.9998}]
✅ Setup verified! You are ready to go!
```

---

## Running Jupyter Notebooks

### IMPORTANT: Launch Jupyter from Virtual Environment

Jupyter notebooks use the Python environment from which they are launched. You **must** activate your virtual environment **before** running `jupyter notebook`.

### Correct Workflow (Every Time)

```bash
# 1. Navigate to project directory
cd path/to/HuggingFace_Tutorial

# 2. Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 3. Verify venv is active (you should see "(venv)" in your prompt)

# 4. Launch Jupyter
jupyter notebook
```

### Launch Options

**Jupyter Notebook** (classic interface):
```bash
jupyter notebook
```

**JupyterLab** (modern interface):
```bash
jupyter lab
```

Your browser should open automatically to `http://localhost:8888`

### Navigate to Notebooks

The notebooks are organized by domain:

1. Click on `notebooks/` folder
2. Choose a domain folder:
   - `01_nlp/` - Natural Language Processing (5 notebooks)
   - `02_computer_vision/` - Computer Vision (3 notebooks)
   - `03_audio/` - Audio Processing (2 notebooks)
   - `04_multimodal/` - Multimodal AI (1 notebook)
   - `05_best_practices/` - Best Practices & Production (3 notebooks)
   - `06_agentic_workflows/` - Agentic AI with MCP (4 notebooks)
3. **Recommended starting point**: `01_nlp/01_nlp_text_generation.ipynb`
4. Or choose any notebook that interests you!

### Verify Your Environment in Notebook

**In the first cell of any notebook**, run:

```python
import sys
print("Python executable:", sys.executable)
print("\nShould contain 'venv' in the path above ✅")
```

**Expected output (venv is correct)**:
```
Python executable: /path/to/HuggingFace_Tutorial/venv/bin/python
Should contain 'venv' in the path above ✅
```

**Wrong output (venv NOT active)**:
```
Python executable: /usr/bin/python
```

If you don't see `venv` in the path:
1. Exit Jupyter (Ctrl+C in terminal)
2. Activate venv: `source venv/bin/activate`
3. Relaunch Jupyter

### Alternative: Register Virtual Environment as Jupyter Kernel

**Advanced option** - allows you to select venv from Jupyter's kernel menu:

```bash
# With venv activated
pip install ipykernel
python -m ipykernel install --user --name=hf_tutorial --display-name "HuggingFace Tutorial"
```

Now you can:
1. Launch Jupyter from anywhere
2. In any notebook: Kernel → Change Kernel → "HuggingFace Tutorial"

---

## Troubleshooting

### General Issues

#### Issue: `python: command not found`

**Solution**:
- Windows: Reinstall Python with "Add to PATH" checked
- macOS/Linux: Try `python3` instead of `python`

#### Issue: `pip: command not found`

**Solution**:
```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

#### Issue: "No module named 'transformers'"

**Symptoms**: Import error in notebooks

**Solution**:
1. Check venv is activated: `which python` should show venv path
2. Reinstall: `pip install transformers`
3. Restart Jupyter kernel

### Installation Issues

#### Issue: "Microsoft Visual C++ 14.0 is required" (Windows)

**Solution**:
Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### Issue: PaddleOCR installation fails

**Solution**:
```bash
# Install paddlepaddle first
pip install paddlepaddle  # CPU version
# OR
pip install paddlepaddle-gpu  # GPU version

# Then install paddleocr
pip install paddleocr opencv-python
```

#### Issue: Audio libraries fail to install (librosa, soundfile)

**Windows**:
- Install Visual C++ Redistributable
- Try: `pip install soundfile --no-cache-dir`

**macOS**:
```bash
brew install libsndfile
pip install soundfile
```

**Linux**:
```bash
sudo apt-get install libsndfile1
pip install soundfile
```

### GPU/CUDA Issues

#### Issue: `torch.cuda.is_available()` returns `False`

**Solution**:
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall CUDA-enabled PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify CUDA version matches PyTorch CUDA version

#### Issue: "CUDA out of memory"

**Solutions**:
1. **Use smaller model**: Switch to CPU model option in notebook
2. **Reduce batch size**: Change `batch_size=1` in code
3. **Clear GPU memory**: Restart Jupyter kernel
4. **Close other programs**: Free up VRAM

#### Issue: "cudnn error" or "cublas error"

**Solution**:
Update GPU drivers from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)

### Model Download Issues

#### Issue: Model download is very slow

**Explanation**: First run downloads models (100MB-3GB depending on notebook). Subsequent runs use cached models.

**Solution**:
- Be patient on first run
- Check internet connection
- Models cache in: `~/.cache/huggingface/hub/`

#### Issue: "Access denied" or "401 Unauthorized"

**Solution**:
1. Model requires authentication - get HuggingFace token
2. Visit model page and accept terms of use
3. Login: `huggingface-cli login`

### Jupyter Issues

#### Issue: Jupyter kernel dies or crashes

**Solutions**:
1. **Out of memory**: Use smaller models or reduce batch size
2. **Corrupt cache**: `pip install --upgrade jupyter ipykernel`
3. **Restart**: Kernel → Restart Kernel in Jupyter

#### Issue: Jupyter shows wrong Python version

**Solution**:
Install kernel from venv:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install ipykernel
python -m ipykernel install --user --name=hf_tutorial
```

Then: Kernel → Change Kernel → "hf_tutorial"

### Platform-Specific Issues

#### macOS M1/M2 (Apple Silicon)

**Issue**: Some packages not compatible

**Solution**:
- PyTorch has native MPS (Metal Performance Shaders) support
- Use `device="mps"` instead of `device="cuda"` in some notebooks
- Some dependencies may require Rosetta 2

#### Windows Subsystem for Linux (WSL)

**Issue**: CUDA not working in WSL

**Solution**:
- Use WSL 2 (not WSL 1)
- Install NVIDIA drivers for WSL
- Follow [WSL CUDA setup guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

---

## Next Steps

### ✅ Setup Complete!

You're now ready to start learning. Here's your path forward:

**1. Start with the basics** (Notebooks 01-04):
- `01_nlp_text_generation.ipynb` - Generate text
- `02_nlp_text_classification.ipynb` - Sentiment analysis
- `03_nlp_text_summarization.ipynb` - Summarize documents
- `04_cv_image_classification.ipynb` - Classify images

**2. Explore specialized tasks** (Notebooks 05-09):
- `05_cv_object_detection.ipynb` - Detect objects in images
- `06_cv_ocr.ipynb` - Extract text from images
- `07_audio_speech_recognition.ipynb` - Transcribe audio
- `08_audio_text_to_speech.ipynb` - Generate speech
- `09_multimodal_image_to_text.ipynb` - Image captioning

**3. Advanced topics** (Notebooks 10-12):
- `10_ollama_integration.ipynb` - Run local LLMs
- `11_performance_caching_costs.ipynb` - Optimize for production
- `12_model_cards_responsible_ai.ipynb` - AI ethics and safety

**4. Agentic workflows** (Notebooks 14-17):
- `14_mcp_basics.ipynb` - Tool-using agents with MCP
- `15_mcp_servers.ipynb` - Build reusable MCP servers
- `16_multi_tool_agents.ipynb` - Advanced agent patterns
- `17_rag_local_llms.ipynb` - RAG with vector databases

### Additional Resources

- 📖 **README.md** - Project overview and model reference
- 📚 [HuggingFace Course](https://huggingface.co/course) - Free comprehensive course
- 💬 [HuggingFace Forums](https://discuss.huggingface.co/) - Community help
- 📘 [Transformers Docs](https://huggingface.co/docs/transformers) - Official documentation

---

## Getting Help

**Having issues?**

1. ✅ Check this troubleshooting guide
2. ✅ Read error messages carefully (they often tell you what's wrong)
3. ✅ Search [HuggingFace Forums](https://discuss.huggingface.co/)
4. ✅ Check [PyTorch Forums](https://discuss.pytorch.org/) for GPU/CUDA issues
5. ✅ Ask your instructor or TA

**Common commands for debugging**:

```bash
# Check Python version
python --version

# Check pip version
pip --version

# List installed packages
pip list

# Check CUDA
nvidia-smi

# Test PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Test transformers
python -c "import transformers; print(transformers.__version__)"
```

Happy learning! 🚀
