# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **educational HuggingFace tutorial** repository designed to introduce students to the HuggingFace ecosystem through 36 hands-on Jupyter notebooks organized in 7 sections. The project is optimized for students with limited hardware resources while providing GPU-optimized alternatives for those with access to better equipment (RTX 4080 or similar).

**Target Audience**: Students learning machine learning with varying hardware capabilities (CPU-only to consumer-grade GPUs)

**Key Design Principle**: Each notebook offers dual model options:
- Small models that run on CPU (4-8GB RAM)
- Medium models optimized for RTX 4080-class GPUs (6-12GB VRAM)

## Project Architecture

### Dual Interface Design

The project provides **two ways to use HuggingFace models**:

1. **Jupyter Notebooks** (`notebooks/`): Interactive, educational, step-by-step learning
   - Best for understanding concepts
   - Includes explanations, examples, and exercises
   - Organized by domain (NLP, CV, Audio, Multimodal, Best Practices, Agentic)

2. **CLI Tools** (`functions/`): Standalone Python scripts for direct command-line usage
   - Fully self-contained with `if __name__ == "__main__"`
   - Use argparse for parameter handling
   - Include example outputs in docstrings (visible with `--help`)
   - Organized by domain (nlp/, cv/, audio/, multimodal/, agentic/)
   - **12 total CLI tools** matching core notebook functionality

### Notebook Organization

```
notebooks/
├── 00_fundamentals/           # Transformer Fundamentals (5 notebooks)
│   ├── 00_01: Tokenization & Embeddings
│   ├── 00_02: Transformer Architecture
│   ├── 00_03: HuggingFace Ecosystem Tour
│   ├── 00_04: Preprocessors & Feature Extractors
│   └── 00_05: Model Configuration & Customization
│
├── 01_nlp/                    # Natural Language Processing (8 notebooks)
│   ├── 01_01-01_03: Core NLP (generation, classification, summarization)
│   ├── 01_04-01_06: NER, Question Answering, Translation
│   └── 01_07-01_08: Fine-tuning (Unsloth GPU-optimized, LoRA CPU-compatible)
│
├── 02_computer_vision/        # Computer Vision (4 notebooks)
│   ├── 02_01-02_03: Classification, Object Detection, OCR
│   └── 02_04: Image Segmentation
│
├── 03_audio/                  # Audio Processing (3 notebooks)
│   ├── 03_01-03_02: Speech Recognition, Text-to-Speech
│   └── 03_03: Audio Classification
│
├── 04_multimodal/             # Multimodal AI (5 notebooks)
│   ├── 04_01: Image-to-Text (Captioning)
│   ├── 04_02: Visual Question Answering
│   ├── 04_03: Text-to-Image Generation
│   ├── 04_04: Image Editing & Inpainting
│   └── 04_05: Document Understanding
│
├── 05_best_practices/         # Best Practices & Production (8 notebooks)
│   ├── 05_01: Ollama integration (local LLM deployment)
│   ├── 05_02: Performance & caching
│   ├── 05_03: Model cards & responsible AI
│   ├── 05_04: HuggingFace Datasets
│   ├── 05_05: Gradio & HuggingFace Spaces
│   ├── 05_06: Quantization & compression (INT8, ONNX)
│   ├── 05_07: Quantization Deep Dive (GPTQ, AWQ, bitsandbytes)
│   └── 05_08: Training Best Practices (Trainer API, scheduling)
│
└── 06_agentic_workflows/      # Agentic AI with MCP (5 notebooks)
    ├── 06_01: MCP Basics (tool-using agents)
    ├── 06_02: MCP Servers (reusable tool servers)
    ├── 06_03: Multi-Tool Agents (ReAct, Plan-and-Execute, Reflection)
    ├── 06_04: RAG with Local LLMs (vector databases, semantic search)
    └── 06_05: Structured Output & Function Calling
```

### CLI Tools Organization

```
functions/
├── nlp/             # 3 tools: text_generation, text_classification, text_summarization
├── cv/              # 3 tools: image_classification, object_detection, ocr
├── audio/           # 2 tools: speech_recognition, text_to_speech
├── multimodal/      # 1 tool: image_captioning
└── agentic/         # 3 tools: mcp_agent, mcp_server, multi_agent
```

---

## Rules & Conventions

Detailed rules are in the `rules/` folder. The sections below summarize the conventions adapted for this project. When in doubt, the rules files are authoritative.

---

### 1. Self-Contained Notebooks

- Each notebook **runs independently** from top to bottom with a single "Restart & Run All."
- No cross-notebook imports or shared utility modules required at runtime.
- If a helper function is needed in multiple notebooks, duplicate it with a brief comment: `# Helper: <description>`.
- Every notebook begins with its own import cell and ends with a Key Takeaways cell.
- Notebooks must work on **local machines**. Provide fallbacks when optional dependencies (Ollama, PaddleOCR, etc.) are unavailable.

---

### 2. Complete Walkthroughs, Not Exercises

- Every notebook is a **complete, runnable walkthrough**. The reader follows along; they do not fill in blanks.
- All code cells are fully implemented. No `TODO`, `pass`, `# your code here`, or placeholder stubs.
- Markdown cells explain the *why* before the code shows the *how*.
- Exercises sections at the end provide optional challenges but never block the notebook from running.

---

### 3. Notebook Structure Template

**Core notebooks (01-10)** follow this section order:

1. **Title & Learning Objectives** -- H1 with clear educational goals
2. **Prerequisites Table** -- Hardware requirements showing CPU and GPU model options side-by-side
3. **Expected Behaviors** -- What students should observe when running
4. **Overview** -- Conceptual explanation of the task
5. **Setup and Installation** -- Imports, version checks, device setup
6. **Model Selection** -- Commented code showing both CPU and GPU model options
7. **Method 1: Pipeline API** -- Simplest approach for beginners
8. **Method 2: Manual Model Loading** -- Advanced approach with more control
9. **Practical Applications** -- 2-3 real-world examples
10. **Performance Benchmarking** -- Timing and device information
11. **Exercises** -- Optional hands-on challenges
12. **Key Takeaways** -- 3-5 bullet-point summary
13. **Next Steps & Resources** -- Links to documentation and next notebook

**Specialized notebooks** may adapt this template:
- **Notebooks 04-05 NLP (Fine-tuning)**: Training-focused structure
- **Notebooks 10-15 (Best Practices)**: Integration/concept-focused
- **Notebooks 16-19 (Agentic)**: Tool-building structure

### Cell Ordering Convention

Every notebook must follow this cell ordering:

1. **Title cell** (markdown) -- H1 with notebook name, one-sentence objective, prerequisites
2. **Imports cell** (code) -- All imports in a single cell
3. **Device & seed cell** (code) -- Device selection + reproducibility seeds
4. **Configuration cell** (code) -- Model names, hyperparameters, paths (all caps: `MODEL_NAME`, `BATCH_SIZE`)
5. **Content sections** -- Structured per the template above
6. **Key Takeaways cell** (markdown) -- 3-5 bullet point summary

---

### 4. Reproducibility

Set random seeds at the top of every notebook that uses randomness:

```python
import random, numpy as np, torch
SEED = 1103
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

---

### 5. GPU Handling

All notebooks that load models must include a device selection cell:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- Models and tensors are moved to `device` explicitly.
- Notebooks must **work on CPU** even if GPU is available -- never hard-code `"cuda"`.

---

### 6. Model Selection Pattern (2+1 Rule)

All notebooks follow a standardized model pattern:

- **Small Model**: CPU-friendly for inference, trainable on consumer GPU
- **Large Model**: GPU-optimized, inference-only for most hardware
- **SOTA Model**: Only if significantly different from Large model (optional)

```python
# Option 1: Small Model (CPU-friendly, recommended for beginners)
MODEL_NAME = "small-model-name"  # Size in MB, description

# Option 2: Large Model/SOTA (GPU-optimized, production-grade)
# MODEL_NAME = "larger-model-name"  # Size in MB, description
```

Students uncomment the GPU line if they have appropriate hardware.

---

### 7. Code Quality

#### Naming Conventions

- **Functions:** `snake_case` -- e.g., `load_model()`, `compute_score()`
- **Classes:** `PascalCase` -- e.g., `TextGenerator`, `ImageProcessor`
- **Constants:** `ALL_CAPS` -- e.g., `MODEL_NAME`, `BATCH_SIZE`, `MAX_LENGTH`
- **Variables:** `snake_case` -- e.g., `input_text`, `attention_weights`
- Avoid single-letter variables except in mathematical formulas.

#### Style

- Prefer **clarity over cleverness**. A for-loop that is easy to read beats a one-liner that is not.
- Variable names must be descriptive: `learning_rate` not `lr`, `num_epochs` not `n`.
- Keep functions short and single-purpose.
- No global mutable state beyond the seed/config block.

#### Docstrings (Required)

Every function, class, and method must have a docstring. Use Google-style:

```python
def classify_text(
    text: str,
    model_name: str = "distilbert-base-uncased",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Classify text sentiment using a pretrained model.

    Args:
        text: Input text to classify.
        model_name: HuggingFace model identifier.
        threshold: Minimum confidence threshold.

    Returns:
        Dictionary mapping labels to confidence scores.
    """
```

- **Functions:** Describe what it does, all `Args`, and `Returns`.
- **Classes:** Describe purpose and list `Attributes`. Each method gets its own docstring.
- **Short helpers** (<=5 lines): A one-line docstring is acceptable.

#### Type Hints (Required)

All function signatures must include type hints for every parameter and the return type:

```python
def generate_text(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
) -> str:
    """Generate text continuation from a prompt."""
```

- Use `X | None` for optional parameters, not `Optional[X]`.
- For complex return types, use `tuple[float, float]`, `dict[str, float]`, `list[str]`.
- Type hint variables inline when the type is non-obvious: `predictions: list[str] = []`.

---

### 8. Visualization Standards

- Use `matplotlib` as the primary plotting library.
- Always label axes and add titles.
- Use `plt.tight_layout()` or `constrained_layout=True` to prevent clipping.
- Prefer readable colormaps (`viridis`, `coolwarm`). Avoid `jet`.
- Default figure size: `figsize=(8, 5)`, adjust as needed.
- Use `seaborn` for statistical visualizations and color palettes.
- `plotly` for interactive plots (optional).

---

### 9. Output & Print Formatting

- Use f-strings for all formatted output.
- Numeric precision: 4 decimal places for losses (`.4f`), 2 decimal places with % sign for accuracy (`.2%`), 1 for timing (`.1f`).
- Use tables (formatted strings or pandas DataFrames) for multi-model comparisons.
- When comparing multiple approaches, present results in a **pandas DataFrame**, not scattered print statements. Include a baseline row.
- Suppress unnecessary warnings at the top of the notebook:
  ```python
  import warnings
  warnings.filterwarnings("ignore")
  ```

---

### 10. Version & Environment Metadata

Every notebook's setup cell must end with:

```python
import sys
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### 11. Error Handling & Edge Cases

- When loading data or models that may fail (downloads, file I/O), wrap in try/except with a clear error message telling the reader what went wrong and how to fix it.
- Check for optional dependencies gracefully:
  ```python
  try:
      import ollama
      OLLAMA_AVAILABLE = True
  except ImportError:
      OLLAMA_AVAILABLE = False
      print("Ollama not installed. Install with: pip install ollama")
  ```

---

### 12. Long-Running Cells

- Any cell expected to take more than **30 seconds** must include a progress indicator (`tqdm` or periodic print statements).
- Cells expected to take more than **5 minutes** must include a markdown warning before them.
- Never silently block the notebook for extended periods.

---

### 13. Markdown & Documentation

- Every notebook starts with a **title cell** (H1) containing: notebook topic, one-sentence learning objective, prerequisites.
- Major sections use H2 (`##`), subsections use H3 (`###`).
- Each section opens with a short markdown explanation before any code.
- Every notebook ends with a **Key Takeaways** cell summarizing 3-5 bullet points.
- **Transition markdown cells** between major code sections are encouraged -- these serve as the instructor's voice guiding the reader. They should be substantive (2-5 sentences), not just section headers.
- Markdown-to-code cell ratio target: **35-55%** of total cells should be markdown.

---

### 14. Mathematical Notation

Use LaTeX in markdown cells for mathematical expressions:

- Vectors: bold lowercase $\mathbf{x}$, $\mathbf{w}$
- Matrices: bold uppercase $\mathbf{W}$, $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$
- Scalars: italic $\alpha$, $\lambda$, $n$
- Loss functions: $\mathcal{L}$
- Gradient: $\nabla_{\theta} \mathcal{L}$

---

## Development Commands

### Environment Setup

```bash
# Quick setup (see getting_started.md for details)
python -m venv venv
source venv/bin/activate  # macOS/Linux or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Running Notebooks

```bash
jupyter notebook   # Classic interface
jupyter lab        # Modern interface
```

### Running CLI Tools

```bash
# General pattern
python functions/<domain>/<tool>.py [arguments]

# Examples
python functions/nlp/text_generation.py "Once upon a time"
python functions/nlp/text_classification.py "I love this!"
python functions/cv/image_classification.py photo.jpg
python functions/audio/speech_recognition.py audio.wav

# Get help for any tool
python functions/nlp/text_generation.py --help
```

### HuggingFace Authentication

Quick reference: `huggingface-cli login` (paste token from `hf_credential` file). See `getting_started.md` for details.

### Testing Notebooks

```bash
# Run specific notebook programmatically
jupyter nbconvert --execute --to notebook notebooks/01_nlp/01_nlp_text_generation.ipynb
```

### Ollama Setup (for Notebooks 05_01, 06_01-06_05)

```bash
# Install Ollama from ollama.com, then:
ollama pull llama3.2:1b    # Small model, CPU-friendly
ollama pull llama3.1:8b    # Large model, GPU-optimized
```

---

## Important Files

| File | Purpose |
|------|---------|
| `SYLLABUS.md` | Authoritative 36-notebook syllabus with section/notebook listing. |
| `hf_credential` | HuggingFace access token. **Never commit to public repos.** |
| `sample_data/` | Test images and audio files. Notebooks use URLs as fallback. |
| `notebooks/shared_utils.py` | Common helpers (e.g., `load_image_from_url`, `setup_device`). |
| `rules/` | Code quality, structure, and style rules (see below). |
| `requirements.txt` | Python dependencies organized by category. |

### Rules Files

| File | Scope |
|------|-------|
| `rules/RULES_CORE.md` | Foundational conventions: reproducibility, self-containment, naming |
| `rules/RULES_STRUCTURE.md` | Notebook structure template and section ordering |
| `rules/RULES_STYLE.md` | Code quality: docstrings, type hints, visualization, math notation |
| `rules/RULES_QUALITY.md` | Review checklist: error handling, long-running cells, polish |

---

## Model Caching

HuggingFace models are automatically cached in `~/.cache/huggingface/hub/`.

- First run of each notebook downloads models (can be slow)
- Subsequent runs use cached models (much faster)
- Cache can grow to 10-20GB with all notebooks
- Students on shared systems may benefit from `HF_HOME` environment variable

## Model Size Reference

**Tiny (<100MB)**: whisper-tiny (72MB), distilgpt2 (82MB)
**Small (100-500MB)**: distilbert (268MB), vit-base (346MB), detr-resnet-50 (159MB), gpt2 (500MB)
**Medium (500MB-1.5GB)**: trocr-small (558MB), blip-base (990MB), distilbart (1.2GB), gpt2-medium (1.5GB)
**Large (1.5GB+)**: bart-large (1.6GB), blip-large (1.9GB)

**Total Storage**: Plan for 20-30GB total cache storage across all 36 notebooks.

---

## Adding New Notebooks

When creating additional notebooks:

1. **Follow the template structure** and cell ordering convention above
2. **Include Prerequisites table** with both CPU and GPU options
3. **Provide dual model options** with clear size and hardware requirements
4. **Add docstrings and type hints** to all functions
5. **Set seeds** for reproducibility
6. **Print version metadata** in the setup cell
7. **Add to README.md table** with model sizes and hardware specs
8. **Test on CPU** at minimum; test on GPU if available
9. **Include transition markdown cells** between major sections

---

## Validation Checklist

Use this when reviewing a notebook:

### Structure
- [ ] Title cell has notebook name, objective, and prerequisites
- [ ] Imports in a single cell at the top
- [ ] Device selection cell present
- [ ] Seeds set for reproducibility
- [ ] Version metadata printed
- [ ] Key Takeaways at the end (3-5 bullets)

### Code Quality
- [ ] All functions have Google-style docstrings
- [ ] All functions have type hints for parameters and return type
- [ ] No `TODO`, `pass`, or `# your code here`
- [ ] Descriptive variable names (no single-letter except math)
- [ ] Constants in ALL_CAPS

### Correctness
- [ ] Notebook runs end-to-end with "Restart & Run All"
- [ ] Works on CPU (never hard-codes `"cuda"`)
- [ ] Model loads successfully
- [ ] Outputs are reasonable

### Quality
- [ ] Markdown-to-code ratio in 35-55% range
- [ ] Transition markdown cells between major sections
- [ ] Plots have labels, titles, and `tight_layout()`
- [ ] Long-running cells have progress indicators
- [ ] Error handling for downloads and optional dependencies
- [ ] Comparison tables use pandas DataFrames with baseline row

---

## Common Student Issues

### Memory Issues
- Use the CPU (smaller) model option
- Reduce batch size in code examples
- Restart kernel between notebooks

### Download Failures
- Models download from HuggingFace Hub; requires internet
- Corporate firewalls may block; suggest mobile hotspot or VPN
- Some models require authentication via `hf_credential`

### Missing Dependencies
- Audio notebooks need `soundfile` and `librosa`
- Vision notebooks need `Pillow`
- Advanced notebooks need `matplotlib`, `seaborn`, `plotly`
- Fine-tuning notebooks need `peft` and `trl`
- Notebook 11 needs `psutil` for memory profiling
- Some students may need `ffmpeg` for audio
- Windows users may need Visual C++ redistributables

### Ollama Integration (Notebooks 05_01, 06_01-06_05)
- Requires separate Ollama installation from ollama.com
- Models must be pulled before running (`ollama pull llama3.2:1b`)
- Ollama daemon must be running in background

### Best Practices Notebooks (05_04-05_08)
- Notebook 05_04 needs `datasets` library (usually already installed)
- Notebook 05_05 needs `gradio` (`pip install gradio`)
- Notebook 05_06 optionally needs `onnx`, `onnxruntime`, `optimum` for ONNX sections
- Notebook 05_07 optionally needs `bitsandbytes`, `auto-gptq`, `autoawq` (CUDA GPU + Linux)
- Notebook 05_08 needs `datasets` and `evaluate` libraries

### Multimodal Notebooks (04_02-04_05)
- Notebook 04_03/04_04 need `diffusers` library (GPU recommended)
- Notebook 04_05 needs `Pillow` for synthetic document generation

### Agentic Notebooks (06_01-06_05)
- Notebook 06_05 needs `pydantic` for schema validation
- Notebook 06_05 optionally uses `instructor` library
