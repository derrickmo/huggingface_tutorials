# Core Rules — Always Loaded

These rules apply to **every notebook** without exception. They cover packages, datasets, self-containment, reproducibility, naming, and folder structure.

---

## 1. Allowed Packages

Only the following packages (and their submodules) may be imported:

### Core Packages (all modules)

| Package | Use |
|---------|-----|
| `numpy` | Numerical computation |
| `pandas` | Tabular data manipulation |
| `matplotlib` | Visualization (pyplot, patches, etc.) |
| `scikit-learn` | Classical ML algorithms, metrics, preprocessing, pipelines |
| `torch` | Deep learning (nn, optim, utils.data, autograd, etc.) |
| `torchvision` | CV datasets, transforms, pretrained models |
| `torchtext` | NLP datasets and text utilities |
| `torchaudio` | Audio datasets and transforms |

### Module-Restricted Exceptions

These packages may **only** be imported in the specific modules listed. All other notebooks must implement functionality from scratch using the core packages.

| Package | Allowed In | Use |
|---------|-----------|-----|
| `scipy` | Modules 1, 3, 4, 17 | Statistical tests, curve fitting, sparse matrices, optimization |
| `umap-learn` | Module 3 only | UMAP dimensionality reduction |
| `ultralytics` | Module 9 only | YOLO object detection |
| `mediapipe` | Module 9 only | Real-time vision — hand/pose/face |
| `faiss-cpu` | Modules 9, 18 | Approximate nearest neighbor search (HNSW, IVF) |
| `sentencepiece` | Module 7 only | SentencePiece tokenizer demonstration |
| `openai-whisper` | Module 12 only | STT pipeline demonstration |
| `unsloth` | Module 13 only | Efficient LLM fine-tuning |
| `gymnasium` | Module 14 only | RL environments (CartPole, LunarLander, GridWorld) |
| `mlflow` | Module 20 only | Experiment tracking and model registry |
| `gradio` | Module 20 only | Quick model demo UI |
| `fastapi` | Module 20 only | Production model serving |
| `pydantic`, `starlette`, `anyio`, `httpx` | Module 20 only | **Auto-installed as fastapi dependencies** — permitted wherever fastapi is permitted; validator may flag these as false-positive warnings, which can be safely ignored |
| `pytest` | Module 20 only | ML testing — both as a CLI tool (`pytest`) AND as an importable Python package (`import pytest`) |

### System Tools (not Python packages)

| Tool | Allowed In | Use |
|------|-----------|-----|
| `docker` | Module 20 only | Containerization for ML deployment — invoke via `subprocess.run()` with graceful fallback when Docker daemon is unavailable |
| `kubectl` | Module 20 only | Kubernetes CLI — invoke via `subprocess.run()` with graceful fallback when cluster is unavailable |

> **Note on ONNX:** `onnx` and `onnxruntime` are **NOT** in the allowed package list. Use `torch.jit.script` / `torch.jit.trace` for model export in all notebooks. The only exception is 05-10 (Complete MLP Pipeline) which mentions ONNX export conceptually — even there, use `torch.jit.script` for the actual export demonstration.

**Standard library** modules (`math`, `random`, `os`, `json`, `collections`, `typing`, `time`, `warnings`, `http`, `subprocess`, `re`, `struct`, `functools`, `itertools`, `pathlib`, `dataclasses`, `base64`, `statistics`, `abc`, `io`, `tempfile`, `shutil`, `uuid`, `unittest`, etc.) are always permitted.

---

## 2. Self-Contained Notebooks

- Each notebook **runs independently** from top to bottom with a single "Restart & Run All."
- Notebooks must work on both **local machines** and **Google Colab** without modification.
- No external API keys are required for any notebook. All data is downloaded programmatically.
- No cross-notebook imports, shared utility modules, or external helper files.
- If a helper function is needed in multiple notebooks, duplicate it with a brief comment: `# Helper: <description>`.
- Every notebook begins with its own import cell and ends with a summary/takeaway cell.

---

## 3. Datasets

All datasets must come from the **PyTorch ecosystem** or be synthetically generated:

- `torchvision.datasets` — MNIST, FashionMNIST, CIFAR-10, CIFAR-100, ImageNet (subset), STL10, SVHN, CelebA, VOCDetection, VOCSegmentation, CocoDetection, etc.
- `torchtext.datasets` — AG_NEWS, IMDB, WikiText2, WikiText103, SST2, CoNLL2000Chunking, etc.
- `torchaudio.datasets` — SPEECHCOMMANDS, YESNO, LIBRISPEECH, etc.
- `sklearn.datasets` — Permitted for **Modules 1–4** (classical ML), **Module 14** (RL environments), and **Module 19** (ML applications). Includes real datasets (e.g., Iris, California Housing, Digits) and synthetic generators.
- `sklearn.datasets` synthetic generators (`make_*` functions only) — Permitted in **any module** for quick demonstrations or toy examples.

Datasets are downloaded at runtime via their built-in download mechanisms. The notebook must set `download=True` and use a consistent `data/` directory relative to the repo root.

### Data Splits

- **Classical ML (Modules 1–4):** Use `sklearn.model_selection.train_test_split()` with `test_size=0.2, random_state=SEED`.
  Variable names: `X_train, X_test, y_train, y_test`. If validation is needed: apply `train_test_split` twice (first 80/20, then split the 80 into 80/20 again → 64/16/20). Variable names with validation: `X_train, X_val, X_test, y_train, y_val, y_test`.

- **Deep Learning (Modules 5+):** Use `torch.utils.data.random_split()` with a generator seeded by `SEED`.
  Split ratios: **80/10/10** (train/val/test). Variable names: `train_set, val_set, test_set`.

```python
# DL Standard Split
generator = torch.Generator().manual_seed(SEED)
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [0.8, 0.1, 0.1], generator=generator
)
```

- When a dataset provides **official splits** (e.g., CIFAR-10 train/test), use them and only split the official training set into train/val (**90/10**):

```python
# When official train/test splits exist
full_train_set = torchvision.datasets.CIFAR10(root="data/", train=True, ...)
test_set = torchvision.datasets.CIFAR10(root="data/", train=False, ...)

generator = torch.Generator().manual_seed(SEED)
train_size = int(0.9 * len(full_train_set))
val_size = len(full_train_set) - train_size
train_set, val_set = torch.utils.data.random_split(
    full_train_set, [train_size, val_size], generator=generator
)
```

---

## 4. Style — Complete Walkthroughs, Not Exercises

- Every notebook is a **complete, runnable walkthrough**. The reader follows along; they do not fill in blanks.
- All code cells are fully implemented. No `TODO`, `pass`, `# your code here`, or placeholder stubs.
- Markdown cells explain the *why* before the code shows the *how*.
- Use clear section headers (`## Part 0:`, `## Part 1:`, etc.) to structure the narrative per `RULES_STRUCTURE.md`.

---

## 5. No Redundancy — Concept Ownership

Each concept is taught **once**, in the module where it naturally belongs. Later notebooks may *use* a concept but must not re-teach it. A one-line comment referencing the original module is sufficient: `# See Module 5 — Backpropagation`.

The full concept ownership table is in each module's `README.md` file (at `docs/modules/module_XX/README.md`). When generating a notebook, check the target module's file for what this notebook owns and what it must only reference.

---

## 6. Reproducibility

Set **all** random seeds at the top of every notebook:
```python
import random, numpy as np, torch
SEED = 1103
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```
- Use `torch.backends.cudnn.deterministic = True` and `benchmark = False` when GPU reproducibility matters.
- Pin `generator` objects in `DataLoader` with `torch.Generator().manual_seed(SEED)` for shuffled loaders.

---

## 7. Markdown & Documentation

- Every notebook starts with a **title cell** (H1) containing: module number/name, one-sentence learning objective, prerequisites.
- Major sections use H2 (`## Part N: ...`), subsections use H3 (`###`).
- Each section opens with a short markdown explanation before any code.
- Every notebook ends with a **Key Takeaways** cell summarizing 3–5 bullet points.

---

## 8. Visualization Standards

- Use `matplotlib` for all plots.
- Always label axes and add titles.
- Use `plt.tight_layout()` or `constrained_layout=True` to prevent clipping.
- Prefer readable colormaps (`viridis`, `coolwarm`). Avoid `jet`.
- Keep figures to a reasonable size: `figsize=(8, 5)` as default, adjust as needed.

---

## 9. GPU Handling

All DL notebooks (Modules 5+) must include a device selection cell:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
- Models and tensors are moved to `device` explicitly.
- Notebooks must **work on CPU** even if GPU is available — never hard-code `"cuda"`.

---

## 10. Training Loops

All DL training (Modules 5+) must follow this standard structure. Classical ML (Modules 1–4) and exception categories (listed below) may deviate where needed.

### Standard Training Function

```python
def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_inputs, batch_targets in dataloader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_inputs.size(0)
        _, predicted = outputs.max(1)
        total += batch_targets.size(0)
        correct += predicted.eq(batch_targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
```

Adapt the body as needed (e.g., regression tasks omit accuracy), but keep the **function name, parameter list, and return type** consistent.

### Standard Evaluation Function

```python
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The neural network model.
        dataloader: Validation or test DataLoader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            running_loss += loss.item() * batch_inputs.size(0)
            _, predicted = outputs.max(1)
            total += batch_targets.size(0)
            correct += predicted.eq(batch_targets).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
```

### Epoch Output Format

Use this exact print format after each epoch:

```python
print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
      f"Val Acc: {val_acc:.2%} | Time: {elapsed:.1f}s")
```

- Losses: **4 decimal places** (`.4f`).
- Accuracy/percentages: **2 decimal places with % sign** (`.2%`).
- Timing: **1 decimal place** (`.1f`).
- For regression tasks, replace `Val Acc` with `Val MAE` or the task-appropriate metric.

### History Tracking

Store metrics in standard-named lists:

```python
train_losses: list[float] = []
val_losses: list[float] = []
train_accs: list[float] = []
val_accs: list[float] = []
```

Append after each epoch. For regression tasks, replace `_accs` lists with the task-appropriate metric (e.g., `train_maes`, `val_maes`).

### Best Model Tracking

Track and restore the best model based on validation loss:

```python
best_val_loss = float("inf")
best_model_state = None

# Inside training loop, after evaluating:
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_state = model.state_dict().copy()

# After training loop:
model.load_state_dict(best_model_state)
```

### Training Curve Plotting

After training completes, plot training curves:

```python
def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
) -> None:
    """Plot training and validation curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accs: Optional training accuracy per epoch.
        val_accs: Optional validation accuracy per epoch.
    """
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(12, 4))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    if train_accs and val_accs:
        axes[1].plot(train_accs, label="Train")
        axes[1].plot(val_accs, label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curve")
        axes[1].legend()

    plt.tight_layout()
    plt.show()
```

### Exceptions

The following categories may deviate from the standard training loop pattern:

| Category | Deviation | Reason |
|----------|-----------|--------|
| **GAN training** (11-03, 11-04, 11-05) | Dual `train_generator()` / `train_discriminator()` functions, separate optimizers | Two competing models with alternating updates |
| **Reinforcement Learning** (Module 14) | Environment step loop, no DataLoader | RL collects data via interaction, not from a static dataset |
| **Classical ML** (Modules 1–4) | No `train_one_epoch()`, no DataLoader | Uses `model.fit()` / NumPy loops |
| **Theory notebooks** (Category B) | May have lighter or no training | Focus is on mathematical proofs and empirical validation |
| **Evaluation notebooks** (Category C) | Training is not the focus | Focus is on evaluation metrics and methodology |
| **Tool notebooks** (Category D) | Follow the tool's conventions | External tool APIs dictate the training interface |
| **Optimizer topic** (5-09) | Multiple optimizers compared | Topic is specifically about comparing optimizers |

---

## 11. Folder Structure

```
project_root/
├── modules/
│   ├── module_01/
│   │   ├── 01-01_topic_name.ipynb
│   │   └── ...
│   ├── module_02/
│   │   └── ...
│   └── ... (20 module folders)
│
├── data/                            # Downloaded datasets (gitignored, shared)
├── docs/
│   ├── rules/                       # RULES_CORE, STRUCTURE, STYLE, QUALITY
│   ├── modules/                     # module_01/ through module_20/ (each with README.md)
│   └── templates/                   # 6 notebook templates (A–F)
├── scripts/                         # validate, execute, review, check_dataset_reuse
│
├── CLAUDE.md                        # Claude Code generation instructions
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview
```

- One folder per module: `modules/module_MM/`.
- The `data/` directory is shared across all modules and listed in `.gitignore`.

---

## 12. Notebook Naming Convention

```
MM-NN_short_description.ipynb
```
- `MM` = two-digit module number (01–20)
- `NN` = two-digit notebook number within the module (01–10)
- `short_description` = snake_case, max 5 words

---

## 13. Cell Ordering Convention

Every notebook must follow this cell ordering:

1. **Title cell** (markdown) — H1 with module name, objective, prerequisites
2. **Imports cell** (code) — all imports in a single cell
3. **Seed & device cell** (code) — reproducibility seeds + device selection
4. **Configuration cell** (code) — hyperparameters, paths, constants (all caps: `BATCH_SIZE`, `NUM_EPOCHS`)
5. **Content sections** — structured per `RULES_STRUCTURE.md`
6. **Key Takeaways cell** (markdown) — 3–5 bullet point summary

---

## 14. Output & Print Formatting

- Use f-strings for all formatted output. For training loops, follow the exact epoch print format in Section 10.
- Numeric precision: 4 decimal places for losses (`.4f`), 2 decimal places with % sign for accuracy (`.2%`), 1 for timing (`.1f`).
- Use tables (formatted strings or pandas DataFrames) for multi-model comparisons.
- Suppress unnecessary warnings at the top of the notebook:
  ```python
  import warnings
  warnings.filterwarnings("ignore")
  ```

---

## 15. Function & Class Naming

- **Functions:** `snake_case` — e.g., `train_one_epoch()`, `compute_loss()`
- **Classes:** `PascalCase` — e.g., `SimpleRNN`, `TransformerBlock`
- **Constants:** `ALL_CAPS` — e.g., `BATCH_SIZE`, `NUM_EPOCHS`
- **Variables:** `snake_case` — e.g., `train_loader`, `attention_weights`
- Avoid single-letter variables except in mathematical formulas (e.g., `x`, `y`, `W`, `b`).

---

## 16. Version & Environment Metadata

Every notebook's Part 0 imports cell must end with:
```python
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```
