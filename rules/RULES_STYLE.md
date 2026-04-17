# Style Rules — Always Loaded

Code quality, docstrings, type hints, PyTorch conventions, and mathematical notation.

---

## 1. Code Quality

- Prefer **clarity over cleverness**. A for-loop that is easy to read beats a one-liner that is not.
- Variable names must be descriptive: `learning_rate` not `lr`, `num_epochs` not `n`.
- Keep functions short and single-purpose.
- No global mutable state beyond the seed block.

---

## 2. Docstrings (Required)

Every function, class, and method **must** have a docstring. Use Google-style docstrings:

```python
def compute_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len, d_k).
        key: Key tensor of shape (batch_size, num_heads, seq_len, d_k).
        value: Value tensor of shape (batch_size, num_heads, seq_len, d_v).
        mask: Optional attention mask. True values are masked (ignored).

    Returns:
        Attention output of shape (batch_size, num_heads, seq_len, d_v).
    """
```

- **Functions:** Describe what the function does, all `Args`, and `Returns`.
- **Classes:** Describe the class purpose and list `Attributes` in the class-level docstring. Each method gets its own docstring.
- **Short helpers** (≤5 lines): A one-line docstring is acceptable: `"""Normalize tensor to unit length."""`
- Lambda functions and list comprehensions are exempt.

---

## 3. Type Hints (Required)

All function signatures **must** include type hints for every parameter and the return type:

```python
def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
```

- Use `torch.Tensor` (not `Tensor` or `np.ndarray` when the context is PyTorch).
- Use `np.ndarray` for NumPy arrays.
- Use `X | None` (union syntax) for optional parameters, not `Optional[X]`.
- For complex return types, use `tuple[float, float]`, `dict[str, float]`, `list[torch.Tensor]`.
- Class `__init__` methods return `None` (implicit — no return annotation needed).
- Type hint variables inline when the type is non-obvious: `predictions: list[str] = []`.

---

## 4. PyTorch Style Preference

When multiple implementation styles exist, **always choose the PyTorch convention**:

- **Data format:** Channels-first `(N, C, H, W)` — not channels-last `(N, H, W, C)`.
- **Module pattern:** Subclass `torch.nn.Module` with `__init__` + `forward` — not functional-only or Keras-style `Sequential` unless comparing.
- **Optimizer step order:** `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` — never vary this.
- **Dataset/DataLoader pattern:** Subclass `torch.utils.data.Dataset` with `__len__` + `__getitem__` — not generators or raw iteration.
- **Device handling:** Explicit `.to(device)` — not implicit placement.
- **Tensor creation:** `torch.tensor()`, `torch.zeros()`, `torch.randn()` — not numpy-then-convert unless demonstrating the conversion.
- **Loss functions:** Use `torch.nn` loss classes (e.g., `nn.CrossEntropyLoss()`) in training loops — only use raw implementations when the topic is "from scratch" (see "From Scratch" Definition below).
- **Random sampling:** `torch.randint`, `torch.randperm` — not `random.choice` or `np.random` for tensor operations.
- **Saving/loading:** `torch.save` / `torch.load` with `state_dict` — not pickle.
- When comparing our from-scratch implementation against a library, prefer `torch.nn` over `sklearn` equivalents for DL topics (Modules 5+).
- **Default optimizer:** Use `torch.optim.Adam` with `lr=LEARNING_RATE` (where `LEARNING_RATE = 1e-3` in the config cell). Only deviate when the topic is specifically about optimizers (5-09), when a different optimizer is pedagogically justified (e.g., SGD for demonstrating momentum effects), or when the architecture convention requires it (e.g., AdamW for Transformers in Modules 8+).
- **Default loss function:** Use `nn.CrossEntropyLoss()` for classification and `nn.MSELoss()` for regression. Only use raw/from-scratch loss implementations when the topic is specifically about implementing loss functions (5-04, 5-05). For special architectures (e.g., GANs use `nn.BCEWithLogitsLoss()`), use the domain-standard loss.
- **Standard DataLoader creation:** Use this pattern for all DL DataLoaders (Modules 5+):

```python
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=torch.cuda.is_available(),
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available(),
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=torch.cuda.is_available(),
)
```

  Standard variable names: `train_loader`, `val_loader`, `test_loader`. Use `num_workers=0` for Colab compatibility. Use `shuffle=True` only for the training loader.

---

## 5. "From Scratch" Definition

When a topic says "from scratch," this means:

- The core algorithm or component is implemented using **only NumPy, raw PyTorch tensor operations, or Python** — not wrapped library calls.
- Using `torch.nn.functional` for basic ops (like `F.relu`, `F.softmax`) is acceptable. Using `nn.Linear`, `nn.Conv2d`, or other `nn.Module` subclasses is **not** acceptable when building from scratch.
- After the from-scratch implementation, comparing against the library version (e.g., `nn.Transformer`, `sklearn.LinearRegression`) in Part 3 or Part 4 is **required** where feasible — it validates correctness and shows the reader what the library abstracts.
- Topics that are explicitly library-focused (Category D in RULES_STRUCTURE.md) are exempt from from-scratch requirements.

---

## 6. Mathematical Notation Consistency

Use LaTeX in markdown cells for all mathematical expressions:

- Vectors are **bold lowercase**: $\mathbf{x}$, $\mathbf{w}$, $\mathbf{h}$.
- Matrices are **bold uppercase**: $\mathbf{W}$, $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$.
- Scalars are italic: $\alpha$, $\lambda$, $n$, $d$.
- Loss functions: $\mathcal{L}$ (script L).
- Expectations: $\mathbb{E}[\cdot]$.
- Sets: $\mathcal{X}$, $\mathcal{D}$.
- Gradient: $\nabla_{\theta} \mathcal{L}$.
- Use consistent subscript conventions: $\theta_t$ for parameters at step $t$, $x_i$ for the $i$-th sample, $h_l$ for hidden state at layer $l$.

---

## 7. Comparison & Benchmark Tables

When a notebook compares multiple approaches, models, or configurations:

- Present results in a **pandas DataFrame** displayed as a table, not as scattered print statements.
- Include columns for: model/approach name, key hyperparameter(s), metric(s), training time, and parameter count (where relevant).
- Sort by the primary metric.
- Bold or highlight the best result.
- Always include a **baseline row** (e.g., random chance, majority class, or the simplest model).

```python
results_df = pd.DataFrame({
    "Model": ["Random Baseline", "Logistic Regression", "Random Forest", "Gradient Boosting"],
    "Accuracy": [0.50, 0.83, 0.89, 0.91],
    "F1 (Macro)": [0.33, 0.82, 0.88, 0.90],
    "Train Time (s)": [0.0, 0.3, 2.1, 5.4],
})
results_df.style.highlight_max(subset=["Accuracy", "F1 (Macro)"])
```
