# Quality Rules — Loaded During Review Pass

These rules are checked during the review/polish pass after initial generation.

---

## 1. Error Handling & Edge Cases

- Every training loop must handle `KeyboardInterrupt` gracefully — save the best model so far and print a message rather than crashing with a traceback.
- When loading data that may fail (downloads, file I/O), wrap in try/except with a clear error message telling the reader what went wrong and how to fix it.
- When demonstrating numerical instability (vanishing gradients, NaN loss), do so **deliberately** in a controlled cell, not accidentally. Print a warning markdown cell before the demonstration: `⚠️ The following cell intentionally triggers [issue] to demonstrate [concept].`
- Check tensor shapes at key transition points with assert statements: `assert x.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), f"Expected {(BATCH_SIZE, SEQ_LEN, D_MODEL)}, got {x.shape}"`.

---

## 2. Long-Running Cells

- Any cell expected to take more than **30 seconds** must include a progress indicator (tqdm progress bar or periodic print statements).
- Cells expected to take more than **5 minutes** must include a markdown warning before them: `⏱️ This cell takes approximately [X] minutes on [GPU type / CPU].`
- Provide an option to skip long-running cells by loading pre-computed results: `SKIP_LONG_TRAINING = False  # Set True to load saved results`.
- Never silently block the notebook for extended periods.

---

## 3. Checkpoint & Resume Pattern

For any training run lasting more than 5 epochs, implement basic checkpointing:
```python
if (epoch + 1) % SAVE_EVERY == 0:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, f"checkpoint_epoch_{epoch+1}.pt")
```
- In Part 0's configuration, include a `RESUME_FROM` variable (default `None`) that allows re-running from a checkpoint.
- This is required for Modules 8+ where training times become non-trivial.

---

## 4. Notebook Length Guidelines

- **Target length:** 800–1500 logic lines (excluding markdown, comments, and docstrings) per notebook.
- **Minimum:** 800 lines. Every topic should have enough depth — thorough implementations, rich demonstrations, and comprehensive comparisons.
- **Maximum:** 2000 lines. If a topic exceeds this, look for content that belongs in a different notebook.
- Markdown cells should constitute **35–55%** of the notebook's total cells. Generous use of transition blocks, concept explanations, and step-by-step commentary is encouraged.

### Transition & Explanation Blocks

Every notebook should include **transition markdown cells** between major code sections. These serve as the instructor's voice guiding the reader:

- **Before each new concept:** A markdown cell explaining *what* we're about to build and *why*.
- **Between Parts:** A markdown cell summarizing what was accomplished and previewing what comes next.
- **Before complex code:** A markdown cell with the mathematical formulation, algorithm pseudocode, or intuitive explanation.
- **After key results:** A markdown cell interpreting the output — what do the numbers mean? Is this good or bad? What should the reader notice?
- **Before comparisons:** A markdown cell framing what we're comparing and what to look for in the results.

These blocks should be substantive (2–5 sentences), not just section headers. They transform a code dump into a guided learning experience.

---

## 5. Review Checklist

Use this checklist when reviewing a generated notebook:

### Structure
- [ ] Part 0–5 headers all present
- [ ] Title cell has module name, objective, and prerequisites
- [ ] Part 0 includes EDA cell (shapes, class distribution, sample visualization)
- [ ] Part 5 has 3–5 Key Takeaways and "What's Next" link
- [ ] No forward references in Parts 0–4

### Code Quality
- [ ] All functions have Google-style docstrings with Args and Returns
- [ ] **All `__init__` methods have their own Google-style docstring** (this is the most commonly missed item — the validator checks it separately from regular methods)
- [ ] All functions have type hints for every parameter and return type
- [ ] No `TODO`, `pass`, or `# your code here`
- [ ] No single-letter variables outside math formulas
- [ ] PyTorch style conventions followed (channels-first, nn.Module pattern, explicit device)

### Correctness
- [ ] Notebook runs end-to-end with "Restart & Run All"
- [ ] Seeds set in first code cell
- [ ] Device cell present (Modules 5+)
- [ ] Only allowed packages imported for this module
- [ ] From-scratch implementation compared against library version
- [ ] Default optimizer (Adam) used unless topic is specifically about optimizers
- [ ] DataLoader uses standard variable names (`train_loader`, `val_loader`, `test_loader`)
- [ ] Training print format matches standard: `Epoch X/Y | Train Loss: ... | Val Loss: ...`
- [ ] History tracked in standard lists (`train_losses`, `val_losses`, `train_accs`, `val_accs`)
- [ ] Data split uses standard ratios (80/10/10 DL, 80/20 classical ML)

### Quality
- [ ] Markdown-to-code cell ratio in 35–55% range
- [ ] Code line count in 800–1500 range (800–2000 acceptable)
- [ ] Transition markdown cells between major code sections
- [ ] Comparison tables use pandas DataFrames with baseline row
- [ ] Error analysis cell present in Part 4
- [ ] Training curves plotted
- [ ] Long-running cells have progress indicators
- [ ] Checkpoint pattern used for training >5 epochs (Modules 8+)

### Polish
- [ ] No redundant explanations of concepts owned by other notebooks
- [ ] Shape assertions at key transition points
- [ ] KeyboardInterrupt handling in training loops
- [ ] Plots have labels, titles, and `tight_layout()`
- [ ] Version metadata printed in Part 0

### Known Validator False Positives (safe to ignore)
- `pydantic`, `starlette`, `anyio`, `httpx` warnings in Module 20 notebooks — these are auto-installed fastapi sub-packages and are permitted wherever fastapi is permitted
- `uuid` warnings — uuid is Python stdlib and is always permitted
- Validator may warn on any package not in its known-list; cross-check against RULES_CORE.md Section 1 before treating as a real error

### Module 20 Tool Notebook Notes
- Docker and kubectl are system tools invoked via `subprocess.run()` — always include `check_docker_available()` / `check_kubectl_available()` helpers with graceful fallback when the tool is not installed
- FastAPI notebooks must use `fastapi.testclient.TestClient` for all testing — never launch an actual server (it blocks the notebook)
- All statistical tests (z-test, t-test, KS test, chi-square) must be implemented from scratch using numpy/math — scipy is not allowed in Module 20
