# Notebook Structure Rules — Always Loaded

These rules define the Part 0–5 structure every notebook follows and category-specific adaptations.

---

## 1. Notebook Structure Template

Every notebook follows a **6-part structure**. This ensures a consistent learning experience across all 200 topics. Parts may be adapted based on topic category (see Section 2 below), but the overall flow must remain intact.

### Part 0 — Setup & Prerequisites

- Brief markdown overview of **what this notebook covers** and **why it matters** (2–4 sentences).
- Explicitly list which prior notebooks the reader should have completed. Use format: `Prerequisites: 5-06 (Backpropagation), 5-07 (PyTorch Autograd)`.
- All imports, seeds, device setup, and configuration constants.
- Load or generate the dataset. Include a brief EDA cell (shapes, class distribution, sample visualization) so the reader understands the data before any modeling.

```markdown
## Part 0 — Setup & Prerequisites

This notebook covers [topic]. We will [brief description of what we build].

**Prerequisites:** [list of MM-NN notebook references]
```

### Part 1 — Concept Introduction & From-Scratch Implementation

- **Markdown-first:** Explain the core concept(s) with mathematical notation and intuition. Use LaTeX in markdown for equations.
- **Build the key algorithm or component from scratch** using only NumPy or raw PyTorch tensor operations (no high-level library calls for the concept being taught).
- Show intermediate outputs, shapes, and visualizations at each step — the reader should see the data transform at every stage.
- If the topic involves a well-known algorithm, implement it step-by-step and verify against a known result or toy example.
- **Insert transition markdown cells** between each function/class definition explaining what it does, why it's needed, and how it connects to the previous piece. These cells are the instructor's narration — they turn code into a guided walkthrough.

```markdown
## Part 1 — [Concept Name] from Scratch
```

### Part 2 — Assembly & Integration

- Combine the individual functions/components from Part 1 into a **complete, reusable module** (a class or a pipeline function).
- For DL topics: assemble into an `nn.Module` with a clean `forward()` method.
- For classical ML topics: assemble into a class with `fit()` and `predict()` methods.
- For theory topics: assemble the proof or derivation into a runnable experiment that validates the theory empirically.
- Run a **sanity check** on toy data to confirm the assembled module works end-to-end.

```markdown
## Part 2 — Putting It All Together
```

### Part 3 — Training & Application

- **Connect the assembled module to a real or realistic dataset.**
- For DL topics: write the full training loop (forward → loss → backward → optimizer step), using the `device` variable, with epoch-level logging.
- For classical ML topics: fit the model, make predictions, and compare against baselines.
- For theory topics: run the complete experiment (e.g., measure VC dimension empirically, plot learning curves across model complexities).
- For evaluation/pipeline topics: apply the pipeline to a dataset and produce results.
- Include a **comparison cell** where relevant — compare the from-scratch implementation against a library equivalent (e.g., sklearn, pretrained torchvision model) to verify correctness and note speed differences.

```markdown
## Part 3 — Training on [Dataset Name]
```

### Part 4 — Evaluation & Analysis

- Compute and display **quantitative metrics** appropriate to the task (accuracy, F1, perplexity, FID, mAP, etc.).
- Generate **diagnostic visualizations**: confusion matrices, loss curves, attention heatmaps, embedding projections, gradient norms, or whatever is appropriate.
- Include an **error analysis cell** — examine failure cases, identify patterns, and hypothesize why the model fails on specific examples.
- Where applicable, run an **ablation study** — change one hyperparameter or component and measure the impact.

```markdown
## Part 4 — Evaluation & Analysis
```

### Part 5 — Summary & Lessons Learned

- 3–5 bullet-point **Key Takeaways** summarizing the most important insights.
- A **"What's Next"** sentence linking to the next notebook(s) in the sequence: `Next: 5-07 (PyTorch Autograd) builds on the backprop mechanics we implemented here.`
- Optional: a **"Going Further"** cell with pointers to papers, advanced techniques, or related topics not covered.

```markdown
## Part 5 — Summary & Lessons Learned

### Key Takeaways
- ...

### What's Next
→ [MM-NN Topic Name] builds on [concept] we learned here.
```

---

## 2. Category-Specific Structure Guidance

Not all 200 topics fit the same Part 0–5 template identically. The following guidance adapts the template to different topic categories while preserving the overall flow.

### A. Algorithm Notebooks (Modules 2, 3, 5, 6, 7, 8, 11, 14)

The standard template applies directly:
- Part 1: Implement the algorithm from scratch (NumPy or raw tensors).
- Part 2: Wrap into a reusable class (`nn.Module` or sklearn-style API).
- Part 3: Train on a real dataset.
- Part 4: Evaluate with metrics + error analysis.

### B. Theory Notebooks (4-06 through 4-10, 10-09, 17-02)

Theory topics must still be **code-driven**, not lecture notes.
- Part 1: State the theorem/concept mathematically, then **implement it** (e.g., compute VC dimension for a set of classifiers, implement PAC bounds, build a GP from scratch).
- Part 2: Build a complete experiment that **empirically validates** the theoretical claim (e.g., plot sample complexity vs generalization gap, show calibration improvement before/after temperature scaling).
- Part 3: Apply to a real dataset — the theory should produce measurable, visible results.
- Part 4: Compare empirical findings against theoretical predictions. Show where theory matches practice and where it breaks down.

### C. Evaluation & Pipeline Notebooks (4-01 through 4-05, 10-08, 11-07, 12-10, 17-10, 18-05, 18-09, 20-01, 20-07, 20-09)

- Part 1: Implement evaluation metrics or pipeline components from scratch.
- Part 2: Assemble into a complete evaluation or pipeline harness.
- Part 3: **Apply to a pre-trained model's outputs** (load predictions or run a small model) rather than training from scratch — the focus is the evaluation methodology, not model training.
- Part 4: Analyze results, compare metrics, identify failure modes.

### D. Tool/Library Notebooks (9-03 YOLO, 9-04 MediaPipe, 9-09 OCR, 13-10 Unsloth, 20-02 MLflow, 20-04 FastAPI, 20-05 Docker, 20-06 Kubernetes, 20-08 ML Testing)

These notebooks teach external tool usage, not from-scratch implementation.
- Part 1: Explain what the tool does and why it exists (architecture overview, not implementation).
- Part 2: Walk through the tool's API with progressively complex examples.
- Part 3: Build a complete, realistic pipeline using the tool.
- Part 4: Benchmark, profile, or evaluate the tool's output.

### E. Capstone/Integration Notebooks (9-10, 15-10, 16-10, 18-10, 20-10)

These notebooks tie together multiple prior concepts.
- Part 1: Brief recap of the concepts being integrated (1–2 sentences each with references to prior notebooks — do not re-teach).
- Part 2: Design the integrated system/experiment architecture.
- Part 3: Build and run the full system end-to-end.
- Part 4: Comprehensive evaluation comparing against baselines established in prior notebooks.

### F. Comparison/Architecture Notebooks (2-10, 10-03, 12-07, 13-04, 17-01)

Use `TEMPLATE_COMPARISON.ipynb`. When the topic is a comparison or survey of approaches:
- Part 1: Implement or load 2–3 representative approaches.
- Part 2: Run all approaches on the **same task and dataset** for fair comparison.
- Part 3: This becomes the comparison — a side-by-side results table or visualization.
- Part 4: Analysis of tradeoffs (parameter count, speed, quality, memory).

---

## 3. Cross-Notebook Reference Conventions

When a notebook uses a concept taught in an earlier notebook:

- **Do not re-teach the concept.** Instead, provide a one-line comment: `# Backpropagation — see 5-06 for full derivation.`
- **Do not copy-paste implementation code from prior notebooks.** If you need the same function, duplicate it with a brief helper comment: `# Helper: same as 5-06 but adapted for [reason].`
- **If referencing a prior notebook's results:** state what was established (e.g., "In 6-03 we showed ResNet achieves 93.2% on CIFAR-10"), don't re-run the experiment.
- **Forward references** are allowed in Part 5's "What's Next" section but nowhere else. Never reference a concept that hasn't been taught yet in Parts 0–4.
- **Recommended prerequisites** (marked with `| Recommended:` in the module file) are from later modules that deepen understanding but are not required to run the notebook. When a recommended prerequisite is not met, the notebook must still be fully functional — include a brief inline explanation of the concept rather than depending on it.

---

## 4. Dataset Reuse & Consistency

Reuse datasets aggressively across modules. Prefer the standard datasets below before introducing new ones.

When the same dataset appears in multiple notebooks, use identical preprocessing (same splits, same transforms, same seed) unless the notebook's topic specifically requires a different approach. If a notebook changes the standard preprocessing, explain why in a markdown cell.

### Preferred Datasets

| Dataset | Typical Use | Modules |
|---------|------------|---------|
| CIFAR-10 | Image classification, CNN training | 6, 9, 11, 12 |
| CIFAR-100 | Advanced image classification | 9 (9-10 deep dive) |
| MNIST / FashionMNIST | Quick demos, autoencoder, GAN | 3, 5, 6, 11 |
| AG_NEWS | Text classification | 7, 10 |
| WikiText-2 | Language modeling | 7, 8, 10, 17 |
| SST-2 | Sentiment classification | 10, 13 |
| SPEECHCOMMANDS | Audio classification | 12, 19 |
| `sklearn.make_moons` / `make_blobs` | Toy demos | 2, 3, 4, 5, 14 |
| `sklearn.California Housing` | Tabular regression | 2, 4, 19 |
| GridWorld (custom) | RL environments | 14 |
| CartPole / LunarLander (gymnasium) | Deep RL | 14 |

Only introduce a new dataset when none of the preferred datasets fit the task (e.g., CoNLL for NER, SQuAD for QA, VOC for detection). When introducing a new dataset, include a brief EDA cell in Part 0.

---

## 5. Difficulty Progression Within Modules

- Topics within a module are ordered from foundational to advanced. Topic NN-01 should always be more accessible than NN-10.
- The first topic (NN-01) should be approachable even if the reader only completed the listed prerequisites.
- The last topic (NN-10) should be the most advanced or capstone topic.
- Do not introduce advanced concepts in early topics.

---

## 6. Per-Module README

Each module folder must contain a `README.md` that serves as the module's landing page.

### Required Contents

- **Module title and one-sentence description.**
- **Topic list** with brief descriptions (one line each).
- **Prerequisites:** Which prior modules (or specific notebooks) the reader should complete first.
- **Learning path:** Suggested order if topics can be done non-linearly.
- **Cross-module connections:** Mention related topics in other modules that extend or build on this module's content. For example, Module 8's README can mention that Module 17 applies transformer architectures at scale, or that Module 13 uses PPO (taught in 14-07) for RLHF.

### Cross-Module References in READMEs

READMEs are the appropriate place to mention connections across modules. Unlike notebooks (which must not forward-reference untaught concepts in Parts 0–4), READMEs provide a bird's-eye view and may freely reference any module to help the reader navigate the curriculum. Use the format: `See also: MM-NN (Topic Name) for [what it covers].`
