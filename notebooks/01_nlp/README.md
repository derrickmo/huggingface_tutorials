# Natural Language Processing (NLP)

This folder contains 5 notebooks covering core NLP tasks and fine-tuning techniques.

## Notebooks

### Notebook 01: Text Generation
**Concepts**: Autoregressive language models, GPT architecture
**Models**: distilgpt2 (82MB), gpt2-medium (1.5GB)
**Demo**: Generate creative text completions from prompts

**Quick Demo:**
```python
from transformers import pipeline
generator = pipeline('text-generation', model='distilgpt2')
result = generator("Once upon a time", max_length=50)
print(result[0]['generated_text'])
```

---

### Notebook 02: Text Classification
**Concepts**: Sentiment analysis, sequence classification
**Models**: distilbert-base-uncased-finetuned-sst-2-english (268MB)
**Demo**: Classify text sentiment (positive/negative)

**Quick Demo:**
```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
result = classifier("I love this tutorial!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]
```

---

### Notebook 03: Text Summarization
**Concepts**: Sequence-to-sequence, encoder-decoder models
**Models**: distilbart-cnn-12-6 (1.2GB), bart-large-cnn (1.6GB)
**Demo**: Generate summaries of long articles

**Quick Demo:**
```python
from transformers import pipeline
summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
article = "Your long article text here..."
summary = summarizer(article, max_length=130, min_length=30)
print(summary[0]['summary_text'])
```

---

### Notebook 04: Fine-Tuning with Unsloth (GPU-Only)
**Concepts**: Parameter-efficient fine-tuning, LoRA, 2-5x speedup
**Models**: Llama 3.2 (1B/3B)
**Dataset**: TinyStories (~300MB)
**Requirements**: GPU with 8GB+ VRAM, Python 3.10+

**What you'll learn:**
- Install and setup Unsloth for faster training
- Prepare datasets for instruction fine-tuning
- Apply LoRA adapters to Llama models
- Compare before/after model outputs
- Save and load fine-tuned adapters

**Expected Training Time**: 10-20 minutes on RTX 4080 (2-5x faster than standard)

---

### Notebook 05: Fine-Tuning with LoRA (CPU-Compatible)
**Concepts**: Low-Rank Adaptation, efficient fine-tuning
**Models**: GPT-2 (500MB)
**Demo**: Fine-tune language model with minimal resources

**What you'll learn:**
- Standard LoRA implementation (CPU and GPU)
- Configure LoRA parameters (rank, alpha, dropout)
- Train on custom datasets
- Merge adapters back into base model
- Compare performance vs Unsloth (Notebook 04)

**Expected Training Time**: 20-40 minutes on CPU, 5-10 minutes on GPU

---

## Hardware Requirements

| Notebook | Minimum | Recommended | Purpose |
|----------|---------|-------------|---------|
| 01-03 | 8GB RAM (CPU) | 16GB RAM | Inference only |
| 04 (Unsloth) | N/A | 16GB VRAM (GPU) | Fast fine-tuning |
| 05 (LoRA) | 8GB RAM (CPU) | 12GB VRAM (GPU) | Standard fine-tuning |

## Running the Demos

1. Activate your virtual environment
2. Ensure dependencies are installed: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open any notebook and run cells in order

## CLI Tools

Corresponding CLI tools in `functions/nlp/`:
- `text_generation.py` - Generate text from terminal
- `text_classification.py` - Classify text sentiment
- `text_summarization.py` - Summarize articles

Example:
```bash
python functions/nlp/text_generation.py "Hello world" --model large
```

## Next Steps

After completing this section:
- Try **Computer Vision** (02_computer_vision/) for image tasks
- Explore **Best Practices** (05_best_practices/) for production optimization
- Build **Agentic Workflows** (06_agentic_workflows/) to combine NLP with tools
