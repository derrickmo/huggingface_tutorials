# Multimodal AI

This folder contains 1 notebook covering image captioning with vision-language models.

## Notebook

### Notebook 09: Image Captioning (Image-to-Text)
**Concepts**: Vision-language models, cross-modal attention, BLIP architecture
**Models**: Salesforce/blip-image-captioning-base (990MB), blip-image-captioning-large (1.9GB)
**Demo**: Generate natural language descriptions of images

**Quick Demo:**
```python
from transformers import pipeline
from PIL import Image

captioner = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')
image = Image.open('photo.jpg')
result = captioner(image)
print(result[0]['generated_text'])
```

**Expected Output:**
```
A golden retriever playing fetch in a sunny park with trees in the background
```

**Multiple Captions:**
```python
# Generate diverse captions
captions = captioner(image, max_new_tokens=50, num_beams=5, num_return_sequences=3)
for i, caption in enumerate(captions, 1):
    print(f"{i}. {caption['generated_text']}")
```

**Expected Output:**
```
1. A golden retriever playing fetch in a sunny park with trees in the background
2. A dog catching a frisbee outdoors on a beautiful day
3. A happy golden retriever enjoying playtime in a green park
```

---

## What is Multimodal AI?

**Multimodal AI** processes and combines multiple types of data (vision, language, audio) to understand and generate content.

### Key Concepts:

**Vision-Language Models (VLMs):**
- Process both images and text
- Learn relationships between visual and linguistic concepts
- Enable tasks like captioning, VQA (Visual Question Answering), image search

**Cross-Modal Attention:**
- Allows the model to focus on relevant image regions when generating text
- Connects visual features (from vision encoder) with text features (from language decoder)

**Common Architectures:**
- **BLIP** (Bootstrapped Language-Image Pre-training) - Used in this notebook
- **CLIP** (Contrastive Language-Image Pre-training) - Image-text similarity
- **LLaVA** (Large Language and Vision Assistant) - Chat with images
- **Flamingo/IDEFICS** - Few-shot multimodal learning

---

## Hardware Requirements

| Model | Size | Minimum | Recommended | Performance |
|-------|------|---------|-------------|-------------|
| BLIP-base | 990MB | 8GB RAM (CPU) | 8GB VRAM (GPU) | 2-4s per image (CPU), <1s (GPU) |
| BLIP-large | 1.9GB | 16GB RAM (CPU) | 12GB VRAM (GPU) | 4-8s per image (CPU), 1-2s (GPU) |

## Visual Outputs

The notebook includes visualization code to display:
- Input images with generated captions overlaid
- Multiple caption variations side-by-side
- Attention maps showing which image regions influenced the caption

## Running the Demo

1. **Prepare test images**: Place diverse images in `sample_data/`
   - Nature scenes
   - People and activities
   - Objects and products
   - Complex scenes with multiple subjects

2. **Activate environment**: `source venv/bin/activate`

3. **Launch Jupyter**: `jupyter notebook`

4. **Open Notebook 09**: Run cells sequentially

## CLI Tool

Corresponding CLI tool in `functions/multimodal/`:
- `image_captioning.py` - Generate captions from terminal

Examples:
```bash
# Single caption
python functions/multimodal/image_captioning.py photo.jpg

# Multiple diverse captions
python functions/multimodal/image_captioning.py sunset.jpg --model large --num-captions 3

# Longer, detailed captions
python functions/multimodal/image_captioning.py scene.jpg --max-length 100
```

## Practical Applications

### Content Creation:
- Automatic alt-text for accessibility
- Social media caption generation
- E-commerce product descriptions

### Search and Organization:
- Image search by natural language
- Photo library organization
- Content moderation

### Assistive Technology:
- Describe images for visually impaired users
- Scene understanding for navigation
- Document accessibility

### Creative Applications:
- Story generation from images
- Art description and analysis
- Meme caption generation

## Advanced Use Cases

Beyond basic captioning, vision-language models enable:

1. **Visual Question Answering (VQA)**:
   - Ask questions about images
   - Get factual answers grounded in visual content

2. **Image-Text Retrieval**:
   - Find images matching text descriptions
   - Find descriptions matching images

3. **Visual Reasoning**:
   - Understand spatial relationships
   - Count objects
   - Identify actions and interactions

4. **Cross-Modal Generation**:
   - Generate images from text (diffusion models)
   - Edit images with text instructions

## Comparison with Other Models

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **BLIP** (This notebook) | Accurate captions, good generalization | General image description |
| **CLIP** | Excellent for search, zero-shot classification | Image-text matching |
| **LLaVA** | Conversational, can answer questions | Interactive image analysis |
| **GPT-4V / Claude 3.5** | State-of-the-art understanding | Production applications (API) |

## Common Issues

- **Issue**: Generic/vague captions
  - **Solution**: Use `--model large` or adjust `max_length` and `num_beams`

- **Issue**: Incorrect object recognition
  - **Solution**: Ensure image quality is good; try different images

- **Issue**: Slow inference on CPU
  - **Solution**: Use GPU or smaller model

## Next Steps

After completing this notebook:
- **Best Practices** (05_best_practices/) - Learn optimization for production
- **Performance** (Notebook 11) - Measure and improve inference speed
- **Agentic Workflows** (06_agentic_workflows/) - Build agents that understand images
- **RAG** (Notebook 17) - Combine image understanding with knowledge retrieval

## Further Exploration

**Try these experiments:**
1. Caption images from different domains (art, medical, satellite)
2. Compare BLIP-base vs BLIP-large caption quality
3. Generate captions for memes and analyze humor understanding
4. Create an image search system using captions as metadata
5. Build an accessibility tool that describes images for screen readers
