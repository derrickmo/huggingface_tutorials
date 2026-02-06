# Computer Vision (CV)

This folder contains 3 notebooks covering image classification, object detection, and OCR.

## Notebooks

### Notebook 04: Image Classification
**Concepts**: Vision Transformers (ViT), convolutional neural networks
**Models**: google/vit-base-patch16-224 (346MB)
**Demo**: Classify images into 1000 ImageNet categories

**Quick Demo:**
```python
from transformers import pipeline
from PIL import Image

classifier = pipeline('image-classification', model='google/vit-base-patch16-224')
image = Image.open('path/to/image.jpg')
results = classifier(image, top_k=5)
for result in results:
    print(f"{result['label']}: {result['score']:.2%}")
```

**Expected Output:**
```
tabby cat: 94.52%
Egyptian cat: 3.21%
tiger cat: 1.87%
```

---

### Notebook 05: Object Detection
**Concepts**: Bounding boxes, DETR architecture, YOLO
**Models**: facebook/detr-resnet-50 (159MB), YOLOv8 (optional)
**Demo**: Detect and localize multiple objects in images

**Quick Demo (DETR):**
```python
from transformers import pipeline
from PIL import Image

detector = pipeline('object-detection', model='facebook/detr-resnet-50')
image = Image.open('street_scene.jpg')
results = detector(image)

for obj in results:
    print(f"{obj['label']}: {obj['score']:.2f} at {obj['box']}")
```

**Expected Output:**
```
person: 0.99 at {'xmin': 120, 'ymin': 45, 'xmax': 280, 'ymax': 450}
car: 0.95 at {'xmin': 400, 'ymin': 200, 'xmax': 650, 'ymax': 380}
bicycle: 0.87 at {'xmin': 50, 'ymin': 150, 'xmax': 180, 'ymax': 320}
```

**Method 3: YOLOv8 (State-of-the-Art)**
- Faster inference (real-time capable)
- Higher accuracy on small objects
- Requires `ultralytics` package

---

### Notebook 06: Optical Character Recognition (OCR)
**Concepts**: Text detection and recognition, TrOCR
**Models**: microsoft/trocr-small-printed (558MB), PaddleOCR (optional)
**Demo**: Extract text from images of documents

**Quick Demo (TrOCR):**
```python
from transformers import pipeline
from PIL import Image

ocr = pipeline('image-to-text', model='microsoft/trocr-small-printed')
image = Image.open('document.jpg')
result = ocr(image)
print(result[0]['generated_text'])
```

**Expected Output:**
```
This is a sample document with printed text.
The OCR model can read and transcribe it accurately.
```

**Method 2: PaddleOCR (State-of-the-Art)**
- Multi-language support (80+ languages)
- Better accuracy on complex layouts
- Production-grade performance
- Requires `paddleocr` package

---

## Hardware Requirements

| Notebook | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| 04 (Classification) | 8GB RAM (CPU) | 8GB VRAM (GPU) | Fast inference |
| 05 (Detection) | 8GB RAM (CPU) | 8GB VRAM (GPU) | Real-time with GPU |
| 06 (OCR) | 8GB RAM (CPU) | 8GB VRAM (GPU) | Multi-page docs benefit from GPU |

## Visual Outputs

All notebooks include visualization code to display:
- Input images with predictions overlaid
- Bounding boxes for detected objects
- Highlighted text regions for OCR
- Top-K predictions as bar charts

## Running the Demos

1. **Prepare test images**: Place `.jpg` or `.png` images in `sample_data/`
2. **Activate environment**: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. **Launch Jupyter**: `jupyter notebook`
4. **Open notebook**: Start with Notebook 04 for basics

## CLI Tools

Corresponding CLI tools in `functions/cv/`:
- `image_classification.py` - Classify images from terminal
- `object_detection.py` - Detect objects in images
- `ocr.py` - Extract text from document images

Example:
```bash
python functions/cv/image_classification.py photo.jpg --model large --top-k 5
python functions/cv/object_detection.py street.jpg --method detr
python functions/cv/ocr.py document.png --model small
```

## Common Use Cases

**Image Classification:**
- Content moderation
- Product categorization
- Wildlife identification

**Object Detection:**
- Autonomous vehicles
- Security/surveillance
- Retail analytics

**OCR:**
- Document digitization
- Invoice processing
- License plate recognition

## Next Steps

After completing this section:
- Try **Multimodal** (04_multimodal/) to combine vision and language
- Explore **Best Practices** (05_best_practices/) for optimization
- Learn **Performance** (Notebook 11) for production deployment
