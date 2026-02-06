#!/usr/bin/env python3
"""
Object Detection CLI Tool

Detect and localize objects in images using DETR models.

Examples:
    python object_detection.py image.jpg
    python object_detection.py photo.png --model large --threshold 0.7
    python object_detection.py street.jpg --save-image output.jpg

Example Output:
    $ python object_detection.py street_scene.jpg

    Loading model: facebook/detr-resnet-50...
    Using device: CPU
    Loading image: street_scene.jpg
    Image size: (1920, 1080)

    Detecting objects...

    ======================================================================
    OBJECT DETECTION RESULTS
    ======================================================================

    Detected 8 objects:

    1. person
       Confidence: 0.9987
       Bounding box: [245, 180, 420, 650]

    2. car
       Confidence: 0.9845
       Bounding box: [680, 320, 1150, 720]

    3. bicycle
       Confidence: 0.9234
       Bounding box: [50, 280, 180, 580]

    4. person
       Confidence: 0.9156
       Bounding box: [890, 160, 1020, 590]

    5. traffic light
       Confidence: 0.8967
       Bounding box: [1650, 45, 1720, 210]

    6. car
       Confidence: 0.8523
       Bounding box: [1200, 380, 1580, 680]

    7. backpack
       Confidence: 0.7845
       Bounding box: [420, 280, 480, 380]

    8. handbag
       Confidence: 0.7234
       Bounding box: [900, 420, 960, 520]

    ======================================================================

    Object counts:
      person: 2
      car: 2
      bicycle: 1
      traffic light: 1
      backpack: 1
      handbag: 1

    $ python object_detection.py photo.jpg --threshold 0.9 --save-image detected.jpg

    Detected 3 objects (threshold >= 0.9)
    Image with bounding boxes saved to: detected.jpg
"""

import argparse
import sys
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def detect_objects(image_path, model_size='small', threshold=0.5):
    """
    Detect objects in an image.

    Args:
        image_path: Path to image file
        model_size: 'small' (detr-resnet-50) or 'large' (detr-resnet-101)
        threshold: Confidence threshold for detections (0.0-1.0)

    Returns:
        Tuple of (detections list, original image)
    """
    # Select model (2+1 pattern: small + large)
    models = {
        'small': 'facebook/detr-resnet-50',   # 159MB, CPU-friendly
        'large': 'facebook/detr-resnet-101'   # 232MB, better accuracy
    }
    model_name = models.get(model_size, models['small'])

    print(f"Loading model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'GPU' if device == 'cuda' else 'CPU'}")

    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model = model.to(device)

    # Load image
    print(f"Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image size: {image.size}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    print(f"\nDetecting objects (threshold >= {threshold})...")

    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Run detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor([image.size[::-1]]).to(device)  # (height, width)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]

    # Format detections
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        detections.append({
            'label': label_name,
            'score': score.item(),
            'box': {
                'xmin': int(box[0].item()),
                'ymin': int(box[1].item()),
                'xmax': int(box[2].item()),
                'ymax': int(box[3].item())
            }
        })

    return detections, image


def draw_bounding_boxes(image, detections, output_path):
    """
    Draw bounding boxes on image and save.

    Args:
        image: PIL Image
        detections: List of detection dictionaries
        output_path: Path to save annotated image
    """
    draw = ImageDraw.Draw(image)

    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for detection in detections:
        box = detection['box']
        label = detection['label']
        score = detection['score']

        # Draw rectangle
        x1, y1 = box['xmin'], box['ymin']
        x2, y2 = box['xmax'], box['ymax']
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Draw label background
        text = f"{label}: {score:.2f}"
        bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((x1, y1 - 20), text, fill="white", font=font)

    # Save image
    image.save(output_path)
    print(f"\nImage with bounding boxes saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect objects in images using DETR models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg
  %(prog)s photo.png --model large --threshold 0.7
  %(prog)s street.jpg --save-image output.jpg
  %(prog)s scene.jpg --threshold 0.9

Supported formats: JPG, PNG, TIFF, BMP

COCO Object Classes (80 categories):
  People: person
  Vehicles: bicycle, car, motorcycle, airplane, bus, train, truck, boat
  Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
  Household: chair, couch, bed, dining table, toilet, tv, laptop, mouse, etc.
  And many more...

Threshold Guide:
  0.5: More detections, some false positives
  0.7: Balanced (recommended)
  0.9: Only very confident detections, may miss some objects
        """
    )

    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', choices=['small', 'large'], default='small',
                        help='Model size: small (detr-resnet-50, 159MB) or large (detr-resnet-101, 232MB)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence threshold (0.0-1.0, default: 0.7)')
    parser.add_argument('--save-image', type=str,
                        help='Save image with bounding boxes to file')

    args = parser.parse_args()

    # Validate threshold
    if args.threshold < 0 or args.threshold > 1:
        print("Error: threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    try:
        # Detect objects
        detections, image = detect_objects(
            args.image,
            model_size=args.model,
            threshold=args.threshold
        )

        # Print results
        print("\n" + "=" * 70)
        print("OBJECT DETECTION RESULTS")
        print("=" * 70)

        if not detections:
            print(f"\nNo objects detected with confidence >= {args.threshold}")
            print("Try lowering the threshold with --threshold 0.5")
        else:
            print(f"\nDetected {len(detections)} object(s):\n")

            for i, detection in enumerate(detections, 1):
                print(f"{i}. {detection['label']}")
                print(f"   Confidence: {detection['score']:.4f}")
                box = detection['box']
                print(f"   Bounding box: [{box['xmin']}, {box['ymin']}, {box['xmax']}, {box['ymax']}]")
                print()

            # Count objects by category
            labels = [d['label'] for d in detections]
            counts = Counter(labels)

            print("Object counts:")
            for label, count in counts.most_common():
                print(f"  {label}: {count}")

        print("\n" + "=" * 70)

        # Save annotated image if requested
        if args.save_image and detections:
            draw_bounding_boxes(image.copy(), detections, args.save_image)
        elif args.save_image and not detections:
            print(f"\nNo bounding boxes to draw (no objects detected)")

    except FileNotFoundError:
        print(f"Error: Image file '{args.image}' not found", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDetection cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
