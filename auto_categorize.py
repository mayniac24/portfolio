"""
Auto-categorize portfolio images using CLIP vision model.

This script analyzes images and automatically assigns categories
(landscape, portrait, nature, urban) based on image content.

Requirements:
    pip install torch torchvision transformers pillow

Usage:
    python auto_categorize.py
"""

import os
import sys
from pathlib import Path

# Check for required packages
try:
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
except ImportError as e:
    print("Missing required packages. Install them with:")
    print("    pip install torch torchvision transformers pillow")
    sys.exit(1)


# Configuration
IMAGES_DIR = Path(r"M:\Photography\Portfolio Selections")
HTML_FILE = Path(r"M:\Photography\Portfolio Website\index.html")
CATEGORIES = ["landscape", "portrait", "nature", "urban"]

# Detailed prompts for better classification accuracy
CATEGORY_PROMPTS = {
    "landscape": [
        "a landscape photograph with mountains, hills, or scenic vista",
        "a wide scenic photograph of nature with sky and horizon",
        "a photograph of mountains, valleys, or open terrain",
        "a sunset or sunrise landscape photograph",
    ],
    "portrait": [
        "a portrait photograph of a person",
        "a close-up photograph of a human face",
        "a photograph focused on a person or people",
        "a headshot or portrait with a person as the main subject",
    ],
    "nature": [
        "a nature photograph of plants, flowers, or wildlife",
        "a close-up photograph of animals, insects, or flora",
        "a macro photograph of natural elements",
        "a photograph of trees, forests, or natural details",
    ],
    "urban": [
        "an urban photograph of buildings or city architecture",
        "a street photography image of city life",
        "a photograph of urban infrastructure or cityscape",
        "an architectural photograph of man-made structures",
    ],
}


def load_model():
    """Load CLIP model and processor."""
    print("Loading CLIP model (this may take a moment on first run)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device.upper()}")

    return model, processor, device


def classify_image(image_path: Path, model, processor, device) -> tuple[str, dict]:
    """
    Classify an image into one of the categories.

    Returns:
        tuple: (best_category, confidence_scores)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Build all prompts
    all_prompts = []
    prompt_to_category = {}
    for category, prompts in CATEGORY_PROMPTS.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_category[prompt] = category

    # Process image and text
    inputs = processor(
        text=all_prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # Aggregate scores by category
    category_scores = {cat: 0.0 for cat in CATEGORIES}
    for prompt, prob in zip(all_prompts, probs):
        category = prompt_to_category[prompt]
        category_scores[category] += prob

    # Normalize scores
    total = sum(category_scores.values())
    category_scores = {k: v / total for k, v in category_scores.items()}

    # Get best category
    best_category = max(category_scores, key=category_scores.get)

    return best_category, category_scores


def generate_alt_text(image_path: Path, category: str) -> str:
    """Generate descriptive alt text based on filename and category."""
    filename = image_path.stem

    # Base descriptions by category
    descriptions = {
        "landscape": [
            "Scenic landscape with dramatic lighting",
            "Expansive vista capturing natural beauty",
            "Landscape photograph at golden hour",
            "Panoramic view of natural terrain",
            "Serene landscape with atmospheric conditions",
        ],
        "portrait": [
            "Portrait photograph with natural lighting",
            "Candid portrait capturing a genuine moment",
            "Environmental portrait with thoughtful composition",
            "Portrait showcasing personality and emotion",
            "Artistic portrait with creative lighting",
        ],
        "nature": [
            "Nature photography capturing organic beauty",
            "Close-up of natural elements and textures",
            "Wildlife or flora in natural habitat",
            "Macro photograph revealing natural details",
            "Nature scene with rich colors and patterns",
        ],
        "urban": [
            "Urban architecture and geometric forms",
            "Street photography capturing city life",
            "Architectural details and urban patterns",
            "Cityscape with dynamic composition",
            "Urban environment with striking contrast",
        ],
    }

    # Use hash of filename to consistently pick a description
    desc_list = descriptions.get(category, descriptions["landscape"])
    index = hash(filename) % len(desc_list)

    # Add B&W note if applicable
    base_desc = desc_list[index]
    if "_BW" in filename or "_bw" in filename:
        base_desc = f"Black and white {base_desc.lower()}"

    return base_desc


def scan_images(images_dir: Path) -> list[Path]:
    """Find all image files in the directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    images = []

    for file in images_dir.iterdir():
        if file.suffix.lower() in extensions:
            images.append(file)

    return sorted(images)


def update_html(html_file: Path, categorized_images: dict):
    """Update the HTML file with new categories and alt text."""
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Update each image entry
    for filename, data in categorized_images.items():
        category = data["category"]
        alt_text = data["alt_text"]

        # Find and replace the gallery item for this image
        # Match pattern: data-category="anything" ... src=".../{filename}"
        import re

        # Pattern to find the gallery item containing this image
        pattern = rf'(<div class="gallery-item"[^>]*data-category=")[^"]*("[^>]*><img[^>]*src="[^"]*{re.escape(filename)}"[^>]*alt=")[^"]*(")'
        replacement = rf'\g<1>{category}\g<2>{alt_text}\g<3>'

        content = re.sub(pattern, replacement, content)

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nUpdated {html_file}")


def main():
    print("=" * 60)
    print("Portfolio Image Auto-Categorizer")
    print("=" * 60)

    # Check if images directory exists
    if not IMAGES_DIR.exists():
        print(f"Error: Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    # Scan for images
    images = scan_images(IMAGES_DIR)
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    print(f"\nFound {len(images)} images to analyze")

    # Load model
    model, processor, device = load_model()

    # Classify each image
    print("\nAnalyzing images...")
    print("-" * 60)

    categorized = {}

    for i, image_path in enumerate(images, 1):
        filename = image_path.name
        print(f"[{i}/{len(images)}] {filename}...", end=" ", flush=True)

        try:
            category, scores = classify_image(image_path, model, processor, device)
            alt_text = generate_alt_text(image_path, category)

            categorized[filename] = {
                "category": category,
                "scores": scores,
                "alt_text": alt_text,
            }

            # Show confidence
            confidence = scores[category] * 100
            print(f"{category.upper()} ({confidence:.1f}%)")

        except Exception as e:
            print(f"ERROR: {e}")
            categorized[filename] = {
                "category": "landscape",  # fallback
                "scores": {},
                "alt_text": "Portfolio photograph",
            }

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    category_counts = {cat: 0 for cat in CATEGORIES}
    for data in categorized.values():
        category_counts[data["category"]] += 1

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {cat.capitalize():12} {count:3} {bar}")

    # Show detailed results
    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)

    for filename, data in categorized.items():
        print(f"\n{filename}")
        print(f"  Category: {data['category'].upper()}")
        if data["scores"]:
            scores_str = " | ".join(
                f"{cat}: {score*100:.0f}%"
                for cat, score in sorted(data["scores"].items(), key=lambda x: -x[1])
            )
            print(f"  Scores:   {scores_str}")
        print(f"  Alt text: {data['alt_text']}")

    # Ask to update HTML
    print("\n" + "=" * 60)
    response = input("Update index.html with these categories? [y/N]: ").strip().lower()

    if response == "y":
        update_html(HTML_FILE, categorized)
        print("\nDone! Categories and alt text have been updated.")
        print("Review the changes and adjust any misclassifications manually.")
    else:
        print("\nNo changes made. You can manually update the HTML using the results above.")

    # Generate copy-paste ready HTML (optional output)
    print("\n" + "-" * 60)
    print("COPY-PASTE HTML (if needed):")
    print("-" * 60)

    for filename, data in categorized.items():
        print(f'<div class="gallery-item" data-category="{data["category"]}"><img src="../Portfolio Selections/{filename}" alt="{data["alt_text"]}" loading="lazy"></div>')


if __name__ == "__main__":
    main()
