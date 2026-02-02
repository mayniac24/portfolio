"""
Auto-categorize portfolio images using LOCAL LM Studio vision model.

This script analyzes images LOCALLY using your LM Studio server -
no images are sent to external services.

Requirements:
    pip install pillow requests

Setup:
    1. Open LM Studio
    2. Load a vision-capable model (e.g., LLaVA, Qwen2-VL, Pixtral)
    3. Start the local server (usually at http://localhost:1234)
    4. Run this script

Usage:
    python auto_categorize.py
"""

import base64
import json
import re
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import requests
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    print("Missing required packages. Install them with:")
    print("    pip install pillow requests")
    sys.exit(1)


# Configuration
IMAGES_DIR = Path(r"M:\Photography\Portfolio Selections")
HTML_FILE = Path(r"M:\Photography\Portfolio Website\index.html")
CATEGORIES = ["landscape", "portrait", "nature", "urban"]

# LM Studio local server settings
LM_STUDIO_URL = "http://localhost:11434/v1/chat/completions"
MAX_IMAGE_SIZE = 1024  # Resize images to this max dimension for faster processing


def resize_image_for_analysis(image_path: Path) -> bytes:
    """Resize image to reasonable size and convert to base64."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Resize if too large
        if max(img.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.thumbnail(new_size, Image.Resampling.LANCZOS)
            img = Image.open(image_path).convert('RGB')
            img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)

        # Save to bytes
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 string."""
    image_bytes = resize_image_for_analysis(image_path)
    return base64.b64encode(image_bytes).decode('utf-8')


def check_lm_studio_connection():
    """Check if LM Studio server is running."""
    try:
        response = requests.get("http://localhost:11434/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('data', [])
            if models:
                model_id = models[0].get('id', 'unknown')
                print(f"Connected to LM Studio - Model: {model_id}")
                return True
        return False
    except requests.exceptions.ConnectionError:
        return False


def classify_image(image_path: Path) -> tuple[str, str, dict]:
    """
    Classify an image using local LM Studio vision model.

    Returns:
        tuple: (category, alt_text, confidence_info)
    """
    # Convert image to base64
    image_b64 = image_to_base64(image_path)

    # Build the prompt
    prompt = """Analyze this photograph and classify it into ONE of these categories:
- landscape: scenic vistas, mountains, horizons, wide nature views, skies, sunsets
- portrait: photos of people, faces, human subjects as main focus
- nature: close-ups of plants, animals, wildlife, flowers, natural details
- urban: cities, buildings, architecture, streets, man-made structures

Also write a brief, descriptive alt text (10-15 words) describing what's in the image.

Respond in this exact JSON format only:
{"category": "landscape|portrait|nature|urban", "alt_text": "description here", "confidence": "high|medium|low"}"""

    # Call local LM Studio API
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1,
        "stream": False
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        content = result['choices'][0]['message']['content'].strip()

        # Parse JSON response
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            data = json.loads(json_match.group())
            category = data.get('category', 'landscape').lower()
            alt_text = data.get('alt_text', 'Portfolio photograph')
            confidence = data.get('confidence', 'medium')

            # Validate category
            if category not in CATEGORIES:
                category = 'landscape'

            return category, alt_text, {'confidence': confidence, 'raw': content}
        else:
            # Fallback: try to find category keyword in response
            content_lower = content.lower()
            for cat in CATEGORIES:
                if cat in content_lower:
                    return cat, "Portfolio photograph", {'confidence': 'low', 'raw': content}

            return 'landscape', "Portfolio photograph", {'confidence': 'low', 'raw': content}

    except requests.exceptions.Timeout:
        return 'landscape', "Portfolio photograph", {'error': 'timeout'}
    except Exception as e:
        return 'landscape', "Portfolio photograph", {'error': str(e)}


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

    for filename, data in categorized_images.items():
        category = data["category"]
        alt_text = data["alt_text"].replace('"', '&quot;')

        # Pattern to find and replace the gallery item
        pattern = rf'(<div class="gallery-item"[^>]*data-category=")[^"]*("[^>]*><img[^>]*src="[^"]*{re.escape(filename)}"[^>]*alt=")[^"]*(")'
        replacement = rf'\g<1>{category}\g<2>{alt_text}\g<3>'

        content = re.sub(pattern, replacement, content)

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nUpdated {html_file}")


def main():
    print("=" * 60)
    print("Portfolio Image Auto-Categorizer (LOCAL)")
    print("Using LM Studio - No images sent externally")
    print("=" * 60)

    # Check LM Studio connection
    print("\nChecking LM Studio connection...")
    if not check_lm_studio_connection():
        print("\nERROR: Cannot connect to LM Studio server!")
        print("\nPlease ensure:")
        print("  1. LM Studio is open")
        print("  2. A vision model is loaded (e.g., LLaVA, Qwen2-VL)")
        print("  3. Local server is started (Server tab -> Start Server)")
        print("  4. Server is running at http://localhost:11434")
        sys.exit(1)

    # Check images directory
    if not IMAGES_DIR.exists():
        print(f"\nError: Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    # Scan for images
    images = scan_images(IMAGES_DIR)
    if not images:
        print(f"\nNo images found in {IMAGES_DIR}")
        sys.exit(1)

    print(f"\nFound {len(images)} images to analyze")
    print("This may take a while depending on your hardware...")

    # Classify each image
    print("\n" + "-" * 60)

    categorized = {}

    for i, image_path in enumerate(images, 1):
        filename = image_path.name
        print(f"[{i}/{len(images)}] {filename}...", end=" ", flush=True)

        try:
            category, alt_text, info = classify_image(image_path)

            categorized[filename] = {
                "category": category,
                "alt_text": alt_text,
                "info": info,
            }

            confidence = info.get('confidence', '?')
            print(f"{category.upper()} ({confidence})")

            if 'error' in info:
                print(f"    Warning: {info['error']}")

        except Exception as e:
            print(f"ERROR: {e}")
            categorized[filename] = {
                "category": "landscape",
                "alt_text": "Portfolio photograph",
                "info": {"error": str(e)},
            }

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    category_counts = {cat: 0 for cat in CATEGORIES}
    for data in categorized.values():
        category_counts[data["category"]] += 1

    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"  {cat.capitalize():12} {count:3} [{bar}]")

    # Show detailed results
    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)

    for filename, data in categorized.items():
        print(f"\n{filename}")
        print(f"  Category: {data['category'].upper()}")
        print(f"  Alt text: {data['alt_text']}")
        if 'error' in data['info']:
            print(f"  Error: {data['info']['error']}")

    # Ask to update HTML
    print("\n" + "=" * 60)
    response = input("Update index.html with these categories? [y/N]: ").strip().lower()

    if response == "y":
        update_html(HTML_FILE, categorized)
        print("\nDone! Categories and alt text have been updated.")
        print("Review the changes and adjust any misclassifications manually.")
    else:
        print("\nNo changes made.")

    # Generate copy-paste ready HTML
    print("\n" + "-" * 60)
    print("COPY-PASTE HTML (if needed):")
    print("-" * 60)

    for filename, data in categorized.items():
        alt_escaped = data["alt_text"].replace('"', '&quot;')
        print(f'<div class="gallery-item" data-category="{data["category"]}"><img src="../Portfolio Selections/{filename}" alt="{alt_escaped}" loading="lazy"></div>')


if __name__ == "__main__":
    main()
