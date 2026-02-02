# Photography Portfolio

A responsive photography portfolio showcasing mountain portraits, elopements, and outdoor events in Colorado.

**Live site:** https://mayniac24.github.io/portfolio/

## Features

- Responsive gallery with category filtering (Landscapes, Portraits, Nature, Urban)
- Lightbox image viewer with keyboard and touch navigation
- Progressive Web App (PWA) - installable and works offline
- Contact form powered by Formspree
- Lazy loading images with skeleton placeholders

## Local Development

No build step required. Open `index.html` in a browser or use a local server:

```bash
python -m http.server 8000
```

Then visit http://localhost:8000

## Image Categorization

The `auto_categorize.py` script uses a local LM Studio vision model to automatically categorize images and generate alt text:

```bash
pip install pillow requests
python auto_categorize.py
```

Requires [LM Studio](https://lmstudio.ai/) running locally with a vision model.

## License

All photographs are copyrighted. Please contact for licensing inquiries.
