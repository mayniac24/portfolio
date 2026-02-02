# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static photography portfolio website with PWA support. No build system required—deploy files directly to any static hosting provider.

## Architecture

**Single-page application structure:**
- `index.html` - Complete site (HTML + inline CSS + inline JS). Contains gallery grid, lightbox viewer, category filtering, contact form, and service worker registration.
- `service-worker.js` - Offline caching with stale-while-revalidate strategy. Caches HTML, manifest, favicon, and images automatically.
- `manifest.json` - PWA configuration for installable app experience.

**Image handling:**
- Gallery images are sourced from `../Portfolio Selections/` (sibling directory)
- Each gallery item has `data-category` attribute for filtering (landscape, portrait, nature, urban)
- Images use native lazy loading (`loading="lazy"`)

**External dependencies:**
- Contact form uses Formspree (requires form ID configuration at `https://formspree.io/f/YOUR_FORM_ID`)

## Utility Script

`auto_categorize.py` - Uses local LM Studio vision model to automatically categorize images and generate alt text. Updates `index.html` directly.

```bash
# Requires LM Studio running locally with a vision model
pip install pillow requests
python auto_categorize.py
```

## Deployment Notes

Before deploying:
1. Replace `YOUR_FORM_ID` in the contact form action URL with actual Formspree endpoint
2. Copy portfolio images into the project or update image paths in `index.html`
3. Generate `icon-192.png` and `icon-512.png` for PWA icons (referenced in manifest.json but not present)
4. Update `og:url` meta tag with actual site URL
