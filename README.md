# EpubNarrate 🎧

A local, offline pipeline for converting EPUB ebooks into MP3 audiobooks. 

EpubNarrate extracts text, filters out non-content chapters (TOCs, acknowledgments), and synthesizes audio using Coqui's XTTS-v2 on your own GPU. Built with FastAPI, Celery, and Redis to handle long-running, asynchronous batch conversions.

**Key features:**
* **100% Offline:** Runs locally, no API keys or subscriptions.
* **Voice Cloning:** Narrates using any voice from a short `.wav` reference.
* **Smart Parsing:** Automatically ignores link-heavy chapters and metadata.
* **Queue System:** Drop multiple books into the Gradio UI for overnight rendering.
