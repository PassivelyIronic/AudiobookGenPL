# EpubNarrate 🎧

> Convert any EPUB ebook into a fully narrated Audiobook (MP3) using local, high-quality AI (XTTS-v2).

EpubNarrate is a self-hosted, asynchronous pipeline (FastAPI + Celery + Redis + Gradio) that parses EPUB files, intelligently chunks the text, and synthesizes speech using Coqui's XTTS-v2 model directly on your local GPU.

## Features
* **Local Processing:** 100% offline. No cloud APIs, no subscription fees.
* **Smart Parsing:** Automatically skips TOCs, acknowledgments, and link-dense garbage chapters.
* **Asynchronous Queue:** Built on Celery & Redis. Queue multiple books for overnight processing.
* **Voice Cloning:** Provide a short `.wav` sample, and the AI will narrate the entire book in that voice.

## Installation

### Method A: Docker Compose (Linux / WSL2)
*Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed for GPU access.*

1. Clone the repository and configure your `.env` (use `.env.example` as a template).
2. Place your reference voice sample (e.g., `lektor.wav`) in the project root.
3. Run the stack:
   ```bash
<<<<<<< HEAD
   docker compose up -d
=======
   docker compose up -d
>>>>>>> dc8eb1acd09527a2bdb88d62624529dbd9f72462
