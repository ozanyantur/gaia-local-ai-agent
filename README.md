# Autonomous GAIA Benchmark Agent — Version 1 (Local LLM)

**Version:** `v1.0`  
**Benchmark Result:** `6/20 (30.0%)` on GAIA Level 1 — *Hugging Face AI Agents Course Certificate of Excellence*

An autonomous code agent built with Hugging Face `smolagents` and a locally hosted quantized LLM (**Ollama + `qwen2.5:7b`**). This initial release (`v1.0`) focuses on running multi-step reasoning and tool-use pipelines completely locally on consumer hardware without relying on paid cloud APIs.

---

## Performance & Latency Notes
- **Local Small-Parameter Model:** Because this agent runs locally on a 7-billion-parameter quantized model (`qwen2.5:7b`, `num_ctx=12288`, `temperature=0.0`) on a 6 GB VRAM GPU (RTX 3060), **response times are relatively high** (typically 45–90 seconds per multi-step task) compared to large cloud-hosted models.
- **Deterministic Post-Processing:** Includes a regex-based `clean_answer` pipeline (`src/agent.py`) to enforce GAIA's strict Exact Match formatting rules.

---

## Tool Status (`v1.0`)

### Working Tools
- **`search_wikipedia` (`src/tools.py`):** Queries English Wikipedia and supports keyword-based section jumping to extract specific tables or sections (e.g., discographies, participant lists) from long articles without overflowing the context window.
- **`DuckDuckGoSearchTool` & `VisitWebpageTool`:** Performs live web searches and extracts webpage text for fact-checking and academic/news lookups.
- **`get_youtube_transcript` (`src/tools.py`):** Retrieves video captions via `youtube-transcript-api` with automatic web-search fallback when transcripts are disabled.
- **Built-in Python Interpreter (`CodeAgent`):** Executes arithmetic, string/cipher manipulation, set theory logic, and data filtering directly in Python.

### Currently Non-Working / Limited Tools
- **Remote Task Attachment Pipeline (`read_pdf_file`, `read_tabular_file`, `read_text_or_code_file`):** Although local file parsers (`PyMuPDF`, `pandas`, `openpyxl`) are implemented in `src/tools.py`, the evaluation server's `/files/{task_id}` endpoint currently returns `404` due to ephemeral storage resets, preventing external `.xlsx`, `.py`, `.pdf`, `.png`, and `.mp3` task files from being downloaded during remote evaluation.
- **Audio Transcription (`transcribe_audio_file` via `faster-whisper`) & Vision Tasks:** Implemented locally, but constrained by the missing remote audio/image attachments on the scoring server and VRAM sharing limits alongside the 7B LLM.

---

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure Ollama is running with Qwen 2.5 7B
ollama pull qwen2.5:7b

# 3. Test on a single random GAIA question
python src/run_local.py

# 4. Run full 20-question evaluation and submit to leaderboard
python src/run_local.py submit
