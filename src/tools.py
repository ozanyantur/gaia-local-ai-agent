import json
import re
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
import wikipedia
from faster_whisper import WhisperModel
from smolagents import tool
from youtube_transcript_api import YouTubeTranscriptApi

_whisper_model = None  # Global variable to hold the Whisper model instance

def get_whisper_model() -> WhisperModel:
    """Load and return the Whisper model instance, initializing it if necessary."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("small", device="cuda", compute_type="float16")  # Change device as needed
    return _whisper_model

@tool
def transcribe_audio_file(file_path: str) -> str:
    """
    Transcribes speech from a local audio file (.mp3, .wav, .m4a, .ogg) into text.

    Args:
        file_path: The local file path to the audio file.
    """
    path = Path(file_path)
    if not path.is_file():
        return f"Error: Audio file not found at {file_path}"

    try:
        model = get_whisper_model()
        segments, info = model.transcribe(str(path), beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments)
        return f"Detected language: {info.language}\nTranscript:\n{transcript}"
    except Exception as e:
        return f"Audio transcription error: {str(e)}"

@tool
def get_youtube_transcript(video_url: str) -> str:
    """
    Fetches the text transcript/subtitles of a YouTube video given its URL.

    Args:
        video_url: Full YouTube video URL (e.g., https://www.youtube.com/watch?v=...).
    """
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", video_url)
    if not match:
        return "Error: Could not extract a valid YouTube video ID from the URL."

    video_id = match.group(1)
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        text = " ".join(snippet.text for snippet in fetched)
        return text[:15000]
    except Exception as e:
        return f"YouTube transcript error: {str(e)}"

@tool
def read_pdf_file(file_path: str) -> str:
    """
    Extracts and returns all text from a local PDF file page by page.

    Args:
        file_path: The local file path to the PDF document.
    """
    path = Path(file_path)
    if not path.is_file():
        return f"Error: The file {file_path} does not exist."

    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        pages.append(f"--- Page {i} ---\n{page.get_text()}")

    content = "\n".join(pages)
    if len(content) > 12000:
        return (
            f"{content[:12000]}\n\n"
            f"[Truncated: PDF has {len(content)} characters total. "
            "Use Python code with fitz to inspect specific pages if needed.]"
        )
    return content


@tool
def read_tabular_file(file_path: str) -> str:
    """
    Reads a CSV or Excel (.xlsx, .xls) file and returns its shape, column names, and full content.

    Args:
        file_path: The local file path to the CSV or Excel file.
    """
    path = Path(file_path)
    if not path.is_file():
        return f"Error: The file {file_path} does not exist."

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    return f"Shape: {df.shape}\nColumns: {list(df.columns)}\n\nData:\n{df.to_string()}"


@tool
def read_text_or_code_file(file_path: str) -> str:
    """
    Reads and returns the content of plain text, code, or structured text files
    (.txt, .py, .json, .md, .xml, .yaml, .yml, .log, .html).

    Args:
        file_path: The local file path to the text or code file.
    """
    path = Path(file_path)
    if not path.is_file():
        return f"Error: The file {file_path} does not exist."

    supported_extensions = {".txt", ".py", ".json", ".md", ".xml", ".yaml", ".yml", ".log", ".html"}
    if path.suffix.lower() not in supported_extensions:
        return (
            f"Error: Unsupported file type '{path.suffix}'. "
            "Expected a .txt, .py, .json, .md, .xml, .yaml, .yml, .log, or .html file."
        )

    content = path.read_text(encoding="utf-8-sig", errors="replace")
    max_chars = 12000
    if len(content) > max_chars:
        return (
            f"{content[:max_chars]}\n\n"
            f"[Truncated: File has {len(content)} characters total. "
            "Use Python code to inspect specific sections if needed.]"
        )
    return content


@tool
def search_wikipedia(query: str, keyword: str = "") -> str:
    """
    Searches Wikipedia for a topic and returns the title, URL, and relevant article text.
    If 'keyword' is provided (e.g., 'Discography', '2000', 'Participants'), returns the section around that keyword.

    Args:
        query: The topic, person, event, or term to search on Wikipedia.
        keyword: Optional specific word or section heading to jump to inside long articles.
    """
    if not query.strip():
        return "Error: The Wikipedia search query cannot be empty."

    try:
        results = wikipedia.search(query)
        if not results:
            return f"No Wikipedia articles found for {query!r}."

        page = wikipedia.page(results[0], auto_suggest=False)
        content = page.content
        if keyword and keyword.lower() in content.lower():
            idx = content.lower().index(keyword.lower())
            start = max(0, idx - 2000)
            extracted = content[start : start + 22000]
        else:
            extracted = content[:22000]

        return json.dumps(
            {"title": page.title, "url": page.url, "content": extracted},
            ensure_ascii=False,
        )
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation error for {query!r}. Try one of these titles: {e.options[:10]}"
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"