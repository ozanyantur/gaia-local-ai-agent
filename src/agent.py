import re
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    VisitWebpageTool,
)
from tools import (
    get_youtube_transcript,
    read_pdf_file,
    read_tabular_file,
    read_text_or_code_file,
    search_wikipedia,
    transcribe_audio_file,
)

CUSTOM_INSTRUCTIONS = """
You MUST always wrap your Python code inside <code> and </code> tags, and return the final result using final_answer(...).
ONLY call functions that are explicitly provided in your tools list. NEVER invent function names.

CRITICAL PROBLEM-SOLVING & FORMATTING RULES:
1. NEVER pass None or empty string to final_answer(). Verify your variable has a real value before calling final_answer().
2. If the question text looks backwards or reversed, write Python code `print(text[::-1])` FIRST to read the actual instructions accurately.
3. For YouTube questions: if get_youtube_transcript fails or returns no transcript, ALWAYS fall back to duckduckgo_search using the exact quote or video title to find the answer.
4. For long Wikipedia pages (e.g., Olympics participant tables or discographies), use the `keyword` parameter in `search_wikipedia(query=..., keyword=...)` to jump directly to the relevant section.
5. Pass ONLY the exact final answer value inside final_answer("..."). Do NOT include explanations, full sentences, or prefixes like "FINAL ANSWER:".
6. If the answer is a number, pass ONLY the digits without thousand-separator commas or units unless explicitly requested.
7. If the answer is a comma-separated list, ALWAYS separate items with a comma followed by a single space (e.g., final_answer("item1, item2, item3")).
"""


def create_agent(model_id: str = "ollama_chat/qwen2.5:7b") -> CodeAgent:
    model = LiteLLMModel(
        model_id=model_id,
        api_base="http://127.0.0.1:11434",
        num_ctx=12288,
        temperature=0.0,
    )

    agent = CodeAgent(
        tools=[
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            search_wikipedia,
            transcribe_audio_file,
            get_youtube_transcript,
            read_pdf_file,
            read_tabular_file,
            read_text_or_code_file,
        ],
        model=model,
        instructions=CUSTOM_INSTRUCTIONS,
        additional_authorized_imports=[
            "pandas",
            "numpy",
            "openpyxl",
            "math",
            "datetime",
            "json",
            "re",
            "csv",
            "pathlib",
        ],
        max_steps=18,
    )
    return agent


def clean_answer(raw_output: object) -> str:
    """Exact Match: Cleans the raw output from the agent to extract the final answer, removing any prefixes, suffixes, or formatting artifacts."""
    if isinstance(raw_output, (list, tuple)):
        return ", ".join(str(x).strip() for x in raw_output if str(x).strip())

    text = str(raw_output).strip()

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].replace("'", "").replace('"', "").strip()

    text = re.sub(r"^(FINAL ANSWER|Final Answer|Answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip('"').strip("'").strip()

    if text.endswith(".") and not text.replace(".", "", 1).isdigit():
        text = text[:-1].strip()

    if "," in text and len(text) < 150 and "\n" not in text:
        text = re.sub(r"\s*,\s*", ", ", text)

    return text


def solve_question(agent: CodeAgent, question: str, attached_file_path: str | None = None) -> str:
    prompt = question.strip()
    if attached_file_path:
        prompt += (
            f"\n\nATTACHED FILE PATH: '{attached_file_path}'\n"
            "Use this exact file path with your tools."
        )

    raw_result = agent.run(prompt)
    return clean_answer(raw_result)