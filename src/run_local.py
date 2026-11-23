import json
import sys
from pathlib import Path
import requests
from agent import create_agent, solve_question

BASE_URL = "https://agents-course-unit4-scoring.hf.space"
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)


def download_task_file(task_id: str, file_name: str | None) -> str | None:
    """If a file name is provided, download the file from the evaluation server and save it temp_files/<task_id>/. Returns the local file path or None if no file is provided."""
    if not file_name:
        return None

    url = f"{BASE_URL}/files/{task_id}"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        task_dir = TEMP_DIR / task_id   
        task_dir.mkdir(parents=True, exist_ok=True)
        save_path = task_dir / file_name
        save_path.write_bytes(response.content)
        return str(save_path.as_posix())
    return None


def test_single_random_question():
    print("Fetching a random question from evaluation server...")
    resp = requests.get(f"{BASE_URL}/random-question", timeout=30)
    resp.raise_for_status()
    item = resp.json()

    task_id = item["task_id"]
    question = item["question"]
    file_name = item.get("file_name")

    print(f"\nTask ID: {task_id}")
    print(f"File: {file_name}")
    print(f"Question: {question}\n")

    local_file = download_task_file(task_id, file_name)
    agent = create_agent(model_id="ollama_chat/qwen2.5:7b")

    answer = solve_question(agent, question, attached_file_path=local_file)
    print(f"\n>>> CLEANED FINAL ANSWER: {answer}")


def run_full_evaluation(hf_username: str, space_code_url: str):
    print("Fetching all 20 evaluation questions...")    
    resp = requests.get(f"{BASE_URL}/questions", timeout=30)
    resp.raise_for_status()
    questions = resp.json()

    agent = create_agent(model_id="ollama_chat/qwen2.5:7b")
    answers = []

    for idx, item in enumerate(questions, start=1):
        task_id = item["task_id"]
        question = item["question"]
        file_name = item.get("file_name")

        print(f"\n[{idx}/{len(questions)}] Solving Task ID: {task_id}")
        local_file = download_task_file(task_id, file_name)

        try:
            answer = solve_question(agent, question, attached_file_path=local_file)
        except Exception as e:
            print(f"Error on {task_id}: {e}")
            answer = ""

        print(f"Submitted Answer: {answer}")
        answers.append({"task_id": task_id, "submitted_answer": answer})

        # Save intermediate results to a backup file after each question
        Path("answers_backup.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")

    payload = {
        "username": hf_username,
        "agent_code": space_code_url,
        "answers": answers,
    }

    print("\nSubmitting answers to leaderboard...")
    submit_resp = requests.post(f"{BASE_URL}/submit", json=payload, timeout=60)
    print("Status Code:", submit_resp.status_code)
    print("Server Response:", submit_resp.json())


if __name__ == "__main__":
    # By default, run a single random question for local testing. Use "submit" argument to run the full evaluation and submit answers.
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        HF_USERNAME = "ozanyantur"
        SPACE_URL = "https://huggingface.co/spaces/ozanyantur/Final_Assignment_Template/tree/main"
        run_full_evaluation(HF_USERNAME, SPACE_URL)
    else:
        test_single_random_question()