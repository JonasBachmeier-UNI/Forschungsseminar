from openai import OpenAI
import os
import json
from pathlib import Path
import re
import time
import threading

# 1. Setup API Client
# First, add your Key to your shell environment:
# export LLMAPI_KEY="paste-your-key-here"

api_key = os.getenv("LLMAPI_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing API key. Set LLMAPI_KEY (preferred for this project) "
        "or OPENAI_API_KEY in your shell environment."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://hub.nhr.fau.de/api/llmgw/v1"
)

REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "240"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))


def is_retryable_error(error: Exception) -> bool:
    message = str(error).lower()
    retryable_markers = [
        "502",
        "proxy error",
        "error reading from remote server",
        "timed out",
        "timeout",
        "connection",
        "temporar",
        "upstream",
    ]
    return any(marker in message for marker in retryable_markers)


def create_completion_with_live_timer(model_id: str, prompt_text: str):
    start_time = time.perf_counter()
    stop_event = threading.Event()

    def timer_worker():
        while not stop_event.wait(1):
            elapsed = int(time.perf_counter() - start_time)
            print(f"  Call running: {elapsed}s", end="\r", flush=True)

    timer_thread = threading.Thread(target=timer_worker, daemon=True)
    timer_thread.start()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Du bist ein erfahrener Richter am Landgericht, spezialisiert auf Kapitaldelikte (Totschlag, § 212 StGB). Deine Aufgabe ist die dogmatische Analyse der Strafzumessungserwägungen in einem Urteil."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return response, time.perf_counter() - start_time
    finally:
        stop_event.set()
        timer_thread.join(timeout=0.2)
        elapsed = int(time.perf_counter() - start_time)
        print(f"  Call running: {elapsed}s", flush=True)


def normalize_response_content(raw_content) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if raw_content is None:
        return ""
    return str(raw_content)

# 2. Load Models and Prompt
with open('available_models.json') as f:
    available_models = json.load(f)

with open("knowledge-extraction-zero-shot-prompt.txt", encoding="utf-8") as f:
    base_prompt = f.read()

input_files = sorted(Path("ergebnis_kapitel_5").glob("*.txt"))

def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)

def resolve_model_output_dir(base_dir: Path, model_id: str) -> Path:
    # Prefer an existing folder that matches the model name, then fall back to sanitized.
    exact_dir = base_dir / model_id
    if exact_dir.exists() and exact_dir.is_dir():
        return exact_dir

    safe_model_id = sanitize_name(model_id)
    safe_dir = base_dir / safe_model_id
    safe_dir.mkdir(parents=True, exist_ok=True)
    return safe_dir

# 3. Process Models and Files
base_output_dir = Path("annotationen_uni_models_zero_shot")
base_output_dir.mkdir(parents=True, exist_ok=True)

selected_models = available_models['data'][:6]

tasks = []
for model in selected_models:
    model_id = model['id']
    safe_model_id = sanitize_name(model_id)
    model_output_dir = resolve_model_output_dir(base_output_dir, model_id)

    for text_file in input_files:
        safe_filename = sanitize_name(text_file.stem)
        output_filename = model_output_dir / f"{safe_model_id}_{safe_filename}_extraction.json"
        tasks.append((model_id, safe_model_id, text_file, output_filename))

total_tasks = len(tasks)
existing_tasks = sum(1 for _, _, _, output_filename in tasks if output_filename.exists())
remaining_tasks = total_tasks - existing_tasks
processed_tasks = 0

print(
    f"Total tasks: {total_tasks} | "
    f"Already existing: {existing_tasks} | "
    f"To process: {remaining_tasks}"
)

for model_id, safe_model_id, text_file, output_filename in tasks:
    processed_tasks += 1
    progress_pct = (processed_tasks / total_tasks * 100) if total_tasks else 100.0

    if output_filename.exists():
        print(
            f"[{processed_tasks}/{total_tasks} | {progress_pct:.1f}%] "
            f"Skipping existing: {output_filename.name}"
        )
        continue

    print(
        f"[{processed_tasks}/{total_tasks} | {progress_pct:.1f}%] "
        f"Processing {text_file.name} with {model_id}..."
    )

    text_content = text_file.read_text(encoding="utf-8")

    try:
        response = None
        call_elapsed_seconds = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  Attempt {attempt}/{MAX_RETRIES} started...")
                response, call_elapsed_seconds = create_completion_with_live_timer(
                    model_id=model_id,
                    prompt_text=base_prompt + "\n\nUrteilstext:\n" + text_content,
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as request_error:
                if attempt >= MAX_RETRIES or not is_retryable_error(request_error):
                    raise
                wait_seconds = min(2 ** attempt, 20)
                print(
                    f"Retryable error for {text_file.name} "
                    f"(attempt {attempt}/{MAX_RETRIES}): {request_error}"
                )
                print(f"Waiting {wait_seconds}s before retry...")
                time.sleep(wait_seconds)

        if response is None:
            raise RuntimeError("No response after retries.")

        print(
            f"Call succeeded in {call_elapsed_seconds:.2f}s "
            f"(attempt {attempt}/{MAX_RETRIES})"
        )

        # Extract content from response
        raw_content = normalize_response_content(response.choices[0].message.content)

        if not raw_content.strip():
            print(
                f"Warning: Empty response content for {text_file.name}. "
                "Skipping file save."
            )
            continue
        else:
            # Clean Markdown formatting if present (e.g., ```json ... ```)
            clean_json_str = re.sub(r'^```json\s*|```$', '', raw_content.strip(), flags=re.MULTILINE)

            # Parse the string as JSON
            try:
                data_to_save = json.loads(clean_json_str)
            except json.JSONDecodeError:
                print(
                    f"Warning: Output for {text_file.name} was not valid JSON. "
                    "Skipping file save."
                )
                continue

        # 4. Save to File
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        print(f"Successfully saved: {output_filename}")

    except Exception as e:
        print(f"An error occurred while processing {text_file.name}: {e}")

print("\nProcessing complete.")