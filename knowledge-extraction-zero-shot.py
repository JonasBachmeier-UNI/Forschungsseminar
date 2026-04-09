from openai import OpenAI
import os
import json
from pathlib import Path
import re

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

# 2. Load Models and Prompt
with open('available_models.json') as f:
    available_models = json.load(f)

with open("knowledge-extraction-zero-shot-prompt.txt", encoding="utf-8") as f:
    base_prompt = f.read()

input_files = sorted(Path("ergebnis_kapitel_5").glob("*.txt"))

def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)

# 3. Process Models and Files
for model in available_models['data'][:3]:
    model_id = model['id']
    safe_model_id = sanitize_name(model_id)
    
    for text_file in input_files:
        print(f"Processing {text_file.name} with {model_id}...")
        
        text_content = text_file.read_text(encoding="utf-8")
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Du bist ein erfahrener Richter am Landgericht, spezialisiert auf Kapitaldelikte (Totschlag, § 212 StGB). Deine Aufgabe ist die dogmatische Analyse der Strafzumessungserwägungen in einem Urteil."},
                    {"role": "user", "content": base_prompt + "\n\nUrteilstext:\n" + text_content}
                ],
                temperature=0.7
            )

            # Extract content from response
            raw_content = response.choices[0].message.content

            # Clean Markdown formatting if present (e.g., ```json ... ```)
            clean_json_str = re.sub(r'^```json\s*|```$', '', raw_content.strip(), flags=re.MULTILINE)

            # Parse the string as JSON
            try:
                data_to_save = json.loads(clean_json_str)
            except json.JSONDecodeError:
                # If the model fails to provide valid JSON, we save the raw text in a JSON wrapper
                print(f"Warning: Output for {text_file.name} was not valid JSON. Saving as raw string.")
                data_to_save = {"error": "Invalid JSON from model", "raw_response": raw_content}

            # Prepare Output Path
            safe_filename = sanitize_name(text_file.stem)
            output_dir = Path("annotationen_uni_models_zero_shot")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = output_dir / f"{safe_model_id}_{safe_filename}_extraction.json"

            # 4. Save to File
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
            print(f"Successfully saved: {output_filename}")

        except Exception as e:
            print(f"An error occurred while processing {text_file.name}: {e}")

print("\nProcessing complete.")