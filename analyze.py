import json
from pathlib import Path
from itertools import combinations
from difflib import SequenceMatcher
import pandas as pd

def get_similarity_ratio(str1, str2):
    """Calculates fuzzy string similarity between two quotes."""
    if not str1 or not str2: return 0.0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def extract_annotations(filepath):
    """Safely extracts the annotations list from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("annotations", [])
    except Exception:
        return []

# 1. Load your files (Update the path if necessary)
output_dir = Path("annotationen_uni_models_zero_shot")
all_files = list(output_dir.glob("*_extraction.json"))

# Group files by the original Text ID
# Adjust the splitting logic based on your exact filenames
texts_dict = {}
for f in all_files:
    # Assuming filename: {model_id}_{text_filename}_extraction.json
    parts = f.stem.replace("_extraction", "").split("_", 1)
    if len(parts) == 2:
        model_id, text_id = parts
        if text_id not in texts_dict:
            texts_dict[text_id] = {}
        texts_dict[text_id][model_id] = extract_annotations(f)

results = []

# 2. Analyze Model Consistencies per Text
for text_id, model_outputs in texts_dict.items():
    models = list(model_outputs.keys())
    
    # Compare every model against every other model (e.g., A vs B, B vs C, A vs C)
    for model_a, model_b in combinations(models, 2):
        ann_a = model_outputs[model_a]
        ann_b = model_outputs[model_b]
        
        # Extract Sets of Tags
        tags_a = set([item.get('tag') for item in ann_a if item.get('tag')])
        tags_b = set([item.get('tag') for item in ann_b if item.get('tag')])
        
        # Calculate Tag Overlap (Jaccard Similarity)
        intersection = tags_a.intersection(tags_b)
        union = tags_a.union(tags_b)
        tag_jaccard = len(intersection) / len(union) if union else 0.0
        
        # Calculate Quote Similarity for overlapping tags
        quote_similarities = []
        for tag in intersection:
            # Find all quotes for this specific tag in both models
            quotes_a = [item.get('quote', '') for item in ann_a if item.get('tag') == tag]
            quotes_b = [item.get('quote', '') for item in ann_b if item.get('tag') == tag]
            
            # Since a tag might appear multiple times (like 'Strafhoehe'), 
            # we check the best match between the lists of quotes
            for qa in quotes_a:
                best_match = max([get_similarity_ratio(qa, qb) for qb in quotes_b] + [0])
                quote_similarities.append(best_match)
                
        avg_quote_sim = sum(quote_similarities) / len(quote_similarities) if quote_similarities else 0.0

        results.append({
            "Text_File": text_id,
            "Model_Pair": f"{model_a} vs {model_b}",
            "Shared_Tags_Count": len(intersection),
            "Tag_Overlap_Score": round(tag_jaccard, 3),
            "Quote_Similarity_Score": round(avg_quote_sim, 3)
        })

# 3. View Results
df_results = pd.DataFrame(results)

print("--- Pairwise Agreement ---")
print(df_results.to_string(index=False))

# Calculate Average Performance across all texts
print("\n--- Average Agreement by Model Pair ---")
avg_agreement = df_results.groupby("Model_Pair")[["Tag_Overlap_Score", "Quote_Similarity_Score"]].mean().reset_index()
print(avg_agreement.to_string(index=False))