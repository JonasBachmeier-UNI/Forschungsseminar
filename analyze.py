import json
from pathlib import Path
from itertools import combinations
from difflib import SequenceMatcher
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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


def load_model_prefixes(base_dir):
    """Load known model ids and convert them to filename prefixes."""
    models_path = base_dir / "available_models.json"
    try:
        with open(models_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    prefixes = []
    for item in data.get("data", []):
        model_id = item.get("id")
        if model_id:
            prefixes.append(model_id.replace("/", "_"))

    return sorted(set(prefixes), key=len, reverse=True)


def parse_extraction_filename(filename, model_prefixes):
    """Split `model_prefix + text_id + _extraction` filenames safely."""
    stem = filename.stem.replace("_extraction", "")
    for prefix in model_prefixes:
        if stem.startswith(prefix + "_"):
            return prefix, stem[len(prefix) + 1 :]

    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

# 1. Load your files
base_dir = Path(__file__).resolve().parent
output_dir = base_dir / "annotationen_uni_models_zero_shot"
all_files = list(output_dir.glob("*_extraction.json"))
model_prefixes = load_model_prefixes(base_dir)

# Group files by the original Text ID
# Adjust the splitting logic based on your exact filenames
texts_dict = {}
for f in all_files:
    model_id, text_id = parse_extraction_filename(f, model_prefixes)
    if model_id and text_id:
        if text_id not in texts_dict:
            texts_dict[text_id] = {}
        texts_dict[text_id][model_id] = extract_annotations(f)

results = []
model_level_results = []

# 2. Analyze Model Consistencies per Text
for text_id, model_outputs in texts_dict.items():
    models = list(model_outputs.keys())
    
    # Compare every model against every other model (e.g., A vs B, B vs C, A vs C)
    for model_a, model_b in combinations(models, 2):
        model_left, model_right = sorted([model_a, model_b])
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
            "Model_Pair": f"{model_left} vs {model_right}",
            "Shared_Tags_Count": len(intersection),
            "Tag_Overlap_Score": round(tag_jaccard, 3),
            "Quote_Similarity_Score": round(avg_quote_sim, 3),
            "Quality_Score": round((tag_jaccard + avg_quote_sim) / 2, 3)
        })

        pair_quality_score = (tag_jaccard + avg_quote_sim) / 2
        model_level_results.append({
            "Model": model_a,
            "Tag_Overlap_Score": tag_jaccard,
            "Quote_Similarity_Score": avg_quote_sim,
            "Quality_Score": pair_quality_score,
        })
        model_level_results.append({
            "Model": model_b,
            "Tag_Overlap_Score": tag_jaccard,
            "Quote_Similarity_Score": avg_quote_sim,
            "Quality_Score": pair_quality_score,
        })

# 3. View Results
df_results = pd.DataFrame(results)

print("--- Pairwise Agreement ---")
print(df_results.to_string(index=False))

# Calculate Average Performance across all texts
print("\n--- Average Agreement by Model Pair ---")
avg_agreement = df_results.groupby("Model_Pair")[["Tag_Overlap_Score", "Quote_Similarity_Score"]].mean().reset_index()
avg_agreement["Quality_Score"] = avg_agreement[["Tag_Overlap_Score", "Quote_Similarity_Score"]].mean(axis=1)
print(avg_agreement.to_string(index=False))

model_scores = pd.DataFrame(model_level_results)
if not model_scores.empty:
    model_scores = (
        model_scores
        .groupby("Model")[["Tag_Overlap_Score", "Quote_Similarity_Score", "Quality_Score"]]
        .mean()
        .reset_index()
        .sort_values("Quality_Score", ascending=True)
    )

    pair_index = {}
    for _, row in avg_agreement.iterrows():
        model_a, model_b = row["Model_Pair"].split(" vs ")
        pair_index[(model_a, model_b)] = row["Quality_Score"]
        pair_index[(model_b, model_a)] = row["Quality_Score"]

    models = model_scores["Model"].tolist()
    heatmap = []
    for model_a in models:
        row_values = []
        for model_b in models:
            if model_a == model_b:
                row_values.append(1.0)
            else:
                row_values.append(pair_index.get((model_a, model_b), float("nan")))
        heatmap.append(row_values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1.2]})

    score_norm = Normalize(vmin=0, vmax=1)
    score_cmap = plt.cm.RdYlGn

    bar_colors = score_cmap(score_norm(model_scores["Quality_Score"]))
    axes[0].barh(model_scores["Model"], model_scores["Quality_Score"], color=bar_colors)
    axes[0].set_title("Average model agreement")
    axes[0].set_xlabel("Quality score")
    axes[0].set_xlim(0, 1)
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    for index, value in enumerate(model_scores["Quality_Score"]):
        axes[0].text(min(value + 0.01, 0.99), index, f"{value:.2f}", va="center", fontsize=9)

    heatmap_axis = axes[1]
    image = heatmap_axis.imshow(heatmap, cmap=score_cmap, vmin=0, vmax=1)
    heatmap_axis.set_title("Pairwise model agreement")
    heatmap_axis.set_xticks(range(len(models)))
    heatmap_axis.set_yticks(range(len(models)))
    heatmap_axis.set_xticklabels(models, rotation=45, ha="right")
    heatmap_axis.set_yticklabels(models)

    for row_index, row_values in enumerate(heatmap):
        for column_index, value in enumerate(row_values):
            if pd.notna(value):
                heatmap_axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04, label="Quality score")
    fig.suptitle("LLM response quality comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    figure_path = base_dir / "llm_quality_comparison.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    print(f"\nVisualization saved to: {figure_path}")
else:
    print("\nNo model-level scores available for visualization.")