"""
Annotation Agreement Analysis für KI-Modell-Vergleich
======================================================
Verbesserungen gegenüber der ursprünglichen Version:
  1. Ground-Truth Evaluation (Precision / Recall / F1) für annotierte Dateien
  2. Majority-Vote Pseudo-GT für nicht annotierte Dateien
  3. Per-Tag / Per-Kategorie Breakdown
  4. Korrekte individuelle Modell-Scores (nicht mehr Paar-Score für beide)
  5. Multiplizität berücksichtigt (multiset statt set)
  6. Bootstrap-Konfidenzintervalle für Gesamt-Metriken
"""

import json
import re
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.75   # Ab wann gilt ein Quote-Match als "gefunden"
MAJORITY_THRESHOLD   = 0.35    # Anteil Modelle, die einen Tag brauchen für Pseudo-GT
BOOTSTRAP_N          = 1_000  # Iterationen für Konfidenzintervalle
RANDOM_SEED          = 42

base_dir   = Path(__file__).resolve().parent
output_dir = base_dir / "annotationen_uni_models_zero_shot"
gt_dir     = base_dir / "ground_truth"   # Ordner mit professionellen Annotationen
figure_path = base_dir / "llm_quality_comparison_2.png"
hallucination_figure_path = base_dir / "zero_shot_halluzination.png"


# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────

def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def similarity(s1: str, s2: str) -> float:
    """Fuzzy-Ähnlichkeit zweier Strings."""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def extract_annotations(filepath: Path) -> list[dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f).get("annotations", [])
    except Exception:
        return []


def load_model_prefixes(base_dir: Path) -> list[str]:
    try:
        with open(base_dir / "available_models.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    prefixes = [sanitize_name(item["id"]) for item in data.get("data", []) if item.get("id")]
    return sorted(set(prefixes), key=len, reverse=True)


def parse_extraction_filename(filepath: Path, model_prefixes: list[str]) -> tuple[str, str]:
    stem = filepath.stem.replace("_extraction", "")
    for prefix in model_prefixes:
        if stem.startswith(prefix + "_"):
            return prefix, stem[len(prefix) + 1:]
    parts = stem.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


# ─────────────────────────────────────────────
# Annotation-Matching (NER-Style)
# ─────────────────────────────────────────────

def ann_to_items(annotations: list[dict]) -> list[tuple[str, str]]:
    """Wandelt Annotationen in (tag, quote)-Tupel um (Multiplizität erhalten)."""
    return [(a.get("tag", ""), a.get("quote", "")) for a in annotations if a.get("tag")]


def match_items(
    pred: list[tuple[str, str]],
    gold: list[tuple[str, str]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[int, int, int]:
    """
    Greedy Matching: ein Gold-Item kann nur einmal gematcht werden.
    Gibt (True Positives, False Positives, False Negatives) zurück.
    """
    gold_remaining = list(gold)
    tp = 0
    for p_tag, p_quote in pred:
        for i, (g_tag, g_quote) in enumerate(gold_remaining):
            if p_tag == g_tag and similarity(p_quote, g_quote) >= threshold:
                tp += 1
                gold_remaining.pop(i)
                break
    fp = len(pred) - tp
    fn = len(gold_remaining)
    return tp, fp, fn


def match_annotations_with_hallucinations(
    pred: list[dict],
    gold: list[dict],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[int, int, int, Counter]:
    """Greedy-Matching plus Kategorien der nicht gematchten Vorhersagen."""
    gold_remaining = list(gold)
    tp = 0
    hallucinated_categories: Counter = Counter()

    for pred_ann in pred:
        p_tag = pred_ann.get("tag", "")
        p_quote = pred_ann.get("quote", "")
        matched = False

        for i, gold_ann in enumerate(gold_remaining):
            if p_tag == gold_ann.get("tag", "") and similarity(p_quote, gold_ann.get("quote", "")) >= threshold:
                tp += 1
                gold_remaining.pop(i)
                matched = True
                break

        if not matched:
            category = pred_ann.get("category", "Unbekannt") or "Unbekannt"
            hallucinated_categories[category] += 1

    fp = len(pred) - tp
    fn = len(gold_remaining)
    return tp, fp, fn, hallucinated_categories


def compute_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_hallucination_rate(tp: int, fp: int) -> float:
    """Anteil der vorhergesagten Aussagen, die nicht durch die Referenz gedeckt sind."""
    total_predicted = tp + fp
    return fp / total_predicted if total_predicted > 0 else 0.0


# ─────────────────────────────────────────────
# Majority-Vote Pseudo-Ground-Truth
# ─────────────────────────────────────────────

def build_majority_vote(model_outputs: dict[str, list[dict]], threshold: float = MAJORITY_THRESHOLD) -> list[dict]:
    """
    Erzeugt eine Pseudo-GT aus Mehrheitsentscheid:
      - Ein Tag wird aufgenommen, wenn >= threshold*n Modelle ihn finden
      - Quote wird der am häufigsten vorkommende / ähnlichste Konsens-String
    """
    n_models = len(model_outputs)
    min_votes = max(2, int(np.ceil(threshold * n_models)))

    # Sammle alle (tag, quote)-Paare über alle Modelle
    tag_counter: dict[str, list[str]] = defaultdict(list)
    for annotations in model_outputs.values():
        for ann in annotations:
            tag = ann.get("tag", "")
            quote = ann.get("quote", "")
            if tag:
                tag_counter[tag].append(quote)

    pseudo_gt = []
    for tag, quotes in tag_counter.items():
        # Nur aufnehmen wenn genug Modelle diesen Tag haben
        # (Zähle Modelle, die den Tag mindestens einmal haben)
        models_with_tag = sum(
            1 for anns in model_outputs.values()
            if any(a.get("tag") == tag for a in anns)
        )
        if models_with_tag >= min_votes:
            # Konsens-Quote: wähle Quote mit höchster Durchschnitts-Ähnlichkeit zu allen anderen
            best_quote = max(quotes, key=lambda q: sum(similarity(q, other) for other in quotes))
            pseudo_gt.append({"tag": tag, "quote": best_quote})

    return pseudo_gt


# ─────────────────────────────────────────────
# Bootstrap Konfidenzintervall
# ─────────────────────────────────────────────

def bootstrap_ci(values: list[float], n: int = BOOTSTRAP_N, seed: int = RANDOM_SEED) -> tuple[float, float]:
    """95%-Konfidenzintervall via Bootstrap."""
    rng = random.Random(seed)
    if not values:
        return (0.0, 0.0)
    means = [
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(n)
    ]
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return lo, hi


# ─────────────────────────────────────────────
# Daten laden
# ─────────────────────────────────────────────

model_prefixes = load_model_prefixes(base_dir)
all_files      = list(output_dir.rglob("*_extraction.json"))

texts_dict: dict[str, dict[str, list[dict]]] = {}
for f in all_files:
    model_id, text_id = parse_extraction_filename(f, model_prefixes)
    if model_id and text_id:
        texts_dict.setdefault(text_id, {})[model_id] = extract_annotations(f)

# Ground-Truth laden (Dateien müssen text_id.json oder text_id_gt.json heißen)
ground_truth: dict[str, list[dict]] = {}
if gt_dir.exists():
    for gt_file in gt_dir.glob("*.json"):
        text_id = gt_file.stem.replace("_gt", "")
        ground_truth[text_id] = extract_annotations(gt_file)

print(f"Texte geladen:        {len(texts_dict)}")
print(f"Mit Ground-Truth:     {len(ground_truth)}")
print(f"Ohne Ground-Truth:    {len(texts_dict) - len(ground_truth)}")
print(f"Modelle erkannt:      {sorted({m for t in texts_dict.values() for m in t})}\n")


# ─────────────────────────────────────────────
# 1. Paarweise Agreement (Jaccard + Quote-Ähnlichkeit)
#    – nur für Dateien OHNE GT (oder generell als Konsistenz-Metrik)
# ─────────────────────────────────────────────

pairwise_rows = []
# Für individuelle Modell-Scores: score pro (modell, paarung) separat speichern
model_pair_scores: dict[str, list[float]] = defaultdict(list)

for text_id, model_outputs in texts_dict.items():
    models = list(model_outputs.keys())
    for model_a, model_b in combinations(models, 2):
        ml, mr = sorted([model_a, model_b])
        items_a = ann_to_items(model_outputs[model_a])
        items_b = ann_to_items(model_outputs[model_b])

        tags_a = Counter(t for t, _ in items_a)
        tags_b = Counter(t for t, _ in items_b)
        all_tags = set(tags_a) | set(tags_b)

        # Jaccard auf Multiset-Basis (tag-level, ohne Quote)
        intersection = sum(min(tags_a[t], tags_b[t]) for t in all_tags)
        union        = sum(max(tags_a[t], tags_b[t]) for t in all_tags)
        tag_jaccard  = intersection / union if union else 0.0

        # Quote-Similarity nur für gemeinsame Tags
        tp, fp, fn = match_items(items_a, items_b)
        total = tp + fp + fn
        quote_sim = tp / total if total > 0 else 0.0

        quality = (tag_jaccard + quote_sim) / 2

        pairwise_rows.append({
            "Text_File":            text_id,
            "Model_Pair":           f"{ml} vs {mr}",
            "Tag_Jaccard":          round(tag_jaccard, 3),
            "Quote_Similarity":     round(quote_sim, 3),
            "Quality_Score":        round(quality, 3),
        })

        # Individuelle Scores: jedes Modell bekommt seinen eigenen Beitrag
        # (Precision-Perspektive: wie gut trifft A auf B und umgekehrt)
        tp_a, fp_a, fn_a = match_items(items_a, items_b)  # A als "pred", B als "gold"
        tp_b, fp_b, fn_b = match_items(items_b, items_a)

        prf_a = compute_prf(tp_a, fp_a, fn_a)
        prf_b = compute_prf(tp_b, fp_b, fn_b)

        model_pair_scores[model_a].append(prf_a["f1"])
        model_pair_scores[model_b].append(prf_b["f1"])


df_pairwise = pd.DataFrame(pairwise_rows)

print("=" * 70)
print("PAARWEISE AGREEMENT")
print("=" * 70)
print(df_pairwise.to_string(index=False))

avg_pair = (
    df_pairwise.groupby("Model_Pair")[["Tag_Jaccard", "Quote_Similarity", "Quality_Score"]]
    .mean()
    .reset_index()
    .sort_values("Quality_Score", ascending=False)
)
print("\n--- Durchschnitt pro Modellpaar ---")
print(avg_pair.to_string(index=False))


# ─────────────────────────────────────────────
# 2. Ground-Truth Evaluation (Precision / Recall / F1)
# ─────────────────────────────────────────────

gt_rows = []
gt_hallucinated_by_model: dict[str, Counter] = defaultdict(Counter)

for text_id, gt_annotations in ground_truth.items():
    if text_id not in texts_dict:
        continue
    gt_items = ann_to_items(gt_annotations)

    for model_id, model_annotations in texts_dict[text_id].items():
        pred_items = ann_to_items(model_annotations)
        tp, fp, fn = match_items(pred_items, gt_items)
        prf = compute_prf(tp, fp, fn)
        _, _, _, hallucinated_categories = match_annotations_with_hallucinations(model_annotations, gt_annotations)
        gt_hallucinated_by_model[model_id].update(hallucinated_categories)

        # Per-Tag Breakdown
        all_tags = set(t for t, _ in gt_items) | set(t for t, _ in pred_items)
        for tag in all_tags:
            pred_tag = [(t, q) for t, q in pred_items if t == tag]
            gt_tag   = [(t, q) for t, q in gt_items   if t == tag]
            tp_t, fp_t, fn_t = match_items(pred_tag, gt_tag)
            prf_t = compute_prf(tp_t, fp_t, fn_t)
            gt_rows.append({
                "Text_File": text_id,
                "Model":     model_id,
                "Tag":       tag,
                "TP": tp_t, "FP": fp_t, "FN": fn_t,
                "Hallucinated_Count": fp_t,
                "Predicted_Count": tp_t + fp_t,
                "Hallucination_Rate": round(compute_hallucination_rate(tp_t, fp_t), 3),
                **{k: round(v, 3) for k, v in prf_t.items()},
            })

df_gt = pd.DataFrame(gt_rows)

if not df_gt.empty:
    print("\n" + "=" * 70)
    print("GROUND-TRUTH EVALUATION")
    print("=" * 70)

    model_gt_summary = (
        df_gt.groupby("Model")[["TP", "FP", "FN"]]
        .sum()
        .reset_index()
    )
    model_gt_summary[["precision", "recall", "f1"]] = model_gt_summary.apply(
        lambda row: pd.Series(
            compute_prf(int(row["TP"]), int(row["FP"]), int(row["FN"]))
        ),
        axis=1,
    )
    model_gt_summary["Hallucination_Rate"] = model_gt_summary.apply(
        lambda row: compute_hallucination_rate(int(row["TP"]), int(row["FP"])),
        axis=1,
    )
    model_gt_summary = model_gt_summary.sort_values("f1", ascending=False)
    print("\n--- Modell-Gesamt (aggregiert über GT-Dateien) ---")
    print(model_gt_summary.to_string(index=False))

    tag_gt_summary = (
        df_gt.groupby(["Tag", "Model"])[["TP", "FP", "FN"]]
        .sum()
        .reset_index()
    )
    tag_gt_summary[["precision", "recall", "f1"]] = tag_gt_summary.apply(
        lambda row: pd.Series(
            compute_prf(int(row["TP"]), int(row["FP"]), int(row["FN"]))
        ),
        axis=1,
    )
    tag_gt_summary["Hallucination_Rate"] = tag_gt_summary.apply(
        lambda row: compute_hallucination_rate(int(row["TP"]), int(row["FP"])),
        axis=1,
    )
    tag_gt_summary = tag_gt_summary.sort_values(["Tag", "f1"], ascending=[True, False])
    print("\n--- Per-Tag Breakdown ---")
    print(tag_gt_summary.to_string(index=False))

    # Bootstrap CIs für F1 pro Modell
    print("\n--- F1 mit 95%-Konfidenzintervall (Bootstrap) ---")
    for model_id, grp in df_gt.groupby("Model"):
        f1_vals = grp["f1"].tolist()
        mean_f1 = np.mean(f1_vals)
        lo, hi  = bootstrap_ci(f1_vals)
        hallucination_rate = (
            grp["FP"].sum() / (grp["TP"].sum() + grp["FP"].sum())
            if (grp["TP"].sum() + grp["FP"].sum()) > 0
            else 0.0
        )
        print(
            f"  {model_id:<40}  F1={mean_f1:.3f}  "
            f"Hallucination={hallucination_rate:.3f}  95%-CI=[{lo:.3f}, {hi:.3f}]"
        )
else:
    print("\nKeine Ground-Truth-Dateien gefunden – GT-Evaluation übersprungen.")


# ─────────────────────────────────────────────
# 3. Majority-Vote Pseudo-GT für alle Texte
# ─────────────────────────────────────────────

pseudo_gt_rows = []
pseudo_hallucinated_by_model: dict[str, Counter] = defaultdict(Counter)

for text_id, model_outputs in texts_dict.items():
    if len(model_outputs) < 2:
        continue
    pseudo_gt = build_majority_vote(model_outputs)
    if not pseudo_gt:
        continue
    pseudo_items = ann_to_items(pseudo_gt)

    for model_id, model_annotations in model_outputs.items():
        pred_items = ann_to_items(model_annotations)
        tp, fp, fn = match_items(pred_items, pseudo_items)
        prf = compute_prf(tp, fp, fn)
        _, _, _, hallucinated_categories = match_annotations_with_hallucinations(model_annotations, pseudo_gt)
        pseudo_hallucinated_by_model[model_id].update(hallucinated_categories)
        pseudo_gt_rows.append({
            "Text_File": text_id,
            "Model":     model_id,
            "TP": tp, "FP": fp, "FN": fn,
            "Hallucinated_Count": fp,
            "Predicted_Count": tp + fp,
            "Hallucination_Rate": round(compute_hallucination_rate(tp, fp), 3),
            **{k: round(v, 3) for k, v in prf.items()},
        })

df_pseudo = pd.DataFrame(pseudo_gt_rows)

if not df_pseudo.empty:
    print("\n" + "=" * 70)
    print("MAJORITY-VOTE PSEUDO-GT EVALUATION (alle 75 Texte)")
    print("=" * 70)
    pseudo_summary = (
        df_pseudo.groupby("Model")[["TP", "FP", "FN"]]
        .sum()
        .reset_index()
    )
    pseudo_summary[["precision", "recall", "f1"]] = pseudo_summary.apply(
        lambda row: pd.Series(
            compute_prf(int(row["TP"]), int(row["FP"]), int(row["FN"]))
        ),
        axis=1,
    )
    pseudo_summary["Hallucination_Rate"] = pseudo_summary.apply(
        lambda row: compute_hallucination_rate(int(row["TP"]), int(row["FP"])),
        axis=1,
    )
    pseudo_summary["Pseudo_Disagreement_Rate"] = pseudo_summary["Hallucination_Rate"]
    pseudo_summary = pseudo_summary.sort_values("f1", ascending=False)
    if df_gt.empty:
        print(
            "Hinweis: Ohne Ground-Truth ist Hallucination_Rate eine "
            "Pseudo-GT-Konsens-Abweichung (nicht zwingend echte Halluzination im Text)."
        )
    print(pseudo_summary.to_string(index=False))


# ─────────────────────────────────────────────
# 4. Individuelle Modell-Scores (korrigiert)
# ─────────────────────────────────────────────

model_score_rows = []
for model_id, f1_list in model_pair_scores.items():
    lo, hi = bootstrap_ci(f1_list)
    model_score_rows.append({
        "Model":      model_id,
        "Pair_F1":    round(np.mean(f1_list), 3),
        "CI_lo":      round(lo, 3),
        "CI_hi":      round(hi, 3),
    })

df_model_scores = pd.DataFrame(model_score_rows).sort_values("Pair_F1", ascending=True)

print("\n" + "=" * 70)
print("INDIVIDUELLE MODELL-SCORES (Paar-F1, korrigiert)")
print("=" * 70)
print(df_model_scores.to_string(index=False))


# ─────────────────────────────────────────────
# 5. Visualisierung
# ─────────────────────────────────────────────

models_sorted = df_model_scores["Model"].tolist()
n_models = len(models_sorted)

pair_index: dict[tuple[str, str], float] = {}
for _, row in avg_pair.iterrows():
    a, b = row["Model_Pair"].split(" vs ")
    pair_index[(a, b)] = row["Quality_Score"]
    pair_index[(b, a)] = row["Quality_Score"]

heatmap = np.full((n_models, n_models), np.nan)
for i, ma in enumerate(models_sorted):
    for j, mb in enumerate(models_sorted):
        heatmap[i, j] = 1.0 if ma == mb else pair_index.get((ma, mb), np.nan)

cmap = plt.cm.RdYlGn
norm = Normalize(vmin=0, vmax=1)

has_gt  = not df_gt.empty
has_pgt = not df_pseudo.empty
n_plots = 2 + (1 if has_gt else 0) + (1 if has_pgt else 0)

fig = plt.figure(figsize=(7 * n_plots, 8))
gs  = gridspec.GridSpec(1, n_plots, figure=fig)
ax_idx = 0

# — Balkendiagramm: Paar-F1 mit CI —
ax = fig.add_subplot(gs[ax_idx]); ax_idx += 1
colors = cmap(norm(df_model_scores["Pair_F1"]))
bars = ax.barh(df_model_scores["Model"], df_model_scores["Pair_F1"], color=colors)
ax.errorbar(
    df_model_scores["Pair_F1"], df_model_scores["Model"],
    xerr=[
        df_model_scores["Pair_F1"] - df_model_scores["CI_lo"],
        df_model_scores["CI_hi"]   - df_model_scores["Pair_F1"],
    ],
    fmt="none", color="black", capsize=4, linewidth=1.2,
)
ax.set_title("Paar-F1 (± 95%-CI)")
ax.set_xlabel("F1-Score")
ax.set_xlim(0, 1)
ax.grid(axis="x", linestyle="--", alpha=0.3)
for bar, val in zip(bars, df_model_scores["Pair_F1"]):
    ax.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)

# — Heatmap: Paarweises Agreement —
ax = fig.add_subplot(gs[ax_idx]); ax_idx += 1
img = ax.imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
ax.set_title("Paarweises Agreement (Quality Score)")
ax.set_xticks(range(n_models)); ax.set_xticklabels(models_sorted, rotation=45, ha="right")
ax.set_yticks(range(n_models)); ax.set_yticklabels(models_sorted)
for i in range(n_models):
    for j in range(n_models):
        v = heatmap[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

# — GT F1 wenn vorhanden —
if has_gt:
    ax = fig.add_subplot(gs[ax_idx]); ax_idx += 1
    gt_plot = model_gt_summary.sort_values("f1")
    colors_gt = cmap(norm(gt_plot["f1"]))
    ax.barh(gt_plot["Model"], gt_plot["f1"], color=colors_gt)
    ax.barh(gt_plot["Model"], gt_plot["precision"], color="none", edgecolor="steelblue", linestyle="--", label="Precision")
    ax.barh(gt_plot["Model"], gt_plot["recall"],    color="none", edgecolor="salmon",   linestyle=":",  label="Recall")
    ax.set_title("Ground-Truth F1 / P / R")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

# — Pseudo-GT F1 wenn vorhanden —
if has_pgt:
    ax = fig.add_subplot(gs[ax_idx]); ax_idx += 1
    pgt_plot = pseudo_summary.sort_values("f1")
    colors_pgt = cmap(norm(pgt_plot["f1"]))
    ax.barh(pgt_plot["Model"], pgt_plot["f1"], color=colors_pgt)
    ax.set_title("Majority-Vote Pseudo-GT F1")
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for _, row in pgt_plot.iterrows():
        ax.text(min(row["f1"] + 0.01, 0.97), pgt_plot["Model"].tolist().index(row["Model"]),
                f"{row['f1']:.2f}", va="center", fontsize=8)

fig.suptitle("LLM Annotation Quality – Strafzumessungsumstände", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
print(f"\nVisualisierung gespeichert: {figure_path}")

# — Separate Halluzinationsgrafik —
if has_gt or has_pgt:
    if has_gt:
        hallucination_plot = model_gt_summary.sort_values("Hallucination_Rate")
        hallucination_categories = gt_hallucinated_by_model
        hallucination_title = "Zero-Shot Halluzination (Ground Truth)"
        hallucination_rate_col = "Hallucination_Rate"
        hallucination_x_label = "Halluzinationsrate"
    else:
        hallucination_plot = pseudo_summary.sort_values("Pseudo_Disagreement_Rate")
        hallucination_categories = pseudo_hallucinated_by_model
        hallucination_title = "Zero-Shot Pseudo-GT Abweichung"
        hallucination_rate_col = "Pseudo_Disagreement_Rate"
        hallucination_x_label = "Abweichungsrate zu Pseudo-GT"

    category_names = sorted({cat for counts in hallucination_categories.values() for cat in counts.keys()})
    if not category_names:
        category_names = ["Unbekannt"]

    category_totals = Counter()
    for counts in hallucination_categories.values():
        category_totals.update(counts)
    category_order = [cat for cat, _ in category_totals.most_common()]
    if not category_order:
        category_order = category_names

    category_colors = plt.cm.tab20(np.linspace(0, 1, max(len(category_order), 1)))

    hallu_fig, (hallu_ax_rate, hallu_ax_cat) = plt.subplots(
        2,
        1,
        figsize=(11, max(6, 0.45 * len(hallucination_plot) + 4)),
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    colors_hall = plt.cm.OrRd(norm(1 - hallucination_plot[hallucination_rate_col]))
    hallu_ax_rate.barh(hallucination_plot["Model"], hallucination_plot[hallucination_rate_col], color=colors_hall)
    hallu_ax_rate.set_title(hallucination_title)
    hallu_ax_rate.set_xlabel(hallucination_x_label)
    hallu_ax_rate.set_xlim(0, 1)
    hallu_ax_rate.grid(axis="x", linestyle="--", alpha=0.3)
    for idx, (_, row) in enumerate(hallucination_plot.iterrows()):
        hallu_ax_rate.text(
            min(row[hallucination_rate_col] + 0.01, 0.97),
            idx,
            f"{row[hallucination_rate_col]:.2f}",
            va="center",
            fontsize=8,
        )

    left_offsets = np.zeros(len(hallucination_plot))
    model_names = hallucination_plot["Model"].tolist()
    for cat_idx, category in enumerate(category_order):
        category_values = np.array([
            hallucination_categories[model].get(category, 0)
            for model in model_names
        ])
        hallu_ax_cat.barh(
            model_names,
            category_values,
            left=left_offsets,
            color=category_colors[cat_idx % len(category_colors)],
            label=category,
        )
        left_offsets += category_values

    hallu_ax_cat.set_title("Halluzinierte Annotationen nach Category")
    hallu_ax_cat.set_xlabel("Anzahl halluzinierter Annotationen")
    hallu_ax_cat.grid(axis="x", linestyle="--", alpha=0.3)
    hallu_ax_cat.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")

    hallu_fig.tight_layout()
    hallu_fig.savefig(hallucination_figure_path, dpi=200, bbox_inches="tight")
    print(f"Separate Halluzinationsgrafik gespeichert: {hallucination_figure_path}")