#!/usr/bin/env python3
"""
Auswertung der KI-Modell-Annotationen (Zero-Shot)
===================================================
Metriken: Precision, Recall, F1-Score, MCC, Cohen's Kappa
Analyse:  Paarweises IAA, Tag-Konsens, Zitat-Matching, Fehlertypen
Ground Truth: Optional (Pfad in GROUND_TRUTH_DIR setzen)

Ausgabe: Konsole + auswertung_ergebnis.xlsx + auswertung_diagramme/
"""

import json
import re
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from difflib import SequenceMatcher
import math

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # kein Display nötig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ============================================================
# Konfiguration
# ============================================================

ANNOTATION_DIR   = Path("annotationen_uni_models_zero_shot")
GROUND_TRUTH_DIR = None          # z.B. Path("ground_truth") – Ordner mit {text_id}.json
OUTPUT_FILE      = Path("auswertung_ergebnis.xlsx")
QUOTE_SIM_THRESHOLD = 0.5        # Ab welcher Ähnlichkeit gilt ein Zitat als "Match"

# Wird zur Laufzeit aus den vorhandenen Dateien befüllt (leer lassen)
KNOWN_MODELS: list[str] = []
MODEL_SHORT:  dict[str, str] = {}
MODEL_COLORS: dict[str, str] = {}

# Normalisierung von Varianten (Umlaute, Großschreibung)
TAG_ALIASES = {
    "Geständnis": "Gestaendnis",
}
CATEGORY_ALIASES = {
    "Abwaegung":  "Abwaegung",
    "Abwaegung":  "Abwaegung",
    "Schaerfend": "Schaerfend",
    "Schaerfend": "Schaerfend",
}

# ============================================================
# Hilfsfunktionen
# ============================================================

def normalize_tag(tag: str) -> str:
    return TAG_ALIASES.get(tag, tag)

def normalize_category(cat: str) -> str:
    return CATEGORY_ALIASES.get(cat, cat)

def quote_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def parse_filename(stem: str) -> tuple[str, str]:
    """
    Trennt Modellname und Text-ID automatisch.
    Alle Texte beginnen mit GesR_ oder LG_, daher wird am ersten
    Vorkommen von _(GesR_|LG_) gesplittet.
    """
    stem = stem.replace("_extraction", "")
    match = re.search(r'_((?:GesR|LG)_)', stem)
    if match:
        return stem[:match.start()], stem[match.start() + 1:]
    # Fallback: erster Unterstrich
    parts = stem.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (stem, "")

def load_annotations(filepath: Path) -> list[dict]:
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        result = []
        for a in data.get("annotations", []):
            tag = normalize_tag(a.get("tag", "").strip())
            cat = normalize_category(a.get("category", "").strip())
            quote = a.get("quote", "").strip()
            if tag:
                result.append({"tag": tag, "category": cat, "quote": quote})
        return result
    except Exception as e:
        print(f"  [WARN] {filepath.name}: {e}")
        return []

def load_all_annotations(directory: Path) -> dict[str, dict[str, list]]:
    """Gibt {text_id: {model_id: [annotations]}} zurück und befüllt KNOWN_MODELS."""
    global KNOWN_MODELS, MODEL_SHORT, MODEL_COLORS
    result: dict[str, dict] = defaultdict(dict)
    for f in sorted(directory.glob("*_extraction.json")):
        model_id, text_id = parse_filename(f.stem)
        if text_id:  # ungültige Dateinamen überspringen
            result[text_id][model_id] = load_annotations(f)

    # Modelle alphabetisch sortieren
    KNOWN_MODELS = sorted({m for models in result.values() for m in models})

    # Kurznamen: letzter Teil nach "_", max. 14 Zeichen
    def make_short(name: str) -> str:
        parts = name.replace("-", "_").split("_")
        # Suche nach dem informativen Teil (Zahl im Namen = Parametergröße)
        for i, p in enumerate(parts):
            if any(c.isdigit() for c in p):
                return "_".join(parts[max(0, i-1):i+2])[:14]
        return name[:14]

    MODEL_SHORT  = {m: make_short(m) for m in KNOWN_MODELS}

    # Farben: seaborn-Palette mit genug Farben für beliebig viele Modelle
    palette = sns.color_palette("tab10", len(KNOWN_MODELS))
    MODEL_COLORS = {m: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                    for m, (r, g, b) in zip(KNOWN_MODELS, palette)}

    return dict(result)

def shorten(text: str, n: int = 50) -> str:
    return text[:n] + "…" if len(text) > n else text

# ============================================================
# Tag-Präsenz-Matrix  {model -> {tag -> 0/1}}
# ============================================================

def build_presence_matrix(model_anns: dict[str, list], all_tags: list[str]) -> dict[str, list[int]]:
    matrix = {}
    for model, anns in model_anns.items():
        used = {a["tag"] for a in anns}
        matrix[model] = [int(t in used) for t in all_tags]
    return matrix

# ============================================================
# Metriken (ohne externe Abhängigkeit außer numpy)
# ============================================================

def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    denom     = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc       = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                Precision=round(precision, 4), Recall=round(recall, 4),
                F1=round(f1, 4), MCC=round(mcc, 4))

def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """Cohen's Kappa für zwei binäre Annotator-Vektoren."""
    n = len(y1)
    if n == 0:
        return 0.0
    p_obs = sum(a == b for a, b in zip(y1, y2)) / n
    p1_a  = sum(y1) / n
    p1_b  = sum(y2) / n
    p_exp = p1_a * p1_b + (1 - p1_a) * (1 - p1_b)
    if p_exp == 1.0:
        return 1.0 if p_obs == 1.0 else 0.0
    return round((p_obs - p_exp) / (1 - p_exp), 4)

# ============================================================
# 1. Modell-Übersicht
# ============================================================

def compute_model_summary(texts_data: dict, all_tags: list[str]) -> pd.DataFrame:
    stats: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for model_anns in texts_data.values():
        for model, anns in model_anns.items():
            stats[model]["Texte"] += 1
            stats[model]["Annotationen_gesamt"] += len(anns)
            for a in anns:
                stats[model][f"#{a['tag']}"] += 1
                stats[model][f"@{a['category']}"] += 1

    rows = []
    for model in KNOWN_MODELS:
        s = stats[model]
        row = {
            "Modell":               MODEL_SHORT.get(model, model),
            "Texte":                s["Texte"],
            "Annotationen_gesamt":  s["Annotationen_gesamt"],
            "Ø_pro_Text":           round(s["Annotationen_gesamt"] / max(s["Texte"], 1), 1),
        }
        for t in all_tags:
            row[t] = s.get(f"#{t}", 0)
        rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# 2. Paarweises Inter-Annotator Agreement
# ============================================================

def compute_pairwise_iaa(texts_data: dict, all_tags: list[str]) -> pd.DataFrame:
    rows = []
    for text_id, model_anns in texts_data.items():
        models = [m for m in KNOWN_MODELS if m in model_anns]
        matrix = build_presence_matrix(model_anns, all_tags)

        for m_a, m_b in combinations(models, 2):
            va = matrix[m_a]
            vb = matrix[m_b]
            kappa  = cohen_kappa(va, vb)
            m_ab   = binary_metrics(va, vb)   # A=Referenz, B=Vorhersage
            m_ba   = binary_metrics(vb, va)   # B=Referenz, A=Vorhersage

            shared = sum(a == 1 and b == 1 for a, b in zip(va, vb))
            only_a = sum(a == 1 and b == 0 for a, b in zip(va, vb))
            only_b = sum(a == 0 and b == 1 for a, b in zip(va, vb))

            rows.append({
                "Text":                shorten(text_id, 45),
                "Modell_A":            MODEL_SHORT.get(m_a, m_a),
                "Modell_B":            MODEL_SHORT.get(m_b, m_b),
                "Cohen_Kappa":         kappa,
                "F1_(A_ref)":          m_ab["F1"],
                "Precision_(B_pred)":  m_ab["Precision"],
                "Recall_(B_pred)":     m_ab["Recall"],
                "F1_(B_ref)":          m_ba["F1"],
                "Precision_(A_pred)":  m_ba["Precision"],
                "Recall_(A_pred)":     m_ba["Recall"],
                "MCC":                 m_ab["MCC"],
                "Geteilte_Tags":       shared,
                "Nur_A":               only_a,
                "Nur_B":               only_b,
            })
    return pd.DataFrame(rows)

# ============================================================
# 3. Tag-Konsens-Analyse
# ============================================================

def compute_tag_consensus(texts_data: dict, all_tags: list[str]) -> pd.DataFrame:
    rows = []
    for text_id, model_anns in texts_data.items():
        models = [m for m in KNOWN_MODELS if m in model_anns]
        matrix = build_presence_matrix(model_anns, all_tags)

        for tag in all_tags:
            votes    = {m: matrix[m][all_tags.index(tag)] for m in models}
            n_used   = sum(votes.values())
            n_models = len(models)

            if   n_used == n_models: konsens = "Alle"
            elif n_used == 0:        konsens = "Keiner"
            else:                    konsens = f"{n_used}/{n_models}"

            row = {
                "Text":          shorten(text_id, 45),
                "Tag":           tag,
                "Konsens":       konsens,
                "Anzahl_Modelle": n_used,
            }
            for m in models:
                row[f"Nutzt_{MODEL_SHORT.get(m, m)}"] = "✓" if votes[m] else ""
                if votes[m]:
                    quotes = [a["quote"] for a in model_anns[m] if a["tag"] == tag]
                    row[f"Zitat_{MODEL_SHORT.get(m, m)}"] = " | ".join(q[:100] for q in quotes[:2])
                else:
                    row[f"Zitat_{MODEL_SHORT.get(m, m)}"] = ""
            rows.append(row)
    return pd.DataFrame(rows)

# ============================================================
# 4. Zitat-Matching-Analyse
# ============================================================

def compute_quote_matching(texts_data: dict) -> pd.DataFrame:
    rows = []
    for text_id, model_anns in texts_data.items():
        models = [m for m in KNOWN_MODELS if m in model_anns]

        all_used_tags = sorted({a["tag"] for anns in model_anns.values() for a in anns})
        for tag in all_used_tags:
            for m_a, m_b in combinations(models, 2):
                qa_list = [a["quote"] for a in model_anns.get(m_a, []) if a["tag"] == tag]
                qb_list = [a["quote"] for a in model_anns.get(m_b, []) if a["tag"] == tag]

                if not qa_list and not qb_list:
                    continue

                if not qa_list or not qb_list:
                    status = "Nur_A" if qa_list else "Nur_B"
                    rows.append({
                        "Text":           shorten(text_id, 45),
                        "Tag":            tag,
                        "Modell_A":       MODEL_SHORT.get(m_a, m_a),
                        "Modell_B":       MODEL_SHORT.get(m_b, m_b),
                        "Status":         status,
                        "Ähnlichkeit":    0.0,
                        "Zitat_A":        (qa_list[0][:100] if qa_list else ""),
                        "Zitat_B":        (qb_list[0][:100] if qb_list else ""),
                    })
                    continue

                # Beste paarweise Ähnlichkeit
                best_sim, best_qa, best_qb = 0.0, "", ""
                for qa in qa_list:
                    for qb in qb_list:
                        s = quote_similarity(qa, qb)
                        if s > best_sim:
                            best_sim, best_qa, best_qb = s, qa, qb

                status = "Match" if best_sim >= QUOTE_SIM_THRESHOLD else "Divergent"
                rows.append({
                    "Text":        shorten(text_id, 45),
                    "Tag":         tag,
                    "Modell_A":    MODEL_SHORT.get(m_a, m_a),
                    "Modell_B":    MODEL_SHORT.get(m_b, m_b),
                    "Status":      status,
                    "Ähnlichkeit": round(best_sim, 3),
                    "Zitat_A":     best_qa[:100],
                    "Zitat_B":     best_qb[:100],
                })
    return pd.DataFrame(rows)

# ============================================================
# 5. Qualitative Fehleranalyse
# ============================================================

def compute_error_analysis(texts_data: dict, all_tags: list[str]) -> pd.DataFrame:
    rows = []
    for text_id, model_anns in texts_data.items():
        models = [m for m in KNOWN_MODELS if m in model_anns]
        matrix = build_presence_matrix(model_anns, all_tags)

        for i, tag in enumerate(all_tags):
            votes   = [matrix[m][i] for m in models]
            n_used  = sum(votes)
            if n_used == 0:
                continue

            if n_used == len(models):
                fehlertyp = "Konsens"
            elif n_used == 1:
                einzel = MODEL_SHORT.get(models[votes.index(1)], models[votes.index(1)])
                fehlertyp = f"Einzelmeinung [{einzel}]"
            else:
                mehrheit = [MODEL_SHORT.get(models[j], models[j]) for j, v in enumerate(votes) if v]
                fehlertyp = f"Teilkonsens [{', '.join(mehrheit)}]"

            # Zitat-Divergenz bei gemeinsamen Tags
            zitat_info = ""
            if n_used > 1:
                active_models = [models[j] for j, v in enumerate(votes) if v]
                if len(active_models) >= 2:
                    q_a = [a["quote"] for a in model_anns[active_models[0]] if a["tag"] == tag]
                    q_b = [a["quote"] for a in model_anns[active_models[1]] if a["tag"] == tag]
                    if q_a and q_b:
                        best = max(
                            quote_similarity(qa, qb)
                            for qa in q_a for qb in q_b
                        )
                        if best < QUOTE_SIM_THRESHOLD:
                            zitat_info = f"Zitat-Divergenz (Sim={best:.2f})"

            rows.append({
                "Text":           shorten(text_id, 45),
                "Tag":            tag,
                "Fehlertyp":      fehlertyp,
                "Zitat_Hinweis":  zitat_info,
                "Modelle_gesamt": f"{n_used}/{len(models)}",
                **{
                    MODEL_SHORT.get(models[j], models[j]): ("✓" if votes[j] else "")
                    for j in range(len(models))
                },
            })
    return pd.DataFrame(rows)

# ============================================================
# 6. Evaluation gegen Ground Truth (optional)
# ============================================================

def compute_ground_truth_eval(
    texts_data: dict,
    gt_data: dict[str, list],
    all_tags: list[str],
) -> pd.DataFrame:
    """
    gt_data: {text_id -> [{"tag": ..., "category": ..., "quote": ...}]}
    """
    rows = []
    for text_id, model_anns in texts_data.items():
        gt_anns = gt_data.get(text_id, [])
        gt_tags = {normalize_tag(a.get("tag", "")) for a in gt_anns}
        gt_vec  = [int(t in gt_tags) for t in all_tags]

        for model in KNOWN_MODELS:
            if model not in model_anns:
                continue
            pred_tags = {a["tag"] for a in model_anns[model]}
            pred_vec  = [int(t in pred_tags) for t in all_tags]
            m = binary_metrics(gt_vec, pred_vec)
            rows.append({
                "Text":      shorten(text_id, 45),
                "Modell":    MODEL_SHORT.get(model, model),
                **m,
            })

    # Tag-spezifische Metriken
    tag_rows = []
    for text_id, model_anns in texts_data.items():
        gt_anns = gt_data.get(text_id, [])
        gt_tags = {normalize_tag(a.get("tag", "")) for a in gt_anns}

        for model in KNOWN_MODELS:
            if model not in model_anns:
                continue
            pred_tags = {a["tag"] for a in model_anns[model]}
            for tag in all_tags:
                gt_val   = int(tag in gt_tags)
                pred_val = int(tag in pred_tags)
                tag_rows.append({
                    "Text":   shorten(text_id, 45),
                    "Modell": MODEL_SHORT.get(model, model),
                    "Tag":    tag,
                    "GT":     gt_val,
                    "Pred":   pred_val,
                    "Correct": int(gt_val == pred_val),
                    "TP":     int(gt_val == 1 and pred_val == 1),
                    "FP":     int(gt_val == 0 and pred_val == 1),
                    "FN":     int(gt_val == 1 and pred_val == 0),
                    "TN":     int(gt_val == 0 and pred_val == 0),
                })

    return pd.DataFrame(rows), pd.DataFrame(tag_rows)

# ============================================================
# 7. MCC & F1 pro Tag (paarweise als Pseudo-GT)
# ============================================================

def compute_per_tag_metrics(texts_data: dict, all_tags: list[str]) -> pd.DataFrame:
    """
    Für jeden Tag über alle Texte: paarweise F1 / MCC zwischen den Modellen.
    Interpretierbar als: Wie stabil ist die Tag-Erkennung pro Modellpaar?
    """
    # Vektoren über alle Texte zusammenführen
    vecs: dict[str, list[int]] = {m: [] for m in KNOWN_MODELS}
    for text_id, model_anns in sorted(texts_data.items()):
        matrix = build_presence_matrix(model_anns, all_tags)
        for m in KNOWN_MODELS:
            if m in matrix:
                vecs[m].extend(matrix[m])
            else:
                vecs[m].extend([0] * len(all_tags))

    rows = []
    for tag_idx, tag in enumerate(all_tags):
        for m_a, m_b in combinations(KNOWN_MODELS, 2):
            # Tag-spezifische Vektoren (alle Texte)
            n_texts = len(texts_data)
            va = vecs[m_a][tag_idx::len(all_tags)]
            vb = vecs[m_b][tag_idx::len(all_tags)]
            m = binary_metrics(va, vb)
            kappa = cohen_kappa(va, vb)
            rows.append({
                "Tag":         tag,
                "Modell_A":    MODEL_SHORT.get(m_a, m_a),
                "Modell_B":    MODEL_SHORT.get(m_b, m_b),
                "Cohen_Kappa": kappa,
                "F1":          m["F1"],
                "MCC":         m["MCC"],
                "TP":          m["TP"],
                "FP":          m["FP"],
                "FN":          m["FN"],
                "TN":          m["TN"],
            })
    return pd.DataFrame(rows)

# ============================================================
# Diagramme
# ============================================================

# Einheitliches Farbschema für die drei Modelle
def _save(fig: plt.Figure, path: Path, title: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path.name}  –  {title}")


def create_diagrams(
    out_dir: Path,
    df_summary: pd.DataFrame,
    df_iaa: pd.DataFrame,
    iaa_agg: pd.DataFrame,
    df_consensus: pd.DataFrame,
    df_quotes: pd.DataFrame,
    df_tag_metrics: pd.DataFrame,
    tag_agg: pd.DataFrame,
    all_tags: list[str],
    df_gt_overall: pd.DataFrame | None = None,
):
    out_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    short_tags = [t.replace("_", "\n") for t in all_tags]   # Zeilenumbruch in Achsenbeschriftung

    # ----------------------------------------------------------
    # Diagramm 1 · Heatmap: Tag-Konsens pro Text
    # Zeigt für jede (Text × Tag)-Kombination, wie viele Modelle
    # den Tag genutzt haben (0 = keiner, 3 = alle).
    # ----------------------------------------------------------
    pivot = df_consensus.pivot_table(
        index="Tag", columns="Text", values="Anzahl_Modelle", aggfunc="sum"
    ).fillna(0).astype(int)
    # Kurze Textnamen für die Spalten
    pivot.columns = [c[:30] + "…" if len(c) > 30 else c for c in pivot.columns]
    pivot.index   = [i.replace("_", " ") for i in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.8), max(5, len(pivot) * 0.55)))
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt="d", cmap=cmap,
        vmin=0, vmax=3, linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"label": "Anzahl Modelle", "ticks": [0, 1, 2, 3]},
    )
    ax.set_title("Diagramm 1 · Tag-Konsens pro Text\n(0 = kein Modell, 3 = alle Modelle)", fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=9)
    _save(fig, out_dir / "D1_tag_konsens_heatmap.png", "Tag-Konsens-Heatmap")

    # ----------------------------------------------------------
    # Diagramm 2 · Grouped Bar: Annotationsanzahl pro Modell & Tag
    # Vergleicht, wie viele Annotationen je Modell pro Tag-Typ
    # über alle 5 Texte vergeben wurden.
    # ----------------------------------------------------------
    tag_cols = [c for c in df_summary.columns if c in all_tags]
    df_melt = df_summary[["Modell"] + tag_cols].melt(
        id_vars="Modell", var_name="Tag", value_name="Anzahl"
    )
    df_melt["Tag_kurz"] = df_melt["Tag"].str.replace("_", "\n")

    fig, ax = plt.subplots(figsize=(max(10, len(tag_cols) * 0.85), 5))
    x    = np.arange(len(tag_cols))
    w    = 0.26
    models_in_summary = df_summary["Modell"].tolist()
    for i, model in enumerate(models_in_summary):
        vals = df_melt[df_melt["Modell"] == model]["Anzahl"].values
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=model,
                      color=MODEL_COLORS.get(model, list(MODEL_COLORS.values())[i % max(len(MODEL_COLORS), 1)]),
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(int(v)), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tag_cols], fontsize=8)
    ax.set_ylabel("Anzahl Annotationen (gesamt)")
    ax.set_title("Diagramm 2 · Annotationsanzahl pro Modell und Tag", fontsize=12)
    ax.legend(title="Modell", fontsize=9)
    ax.set_ylim(0, df_melt["Anzahl"].max() * 1.2 + 1)
    _save(fig, out_dir / "D2_annotationen_pro_tag.png", "Annotationsanzahl pro Modell & Tag")

    # ----------------------------------------------------------
    # Diagramm 3 · Balken: IAA-Metriken pro Modellpaar
    # Zeigt Cohen's Kappa, F1 und MCC im direkten Vergleich
    # der drei Modellpaare (aggregiert über alle 5 Texte).
    # ----------------------------------------------------------
    pairs = iaa_agg["Modell_A"] + "\nvs\n" + iaa_agg["Modell_B"]
    metrics = ["Cohen_Kappa", "F1_(A_ref)", "MCC"]
    metric_labels = ["Cohen's κ", "F1", "MCC"]
    bar_colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(pairs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, len(pairs) * 2), 5))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, bar_colors)):
        vals = iaa_agg[metric].values
        bars = ax.bar(x + (i - 1) * w, vals, width=w, label=label,
                      color=color, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height() + 0.01, 0.02),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.6, color="gray", linestyle="--", linewidth=0.8, label="κ = 0.6 (gute Übereinstimmung)")
    ax.set_ylabel("Metrik-Wert (0–1)")
    ax.set_title("Diagramm 3 · IAA-Metriken pro Modellpaar\n(gemittelt über alle 5 Texte)", fontsize=12)
    ax.legend(fontsize=9)
    _save(fig, out_dir / "D3_iaa_modellpaar.png", "IAA-Metriken pro Modellpaar")

    # ----------------------------------------------------------
    # Diagramm 4 · Heatmap: Tag-Stabilität (F1 pro Tag × Modellpaar)
    # Zeigt, für welche Tags die Modelle zuverlässig übereinstimmen
    # und wo die größten Unterschiede auftreten.
    # ----------------------------------------------------------
    df_tag_metrics["Paar"] = df_tag_metrics["Modell_A"] + " vs " + df_tag_metrics["Modell_B"]
    pivot_f1 = df_tag_metrics.pivot_table(
        index="Tag", columns="Paar", values="F1", aggfunc="mean"
    ).fillna(0)
    pivot_f1.index = [i.replace("_", " ") for i in pivot_f1.index]

    fig, ax = plt.subplots(figsize=(max(7, len(pivot_f1.columns) * 2.2), max(5, len(pivot_f1) * 0.55)))
    sns.heatmap(
        pivot_f1, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1, linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"label": "F1-Score"},
    )
    ax.set_title("Diagramm 4 · Tag-Stabilität: F1-Score pro Tag und Modellpaar\n"
                 "(1.0 = vollständige Übereinstimmung, 0.0 = keine)", fontsize=12, pad=10)
    ax.set_xlabel("Modellpaar")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=15, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=9)
    _save(fig, out_dir / "D4_tag_stabilitaet_heatmap.png", "Tag-Stabilität-Heatmap")

    # ----------------------------------------------------------
    # Diagramm 5 · Stacked Bar: Fehlertypen pro Text
    # Zeigt die Verteilung von Konsens, Teilkonsens und
    # Einzelmeinungen für jeden der 5 Texte.
    # ----------------------------------------------------------
    konsens_data = df_consensus.copy()
    konsens_data["Typ"] = konsens_data["Konsens"].map(
        lambda k: "Keiner" if k == "Keiner" else
                  ("Alle"  if k == "Alle"  else
                   ("Teilkonsens" if "/" in k else "Einzelmeinung"))
    )
    # Nur Fälle wo mindestens ein Modell den Tag nutzt
    konsens_data = konsens_data[konsens_data["Typ"] != "Keiner"]

    typ_order  = ["Alle", "Teilkonsens", "Einzelmeinung"]
    typ_colors = {"Alle": "#55A868", "Teilkonsens": "#FFC857", "Einzelmeinung": "#E84855"}

    grouped = (
        konsens_data.groupby(["Text", "Typ"])["Tag"]
        .count()
        .unstack(fill_value=0)
        .reindex(columns=typ_order, fill_value=0)
    )
    grouped.index = [i[:35] + "…" if len(i) > 35 else i for i in grouped.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(grouped) * 0.8)))
    bottom = np.zeros(len(grouped))
    for typ in typ_order:
        if typ in grouped.columns:
            vals = grouped[typ].values
            ax.barh(range(len(grouped)), vals, left=bottom,
                    label=typ, color=typ_colors[typ], edgecolor="white")
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(b + v / 2, j, str(int(v)), ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += vals

    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped.index, fontsize=9)
    ax.set_xlabel("Anzahl (Tag × Text)")
    ax.set_title("Diagramm 5 · Fehlertypen pro Text\n"
                 "(Konsens / Teilkonsens / Einzelmeinung)", fontsize=12)
    ax.legend(title="Typ", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.invert_yaxis()
    _save(fig, out_dir / "D5_fehlertypen_pro_text.png", "Fehlertypen pro Text")

    # ----------------------------------------------------------
    # Diagramm 6 · Violin/Strip: Zitat-Ähnlichkeit pro Modellpaar
    # Zeigt die Verteilung der Fuzzy-Ähnlichkeitswerte für
    # Fälle, bei denen beide Modelle denselben Tag gesetzt haben.
    # Niedrige Werte = gleicher Tag, aber anderer Textausschnitt.
    # ----------------------------------------------------------
    shared = df_quotes[df_quotes["Status"].isin(["Match", "Divergent"])].copy()
    if not shared.empty:
        shared["Paar"] = shared["Modell_A"] + "\nvs\n" + shared["Modell_B"]
        pair_order = sorted(shared["Paar"].unique())

        _pal = sns.color_palette("tab10", len(pair_order))
        pair_colors = {p: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                       for p, (r, g, b) in zip(pair_order, _pal)}
        fig, ax = plt.subplots(figsize=(max(7, len(pair_order) * 2.2), 5))
        sns.violinplot(
            data=shared, x="Paar", y="Ähnlichkeit", hue="Paar",
            order=pair_order, hue_order=pair_order,
            palette=pair_colors, inner="box", ax=ax, cut=0, legend=False,
        )
        # Datenpunkte überlagern
        sns.stripplot(
            data=shared, x="Paar", y="Ähnlichkeit", order=pair_order,
            color="black", size=3, alpha=0.4, ax=ax,
        )
        ax.axhline(QUOTE_SIM_THRESHOLD, color="red", linestyle="--", linewidth=1,
                   label=f"Match-Schwelle ({QUOTE_SIM_THRESHOLD})")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Zitat-Ähnlichkeit (SequenceMatcher)")
        ax.set_xlabel("")
        ax.set_title("Diagramm 6 · Verteilung der Zitat-Ähnlichkeit pro Modellpaar\n"
                     "(nur Tags die beide Modelle gesetzt haben)", fontsize=12)
        ax.legend(fontsize=9)
        _save(fig, out_dir / "D6_zitat_aehnlichkeit.png", "Zitat-Ähnlichkeitsverteilung")
    else:
        print("  [skip] D6: keine gemeinsamen Tags zum Vergleich gefunden.")

    # ----------------------------------------------------------
    # Diagramm 7 · Heatmap: Kappa pro Tag × Modellpaar
    # Ergänzt Diagramm 4 (F1) um Cohen's Kappa als robustere
    # Übereinstimmungsmetrik – besonders bei seltenen Tags.
    # ----------------------------------------------------------
    pivot_kappa = df_tag_metrics.pivot_table(
        index="Tag", columns="Paar", values="Cohen_Kappa", aggfunc="mean"
    ).fillna(0)
    pivot_kappa.index = [i.replace("_", " ") for i in pivot_kappa.index]

    fig, ax = plt.subplots(figsize=(max(7, len(pivot_kappa.columns) * 2.2), max(5, len(pivot_kappa) * 0.55)))
    sns.heatmap(
        pivot_kappa, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"label": "Cohen's κ"},
    )
    ax.set_title("Diagramm 7 · Cohen's Kappa pro Tag und Modellpaar\n"
                 "(robuster bei seltenen Tags; 1=perfekt, 0=zufällig, <0=schlechter als Zufall)",
                 fontsize=12, pad=10)
    ax.set_xlabel("Modellpaar")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=15, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=9)
    _save(fig, out_dir / "D7_kappa_pro_tag.png", "Kappa pro Tag und Modellpaar")

    # ----------------------------------------------------------
    # Diagramm 8 · Ground Truth (nur wenn vorhanden)
    # Balkendiagramm: Precision / Recall / F1 / MCC pro Modell
    # ----------------------------------------------------------
    if df_gt_overall is not None:
        gt_agg = (
            df_gt_overall.groupby("Modell")[["Precision", "Recall", "F1", "MCC"]]
            .mean()
            .round(4)
            .reset_index()
        )
        gt_melt = gt_agg.melt(id_vars="Modell", var_name="Metrik", value_name="Wert")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=gt_melt, x="Metrik", y="Wert", hue="Modell",
                    palette=list(MODEL_COLORS.values()), ax=ax)
        ax.set_ylim(0, 1.1)
        ax.set_title("Diagramm 8 · Evaluation gegen Ground Truth\n"
                     "Precision / Recall / F1 / MCC pro Modell", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Wert (0–1)")
        ax.legend(title="Modell", fontsize=9)
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        _save(fig, out_dir / "D8_ground_truth_eval.png", "Ground-Truth-Evaluation")

    print(f"\n  Alle Diagramme gespeichert in: {out_dir.resolve()}/")


# ============================================================
# Hauptprogramm
# ============================================================

def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)

def main():
    print_section("KI-Modell Annotationsauswertung – Zero-Shot")

    # --- Daten laden ---
    texts_data = load_all_annotations(ANNOTATION_DIR)
    if not texts_data:
        print(f"FEHLER: Keine Dateien in {ANNOTATION_DIR} gefunden.")
        return

    print(f"\n{len(texts_data)} Texte, {len(KNOWN_MODELS)} Modelle\n")
    for text_id, models in sorted(texts_data.items()):
        total_ann = sum(len(a) for a in models.values())
        print(f"  {shorten(text_id, 55):<57} {total_ann:>3} Annot. ({len(models)} Modelle)")

    # Alle normalisierten Tags (Gesamtvokabular)
    all_tags = sorted({
        a["tag"]
        for model_anns in texts_data.values()
        for anns in model_anns.values()
        for a in anns
    })
    print(f"\n{len(all_tags)} Tags im Vokabular: {', '.join(all_tags)}")

    # --- Berechnungen ---
    print_section("1 · Modell-Übersicht")
    df_summary = compute_model_summary(texts_data, all_tags)
    disp_cols = ["Modell", "Texte", "Annotationen_gesamt", "Ø_pro_Text"]
    print(df_summary[disp_cols].to_string(index=False))

    print_section("2 · Paarweises Inter-Annotator Agreement (IAA)")
    df_iaa = compute_pairwise_iaa(texts_data, all_tags)

    # Aggregiert über alle Texte
    iaa_agg = (
        df_iaa.groupby(["Modell_A", "Modell_B"])[
            ["Cohen_Kappa", "F1_(A_ref)", "MCC"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    print("\n--- Durchschnitt über alle Texte ---")
    print(iaa_agg.to_string(index=False))
    print("\n--- Pro Text ---")
    text_cols = ["Text", "Modell_A", "Modell_B", "Cohen_Kappa", "F1_(A_ref)", "MCC",
                 "Geteilte_Tags", "Nur_A", "Nur_B"]
    print(df_iaa[text_cols].to_string(index=False))

    print_section("3 · Tag-Konsens")
    df_consensus = compute_tag_consensus(texts_data, all_tags)
    konsens_sum = (
        df_consensus.groupby("Konsens")["Tag"]
        .count()
        .reset_index()
        .rename(columns={"Tag": "Anzahl (Text×Tag)"})
        .sort_values("Anzahl (Text×Tag)", ascending=False)
    )
    print(konsens_sum.to_string(index=False))

    # Detailansicht: nur Texte mit Teilkonsens oder Einzelmeinung
    strittige = df_consensus[df_consensus["Konsens"].str.startswith(("1/", "2/"))]
    if not strittige.empty:
        print(f"\n{len(strittige)} strittige (Tag × Text) Kombinationen:")
        print(strittige[["Text", "Tag", "Konsens", "Anzahl_Modelle"]].to_string(index=False))

    print_section("4 · Zitat-Matching (Fuzzy-Ähnlichkeit)")
    df_quotes = compute_quote_matching(texts_data)
    qsum = df_quotes.groupby("Status")["Tag"].count().reset_index()
    qsum.columns = ["Status", "Anzahl"]
    print(qsum.to_string(index=False))

    divergent = df_quotes[df_quotes["Status"] == "Divergent"]
    if not divergent.empty:
        print(f"\nTop divergente Zitat-Paare (Ähnlichkeit < {QUOTE_SIM_THRESHOLD}):")
        print(
            divergent[["Text", "Tag", "Modell_A", "Modell_B", "Ähnlichkeit", "Zitat_A", "Zitat_B"]]
            .sort_values("Ähnlichkeit")
            .head(10)
            .to_string(index=False)
        )

    print_section("5 · Tag-Stabilität (MCC & F1 pro Tag, paarweise)")
    df_tag_metrics = compute_per_tag_metrics(texts_data, all_tags)
    tag_agg = (
        df_tag_metrics.groupby("Tag")[["Cohen_Kappa", "F1", "MCC"]]
        .mean()
        .round(4)
        .reset_index()
        .sort_values("F1", ascending=False)
    )
    print(tag_agg.to_string(index=False))

    print_section("6 · Qualitative Fehleranalyse")
    df_errors = compute_error_analysis(texts_data, all_tags)

    einzelmeinungen = df_errors[df_errors["Fehlertyp"].str.startswith("Einzelmeinung")]
    print(f"Einzelmeinungen (ein Modell sieht Tag, andere nicht): {len(einzelmeinungen)}")
    if not einzelmeinungen.empty:
        em_by_model = (
            einzelmeinungen["Fehlertyp"]
            .value_counts()
            .reset_index()
            .rename(columns={"Fehlertyp": "Fehlertyp", "count": "Anzahl"})
        )
        print(em_by_model.to_string(index=False))
        print("\nBeispiele:")
        print(einzelmeinungen[["Text", "Tag", "Fehlertyp"]].head(12).to_string(index=False))

    zitat_divergenz = df_errors[df_errors["Zitat_Hinweis"] != ""]
    if not zitat_divergenz.empty:
        print(f"\nZitat-Divergenzen bei geteilten Tags: {len(zitat_divergenz)}")
        print(zitat_divergenz[["Text", "Tag", "Fehlertyp", "Zitat_Hinweis"]].head(8).to_string(index=False))

    # --- Ground Truth (optional) ---
    df_gt_overall, df_gt_tags = None, None
    if GROUND_TRUTH_DIR and Path(GROUND_TRUTH_DIR).exists():
        print_section("7 · Evaluation gegen Ground Truth")
        gt_data = {}
        for f in Path(GROUND_TRUTH_DIR).glob("*.json"):
            gt_data[f.stem] = load_annotations(f)
        if gt_data:
            df_gt_overall, df_gt_tags = compute_ground_truth_eval(texts_data, gt_data, all_tags)
            print(f"Ground Truth für {len(gt_data)} Texte geladen.")
            gt_agg = (
                df_gt_overall.groupby("Modell")[["Precision", "Recall", "F1", "MCC"]]
                .mean()
                .round(4)
            )
            print(gt_agg.to_string())
        else:
            print("  Keine Ground-Truth-Dateien gefunden.")

    # ============================================================
    # Excel-Ausgabe
    # ============================================================
    # ============================================================
    # Diagramm-Ausgabe
    # ============================================================
    diag_dir = Path("auswertung_diagramme")
    print_section("Diagramme")
    create_diagrams(
        out_dir=diag_dir,
        df_summary=df_summary,
        df_iaa=df_iaa,
        iaa_agg=iaa_agg,
        df_consensus=df_consensus,
        df_quotes=df_quotes,
        df_tag_metrics=df_tag_metrics,
        tag_agg=tag_agg,
        all_tags=all_tags,
        df_gt_overall=df_gt_overall,
    )

    print(f"\nSchreibe Ergebnisse → {OUTPUT_FILE} …")
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Modell_Übersicht", index=False)
        df_iaa.to_excel(writer, sheet_name="IAA_pro_Text", index=False)
        iaa_agg.to_excel(writer, sheet_name="IAA_Gesamt", index=False)
        df_consensus.to_excel(writer, sheet_name="Tag_Konsens", index=False)
        df_quotes.to_excel(writer, sheet_name="Zitat_Matching", index=False)
        df_tag_metrics.to_excel(writer, sheet_name="Tag_Stabilität", index=False)
        tag_agg.to_excel(writer, sheet_name="Tag_Stabilität_Agg", index=False)
        df_errors.to_excel(writer, sheet_name="Fehleranalyse", index=False)
        if df_gt_overall is not None:
            df_gt_overall.to_excel(writer, sheet_name="GT_Eval_Gesamt", index=False)
            df_gt_tags.to_excel(writer, sheet_name="GT_Eval_Pro_Tag", index=False)

    print(f"Fertig! → {OUTPUT_FILE.resolve()}")
    print(f"\nEnthaltene Tabellenblätter:")
    sheets = [
        "Modell_Übersicht    – Annotationsanzahl pro Modell und Tag",
        "IAA_pro_Text        – Paarweises IAA je Text",
        "IAA_Gesamt          – Kappa/F1/MCC gemittelt über alle Texte",
        "Tag_Konsens         – Welche Modelle nutzen welchen Tag je Text",
        "Zitat_Matching      – Ähnlichkeit der extrahierten Zitate (paarweise)",
        "Tag_Stabilität      – MCC/F1/Kappa pro Tag und Modellpaar",
        "Tag_Stabilität_Agg  – Aggregiert pro Tag",
        "Fehleranalyse       – Einzelmeinungen, Zitat-Divergenzen",
    ]
    if df_gt_overall is not None:
        sheets += [
            "GT_Eval_Gesamt      – Precision/Recall/F1/MCC gegen Ground Truth",
            "GT_Eval_Pro_Tag     – TP/FP/FN/TN pro Tag und Modell",
        ]
    for s in sheets:
        print(f"  • {s}")
    print(f"\nDiagramme (PNG, 150 dpi) → {diag_dir.resolve()}/")
    diagrams = [
        "D1_tag_konsens_heatmap.png   – Heatmap: wie viele Modelle je Tag & Text",
        "D2_annotationen_pro_tag.png  – Grouped Bar: Annotationsanzahl pro Modell & Tag",
        "D3_iaa_modellpaar.png        – Bar: Kappa / F1 / MCC pro Modellpaar",
        "D4_tag_stabilitaet_heatmap.png – Heatmap: F1 pro Tag × Modellpaar",
        "D5_fehlertypen_pro_text.png  – Stacked Bar: Konsens/Teilkonsens/Einzelmeinung",
        "D6_zitat_aehnlichkeit.png    – Violin: Ähnlichkeitsverteilung der Zitate",
        "D7_kappa_pro_tag.png         – Heatmap: Cohen's κ pro Tag × Modellpaar",
    ]
    if df_gt_overall is not None:
        diagrams.append("D8_ground_truth_eval.png     – Bar: P/R/F1/MCC gegen Ground Truth")
    for d in diagrams:
        print(f"  • {d}")


if __name__ == "__main__":
    main()