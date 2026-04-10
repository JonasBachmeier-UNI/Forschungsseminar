#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)


def collect_expected_stems(input_dirs: list[Path]) -> set[str]:
    expected: set[str] = set()
    for input_dir in input_dirs:
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Warning: Input folder not found, skipping: {input_dir}")
            continue

        for txt_file in sorted(input_dir.glob("*.txt")):
            expected.add(sanitize_name(txt_file.stem))
    return expected


def has_annotation_for_stem(folder: Path, expected_stem: str) -> bool:
    pattern = f"*_{expected_stem}_extraction.json"
    return any(folder.glob(pattern))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check if all source texts from ergebnis_* folders have annotation "
            "JSONs in each subfolder of annotationen_uni_models_zero_shot."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["ergebnis_kapitel_5"],
        help="Folders containing .txt files that should be annotated.",
    )
    parser.add_argument(
        "--annotations-root",
        default="annotationen_uni_models_zero_shot",
        help="Root folder containing model subfolders with annotation JSON files.",
    )
    parser.add_argument(
        "--show-present-count",
        action="store_true",
        help="Also print count of present files per model folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    annotations_root = Path(args.annotations_root)

    if not annotations_root.exists() or not annotations_root.is_dir():
        print(f"Error: Annotation root folder not found: {annotations_root}")
        return 2

    expected_stems = collect_expected_stems(input_dirs)
    expected_count = len(expected_stems)

    if expected_count == 0:
        print("Error: No expected .txt files found in the provided input folders.")
        return 2

    model_folders = sorted([p for p in annotations_root.iterdir() if p.is_dir()])
    if not model_folders:
        print(f"Error: No model subfolders found in: {annotations_root}")
        return 2

    print(f"Expected source files: {expected_count}")
    print(f"Model folders checked: {len(model_folders)}")
    print()

    all_complete = True

    for model_folder in model_folders:
        missing = sorted(
            stem for stem in expected_stems if not has_annotation_for_stem(model_folder, stem)
        )
        present_count = expected_count - len(missing)

        if missing:
            all_complete = False
            print(f"[{model_folder.name}] MISSING {len(missing)}/{expected_count}")
            for stem in missing:
                print(f"  - {stem}")
        else:
            print(f"[{model_folder.name}] OK ({expected_count}/{expected_count})")

        if args.show_present_count and missing:
            print(f"  Present: {present_count}/{expected_count}")

        print()

    if all_complete:
        print("Result: All annotation folders are complete.")
        return 0

    print("Result: Missing annotations found.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
