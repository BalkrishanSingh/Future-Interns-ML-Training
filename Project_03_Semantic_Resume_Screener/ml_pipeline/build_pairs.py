"""Training Pair Construction for Contrastive Fine-Tuning.

Constructs positive (Job Specification, Resume) pairs for contrastive learning
with MultipleNegativesRankingLoss. Oversamples underrepresented categories
(< 100 resumes) to ensure balanced representation across all 24 domains.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import InputExample
except ImportError:  # pragma: no cover
    class InputExample:  # type: ignore[no-redef]
        """Fallback container for sentence-transformers InputExample."""

        def __init__(self, texts: list[str], label: float = 0.0, guid: str = ""):
            self.texts = texts
            self.label = label
            self.guid = guid

        def __repr__(self) -> str:
            return f"<InputExample texts={self.texts!r} label={self.label}>"


def format_jd(jd: dict[str, Any] | str) -> str:
    """Format a job description record into structured text for embedding."""
    if isinstance(jd, str):
        return jd.strip()

    title = str(jd.get("title", "")).strip()
    required = str(jd.get("required_skills", "")).strip()
    preferred = str(jd.get("preferred_skills", "")).strip()
    min_exp = str(jd.get("min_experience_years", "")).strip()

    req_list = [s.strip() for s in required.split("|") if s.strip()] if required else []
    pref_list = [s.strip() for s in preferred.split("|") if s.strip()] if preferred else []

    parts: list[str] = []
    if title:
        parts.append(f"Job Title: {title}")
    if min_exp:
        parts.append(f"Minimum Experience: {min_exp} years")
    if req_list:
        parts.append(f"Required Skills: {', '.join(req_list)}")
    if pref_list:
        parts.append(f"Preferred Skills: {', '.join(pref_list)}")

    return ". ".join(parts) + ("." if parts else "")


def format_resume(resume: dict[str, Any] | str) -> str:
    """Format resume record into clean plain text for embedding."""
    if isinstance(resume, str):
        return resume.strip()

    text = resume.get("Resume_str") or resume.get("Resume_html") or ""
    return str(text).strip()


def oversample_resumes(
    resumes: list[dict[str, Any]],
    min_samples: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Oversample resumes via random duplication until reaching min_samples."""
    if not resumes:
        return []
    if len(resumes) >= min_samples:
        return list(resumes)

    rng = random.Random(seed)
    oversampled = list(resumes)
    deficit = min_samples - len(resumes)
    additional = [rng.choice(resumes) for _ in range(deficit)]
    oversampled.extend(additional)
    return oversampled


def build_pairs(
    resumes_path: str | Path = "Resume.csv",
    jds_path: str | Path = "generated_jds.csv",
    output_path: Optional[str | Path] = None,
    min_samples: int = 100,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Construct positive (jd, resume) pairs across matching categories.

    Applies oversampling to underrepresented categories (< min_samples).
    Optionally outputs pairs to a JSONL file.
    """
    resumes_file = Path(resumes_path)
    jds_file = Path(jds_path)

    if not resumes_file.exists():
        raise FileNotFoundError(f"Resume dataset not found: {resumes_file}")
    if not jds_file.exists():
        raise FileNotFoundError(f"Job descriptions not found: {jds_file}")

    # Load and group resumes by Category
    resumes_by_category: dict[str, list[dict[str, Any]]] = {}
    with open(resumes_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = str(row.get("Category", "")).strip()
            if cat:
                resumes_by_category.setdefault(cat, []).append(row)

    # Load and group JDs by Category
    jds_by_category: dict[str, list[dict[str, Any]]] = {}
    with open(jds_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = str(row.get("Category", "")).strip()
            if cat:
                jds_by_category.setdefault(cat, []).append(row)

    pairs: list[dict[str, str]] = []
    all_categories = sorted(set(resumes_by_category.keys()) | set(jds_by_category.keys()))

    for cat in all_categories:
        cat_resumes = resumes_by_category.get(cat, [])
        cat_jds = jds_by_category.get(cat, [])

        if not cat_resumes or not cat_jds:
            continue

        # Apply oversampling to resumes if below min_samples
        effective_resumes = oversample_resumes(cat_resumes, min_samples=min_samples, seed=seed)

        for idx, resume_row in enumerate(effective_resumes):
            jd_row = cat_jds[idx % len(cat_jds)]
            jd_text = format_jd(jd_row)
            resume_text = format_resume(resume_row)

            if jd_text and resume_text:
                pairs.append({
                    "jd": jd_text,
                    "resume": resume_text,
                    "category": cat,
                })

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, mode="w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return pairs


def get_summary_stats(
    resumes_path: str | Path,
    pairs: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    """Compute summary statistics: category name -> resume count and pair count."""
    resumes_file = Path(resumes_path)
    resume_counts: dict[str, int] = {}
    if resumes_file.exists():
        with open(resumes_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = str(row.get("Category", "")).strip()
                if cat:
                    resume_counts[cat] = resume_counts.get(cat, 0) + 1

    pair_counts: dict[str, int] = {}
    for p in pairs:
        cat = p["category"]
        pair_counts[cat] = pair_counts.get(cat, 0) + 1

    categories = sorted(set(resume_counts.keys()) | set(pair_counts.keys()))
    summary: dict[str, dict[str, int]] = {}
    for cat in categories:
        summary[cat] = {
            "resumes": resume_counts.get(cat, 0),
            "pairs": pair_counts.get(cat, 0),
        }
    return summary


def print_summary_table(summary: dict[str, dict[str, int]]) -> None:
    """Print markdown-compatible summary table of resume and pair counts."""
    header = f"{'Category':<25} | {'Resumes':>8} | {'Pairs':>8}"
    divider = f"{'-'*25}-|-{'-'*8}-|-{'-'*8}"
    print("\n" + header)
    print(divider)
    total_resumes = 0
    total_pairs = 0
    for cat in sorted(summary.keys()):
        stats = summary[cat]
        r_cnt = stats.get("resumes", 0)
        p_cnt = stats.get("pairs", 0)
        total_resumes += r_cnt
        total_pairs += p_cnt
        print(f"{cat:<25} | {r_cnt:>8} | {p_cnt:>8}")
    print(divider)
    print(f"{'TOTAL':<25} | {total_resumes:>8} | {total_pairs:>8}\n")


def to_input_examples(pairs: list[dict[str, str]]) -> list[InputExample]:
    """Convert pair dictionaries into sentence-transformers InputExample objects."""
    return [InputExample(texts=[p["jd"], p["resume"]]) for p in pairs]


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for training pair construction."""
    parser = argparse.ArgumentParser(
        description="Construct positive (JD, Resume) pairs with oversampling for contrastive fine-tuning."
    )
    parser.add_argument(
        "--resumes",
        type=str,
        default="Resume.csv",
        help="Path to Resume.csv dataset (default: Resume.csv)",
    )
    parser.add_argument(
        "--jds",
        type=str,
        default="generated_jds.csv",
        help="Path to generated_jds.csv (default: generated_jds.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pairs.jsonl",
        help="Output path for pairs.jsonl (default: pairs.jsonl)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum effective samples per category after oversampling (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic oversampling (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary table output",
    )

    args = parser.parse_args(argv)

    try:
        pairs = build_pairs(
            resumes_path=args.resumes,
            jds_path=args.jds,
            output_path=args.output,
            min_samples=args.min_samples,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    summary = get_summary_stats(args.resumes, pairs)
    if not args.quiet:
        print_summary_table(summary)
        print(f"Wrote {len(pairs)} pairs to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
