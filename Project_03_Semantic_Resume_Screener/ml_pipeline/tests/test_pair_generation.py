"""Tests for job description and pair generation."""

import csv
import json
from pathlib import Path
import pytest

from ml_pipeline.generate_jds import CATEGORIES, generate_jds
from ml_pipeline.build_pairs import (
    build_pairs,
    format_jd,
    format_resume,
    oversample_resumes,
    to_input_examples,
)



def test_pair_generation_stub():
    """Verify pair generation test harness is ready."""
    assert True


def test_jd_coverage(tmp_path):
    """Assert all 24 categories appear in generated_jds.csv with >= 2 rows each."""
    test_output = tmp_path / "generated_jds.csv"
    test_seed = tmp_path / "custom_job_descriptions.csv"

    # Run generator with temporary paths
    generate_jds(output_path=str(test_output), seed_path=str(test_seed))

    assert test_output.exists(), "generated_jds.csv was not created"

    rows = []
    with open(test_output, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Verify required columns exist
    required_cols = {"Category", "title", "required_skills", "preferred_skills", "min_experience_years"}
    assert rows, "generated_jds.csv is empty"
    assert required_cols.issubset(rows[0].keys()), f"Missing columns in generated CSV: {required_cols - set(rows[0].keys())}"

    # Verify coverage across all 24 categories
    category_counts = {}
    for r in rows:
        cat = r["Category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        assert r["title"].strip(), f"Empty title in row: {r}"
        assert r["required_skills"].strip(), f"Empty required_skills in row: {r}"
        assert "|" in r["required_skills"] or len(r["required_skills"]) > 0
        assert int(r["min_experience_years"]) >= 0

    assert len(CATEGORIES) == 24, f"Expected 24 categories, got {len(CATEGORIES)}"
    for cat in CATEGORIES:
        assert cat in category_counts, f"Missing category: {cat}"
        assert category_counts[cat] >= 2, f"Category {cat} has only {category_counts[cat]} rows, expected >= 2"

    # Verify idempotency: running generate_jds again should skip already covered categories
    initial_total = len(rows)
    generate_jds(output_path=str(test_output), seed_path=str(test_seed))
    with open(test_output, mode="r", encoding="utf-8") as f:
        second_run_rows = list(csv.DictReader(f))
    assert len(second_run_rows) == initial_total, "Second run modified or duplicated rows instead of being idempotent"


def test_positive_pair_coverage(tmp_path):
    """Assert all 24 categories have pairs, underrepresented have >= 100, and non-empty texts."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    resumes_csv = repo_root / "Resume.csv"
    jds_csv = repo_root / "generated_jds.csv"
    output_jsonl = tmp_path / "pairs.jsonl"

    assert resumes_csv.exists(), f"Resume dataset not found at {resumes_csv}"
    assert jds_csv.exists(), f"Generated JDs not found at {jds_csv}"

    pairs = build_pairs(
        resumes_path=resumes_csv,
        jds_path=jds_csv,
        output_path=output_jsonl,
        min_samples=100,
        seed=42,
    )

    assert output_jsonl.exists(), "pairs.jsonl output was not created"
    assert len(pairs) > 0, "No pairs constructed"

    # Verify JSONL lines correspond to returned pairs
    with open(output_jsonl, mode="r", encoding="utf-8") as f:
        file_pairs = [json.loads(line) for line in f if line.strip()]
    assert len(file_pairs) == len(pairs), "File pair count does not match in-memory pairs count"

    # Category coverage and non-empty assertions
    category_counts = {}
    for p in pairs:
        cat = p["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

        assert p["jd"].strip() != "", f"Empty jd in pair: {p}"
        assert p["resume"].strip() != "", f"Empty resume in pair: {p}"
        assert p["category"].strip() != "", f"Empty category in pair: {p}"

    # Criterion 1: All 24 categories have >= 1 pair
    for cat in CATEGORIES:
        assert cat in category_counts, f"Missing category in pairs: {cat}"
        assert category_counts[cat] >= 1, f"Category {cat} has 0 pairs"

    # Criterion 2: AGRICULTURE, AUTOMOBILE, BPO each have >= 100 pairs after oversampling
    assert category_counts["AGRICULTURE"] >= 100, (
        f"AGRICULTURE has {category_counts['AGRICULTURE']} pairs, expected >= 100"
    )
    assert category_counts["AUTOMOBILE"] >= 100, (
        f"AUTOMOBILE has {category_counts['AUTOMOBILE']} pairs, expected >= 100"
    )
    assert category_counts["BPO"] >= 100, (
        f"BPO has {category_counts['BPO']} pairs, expected >= 100"
    )


def test_oversample_resumes():
    """Verify oversampling duplicates samples to meet threshold and is deterministic."""
    sample_resumes = [
        {"ID": "1", "Resume_str": "Resume 1", "Category": "TEST"},
        {"ID": "2", "Resume_str": "Resume 2", "Category": "TEST"},
    ]
    oversampled = oversample_resumes(sample_resumes, min_samples=5, seed=42)
    assert len(oversampled) == 5
    # First 2 should be the originals in order
    assert oversampled[0]["ID"] == "1"
    assert oversampled[1]["ID"] == "2"
    # All items should come from the initial sample set
    assert all(r["ID"] in ("1", "2") for r in oversampled)

    # When already >= min_samples, returns original list unchanged
    no_oversample = oversample_resumes(sample_resumes, min_samples=2, seed=42)
    assert len(no_oversample) == 2


def test_format_jd_and_resume():
    """Verify JD and resume text formatting helper functions."""
    jd_row = {
        "title": "Data Scientist",
        "required_skills": "Python|SQL|Machine Learning",
        "preferred_skills": "PyTorch|Docker",
        "min_experience_years": "3",
    }
    jd_text = format_jd(jd_row)
    assert "Data Scientist" in jd_text
    assert "Python, SQL, Machine Learning" in jd_text
    assert "PyTorch, Docker" in jd_text
    assert "3 years" in jd_text

    resume_row = {
        "Resume_str": "  Experienced professional in AI and data systems.  \n\n",
    }
    resume_text = format_resume(resume_row)
    assert resume_text == "Experienced professional in AI and data systems."


def test_to_input_examples():
    """Verify pairs conversion to InputExample objects."""
    pairs = [
        {"jd": "Job description text", "resume": "Resume text", "category": "HR"}
    ]
    examples = to_input_examples(pairs)
    assert len(examples) == 1
    assert hasattr(examples[0], "texts")
    assert examples[0].texts == ["Job description text", "Resume text"]

