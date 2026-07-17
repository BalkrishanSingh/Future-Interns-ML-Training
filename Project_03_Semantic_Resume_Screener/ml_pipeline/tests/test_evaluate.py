"""Tests for ml_pipeline/evaluate.py standalone evaluation script."""

from pathlib import Path
import pytest
from ml_pipeline.evaluate import (
    plot_loss_curves,
    plot_cosine_separation,
    write_eval_metrics_table,
    compute_evaluation_metrics,
    load_eval_dataset,
    load_pairs,
)


def test_load_eval_dataset(tmp_path: Path):
    """Verify load_eval_dataset loads pairs with 'jd' or 'job_description' seamlessly."""
    import json

    pairs_file = tmp_path / "pairs.jsonl"
    with open(pairs_file, mode="w", encoding="utf-8") as f:
        f.write(json.dumps({"jd": "Python Developer JD", "resume": "Python Resume Text", "category": "Dev"}) + "\n")
        f.write(json.dumps({"job_description": "React Developer JD", "resume": "React Resume Text", "category": "Frontend"}) + "\n")

    items = load_eval_dataset(pairs_file)
    assert len(items) == 2
    assert items[0]["jd"] == "Python Developer JD"
    assert items[0]["resume"] == "Python Resume Text"
    assert items[1]["job_description"] == "React Developer JD"

    alias_items = load_pairs(pairs_file)
    assert len(alias_items) == 2


def test_plot_loss_curves(tmp_path: Path):
    """Verify plot_loss_curves creates a valid non-empty image file."""
    epochs = [1, 2, 3]
    losses = [0.45, 0.28, 0.15]
    out = tmp_path / "loss.png"
    result = plot_loss_curves(epochs, losses, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_plot_cosine_separation(tmp_path: Path):
    """Verify plot_cosine_separation creates a valid histogram image file."""
    pos_sims = [0.85, 0.90, 0.78, 0.92]
    neg_sims = [0.22, 0.15, 0.31, 0.05]
    out = tmp_path / "sep.png"
    result = plot_cosine_separation(pos_sims, neg_sims, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_write_eval_metrics_table(tmp_path: Path):
    """Verify write_eval_metrics_table generates markdown report with MRR and Cosine Delta."""
    metrics = {
        "mean_positive_sim": 0.88,
        "mean_negative_sim": 0.25,
        "cosine_delta": 0.63,
        "mrr": 0.95,
        "sample_count": 50,
    }
    out = tmp_path / "eval_metrics.md"
    result = write_eval_metrics_table(metrics, out)
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "Model Evaluation Metrics" in content
    assert "MRR" in content
    assert "Cosine Delta" in content
    assert "0.6300" in content
