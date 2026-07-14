"""Evaluation script for loss curves, cosine separation, and MRR metrics.

Calculates training loss curves, cosine separation between positive and negative
candidate pairs, and Mean Reciprocal Rank (MRR).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")

DEFAULT_MODEL_PATH = "checkpoints/epoch-3"
DEFAULT_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PAIRS_PATH = "pairs.jsonl"
DEFAULT_OUTPUT_DIR = "outputs"


def load_eval_dataset(pairs_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load evaluation pairs from JSONL file.

    Each row is expected to have 'jd' (or 'job_description'/'query'/'text_a')
    and 'resume' (or 'pos'/'text_b') per build_pairs.py schema.
    """
    path = Path(pairs_path)
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found at: {path}")

    pairs: List[Dict[str, str]] = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            jd = item.get("jd") or item.get("job_description") or item.get("query") or item.get("text_a", "")
            resume = item.get("resume") or item.get("pos") or item.get("text_b", "")
            if jd and resume:
                pairs.append({
                    "jd": jd,
                    "job_description": jd,
                    "resume": resume,
                    "category": item.get("category", "General"),
                })

    logger.info("Loaded %d pairs from %s", len(pairs), path)
    return pairs


# Canonical alias
load_pairs = load_eval_dataset


def load_model(model_name_or_path: Union[str, Path] = DEFAULT_MODEL_PATH, device: str = "cpu") -> SentenceTransformer:
    """Load SentenceTransformer model from checkpoint or base model name."""
    p = Path(model_name_or_path)
    if p.exists() and p.is_dir() and any(p.iterdir()):
        logger.info("Loading model checkpoint from: %s", p)
        return SentenceTransformer(str(p), device=device)

    logger.warning("Checkpoint '%s' not found. Falling back to: %s", model_name_or_path, DEFAULT_FALLBACK_MODEL)
    return SentenceTransformer(DEFAULT_FALLBACK_MODEL, device=device)


def compute_evaluation_metrics(
    model: SentenceTransformer,
    pairs: List[Dict[str, str]],
    batch_size: int = 32,
    device: str = "cpu",
    max_samples: Optional[int] = 500,
) -> Dict[str, Any]:
    """Compute positive cosine similarity, in-batch negative similarity, cosine delta, and MRR.

    Args:
        model: Loaded SentenceTransformer.
        pairs: List of pair dicts containing 'job_description' and 'resume'.
        batch_size: Evaluation batch size.
        device: Device to use for encoding.
        max_samples: Optional cap on pairs for faster evaluation.

    Returns:
        Dictionary with mean_positive_sim, mean_negative_sim, cosine_delta, mrr,
        pos_similarities, and neg_similarities.
    """
    if max_samples and len(pairs) > max_samples:
        eval_pairs = pairs[:max_samples]
    else:
        eval_pairs = pairs

    jds = [p["job_description"] for p in eval_pairs]
    resumes = [p["resume"] for p in eval_pairs]

    model.eval()
    with torch.no_grad():
        jd_embeds = model.encode(
            jds,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        resume_embeds = model.encode(
            resumes,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        sim_matrix = torch.matmul(jd_embeds, resume_embeds.T)
        pos_sims = torch.diagonal(sim_matrix)
        mean_pos = float(torch.mean(pos_sims).item())

        n_samples = sim_matrix.size(0)
        if n_samples > 1:
            mask = ~torch.eye(n_samples, dtype=torch.bool, device=sim_matrix.device)
            neg_sims = sim_matrix[mask]
            mean_neg = float(torch.mean(neg_sims).item())
        else:
            neg_sims = torch.tensor([0.0], device=sim_matrix.device)
            mean_neg = 0.0

        # Mean Reciprocal Rank (MRR)
        ranks: List[float] = []
        for i in range(n_samples):
            target_score = pos_sims[i]
            rank = int(torch.sum(sim_matrix[i] >= target_score).item())
            ranks.append(1.0 / max(1, rank))

        mrr = float(np.mean(ranks)) if ranks else 0.0
        cosine_delta = mean_pos - mean_neg

    return {
        "mean_positive_sim": round(mean_pos, 4),
        "mean_negative_sim": round(mean_neg, 4),
        "cosine_delta": round(cosine_delta, 4),
        "mrr": round(mrr, 4),
        "sample_count": n_samples,
        "pos_similarities": pos_sims.detach().cpu().tolist(),
        "neg_similarities": neg_sims.detach().cpu().tolist(),
    }


def plot_loss_curves(
    epochs: List[int],
    losses: List[float],
    output_path: Union[str, Path] = "outputs/training_loss.png",
) -> Path:
    """Plot training loss vs epoch line curve."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        epochs,
        losses,
        marker="o",
        color="#7287fd",
        linewidth=2.5,
        markersize=8,
        label="Training Loss (MNRL)",
    )
    plt.title("Multiple Negatives Ranking Loss vs Epoch", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss Value", fontsize=11)
    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Saved loss curve plot to: %s", output_path)
    return output_path


def plot_cosine_separation(
    pos_sims: List[float],
    neg_sims: List[float],
    output_path: Union[str, Path] = "outputs/cosine_separation.png",
) -> Path:
    """Plot overlay histogram comparing positive pair vs negative pair cosine similarity distributions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    bins = np.linspace(-0.2, 1.0, 40)
    plt.hist(
        pos_sims,
        bins=bins,
        alpha=0.65,
        color="#40a02b",
        label="Positive Pairs (Matching JD-Resume)",
        density=True,
    )
    plt.hist(
        neg_sims,
        bins=bins,
        alpha=0.65,
        color="#e64553",
        label="In-Batch Negatives (Non-Matching)",
        density=True,
    )
    plt.title(
        "Cosine Similarity Separation: Positive vs In-Batch Negatives",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Cosine Similarity", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info("Saved cosine separation histogram to: %s", output_path)
    return output_path


def write_eval_metrics_table(
    metrics: Dict[str, Any],
    output_path: Union[str, Path] = "outputs/eval_metrics.md",
    epoch_records: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Write evaluation metrics markdown report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "# Model Evaluation Metrics",
        "",
        "Evaluation metrics for fine-tuned SentenceTransformer model (`all-MiniLM-L6-v2`) with `MultipleNegativesRankingLoss`.",
        "",
    ]

    if epoch_records:
        lines.extend([
            "| Epoch | Training Loss | Mean Positive Cosine | Mean Negative Cosine | Cosine Delta | MRR |",
            "|:-----:|:-------------:|:--------------------:|:--------------------:|:------------:|:---:|",
        ])
        for r in epoch_records:
            lines.append(
                f"| {r.get('epoch', 1)} | {r.get('loss', 0.0):.4f} | {r.get('mean_positive_sim', 0.0):.4f} | "
                f"{r.get('mean_negative_sim', 0.0):.4f} | {r.get('cosine_delta', 0.0):.4f} | {r.get('mrr', 0.0):.4f} |"
            )
        lines.append("")
    else:
        lines.extend([
            "| Metric | Value | Target Standard |",
            "|:-------|:-----:|:---------------:|",
            f"| Mean Positive Cosine Similarity | {metrics.get('mean_positive_sim', 0.0):.4f} | >= 0.70 |",
            f"| Mean Negative Cosine Similarity | {metrics.get('mean_negative_sim', 0.0):.4f} | <= 0.40 |",
            f"| Cosine Delta Margin | {metrics.get('cosine_delta', 0.0):.4f} | >= 0.30 |",
            f"| Mean Reciprocal Rank (MRR) | {metrics.get('mrr', 0.0):.4f} | >= 0.85 |",
            f"| Evaluated Pairs Sample Count | {metrics.get('sample_count', 0)} | >= 100 |",
            "",
        ])

    lines.extend([
        "## Evaluation Summary",
        "",
        f"- **Cosine Separation Margin:** +{metrics.get('cosine_delta', 0.0):.4f} delta between matching and non-matching profiles.",
        f"- **Ranking Precision (MRR):** {metrics.get('mrr', 0.0):.4f} (top-ranked correct profile frequency).",
        f"- **Average Positive Alignment:** {metrics.get('mean_positive_sim', 0.0):.4f}.",
        "",
    ])

    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Saved evaluation report to: %s", output_path)
    return output_path


def main() -> None:
    """CLI Entrypoint for ml_pipeline.evaluate."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned sentence transformer on resume-JD pairs."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path or name of model checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=DEFAULT_PAIRS_PATH,
        help=f"Path to evaluation pairs JSONL (default: {DEFAULT_PAIRS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save metrics and plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: cpu, mps, or cuda (default: cpu)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum pairs to evaluate (default: 500)",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs)
    model = load_model(args.model, device=args.device)

    metrics = compute_evaluation_metrics(
        model=model,
        pairs=pairs,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )

    # Generate plots and report
    plot_cosine_separation(
        metrics["pos_similarities"],
        metrics["neg_similarities"],
        output_path=out_dir / "cosine_separation.png",
    )
    write_eval_metrics_table(
        metrics=metrics,
        output_path=out_dir / "eval_metrics.md",
    )

    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:                {args.model}")
    print(f"Evaluated Pairs:      {metrics['sample_count']}")
    print(f"Mean Positive Sim:    {metrics['mean_positive_sim']:.4f}")
    print(f"Mean Negative Sim:    {metrics['mean_negative_sim']:.4f}")
    print(f"Cosine Delta Margin:  {metrics['cosine_delta']:.4f}")
    print(f"Mean Reciprocal Rank: {metrics['mrr']:.4f}")
    print(f"Output Directory:     {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
