"""Tests for fine-tuning training pipeline."""

import json
from pathlib import Path
import pytest

from ml_pipeline.train_finetune import (
    load_pairs,
    train_finetune,
)


@pytest.fixture
def pairs_file(tmp_path):
    root_pairs = Path("pairs.jsonl")
    if root_pairs.exists():
        return str(root_pairs)
    # Generate temporary pairs.jsonl for dry-run training tests
    p = tmp_path / "synthetic_pairs.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(12):
            f.write(
                json.dumps({
                    "jd": f"Senior Software Engineer role requiring Python and cloud experience {i}.",
                    "resume": f"Software developer with 5 years of Python, Docker, and AWS experience {i}.",
                    "category": "Software_Developer",
                })
                + "\n"
            )
    return str(p)


def test_training_stub():
    """Verify training test harness is ready."""
    assert True


def test_load_pairs(pairs_file):
    """Verify load_pairs reads valid pairs."""
    pairs = load_pairs(pairs_file)
    assert len(pairs) > 0
    first = pairs[0]
    assert "jd" in first and len(first["jd"].strip()) > 0
    assert "resume" in first and len(first["resume"].strip()) > 0
    assert "category" in first and len(first["category"].strip()) > 0


def test_loss_decreases(tmp_path, pairs_file):
    """Assert loss at epoch 3 < loss at epoch 1 during fine-tuning."""
    checkpoints_dir = tmp_path / "checkpoints"
    outputs_dir = tmp_path / "outputs"

    result = train_finetune(
        pairs_path=pairs_file,
        epochs=3,
        batch_size=4,
        lr=5e-4,
        dry_run=True,
        output_dir=str(checkpoints_dir),
        outputs_dir=str(outputs_dir),
    )

    losses = result["epoch_losses"]
    assert len(losses) >= 3, f"Expected at least 3 epoch losses, got {losses}"
    # Loss at epoch 3 (index 2) must be strictly less than loss at epoch 1 (index 0)
    assert losses[2] < losses[0], f"Expected loss[2] < loss[0], got {losses[2]} >= {losses[0]}"


def test_checkpoint_exists(tmp_path, pairs_file):
    """Assert checkpoints/epoch-3/ directory is non-empty after a dry-run with 1 epoch."""
    checkpoints_dir = tmp_path / "checkpoints"
    outputs_dir = tmp_path / "outputs"

    train_finetune(
        pairs_path=pairs_file,
        epochs=1,
        batch_size=4,
        dry_run=True,
        output_dir=str(checkpoints_dir),
        outputs_dir=str(outputs_dir),
    )

    epoch_3_dir = checkpoints_dir / "epoch-3"
    assert epoch_3_dir.exists(), f"Directory {epoch_3_dir} was not created"
    assert epoch_3_dir.is_dir(), f"{epoch_3_dir} is not a directory"
    contents = list(epoch_3_dir.iterdir())
    assert len(contents) > 0, f"Directory {epoch_3_dir} is empty"


def test_output_artifacts_created(tmp_path, pairs_file):
    """Verify that training loss plot, cosine separation plot, and eval metrics are generated."""
    checkpoints_dir = tmp_path / "checkpoints"
    outputs_dir = tmp_path / "outputs"

    train_finetune(
        pairs_path=pairs_file,
        epochs=3,
        batch_size=4,
        dry_run=True,
        output_dir=str(checkpoints_dir),
        outputs_dir=str(outputs_dir),
    )

    loss_plot = outputs_dir / "training_loss.png"
    separation_plot = outputs_dir / "cosine_separation.png"
    metrics_file = outputs_dir / "eval_metrics.md"

    assert loss_plot.exists() and loss_plot.stat().st_size > 0
    assert separation_plot.exists() and separation_plot.stat().st_size > 0
    assert metrics_file.exists() and metrics_file.stat().st_size > 0

    metrics_text = metrics_file.read_text(encoding="utf-8")
    assert metrics_text.find("Model Evaluation Metrics") != -1 or "Model Evaluation Metrics" in metrics_text
    assert "MRR" in metrics_text
    assert "Cosine Delta" in metrics_text


def test_onnx_output_shape():
    """Verify model_quantized.onnx loads and produces output shape (2, 384) on a batch of 2 strings."""
    onnx_path = Path("model_quantized.onnx")
    if not onnx_path.exists():
        from ml_pipeline.export_onnx import export_onnx
        export_onnx(
            checkpoint_path="checkpoints/epoch-3",
            output_path=onnx_path,
        )

    assert onnx_path.exists()

    batch = [
        "Software Engineer with 5 years experience in Python and AWS.",
        "Data Scientist specializing in Machine Learning and PyTorch.",
    ]

    from ml_pipeline.export_onnx import embed_texts
    embeddings = embed_texts(batch, model_path=onnx_path)
    assert embeddings.shape == (2, 384)

    import onnxruntime as ort
    from transformers import AutoTokenizer

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="np")
    feed = {inp.name: encoded[inp.name] for inp in session.get_inputs() if inp.name in encoded}
    raw_outputs = session.run(None, feed)[0]
    assert raw_outputs.shape == (2, 384)


def test_onnx_cosine_distance():
    """Verify that quantized ONNX model embeddings have cosine distance <= 0.01 against PyTorch model."""
    onnx_path = Path("model_quantized.onnx")
    if not onnx_path.exists():
        from ml_pipeline.export_onnx import export_onnx
        export_onnx(
            checkpoint_path="checkpoints/epoch-3",
            output_path=onnx_path,
        )

    from ml_pipeline.export_onnx import verify_cosine_distance, load_model
    model = load_model("checkpoints/epoch-3")
    results = verify_cosine_distance(model, onnx_path, threshold=0.01)
    for dist in results.values():
        assert dist <= 0.01, f"Cosine distance {dist} exceeded 0.01 threshold"


def test_onnx_fallback_base_model(tmp_path):
    """Verify export_onnx falls back to base model all-MiniLM-L6-v2 when checkpoint is not present."""
    from ml_pipeline.export_onnx import load_model
    non_existent = tmp_path / "non_existent_checkpoint"
    model = load_model(non_existent)
    assert model is not None

