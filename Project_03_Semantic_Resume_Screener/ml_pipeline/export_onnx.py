"""Export Fine-Tuned PyTorch Model to Quantized INT8 ONNX.

Exports fine-tuned sentence-transformers checkpoints (or falls back to
sentence-transformers/all-MiniLM-L6-v2) to an ONNX computational graph with
dynamic batch and sequence dimensions, followed by INT8 dynamic quantization
via onnxruntime.quantization.quantize_dynamic. Verifies output embeddings
against the PyTorch source within cosine distance <= 0.01.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("export_onnx")

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/epoch-3")
DEFAULT_OUTPUT_PATH = Path("model_quantized.onnx")

DEFAULT_TEST_SENTENCES = [
    "Experienced software engineer with 5 years of Python and cloud architecture experience.",
    "Data scientist specializing in deep learning, natural language processing, and PyTorch.",
    "Project manager leading cross-functional teams in agile software development environments.",
    "Registered nurse with extensive background in emergency medicine and intensive care.",
    "Financial analyst proficient in financial modeling, equity valuation, and forecasting.",
]


class OnnxSentenceTransformer(torch.nn.Module):
    """PyTorch wrapper around SentenceTransformer pipeline for ONNX tracing."""

    def __init__(self, st_model: SentenceTransformer) -> None:
        super().__init__()
        self.st_model = st_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass taking tokenized inputs and returning normalized sentence embeddings."""
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.st_model(features)
        return outputs["sentence_embedding"]


def load_model(checkpoint_path: Optional[str | Path] = None) -> SentenceTransformer:
    """Load SentenceTransformer from checkpoint path, falling back to base model if missing.

    Args:
        checkpoint_path: Optional path to saved checkpoint directory.

    Returns:
        SentenceTransformer model loaded on CPU.
    """
    path_to_try = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    if path_to_try.exists() and path_to_try.is_dir() and any(path_to_try.iterdir()):
        logger.info("Loading fine-tuned model checkpoint from: %s", path_to_try)
        return SentenceTransformer(str(path_to_try), device="cpu")

    logger.warning(
        "Checkpoint directory '%s' not found or empty. Falling back to base model: %s",
        path_to_try,
        BASE_MODEL_NAME,
    )
    return SentenceTransformer(BASE_MODEL_NAME, device="cpu")


def export_onnx(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    fp32_output_path: Optional[str | Path] = None,
    verify: bool = True,
    threshold: float = 0.01,
) -> Path:
    """Export SentenceTransformer checkpoint to quantized INT8 ONNX model.

    Args:
        checkpoint_path: Path to PyTorch checkpoint directory or fallback.
        output_path: Destination path for quantized INT8 ONNX model.
        fp32_output_path: Optional destination for intermediate FP32 ONNX model.
        verify: Whether to verify cosine distance against PyTorch model on test sentences.
        threshold: Maximum allowed cosine distance for verification.

    Returns:
        Path to the quantized ONNX file.
    """
    model = load_model(checkpoint_path)
    wrapper = OnnxSentenceTransformer(model)
    wrapper.eval()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use specified intermediate FP32 path or create a temporary file
    temp_dir = None
    if fp32_output_path is not None:
        fp32_path = Path(fp32_output_path).resolve()
        fp32_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        fp32_path = Path(temp_dir.name) / "model_fp32.onnx"

    logger.info("Exporting PyTorch model to FP32 ONNX at: %s", fp32_path)
    dummy_text = ["Sample text for tracing ONNX graph."]
    tokenizer = model.tokenizer
    encoded = tokenizer(dummy_text, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Export via torch.onnx.export with dynamic axes for batch size and sequence length
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("FP32 ONNX export complete.")

    logger.info("Applying dynamic INT8 quantization via onnxruntime...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
    )
    logger.info("Dynamic quantization complete: %s", output_path)

    # Print file sizes before and after quantization
    fp32_size = fp32_path.stat().st_size
    quant_size = output_path.stat().st_size
    size_reduction = (1.0 - quant_size / fp32_size) * 100.0

    print("=" * 60)
    print("ONNX EXPORT & QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"FP32 Model Size:      {fp32_size / (1024 * 1024):8.2f} MB ({fp32_size:,} bytes)")
    print(f"Quantized (INT8) Size:{quant_size / (1024 * 1024):8.2f} MB ({quant_size:,} bytes)")
    print(f"Size Reduction:       {size_reduction:8.1f}%")
    print(f"Output File:          {output_path}")
    print("=" * 60)

    # Verification on test sentences
    if verify:
        verify_cosine_distance(model, output_path, threshold=threshold)

    if temp_dir is not None:
        temp_dir.cleanup()

    return output_path


def verify_cosine_distance(
    st_model: SentenceTransformer,
    onnx_path: str | Path,
    sentences: Optional[List[str]] = None,
    threshold: float = 0.01,
) -> Dict[str, float]:
    """Verify that ONNX model embeddings closely match PyTorch embeddings.

    Asserts that cosine distance (1.0 - cosine_similarity) <= threshold
    for each test sentence.

    Args:
        st_model: Original SentenceTransformer PyTorch model.
        onnx_path: Path to quantized ONNX model file.
        sentences: Optional list of test sentences.
        threshold: Maximum allowed cosine distance.

    Returns:
        Mapping from sentence to computed cosine distance.
    """
    if sentences is None:
        sentences = DEFAULT_TEST_SENTENCES

    # PyTorch reference embeddings
    with torch.no_grad():
        pt_embeddings = st_model.encode(
            sentences,
            convert_to_numpy=True,
            device="cpu",
            normalize_embeddings=True,
        )

    # ONNX inference
    session = load_onnx_session(onnx_path)
    onnx_embeddings = embed_texts(sentences, session=session)

    results: Dict[str, float] = {}
    print("\nVerifying Cosine Distance vs PyTorch Model on 5 Test Sentences:")
    print("-" * 60)
    for idx, (sent, pt_v, onnx_v) in enumerate(zip(sentences, pt_embeddings, onnx_embeddings), start=1):
        norm_pt = np.linalg.norm(pt_v)
        norm_onnx = np.linalg.norm(onnx_v)
        sim = float(np.dot(pt_v, onnx_v) / (norm_pt * norm_onnx))
        dist = 1.0 - sim
        results[sent] = dist
        status = "PASS" if dist <= threshold else "FAIL"
        print(f"  [{status}] Sentence {idx}: cosine distance = {dist:.6f} (threshold: <= {threshold:.2f})")
        print(f"         \"{sent}\"")
        if dist > threshold:
            raise ValueError(
                f"Cosine distance {dist:.6f} exceeded threshold {threshold:.2f} for sentence: {sent}"
            )

    print("-" * 60)
    print("All sentences verified within acceptable cosine distance threshold.\n")
    return results


def load_onnx_session(model_path: str | Path = DEFAULT_OUTPUT_PATH) -> ort.InferenceSession:
    """Load onnxruntime InferenceSession for CPU execution."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model file not found at: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def embed_texts(
    texts: Union[List[str], str],
    model_path: str | Path = DEFAULT_OUTPUT_PATH,
    session: Optional[ort.InferenceSession] = None,
    tokenizer_name_or_path: str = BASE_MODEL_NAME,
) -> np.ndarray:
    """Compute embeddings for a batch of text strings using the ONNX model.

    Args:
        texts: Single string or list of strings to embed.
        model_path: Path to ONNX model (used if session is None).
        session: Pre-loaded onnxruntime InferenceSession.
        tokenizer_name_or_path: HuggingFace model or tokenizer identifier.

    Returns:
        Numpy array of embeddings of shape (batch_size, 384).
    """
    if isinstance(texts, str):
        texts = [texts]

    if session is None:
        session = load_onnx_session(model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    # Match feeds with session input names
    feed = {inp.name: encoded[inp.name] for inp in session.get_inputs() if inp.name in encoded}
    outputs = session.run(None, feed)[0]

    # Ensure L2 normalization
    norms = np.linalg.norm(outputs, axis=1, keepdims=True)
    outputs = outputs / np.maximum(norms, 1e-12)
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for export_onnx.py."""
    parser = argparse.ArgumentParser(
        description="Export and quantize SentenceTransformer to INT8 ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help=f"Path to checkpoint directory (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Path to output quantized ONNX model (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--fp32-output",
        type=str,
        default=None,
        help="Optional path to retain unquantized FP32 ONNX model",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip cosine distance verification against PyTorch",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Maximum cosine distance threshold for verification (default: 0.01)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        fp32_output_path=args.fp32_output,
        verify=not args.no_verify,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
