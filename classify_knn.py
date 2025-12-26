#!/usr/bin/env python3
"""Embedding-based kNN classifier for phishing detection using E5-instruct."""

import sys
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Task instruction for E5-instruct model
TASK_INSTRUCTION = "Classify this email as phishing or legitimate based on its content, sender patterns, and suspicious indicators"

# Our labeled examples (ground truth from manual analysis)
# NOTE: This dataset is heavily skewed toward phishing (phishing pot).
# Only sample-4 is a legitimate email. More legitimate samples needed for production use.
LABELED_EXAMPLES = {
    # Phishing - Banking/Financial
    "email/sample-1.md": {"is_phishing": True, "label": "Banking points scam (Bradesco)"},
    "email/sample-2.md": {"is_phishing": True, "label": "Bank verification scam (Mashreq)"},
    "email/sample-5.md": {"is_phishing": True, "label": "Account blocked scam (Chase)"},
    # Phishing - Advance fee fraud
    "email/sample-3.md": {"is_phishing": True, "label": "Advance fee fraud (419 scam)"},
    # Phishing - Crypto/Web3
    "email/sample-8.md": {"is_phishing": True, "label": "FTX crypto recovery scam"},
    "email/sample-9.md": {"is_phishing": True, "label": "Fake Coinbase transaction"},
    "email/sample-12.md": {"is_phishing": True, "label": "Fake Binance verification (Cyrillic)"},
    "email/sample-15.md": {"is_phishing": True, "label": "MetaMask seed phrase theft"},
    # Phishing - Tech impersonation
    "email/sample-10.md": {"is_phishing": True, "label": "Fake Microsoft sign-in alert"},
    # Phishing - Prize/Contest scams
    "email/sample-6.md": {"is_phishing": True, "label": "Coca Cola prize scam"},
    # Legitimate
    "email/sample-4.md": {"is_phishing": False, "label": "Newsletter unsubscribe (legitimate)"},
}

# Chunk size in characters (approx 256 tokens ~ 1000 chars)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_model():
    """Load the E5-instruct sentence transformer model."""
    print("Loading E5-instruct embedding model...", file=sys.stderr)
    # intfloat/multilingual-e5-large-instruct supports task instructions
    model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
    return model


def format_query(text):
    """Format text as query with task instruction for E5-instruct."""
    return f"Instruct: {TASK_INSTRUCTION}\nQuery: {text}"


def format_passage(text):
    """Format text as passage for E5-instruct."""
    # For passages, just add 'passage: ' prefix
    return f"passage: {text}"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start + overlap >= len(text):
            break
    return chunks


def embed_texts(model, texts, is_query=False):
    """Embed a list of texts with proper formatting for E5-instruct."""
    if is_query:
        formatted = [format_query(t) for t in texts]
    else:
        formatted = [format_passage(t) for t in texts]
    return model.encode(formatted, show_progress_bar=False, normalize_embeddings=True)


def classify_knn(query_embedding, example_embeddings, example_labels, k=3):
    """Classify using k-nearest neighbors with cosine similarity."""
    # Compute similarities
    similarities = cosine_similarity([query_embedding], example_embeddings)[0]

    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # Vote based on labels
    phishing_votes = 0
    legit_votes = 0
    neighbors = []

    for idx in top_k_indices:
        is_phishing = example_labels[idx]["is_phishing"]
        sim = similarities[idx]
        neighbors.append({
            "file": example_labels[idx]["file"],
            "label": example_labels[idx]["label"],
            "similarity": float(sim),
            "is_phishing": is_phishing
        })
        if is_phishing:
            phishing_votes += sim  # Weight by similarity
        else:
            legit_votes += sim

    is_phishing = bool(phishing_votes > legit_votes)
    confidence = max(phishing_votes, legit_votes) / (phishing_votes + legit_votes) if (phishing_votes + legit_votes) > 0 else 0

    return {
        "is_phishing": is_phishing,
        "confidence": float(confidence),
        "neighbors": neighbors
    }


def classify_with_chunking(model, content, example_embeddings, example_labels, k=3, exclude_file=None):
    """
    Classify email using chunking with any-match = phishing rule.
    If ANY chunk is classified as phishing, the whole email is phishing.

    Args:
        exclude_file: If testing a labeled sample, exclude it from kNN to avoid self-matching.
    """
    # Filter out excluded file from embeddings/labels for leave-one-out
    if exclude_file:
        filtered_embeddings = []
        filtered_labels = []
        for i, label in enumerate(example_labels):
            if label["file"] != exclude_file:
                filtered_embeddings.append(example_embeddings[i])
                filtered_labels.append(label)
        use_embeddings = np.array(filtered_embeddings) if filtered_embeddings else example_embeddings
        use_labels = filtered_labels if filtered_labels else example_labels
    else:
        use_embeddings = example_embeddings
        use_labels = example_labels

    chunks = chunk_text(content)
    chunk_results = []

    for i, chunk in enumerate(chunks):
        query_embedding = embed_texts(model, [chunk], is_query=True)[0]
        result = classify_knn(query_embedding, use_embeddings, use_labels, k=k)
        result["chunk_index"] = i
        result["chunk_preview"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
        chunk_results.append(result)

    # Any-match rule: if ANY chunk is phishing, whole email is phishing
    any_phishing = any(r["is_phishing"] for r in chunk_results)

    # Find highest confidence phishing chunk (or highest legit if none phishing)
    if any_phishing:
        phishing_chunks = [r for r in chunk_results if r["is_phishing"]]
        best_chunk = max(phishing_chunks, key=lambda x: x["confidence"])
    else:
        best_chunk = max(chunk_results, key=lambda x: x["confidence"])

    return {
        "is_phishing": any_phishing,
        "confidence": best_chunk["confidence"],
        "neighbors": best_chunk["neighbors"],
        "num_chunks": len(chunks),
        "phishing_chunks": sum(1 for r in chunk_results if r["is_phishing"]),
        "chunk_results": chunk_results
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: classify_knn.py <email.md> [email2.md ...]", file=sys.stderr)
        sys.exit(1)

    # Load model
    model = load_model()

    # Load and embed labeled examples
    example_texts = []
    example_labels = []

    for filepath, info in LABELED_EXAMPLES.items():
        path = Path(filepath)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            example_texts.append(content)
            example_labels.append({
                "file": filepath,
                "is_phishing": info["is_phishing"],
                "label": info["label"]
            })

    print(f"Loaded {len(example_texts)} labeled examples", file=sys.stderr)

    # Embed examples as passages (not queries)
    example_embeddings = embed_texts(model, example_texts, is_query=False)

    # Classify input files
    results = []
    for md_file in sys.argv[1:]:
        md_path = Path(md_file)
        if not md_path.exists():
            print(f"File not found: {md_file}", file=sys.stderr)
            continue

        content = md_path.read_text(encoding="utf-8")

        # Check if this file is in our labeled examples (for leave-one-out)
        exclude_file = None
        for label in example_labels:
            if str(md_path) == label["file"] or md_file == label["file"]:
                exclude_file = label["file"]
                break

        # Use chunking classification (with leave-one-out if testing labeled sample)
        result = classify_with_chunking(model, content, example_embeddings, example_labels, k=3, exclude_file=exclude_file)
        result["file"] = str(md_path)
        result["was_excluded"] = exclude_file is not None
        results.append(result)

        # Output
        status = "PHISHING" if result["is_phishing"] else "LEGITIMATE"
        print(f"\n{'='*60}")
        print(f"File: {result['file']}")
        print(f"Result: {status} (confidence: {result['confidence']*100:.1f}%)")
        print(f"Chunks analyzed: {result['num_chunks']} ({result['phishing_chunks']} flagged as phishing)")
        print(f"Nearest neighbors (from best matching chunk):")
        for n in result["neighbors"]:
            print(f"  - {n['file']} ({n['label']})")
            print(f"    similarity: {n['similarity']:.3f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    phishing_count = sum(1 for r in results if r["is_phishing"])
    print(f"Total: {len(results)} | Phishing: {phishing_count} | Legitimate: {len(results) - phishing_count}")

    # Save JSON (without verbose chunk_results for cleaner output)
    json_results = []
    for r in results:
        json_results.append({
            "file": r["file"],
            "is_phishing": r["is_phishing"],
            "confidence": r["confidence"],
            "neighbors": r["neighbors"],
            "num_chunks": r["num_chunks"],
            "phishing_chunks": r["phishing_chunks"]
        })

    output_path = Path("knn_results.json")
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
