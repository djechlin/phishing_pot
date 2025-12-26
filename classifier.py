"""
Phishing Email Classifier

Signals:
- url_mismatch: Rule-based check for deceptive links
- is_phishing: Embedding-based kNN using few-shot labeled examples
"""

import email
import re
import sys
from email import policy
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
TASK_INSTRUCTION = "Classify this email as phishing or legitimate based on its content, sender patterns, and suspicious indicators"

# --- LABELED EXAMPLES (few-shot learning) ---
# The instruct model learns what phishing looks like from these examples
# Balanced: 4 phishing, 4 legitimate
LABELED_EXAMPLES = {
    # Phishing - diverse scam types
    "email/sample-1.eml": {"is_phishing": True, "label": "Banking points scam"},
    "email/sample-3.eml": {"is_phishing": True, "label": "Advance fee fraud (419 scam)"},
    "email/sample-10.eml": {"is_phishing": True, "label": "Fake sign-in alert"},
    "email/sample-15.eml": {"is_phishing": True, "label": "Seed phrase theft"},
    # Legitimate - real service emails
    "email/sample-4.eml": {"is_phishing": False, "label": "Newsletter unsubscribe"},
    "email/sample-njtransit.eml": {"is_phishing": False, "label": "NJ Transit alert"},
    "email/sample-deepgram.eml": {"is_phishing": False, "label": "Deepgram product update"},
    "email/sample-chess.eml": {"is_phishing": False, "label": "Chess.com notification"},
}


# --- EMAIL PARSING ---
def parse_eml(eml_path):
    """Parse .eml file and return (text_body, html_body)"""
    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    text_body = ""
    html_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                text_body += part.get_content()
            elif ct == "text/html":
                html_body += part.get_content()
    else:
        ct = msg.get_content_type()
        if ct == "text/plain":
            text_body = msg.get_content()
        elif ct == "text/html":
            html_body = msg.get_content()

    return text_body, html_body


def get_email_text(eml_path):
    """Get combined text content from email for embedding"""
    text_body, html_body = parse_eml(eml_path)

    # Prefer plain text, fall back to stripping HTML
    if text_body:
        return text_body
    elif html_body:
        soup = BeautifulSoup(html_body, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    return ""


# --- URL MISMATCH CLASSIFIER (rule-based) ---
def extract_domain(url_string):
    """Extract domain from URL string"""
    try:
        if not re.match(r"^https?://", url_string, re.I):
            url_string = "http://" + url_string
        parsed = urlparse(url_string)
        return parsed.hostname.lower().replace("www.", "") if parsed.hostname else None
    except Exception:
        return None


def looks_like_url(text):
    """Check if text looks like a URL"""
    text = text.strip()
    return bool(re.match(r"^(https?://)?[\w][\w.-]+\.[a-z]{2,}(/\S*)?$", text, re.I))


def classify_url_mismatch(html_body):
    """Returns 1 if any link displays as a URL but points to a different domain."""
    if not html_body:
        return 0

    soup = BeautifulSoup(html_body, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        display_text = a.get_text().strip()

        if not display_text or not looks_like_url(display_text):
            continue

        href_domain = extract_domain(href)
        display_domain = extract_domain(display_text)

        if href_domain and display_domain and href_domain != display_domain:
            return 1

    return 0


# --- EMBEDDING-BASED CLASSIFIER (few-shot kNN) ---
class PhishingClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._example_embeddings = None
            cls._instance._example_labels = None
        return cls._instance

    @property
    def model(self):
        if self._model is None:
            print(f"Loading {MODEL_NAME}...", file=sys.stderr)
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _embed(self, texts, is_query=False):
        """Embed texts with E5-instruct format"""
        if is_query:
            formatted = [f"Instruct: {TASK_INSTRUCTION}\nQuery: {t}" for t in texts]
        else:
            formatted = [f"passage: {t}" for t in texts]
        return self.model.encode(formatted, normalize_embeddings=True)

    def _load_examples(self):
        """Load and embed labeled examples"""
        if self._example_embeddings is not None:
            return

        texts = []
        labels = []

        for filepath, info in LABELED_EXAMPLES.items():
            path = Path(filepath)
            if path.exists():
                content = get_email_text(filepath)
                if content:
                    texts.append(content[:4000])  # Truncate for embedding
                    labels.append({
                        "file": filepath,
                        "is_phishing": info["is_phishing"],
                        "label": info["label"],
                    })

        if texts:
            self._example_embeddings = self._embed(texts, is_query=False)
            self._example_labels = labels
            print(f"Loaded {len(texts)} labeled examples", file=sys.stderr)
        else:
            self._example_embeddings = np.array([])
            self._example_labels = []

    def classify(self, text, k=3, exclude_file=None):
        """
        Classify text using k-nearest neighbors.
        Returns (is_phishing, confidence, neighbors)
        """
        self._load_examples()

        if len(self._example_labels) == 0:
            return False, 0.0, []

        # Filter out excluded file (for leave-one-out testing)
        if exclude_file:
            mask = [l["file"] != exclude_file for l in self._example_labels]
            embeddings = self._example_embeddings[mask]
            labels = [l for l, m in zip(self._example_labels, mask) if m]
        else:
            embeddings = self._example_embeddings
            labels = self._example_labels

        if len(labels) == 0:
            return False, 0.0, []

        # Embed query
        query_emb = self._embed([text[:4000]], is_query=True)[0]

        # Compute similarities
        similarities = np.dot(embeddings, query_emb)
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        # Weighted voting
        phishing_score = 0.0
        legit_score = 0.0
        neighbors = []

        for idx in top_k_idx:
            sim = float(similarities[idx])
            label = labels[idx]
            neighbors.append({
                "file": label["file"],
                "label": label["label"],
                "similarity": sim,
                "is_phishing": label["is_phishing"],
            })
            if label["is_phishing"]:
                phishing_score += sim
            else:
                legit_score += sim

        is_phishing = phishing_score > legit_score
        total = phishing_score + legit_score
        confidence = max(phishing_score, legit_score) / total if total > 0 else 0.0

        return is_phishing, confidence, neighbors


# --- MAIN API ---
def classify_email(eml_path):
    """
    Classify email and return dict of signals.
    """
    text_body, html_body = parse_eml(eml_path)
    text = text_body or (BeautifulSoup(html_body, "html.parser").get_text() if html_body else "")

    clf = PhishingClassifier()
    is_phishing, confidence, neighbors = clf.classify(text)

    return {
        "url_mismatch": classify_url_mismatch(html_body),
        "is_phishing": 1 if is_phishing else 0,
        "confidence": confidence,
        "neighbors": neighbors,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <email.eml> [email2.eml ...] [--json]")
        sys.exit(1)

    json_mode = "--json" in sys.argv
    files = [f for f in sys.argv[1:] if not f.startswith("--")]

    all_results = []
    phishing_count = 0

    for eml_path in files:
        results = classify_email(eml_path)
        results["file"] = eml_path
        all_results.append(results)

        if results["is_phishing"]:
            phishing_count += 1

        # Print as we go - one line per email
        if not json_mode:
            status = "PHISH" if results["is_phishing"] else "LEGIT"
            nearest = results["neighbors"][0]["label"] if results["neighbors"] else "?"
            name = Path(eml_path).name
            print(f"{status} | {name} | {nearest}")
            sys.stdout.flush()

    if json_mode:
        import json
        print(json.dumps(all_results if len(all_results) > 1 else all_results[0], indent=2))
    elif len(files) > 1:
        print(f"\n--- {phishing_count}/{len(files)} phishing ---")


if __name__ == "__main__":
    main()
