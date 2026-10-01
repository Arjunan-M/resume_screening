import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Cache loaded models by name so switching models in the UI doesn't reload
# from disk every single time this module is used.
_model_cache = {}
def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]
def embed_texts(texts: list, model_name: str = "all-MiniLM-L6-v2"):
    model = load_model(model_name)
    return model.encode(texts, normalize_embeddings=True)


def _split_into_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
                if sentence.strip()
            )
    return sentences


def semantic_chunk_text(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.55,
    max_sentences: int = 5,
    max_chars: int = 1200,
) -> list[str]:
    """Group adjacent sentences while their embedding meaning stays similar."""
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    sentence_vectors = embed_texts(sentences, model_name)
    chunks = []
    current_sentences = []
    current_vectors = []

    for sentence, vector in zip(sentences, sentence_vectors):
        current_text = " ".join(current_sentences)
        if current_sentences:
            centroid = np.mean(current_vectors, axis=0)
            centroid /= np.linalg.norm(centroid)
            similarity = float(np.dot(vector, centroid))
            exceeds_limits = (
                len(current_sentences) >= max_sentences
                or len(current_text) + len(sentence) + 1 > max_chars
            )
            if similarity < similarity_threshold or exceeds_limits:
                chunks.append(current_text)
                current_sentences = []
                current_vectors = []

        current_sentences.append(sentence)
        current_vectors.append(vector)

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def embed_resume_json(
    json_path: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.55,
) -> tuple[dict, list[str], np.ndarray]:
    """Load extracted resume text from JSON, semantically chunk, and embed it."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    text = data.get("extracted_text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The JSON file does not contain non-empty extracted_text.")

    chunks = semantic_chunk_text(
        text,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
    )
    return data, chunks, embed_texts(chunks, model_name)


# Quick manual test: `python embedder.py`
if __name__ == "__main__":
    sample_texts = [
        "Experienced Python developer skilled in machine learning.",
        "Looking for a data scientist with ML and Python experience.",
    ]
    vectors = embed_texts(sample_texts)
    print(f"Embedded {len(sample_texts)} texts into shape {vectors.shape}")
