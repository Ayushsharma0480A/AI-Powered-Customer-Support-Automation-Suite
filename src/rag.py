import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

KB_DIR = "data/kb_articles"
EMBED_MODEL = "all-mpnet-base-v2"


def load_kb_texts():
    files = sorted(glob.glob(os.path.join(KB_DIR, "*.txt")))
    texts = []

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            texts.append(fh.read())

    return texts, files


def build_index(texts):
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def retrieve(query, model, index, texts, files, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        raw_excerpt = texts[idx][:200]
        cleaned_excerpt = raw_excerpt.replace("\n", " ")

        results.append({
            "file": os.path.basename(files[idx]),
            "distance": float(dist),
            "excerpt": cleaned_excerpt
        })

    return results


def main():
    texts, files = load_kb_texts()

    if not texts:
        print("No KB files found in data/kb_articles/")
        return

    model, index = build_index(texts)

    queries = [
        "How do I configure automations in Hiver?",
        "Why is CSAT not appearing?"
    ]

    for q in queries:
        print("\n=== QUERY ===")
        print(q)

        results = retrieve(q, model, index, texts, files)

        for r in results:
            print(f"File: {r['file']} | Distance: {r['distance']:.4f}")
            print("Excerpt:", r["excerpt"])
            print()


if __name__ == "__main__":
    main()
