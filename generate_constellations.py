"""
Semantic clustering pipeline for うたのほし lyrics.

Phases:
1. Load discography and lyrics into a single DataFrame.
2. Vectorize text with TF-IDF (word-level char-inclusive tokens) + optional SVD.
3. Probe k in [6, 14] via k-means using silhouette and Davies-Bouldin scores.
4. Reduce to 2D via UMAP (if available) or PCA for map coordinates.
5. Derive top keywords per cluster from TF-IDF features.
6. Build constellations_auto.json and print a quick sanity sample.
"""
from __future__ import annotations

import glob
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DISCOG_PATH = PROJECT_ROOT / "discography.json"
LYRICS_PATTERN = str(PROJECT_ROOT / "utahoshi_lyrics_*.json")
OUTPUT_PATH = PROJECT_ROOT / "constellations_auto.json"


@dataclass
class VectorizerBundle:
    tfidf_matrix: any
    feature_names: List[str]
    reduced_matrix: np.ndarray
    used_svd: bool
    method: str
    embedding_model: Optional[str] = None


def load_discography(path: Path) -> pd.DataFrame:
    """Flatten discography albums/tracks into a DataFrame keyed by lyric_id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for album in data.get("albums", []):
        album_title = album.get("title_ja") or album.get("title_en") or album.get("key")
        album_year = album.get("year")
        for track in album.get("tracks", []):
            song_id = track.get("lyric_id") or track.get("id")
            if song_id is None:
                continue
            records.append(
                {
                    "id": song_id,
                    "title_discog": track.get("title_ja")
                    or track.get("title_en")
                    or track.get("title"),
                    "album_discog": album_title,
                    "year_discog": album_year,
                }
            )
    df = pd.DataFrame(records)
    print(f"[load_discography] entries: {len(df)}")
    return df


def load_lyrics(pattern: str) -> pd.DataFrame:
    """Load all lyrics files matching the pattern."""
    songs: List[Dict] = []
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No lyrics files found.")
    for fp in files:
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        songs.extend(payload.get("songs", []))
    df = pd.DataFrame(songs)
    df = df.rename(
        columns={
            "id": "id",
            "title": "title",
            "album": "album",
            "date": "date",
            "lyrics": "lyrics",
        }
    )
    print(f"[load_lyrics] files: {len(files)}, songs: {len(df)}")
    return df[["id", "title", "album", "date", "lyrics"]]


def embed_openai(texts: List[str], model: str) -> np.ndarray:
    """Embed texts via OpenAI API. Requires OPENAI_API_KEY."""
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is not installed. pip install openai") from exc

    client = OpenAI()
    embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        embeddings.extend([item.embedding for item in resp.data])
    return np.asarray(embeddings)


def build_dataset() -> pd.DataFrame:
    discog_df = load_discography(DISCOG_PATH)
    if not discog_df.empty:
        discog_df = discog_df.drop_duplicates(subset="id")
    lyrics_df = load_lyrics(LYRICS_PATTERN).drop_duplicates(subset="id")

    merged = lyrics_df.merge(discog_df, how="left", on="id")

    merged["title"] = merged["title"].fillna(merged["title_discog"])
    merged["album"] = merged["album"].fillna(merged["album_discog"])
    merged["date"] = merged["date"].fillna(merged["year_discog"])

    merged = merged.drop(columns=["title_discog", "album_discog", "year_discog"])
    numeric_ids = pd.to_numeric(merged["id"], errors="coerce")
    merged["id"] = numeric_ids.where(~numeric_ids.isna(), merged["id"])
    merged = merged.sort_values("id").reset_index(drop=True)

    print("[build_dataset] shape:", merged.shape)
    print(merged.head())
    return merged


def vectorize_texts(texts: pd.Series, method: str, embedding_model: Optional[str]) -> VectorizerBundle:
    """
    Vectorize texts.
    - Always compute TF-IDF (for keywords)
    - For clustering: use OpenAI embeddings if method == 'openai', otherwise SVD-reduced TF-IDF.
    """
    token_pattern = r"(?u)[\w一-龯ぁ-んァ-ヴー]+"
    vectorizer = TfidfVectorizer(
        token_pattern=token_pattern,
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
    )
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()
    print(f"[vectorize] tfidf shape: {tfidf.shape}")

    if method == "openai":
        embeddings = embed_openai(texts.tolist(), model=embedding_model)
        reduced = embeddings
        used_svd = False
        print(f"[vectorize] using OpenAI embeddings: {embeddings.shape}")
    else:
        used_svd = False
        reduced = tfidf
        if tfidf.shape[1] > 2:
            n_components = min(300, tfidf.shape[1] - 1)
            if n_components >= 2:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                reduced = svd.fit_transform(tfidf)
                used_svd = True
                print(f"[vectorize] SVD reduced shape: {reduced.shape}")

    return VectorizerBundle(
        tfidf_matrix=tfidf,
        feature_names=feature_names,
        reduced_matrix=np.asarray(reduced),
        used_svd=used_svd,
        method=method,
        embedding_model=embedding_model if method == "openai" else None,
    )


def choose_k(matrix: np.ndarray, k_min: int = 6, k_max: int = 14) -> Tuple[int, pd.DataFrame]:
    """Evaluate k-means for k in range and return best k plus score table."""
    n_samples = matrix.shape[0]
    ks = [k for k in range(k_min, k_max + 1) if k < n_samples]
    if not ks:
        raise ValueError("Not enough samples to cluster.")

    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        sil = silhouette_score(matrix, labels)
        db = davies_bouldin_score(matrix, labels)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        print(f"[k-eval] k={k:02d} silhouette={sil:.4f} DB={db:.4f}")

    scores_df = pd.DataFrame(rows)
    best = scores_df.sort_values(by=["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    best_k = int(best["k"])
    print(f"[choose_k] best k={best_k}")
    return best_k, scores_df


def cluster(matrix: np.ndarray, k: int) -> np.ndarray:
    model = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = model.fit_predict(matrix)
    return labels


def reduce_to_2d(matrix: np.ndarray) -> np.ndarray:
    """Project to 2D using UMAP if available, else PCA."""
    coords = None
    try:
        import umap

        reducer = umap.UMAP(random_state=42)
        coords = reducer.fit_transform(matrix)
        print("[reduce] used UMAP")
    except Exception as exc:
        print(f"[reduce] UMAP unavailable ({exc}), falling back to PCA")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(matrix)

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    return coords_scaled


def extract_top_keywords(
    tfidf,
    feature_names: List[str],
    labels: np.ndarray,
    top_n: int = 8,
    max_token_len: int = 12,
) -> Dict[int, List[str]]:
    """Return top keywords per cluster using mean TF-IDF scores."""
    stopwords = {
        "こと",
        "よう",
        "もの",
        "それ",
        "これ",
        "ため",
        "なに",
        "君",
        "僕",
        "私",
        "あなた",
        "いる",
        "ある",
        "なる",
        "する",
        "して",
        "今日",
        "明日",
        "昨日",
    }
    top_keywords: Dict[int, List[str]] = {}
    labels_unique = sorted(np.unique(labels))
    for label in labels_unique:
        mask = labels == label
        cluster_tfidf = tfidf[mask]
        scores = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        keywords = []
        for idx in scores.argsort()[::-1]:
            token = feature_names[idx]
            if token in stopwords or len(token) > max_token_len or len(token) < 2:
                continue
            keywords.append(token)
            if len(keywords) >= top_n:
                break
        top_keywords[label] = keywords
    return top_keywords


def build_output(
    df: pd.DataFrame,
    labels: np.ndarray,
    coords: np.ndarray,
    top_keywords: Dict[int, List[str]],
    k: int,
    method: str,
    embedding_model: Optional[str],
) -> Dict:
    df_out = df.copy()
    df_out["cluster"] = labels
    df_out["x"] = coords[:, 0]
    df_out["y"] = coords[:, 1]

    constellations = []
    for idx, cluster_id in enumerate(sorted(df_out["cluster"].unique())):
        subset = df_out[df_out["cluster"] == cluster_id]
        songs = [
            {
                "id": s["id"],
                "title": s["title"],
                "album": (None if pd.isna(s["album"]) else s["album"]),
                "date": (None if pd.isna(s["date"]) else s["date"]),
                "x": float(s["x"]),
                "y": float(s["y"]),
            }
            for s in subset.to_dict(orient="records")
        ]
        constellations.append(
            {
                "id": f"const_{idx + 1:02d}",
                "provisional_name": f"Constellation {idx + 1:02d}",
                "top_keywords": top_keywords.get(cluster_id, []),
                "songs": songs,
            }
        )

    payload = {
        "meta": {
            "method": "semantic_clustering_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_songs": int(len(df_out)),
            "n_constellations": int(k),
            "vectorizer": method,
            "embedding_model": embedding_model,
        },
        "constellations": constellations,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Semantic clustering for うたのほし")
    parser.add_argument("--method", choices=["tfidf", "openai"], default="tfidf", help="Vectorization method for clustering.")
    parser.add_argument("--openai-model", default="text-embedding-3-large", help="OpenAI embedding model (when --method openai).")
    parser.add_argument("--kmin", type=int, default=6, help="Minimum k to evaluate.")
    parser.add_argument("--kmax", type=int, default=14, help="Maximum k to evaluate.")
    args = parser.parse_args()

    df = build_dataset()
    texts = df["title"].fillna("") + "\n" + df["lyrics"].fillna("")

    vector_bundle = vectorize_texts(texts, method=args.method, embedding_model=args.openai_model)
    best_k, scores_df = choose_k(vector_bundle.reduced_matrix, k_min=args.kmin, k_max=args.kmax)
    labels = cluster(vector_bundle.reduced_matrix, best_k)
    coords = reduce_to_2d(vector_bundle.reduced_matrix)
    keywords = extract_top_keywords(vector_bundle.tfidf_matrix, vector_bundle.feature_names, labels)

    output = build_output(df, labels, coords, keywords, best_k, method=args.method, embedding_model=args.openai_model if args.method == "openai" else None)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"[output] saved to {OUTPUT_PATH}")
    print("[output] score table:")
    print(scores_df)
    print("[output] first constellation sample:")
    print(json.dumps(output["constellations"][0], ensure_ascii=False, indent=2))
    print("[meta]", output["meta"])


if __name__ == "__main__":
    main()
