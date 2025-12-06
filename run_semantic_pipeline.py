"""
Semantic pipeline (offline) for うたのほし:
1) Load lyrics JSONs, build heuristic semantic annotations (axes/seasons/tags/summary).
2) Save semantic_annotations.json.
3) Build feature vectors (axes + seasons + tag one-hot), reduce to 2D (UMAP if available else PCA).
4) k-means clustering with auto k (6–14), pick best via silhouette + Davies–Bouldin.
5) Save constellations_auto.json for frontend.

Note: This pipeline avoids external APIs; annotations are heuristic (keyword-weighted).
"""
from __future__ import annotations

import glob
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
LYRICS_PATTERN = str(PROJECT_ROOT / "utahoshi_lyrics_*.json")
ANNOT_PATH = PROJECT_ROOT / "semantic_annotations.json"
ANNOT_EXTERNAL_CANDIDATES = [
    PROJECT_ROOT / "semantic_annotations_marco_v3_no_seasons.json",
    PROJECT_ROOT / "semantic_annotations_marco_v2.json",
]
CONST_PATH = PROJECT_ROOT / "constellations_auto.json"


# ---------------------------
# Helpers
# ---------------------------

def load_lyrics_df() -> pd.DataFrame:
    files = sorted(glob.glob(LYRICS_PATTERN))
    if not files:
        raise FileNotFoundError("No lyrics files found.")
    songs = []
    for fp in files:
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        songs.extend(payload.get("songs", []))
    df = pd.DataFrame(songs)
    keep_cols = ["id", "title", "album", "date", "lyrics"]
    df = df[keep_cols]
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    return df


def make_keyword_weights() -> Dict[str, Dict[str, float]]:
    return {
        "cosmos": {
            "星": 1.0,
            "宇宙": 1.0,
            "銀河": 1.0,
            "月": 0.8,
            "太陽": 0.6,
            "空": 0.6,
            "光": 0.6,
            "流星": 0.8,
            "夜空": 0.8,
        },
        "reality": {
            "街": 0.8,
            "人": 0.5,
            "君": 0.5,
            "私": 0.5,
            "今日": 0.5,
            "手": 0.4,
            "足": 0.4,
        },
        "spiritual": {
            "祈り": 1.0,
            "魂": 1.0,
            "神": 1.0,
            "精霊": 1.0,
            "祝福": 0.8,
            "赦し": 0.8,
            "祈る": 0.8,
        },
        "material": {
            "欲": 0.6,
            "金": 0.6,
            "身体": 0.4,
            "現実": 0.5,
            "ビル": 0.6,
            "機械": 0.5,
        },
        "hope": {
            "光": 1.0,
            "朝": 0.8,
            "希望": 1.0,
            "未来": 0.8,
            "再生": 0.8,
            "芽吹": 0.8,
            "笑顔": 0.6,
            "祝福": 0.8,
            "癒し": 0.8,
        },
        "warning": {
            "警鐘": 1.0,
            "怒り": 0.8,
            "叫び": 0.7,
            "泣く": 0.6,
            "痛み": 0.6,
            "傷": 0.6,
            "滅び": 0.9,
            "崩れる": 0.7,
            "闇": 0.6,
        },
        "season_spring": {"春": 1.0, "桜": 0.8, "芽吹": 0.6, "花": 0.6},
        "season_summer": {"夏": 1.0, "海": 0.8, "波": 0.8, "青": 0.4, "光": 0.4},
        "season_autumn": {"秋": 1.0, "紅葉": 0.8, "枯": 0.8, "夕焼け": 0.7},
        "season_winter": {"冬": 1.0, "雪": 0.9, "凍": 0.7, "白": 0.5, "寒": 0.7},
    }


STOPWORDS = {
    "こと",
    "もの",
    "それ",
    "これ",
    "よう",
    "ため",
    "君",
    "僕",
    "私",
    "あなた",
    "いる",
    "ある",
    "する",
    "して",
    "いた",
    "なる",
    "そう",
    "から",
    "には",
    "あと",
    "まだ",
    "でも",
    "そして",
    "たち",
    "いつ",
    "どこ",
    "なに",
    "どんな",
    "ひと",
}


def tokenize(text: str) -> List[str]:
    # Very light tokenizer: split on non-word characters, keep Japanese.
    return [t for t in re.split(r"[\\s\\n\\r\\t、。・，．！？!?,.\\-\\(\\)\\[\\]\"'“”‘’]", text) if t]


def score_axis(text: str, pos_weights: Dict[str, float], neg_weights: Dict[str, float]) -> float:
    pos = sum(w for k, w in pos_weights.items() if k in text)
    neg = sum(w for k, w in neg_weights.items() if k in text)
    total = pos + neg
    if total == 0:
        return 0.5
    return max(0.0, min(1.0, pos / total))


def score_season(text: str, weights: Dict[str, float]) -> float:
    s = sum(w for k, w in weights.items() if k in text)
    return max(0.0, min(1.0, s / 3.0))


def annotate_song(row, tfidf_terms: List[str]) -> Dict:
    text = f"{row['title']}\\n{row['lyrics']}"
    kw = make_keyword_weights()

    axes = {
        "reality_to_cosmos": score_axis(text, kw["cosmos"], kw["reality"]),
        "material_to_spiritual": score_axis(text, kw["spiritual"], kw["material"]),
        "warning_to_hope": score_axis(text, kw["hope"], kw["warning"]),
    }
    seasons = {
        "spring": score_season(text, kw["season_spring"]),
        "summer": score_season(text, kw["season_summer"]),
        "autumn": score_season(text, kw["season_autumn"]),
        "winter": score_season(text, kw["season_winter"]),
    }

    tokens = [t for t in tfidf_terms if t not in STOPWORDS and len(t) >= 2][:7]
    if len(tokens) < 3:
        tokens += ["祈り", "光"][: 3 - len(tokens)]

    summary = f"{row['title']}は、{tokens[0]}や{tokens[1] if len(tokens)>1 else '祈り'}を軸に、静かに光へ向かう歌。{tokens[-1]}の余韻が心を澄ませる。"

    return {
        "id": row["id"],
        "title": row["title"],
        "axes": axes,
        "seasons": seasons,
        "tags": tokens,
        "summary": summary,
    }


def build_annotations(df: pd.DataFrame) -> List[Dict]:
    # Use TF-IDF to extract per-song top terms
    corpus = (df["title"].fillna("") + "\\n" + df["lyrics"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)[\\w一-龯ぁ-んァ-ヴー]+",
        ngram_range=(1, 2),
        max_features=8000,
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    annotations = []
    for idx, row in df.iterrows():
        row_vec = tfidf[idx].toarray().ravel()
        top_idx = row_vec.argsort()[::-1]
        top_terms = [feature_names[i] for i in top_idx if row_vec[i] > 0][:10]
        ann = annotate_song(row, top_terms)
        annotations.append(ann)
    return annotations


def save_annotations(annotations: List[Dict], path: Path = ANNOT_PATH):
    path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[annotations] saved {len(annotations)} entries -> {path}")


def normalize_annotation(ann: Dict) -> Dict:
    axes = ann.get("axes", {})
    axes = {
        "reality_to_cosmos": float(axes.get("reality_to_cosmos", 0.5)),
        "material_to_spiritual": float(axes.get("material_to_spiritual", 0.5)),
        "warning_to_hope": float(axes.get("warning_to_hope", 0.5)),
    }
    tags = ann.get("tags", [])
    summary = ann.get("summary", "")
    seasons = ann.get("seasons") or {"spring": 0.0, "summer": 0.0, "autumn": 0.0, "winter": 0.0}
    norm = {
        "id": ann["id"],
        "title": ann.get("title"),
        "axes": axes,
        "seasons": seasons,
        "tags": tags,
        "summary": summary,
    }
    return norm


def load_external_annotations() -> List[Dict]:
    for candidate in ANNOT_EXTERNAL_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, list):
                print(f"[annotations] loaded external {candidate} ({len(data)})")
                return [normalize_annotation(a) for a in data if "id" in a]
        except Exception as exc:
            print(f"[annotations] failed to load {candidate}: {exc}")
    return []


# ---------------------------
# Features & clustering
# ---------------------------

def build_feature_matrix(annotations: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    # axes + tag one-hot (seasons excluded per latest spec)
    tag_counter = Counter()
    for ann in annotations:
        tag_counter.update(ann["tags"])
    # keep reasonably frequent tags
    tag_vocab = [t for t, c in tag_counter.items() if c >= 1]
    tag_index = {t: i for i, t in enumerate(tag_vocab)}
    feats = []
    for ann in annotations:
        axes = ann["axes"]
        vec = [
            axes["reality_to_cosmos"],
            axes["material_to_spiritual"],
            axes["warning_to_hope"],
        ]
        tag_vec = [0.0] * len(tag_vocab)
        for t in ann["tags"]:
            if t in tag_index:
                tag_vec[tag_index[t]] = 1.0
        feats.append(vec + tag_vec)
    return np.array(feats, dtype=float), tag_vocab


def reduce_2d(matrix: np.ndarray) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.15)
        coords = reducer.fit_transform(matrix)
        print("[reduce] using UMAP")
    except Exception as exc:
        print(f"[reduce] UMAP unavailable ({exc}), using PCA")
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    coords = StandardScaler().fit_transform(coords)
    return coords


def choose_k(matrix: np.ndarray, k_min: int = 6, k_max: int = 14) -> Tuple[int, pd.DataFrame]:
    ks = [k for k in range(k_min, k_max + 1) if k < matrix.shape[0]]
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(matrix)
        sil = silhouette_score(matrix, labels)
        db = davies_bouldin_score(matrix, labels)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        print(f"[k-eval] k={k:02d} silhouette={sil:.4f} DB={db:.4f}")
    scores = pd.DataFrame(rows)
    best = scores.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]
    return int(best["k"]), scores


def cluster(matrix: np.ndarray, k: int) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=15, random_state=42)
    return model.fit_predict(matrix)


def extract_top_keywords(annotations: List[Dict], labels: np.ndarray, top_n: int = 8) -> Dict[int, List[str]]:
    label_to_tags = defaultdict(Counter)
    for ann, lbl in zip(annotations, labels):
        label_to_tags[lbl].update(ann["tags"])
    top = {}
    for lbl, counter in label_to_tags.items():
        top[lbl] = [t for t, _ in counter.most_common(top_n)]
    return top


# ---------------------------
# Build constellations
# ---------------------------

def build_constellations(df: pd.DataFrame, annotations: List[Dict], coords: np.ndarray, labels: np.ndarray, k: int) -> Dict:
    df_out = df.copy()
    df_out["cluster"] = labels
    df_out["x"] = coords[:, 0]
    df_out["y"] = coords[:, 1]
    keywords = extract_top_keywords(annotations, labels)

    constellations = []
    for idx, cluster_id in enumerate(sorted(df_out["cluster"].unique())):
        subset = df_out[df_out["cluster"] == cluster_id]
        songs = [
            {
                "id": int(s["id"]),
                "title": s["title"],
                "album": s["album"],
                "x": float(s["x"]),
                "y": float(s["y"]),
            }
            for s in subset.to_dict(orient="records")
        ]
        constellations.append(
            {
                "id": f"const_{idx + 1:02d}",
                "name": f"Constellation {idx + 1:02d}",
                "top_keywords": keywords.get(cluster_id, []),
                "songs": songs,
            }
        )
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_songs": int(len(df_out)),
            "n_constellations": int(k),
            "method": "semantic_axes_v1",
        },
        "constellations": constellations,
    }
    CONST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[output] saved {CONST_PATH}")
    return payload


def main():
    print("[step] load lyrics")
    df = load_lyrics_df()
    print(f"[step] songs: {len(df)}")

    annotations = load_external_annotations()
    if annotations:
        print("[step] using external annotations")
        ann_by_id = {a["id"]: a for a in annotations if "id" in a}
        # add manual fallback entries for missing ids
        manual_add = {
            131: {
                "id": 131,
                "title": "NORTH LAND -北辰の祈り-",
                "axes": {"reality_to_cosmos": 0.72, "material_to_spiritual": 0.86, "warning_to_hope": 0.55},
                "seasons": {"spring": 0.05, "summer": 0.05, "autumn": 0.15, "winter": 0.9},
                "tags": ["北風", "祈り", "白銀", "耐える", "星", "再生", "光"],
                "summary": "凍てつく大地で祈りを抱きしめ、白い静寂の中から光を待つ歌。北の星に願いを託し、凍った記憶を解かしていく。",
            },
            132: {
                "id": 132,
                "title": "冬空のメリーゴーラウンド",
                "axes": {"reality_to_cosmos": 0.8, "material_to_spiritual": 0.82, "warning_to_hope": 0.72},
                "seasons": {"spring": 0.1, "summer": 0.05, "autumn": 0.2, "winter": 0.9},
                "tags": ["冬空", "久遠", "祈り", "記憶", "再会", "宇宙の音", "旅"],
                "summary": "冬空の下、悲しみを越えて久遠の夢に身を委ねる祈りの歌。響きあう宇宙の音に導かれ、いつかの再会を信じて進む。",
            },
            133: {
                "id": 133,
                "title": "クオンタム　久遠魂夢",
                "axes": {"reality_to_cosmos": 0.82, "material_to_spiritual": 0.85, "warning_to_hope": 0.75},
                "seasons": {"spring": 0.1, "summer": 0.05, "autumn": 0.2, "winter": 0.85},
                "tags": ["量子", "久遠", "祈り", "魂", "宇宙の音", "絆", "光"],
                "summary": "胸の奥から届く久遠の夢が、量子のように重なり合う祈りの歌。宇宙の音に溶けて、離れても結ばれる絆をたどる。",
            },
            997: {
                "id": 997,
                "title": "お誕生日ありがとう",
                "axes": {"reality_to_cosmos": 0.3, "material_to_spiritual": 0.62, "warning_to_hope": 0.95},
                "seasons": {"spring": 0.35, "summer": 0.25, "autumn": 0.2, "winter": 0.3},
                "tags": ["誕生日", "感謝", "祝福", "出会い", "生まれる", "光"],
                "summary": "生まれてくれたことへの感謝をそっと手渡す、小さな祝福の歌。出会いの光を抱きしめ、いのちにありがとうを伝える。",
            },
            998: {
                "id": 998,
                "title": "おやすみ",
                "axes": {"reality_to_cosmos": 0.6, "material_to_spiritual": 0.8, "warning_to_hope": 0.8},
                "seasons": {"spring": 0.25, "summer": 0.15, "autumn": 0.3, "winter": 0.6},
                "tags": ["子守歌", "感謝", "眠り", "星", "癒し", "やさしさ"],
                "summary": "眠りに落ちる前の静かな感謝と癒しの子守歌。星に見守られながら、痛みを風に預け、あしたの歌を育てる。",
            },
            999: {
                "id": 999,
                "title": "マイステージ",
                "axes": {"reality_to_cosmos": 0.35, "material_to_spiritual": 0.42, "warning_to_hope": 0.9},
                "seasons": {"spring": 0.6, "summer": 0.45, "autumn": 0.2, "winter": 0.1},
                "tags": ["朝", "街", "ワルツ", "光", "ステージ", "小鳥", "躍動"],
                "summary": "朝の街をステージに見立て、光のワルツを踊る歌。小鳥のリズムに背中を押され、日常が静かに輝きはじめる。",
            },
        }
        for mid, item in manual_add.items():
            if mid not in ann_by_id:
                ann_by_id[mid] = normalize_annotation(item)
        df = df[df["id"].isin(ann_by_id.keys())].reset_index(drop=True)
        annotations = [ann_by_id[i] for i in df["id"].tolist() if i in ann_by_id]
        print(f"[step] aligned songs to annotations: {len(annotations)}")
    else:
        print("[step] build annotations (heuristic)")
        annotations = build_annotations(df)
        save_annotations(annotations)

    print("[step] build feature matrix")
    feats, tag_vocab = build_feature_matrix(annotations)
    print(f"[features] shape {feats.shape}, tags={len(tag_vocab)}")

    coords = reduce_2d(feats)
    best_k, score_df = choose_k(feats)
    labels = cluster(feats, best_k)
    print("[scores]\\n", score_df)

    payload = build_constellations(df, annotations, coords, labels, best_k)
    print("[sample constellation]", json.dumps(payload["constellations"][0], ensure_ascii=False, indent=2))
    print("[meta]", payload["meta"])


if __name__ == "__main__":
    main()
