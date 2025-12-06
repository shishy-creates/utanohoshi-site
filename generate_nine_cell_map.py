#!/usr/bin/env python3
"""
Generate a fresh 3x3 (nine-cell) semantic map based only on lyric keywords.

- Uses manual cell assignments (A1..C3, rows counted from top) provided in manual_cells.csv.
- Discards previous raw axes/annotations and re-scores from lyrics.
- If manual_cells.csv is missing, a template manual_cells_template.csv is created and the script exits.

Outputs:
- nine_cell_map.json with x,y in [0,1], depth for visual size, and song metadata.
"""
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
LYRICS_GLOB = str(ROOT / "utahoshi_lyrics_*.json")
DISCO_PATH = ROOT / "discography.json"
MANUAL_CELLS_PATH = ROOT / "manual_cells.csv"
TEMPLATE_PATH = ROOT / "manual_cells_template.csv"
OUTPUT_PATH = ROOT / "nine_cell_map.json"


# -----------------------------
# Data loading
# -----------------------------
def load_lyrics() -> Dict[int, Dict]:
    songs = {}
    for path in glob.glob(LYRICS_GLOB):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        for s in blob.get("songs", []):
            sid = int(s["id"])
            songs[sid] = {
                "id": sid,
                "title": s.get("title", ""),
                "album": s.get("album", ""),
                "lyrics": s.get("lyrics", ""),
            }
    return songs


def load_album_map() -> Dict[int, str]:
    if not DISCO_PATH.exists():
        return {}
    with open(DISCO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    album_map: Dict[int, str] = {}
    # 優先: albums -> tracks -> lyric_id
    for alb in data.get("albums", []):
        title = alb.get("title_ja") or alb.get("title_en") or ""
        for tr in alb.get("tracks", []):
            lid = tr.get("lyric_id")
            if lid is None:
                continue
            try:
                album_map[int(lid)] = title
            except Exception:
                continue
    # フォールバック: songs 配列があれば使う
    for s in data.get("songs", []):
        if "id" in s and s.get("album"):
            album_map[int(s["id"])] = s.get("album", "")
    return album_map


# -----------------------------
# Manual cell handling
# -----------------------------
COL_MAP = {"A": 0, "B": 1, "C": 2}


def parse_cell(cell: str) -> Tuple[int, int]:
    """Parse cell like 'A1' (top-left) .. 'C3' (bottom-right). Rows counted from top."""
    if not cell or len(cell) != 2:
        raise ValueError(f"invalid cell: {cell}")
    col_char, row_char = cell[0].upper(), cell[1]
    if col_char not in COL_MAP or row_char not in {"1", "2", "3"}:
        raise ValueError(f"invalid cell: {cell}")
    col = COL_MAP[col_char]
    row = int(row_char) - 1  # 0=top, 2=bottom
    return col, row


def ensure_manual_template(songs: Dict[int, Dict]) -> None:
    if MANUAL_CELLS_PATH.exists():
        return
    with open(TEMPLATE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "cell"])  # cell examples: A1, B2, C3
        for sid in sorted(songs):
            writer.writerow([sid, songs[sid]["title"], ""])
    print(f"[info] manual_cells.csv not found. Created template at {TEMPLATE_PATH.name}.")
    print("      Fill the 'cell' column with A1..C3 (row is counted from top).")


def load_manual_cells() -> Dict[int, Tuple[int, int]]:
    mapping: Dict[int, Tuple[int, int]] = {}
    with open(MANUAL_CELLS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id") or not row.get("cell"):
                continue
            sid = int(row["id"])
            mapping[sid] = parse_cell(row["cell"])
    return mapping


# -----------------------------
# Keyword scoring
# -----------------------------
def count_hits(text: str, words: List[str]) -> int:
    return sum(text.count(w) for w in words)


HOPE_WORDS = [
    "希望", "祈り", "祈る", "ありがとう", "感謝", "愛", "赦し", "祝福",
    "光", "朝焼け", "笑う", "歌う", "抱きしめ", "未来", "芽吹き", "微笑",
]
WARNING_WORDS = [
    "涙", "痛み", "失う", "別れ", "悲しみ", "叫ぶ", "孤独", "影", "苦しみ",
    "恐れ", "終わり", "崩れる", "嘆き", "寂しさ", "暗闇",
]
COSMIC_WORDS = [
    "星", "夜空", "宇宙", "銀河", "流星", "月", "太陽", "空", "天",
    "惑星", "北極星", "星影", "星座", "光年", "オーロラ", "宙",
]
EVERYDAY_WORDS = [
    "街", "駅", "帰り道", "家", "部屋", "肩", "写真", "手紙", "雨", "花",
    "バス", "電車", "学校", "友", "友だち", "君", "人", "声", "足音",
    "日々", "窓", "カフェ", "公園", "通り", "歩道",
]
SPIRITUAL_WORDS = [
    "祈り", "神", "魂", "心", "光", "精霊", "天", "祝福", "赦し", "加護",
    "祈る", "守る", "抱く", "導く",
]


def score_from_lyrics(text: str) -> Tuple[float, float, float]:
    """Return (x_score, y_score, depth_score) in 0..1 based on keywords only."""
    hope = count_hits(text, HOPE_WORDS)
    warn = count_hits(text, WARNING_WORDS)
    cosmic = count_hits(text, COSMIC_WORDS)
    daily = count_hits(text, EVERYDAY_WORDS)
    spirit = count_hits(text, SPIRITUAL_WORDS)

    # map counts to a bounded delta; small step to avoid saturation
    x = 0.5 + 0.08 * (hope - warn)
    y = 0.5 + 0.08 * (cosmic - daily)
    depth = 0.4 + 0.07 * spirit
    # clamp
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    depth = max(0.0, min(1.0, depth))
    return x, y, depth


# -----------------------------
# Positioning within 3x3 grid
# -----------------------------
CELL_SIZE = 1.0 / 3.0


def place_in_cell(col: int, row: int, x_score: float, y_score: float) -> Tuple[float, float]:
    """Map normalized scores into the specified cell, allowing slight overlap across borders then clamping to [0,1]."""
    base_x = col * CELL_SIZE
    base_y = (2 - row) * CELL_SIZE  # row: 0=top, 2=bottom
    overlap = 0.08  # allow a larger bleed outside the cell
    jitter = 0.012
    # expand cell bounds slightly, then clamp later to global [0,1]
    eff_x = base_x - overlap
    eff_y = base_y - overlap
    eff_size = CELL_SIZE + overlap * 2
    x = eff_x + eff_size * x_score + random.uniform(-jitter, jitter)
    y = eff_y + eff_size * y_score + random.uniform(-jitter, jitter)
    # pull gently toward global center to avoid corner piling
    bias = 0.1
    x = x + (0.5 - x) * bias
    y = y + (0.5 - y) * bias
    # widen horizontal spread for middle column (B)
    if col == 1:
        x += random.uniform(-0.035, 0.035)
    # soft margin to avoid piling on edges
    margin = 0.02
    if x < margin:
        x = margin + random.uniform(0, margin * 0.5)
    elif x > 1 - margin:
        x = 1 - margin - random.uniform(0, margin * 0.5)
    if y < margin:
        y = margin + random.uniform(0, margin * 0.5)
    elif y > 1 - margin:
        y = 1 - margin - random.uniform(0, margin * 0.5)
    # final clamp
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return x, y


# -----------------------------
# Main builder
# -----------------------------
def build_map():
    songs = load_lyrics()
    if not songs:
        print("[error] No lyrics found.")
        sys.exit(1)

    # manual album overrides for missing entries
    manual_album = {
        75: "メッセージ",  # 春とルルリラ
        103: "歌の癒しシリーズ",  # あいのしるしはイノチではなくこちら
    }

    ensure_manual_template(songs)
    if not MANUAL_CELLS_PATH.exists():
        sys.exit(0)

    manual_cells = load_manual_cells()
    if not manual_cells:
        print("[error] manual_cells.csv has no assignments. Fill it first.")
        sys.exit(1)

    albums = load_album_map()

    results = []
    missing = []
    for sid, meta in songs.items():
        if sid not in manual_cells:
            missing.append((sid, meta["title"]))
            continue
        col, row = manual_cells[sid]
        x_score, y_score, depth = score_from_lyrics(meta["lyrics"])
        # shrink spread to keep points away from absolute edges (0.1..0.9 range)
        def narrow(v: float, span: float = 0.8) -> float:
            # span=0.8 -> 0.5 +/- 0.4 => 0.1..0.9
            return max(0.0, min(1.0, 0.5 + (v - 0.5) * span))

        x_score = narrow(x_score)
        y_score = narrow(y_score)
        x, y = place_in_cell(col, row, x_score, y_score)
        album_name = manual_album.get(sid) or albums.get(sid) or meta.get("album") or ""
        results.append(
            {
                "id": sid,
                "title": meta["title"],
                "album": album_name,
                "x": round(x, 6),
                "y": round(y, 6),
                "depth": round(depth, 6),
                "cell": f"{list(COL_MAP.keys())[col]}{row+1}",
                "scores": {
                    "warning_to_hope": x_score,
                    "reality_to_cosmos": y_score,
                    "material_to_spiritual": depth,
                },
            }
        )

    if missing:
        print("[warn] missing manual cells for the following songs (id, title):")
        for sid, title in sorted(missing):
            print(f"  {sid}, {title}")
        print("      Add them to manual_cells.csv and rerun.")
        sys.exit(1)

    payload = {
        "meta": {
            "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "n_songs": len(results),
            "grid": "A1..C3 (rows counted from top)",
            "axes_definition": {
                "x": "warning_to_hope (0=警鐘,1=希望)",
                "y": "reality_to_cosmos (0=日常,1=宇宙)",
                "depth": "material_to_spiritual proxy from spiritual keywords",
            },
        },
        "songs": sorted(results, key=lambda s: s["id"]),
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ok] saved {OUTPUT_PATH} (n_songs={len(results)})")


if __name__ == "__main__":
    random.seed(42)  # deterministic jitter
    build_map()
