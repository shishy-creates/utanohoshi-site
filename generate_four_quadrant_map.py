"""
Generate a fixed-axis four-quadrant map for うたのほし.

Inputs:
- semantic_annotations_marco_v3_no_seasons.json (axes/tags/summary)
- discography.json (optional: to attach album names via lyric_id)

Outputs:
- four_quadrant_map.json with fixed axes:
    x = warning_to_hope (0=警鐘, 1=希望)
    y = reality_to_cosmos (0=日常, 1=宇宙)
    depth = material_to_spiritual (0=物質, 1=霊性)
  plus quadrant metadata.
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
ANNOT_PATH = PROJECT_ROOT / "semantic_annotations_marco_v3_no_seasons.json"
DISCOG_PATH = PROJECT_ROOT / "discography.json"
LYRICS_PATTERN = str(PROJECT_ROOT / "utahoshi_lyrics_*.json")
OUTPUT_PATH = PROJECT_ROOT / "four_quadrant_map.json"


def load_lyrics_ids() -> List[int]:
    files = sorted(glob.glob(LYRICS_PATTERN))
    songs = []
    for fp in files:
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        songs.extend(payload.get("songs", []))
    ids = [s["id"] for s in songs if "id" in s]
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen


def load_lyrics_map() -> Dict[int, str]:
    """Return id -> concatenated title+lyrics text for rule-based adjustment."""
    files = sorted(glob.glob(LYRICS_PATTERN))
    out = {}
    for fp in files:
        payload = json.loads(Path(fp).read_text(encoding="utf-8"))
        for s in payload.get("songs", []):
            if "id" in s:
                text = f"{s.get('title','')}\n{s.get('lyrics','')}"
                out[s["id"]] = text
    return out


def load_annotations() -> List[Dict]:
    data = json.loads(ANNOT_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Annotation file is not a list.")
    return data


def manual_missing_annotations() -> List[Dict]:
    return [
        {
            "id": 131,
            "title": "NORTH LAND -北辰の祈り-",
            "axes": {"reality_to_cosmos": 0.55, "material_to_spiritual": 0.86, "warning_to_hope": 0.55},
            "tags": ["北風", "祈り", "白銀", "耐える", "星", "再生", "光"],
            "summary": "凍てつく大地で祈りを抱きしめ、白い静寂の中から光を待つ歌。北の星に願いを託し、凍った記憶を解かしていく。",
        },
        {
            "id": 132,
            "title": "冬空のメリーゴーラウンド",
            "axes": {"reality_to_cosmos": 0.7, "material_to_spiritual": 0.82, "warning_to_hope": 0.72},
            "tags": ["冬空", "久遠", "祈り", "記憶", "再会", "宇宙の音", "旅"],
            "summary": "冬空の下、悲しみを越えて久遠の夢に身を委ねる祈りの歌。響きあう宇宙の音に導かれ、いつかの再会を信じて進む。",
        },
        {
            "id": 133,
            "title": "クオンタム　久遠魂夢",
            "axes": {"reality_to_cosmos": 0.7, "material_to_spiritual": 0.85, "warning_to_hope": 0.75},
            "tags": ["量子", "久遠", "祈り", "魂", "宇宙の音", "絆", "光"],
            "summary": "胸の奥から届く久遠の夢が、量子のように重なり合う祈りの歌。宇宙の音に溶けて、離れても結ばれる絆をたどる。",
        },
        {
            "id": 997,
            "title": "お誕生日ありがとう",
            "axes": {"reality_to_cosmos": 0.3, "material_to_spiritual": 0.62, "warning_to_hope": 0.95},
            "tags": ["誕生日", "感謝", "祝福", "出会い", "生まれる", "光"],
            "summary": "生まれてくれたことへの感謝をそっと手渡す、小さな祝福の歌。出会いの光を抱きしめ、いのちにありがとうを伝える。",
        },
        {
            "id": 998,
            "title": "おやすみ",
            "axes": {"reality_to_cosmos": 0.6, "material_to_spiritual": 0.8, "warning_to_hope": 0.8},
            "tags": ["子守歌", "感謝", "眠り", "星", "癒し", "やさしさ"],
            "summary": "眠りに落ちる前の静かな感謝と癒しの子守歌。星に見守られながら、痛みを風に預け、あしたの歌を育てる。",
        },
        {
            "id": 999,
            "title": "マイステージ",
            "axes": {"reality_to_cosmos": 0.35, "material_to_spiritual": 0.42, "warning_to_hope": 0.9},
            "tags": ["朝", "街", "ワルツ", "光", "ステージ", "小鳥", "躍動"],
            "summary": "朝の街をステージに見立て、光のワルツを踊る歌。小鳥のリズムに背中を押され、日常が静かに輝きはじめる。",
        },
    ]


def load_album_map() -> Dict:
    if not DISCOG_PATH.exists():
        return {}
    data = json.loads(DISCOG_PATH.read_text(encoding="utf-8"))
    album_map = {}
    for album in data.get("albums", []):
        album_title = album.get("title_ja") or album.get("title_en") or album.get("key")
        for track in album.get("tracks", []):
            song_id = track.get("lyric_id") or track.get("id")
            if song_id is None:
                continue
            album_map[song_id] = album_title
    return album_map


def quadrant(x: float, y: float) -> Dict:
    if x >= 0.5 and y >= 0.5:
        return {"id": 1, "name": "希望×宇宙"}
    if x < 0.5 and y >= 0.5:
        return {"id": 2, "name": "警鐘×宇宙"}
    if x < 0.5 and y < 0.5:
        return {"id": 3, "name": "警鐘×日常"}
    return {"id": 4, "name": "希望×日常"}


def adjust_reality(raw_y: float, text: str) -> float:
    """Heuristic adjustment for reality_to_cosmos based on lyrics."""
    if not text:
        return raw_y
    cosmos_terms = {
        # 遠いほど強く。単体の「星」も加点に復帰（自己比喩を含むが宇宙性として扱う）
        "宇宙": 1.0,
        "銀河": 1.0,
        "北極星": 0.9,
        "流星": 0.8,
        "星座": 0.7,
        "太陽": 0.6,
        "星": 0.6,
        "月": 0.3,
        "空": 0.2,
        "夜": 0.15,
        "光": 0.0,  # 比喩で多用されるため宇宙加点から除外
        "朝": 0.15,
        "夕焼": 0.1,
        "祈り": 0.6,
        "神": 0.6,
        "精霊": 0.6,
        "海": 0.25,
        "波": 0.25,
        "風": 0.15,
        "山": 0.15,
        "雨": 0.1,
        "雪": 0.1,
    }
    reality_terms = {
        "街": 1.0,
        "人": 0.7,
        "君": 0.8,
        "あなた": 0.8,
        "私": 0.7,
        "仕事": 1.0,
        "学校": 1.0,
        "家": 0.9,
        "部屋": 0.9,
        "電話": 0.8,
        "ビル": 0.9,
        "電車": 0.9,
        "自転車": 0.8,
        "車": 0.7,
        "トンネル": 0.8,
        "歩く": 0.7,
        "走る": 0.6,
        "待つ": 0.9,
        "会う": 0.9,
        "恋": 0.7,
        "恋人": 0.7,
        "約束": 0.6,
        "手": 0.6,
        "抱きしめ": 0.6,
        "別れ": 0.6,
        "友情": 0.7,
        "友よ": 0.7,
        "君と": 0.7,
        "君に": 0.7,
        "あなたに": 0.7,
        "花": 0.6,
        "雨": 0.6,
        "道": 0.5,
        "写真": 0.7,
        "肩": 0.5,
        "笑う": 0.6,
        "歌う": 0.6,
        # 日常的な心情・感情も現実寄りに下げる
        "不安": 0.6,
        "恐れ": 0.6,
        "怖さ": 0.5,
        "寂しさ": 0.6,
        "涙": 0.5,
        "悲しみ": 0.5,
        "後悔": 0.6,
        "思い出": 0.6,
        "胸": 0.3,
    }
    text_lower = text  # Japanese unaffected
    c_score = sum(text_lower.count(k) * w for k, w in cosmos_terms.items())
    r_score = sum(text_lower.count(k) * w for k, w in reality_terms.items())
    # 現実ワードを強めつつ、宇宙加点は控えめ
    delta = 0.01 * c_score - 0.08 * r_score
    adjusted = raw_y + delta
    return max(0.0, min(1.0, adjusted))


def adjust_warning_to_hope(raw_x: float, text: str) -> float:
    """Heuristic adjustment for warning_to_hope based on lyrics."""
    if not text:
        return raw_x
    hope_terms = {
        "希望": 1.0,
        "光": 0.8,
        "朝": 0.6,
        "笑顔": 0.6,
        "ありがとう": 0.8,
        "感謝": 0.8,
        "祈り": 0.7,
        "赦し": 0.7,
        "再生": 0.7,
        "芽吹": 0.6,
        "祝福": 0.7,
        "未来": 0.6,
        "夢": 0.5,
        "笑う": 0.5,
        "歌う": 0.5,
        "肩の力": 0.5,
    }
    warning_terms = {
        "警鐘": 1.0,
        "怒り": 0.8,
        "叫び": 0.7,
        "痛み": 0.7,
        "傷": 0.6,
        "滅び": 0.9,
        "崩れる": 0.8,
        "闇": 0.5,
        "泣く": 0.5,
        "悲しみ": 0.6,
    }
    h_score = sum(text.count(k) * w for k, w in hope_terms.items())
    w_score = sum(text.count(k) * w for k, w in warning_terms.items())
    delta = 0.03 * (h_score - w_score)
    # 感謝ボーナス: 感謝/ありがとうがあれば希望方向に+0.12加算
    if ("ありがとう" in text) or ("感謝" in text):
        delta += 0.12
    adjusted = raw_x + delta
    return max(0.0, min(1.0, adjusted))


def build_payload() -> Dict:
    anns = load_annotations()
    ann_by_id = {a["id"]: a for a in anns if "id" in a}
    for a in manual_missing_annotations():
        if a["id"] not in ann_by_id:
            ann_by_id[a["id"]] = a

    # manual overrides for reality_to_cosmos (bring everyday songs down)
    manual_reality = {
        132: 0.4,  # 冬空のメリーゴーラウンド
        13: 0.35,  # 雨上がりの朝に
        53: 0.4,   # なみのうた
        93: 0.4,   # ストーリー
        133: 0.80, # クオンタム 久遠魂夢
        123: 0.75, # 星影のメヌエット
        6: 0.70,   # 巡りの星
        81: 0.65,  # 歌の星
        20: 0.60,  # ドライビングスターズ
        5: 0.28,   # 桜散る前に
        10: 0.22,  # サカナ
        33: 0.26,  # 恋のお星さま
        35: 0.18,  # リフ
        44: 0.20,  # Chocolate
        55: 0.24,  # ゆうやけ小路
        66: 0.21,  # 風と、君と。
        73: 0.25,  # 僕らの別れ
        109: 0.19, # 帰り道
        112: 0.23, # ふたば
        116: 0.17, # color
        94: 0.50,  # ベロニカ
        46: 0.50,  # 夏のEYES
        82: 0.65,  # 照ラ  Terra
        74: 0.90,  # プラトニカ
        78: 0.304, # 幻影（手動で重ならないよう微調整）
    }

    # manual overrides for warning_to_hope (pull extreme hope values down)
    manual_warning = {
        2: 0.7,   # TOMO
        85: 0.6,  # 黎明 reimei（希望をさらに下げる）
        3: 0.15,  # ファントムシティ
        78: 0.406, # 幻影（手動で重ならないよう微調整）
    }

    lyric_ids = load_lyrics_ids()
    lyrics_map = load_lyrics_map()
    album_map = load_album_map()

    # collect raw axes for scaling
    raw_axes = defaultdict(list)  # values used for scaling (exclude manual overrides for x/y)
    for sid in lyric_ids:
        ann = ann_by_id.get(sid)
        if not ann:
            continue
        text = lyrics_map.get(sid, "")
        # blend lyric-based (80%) and raw (20%) for x,y (user: keep 30% raw -> use 0.7 lyric + 0.3 raw)
        raw_axes_val = ann.get("axes", {})
        y_lyric = adjust_reality(0.5, text)
        y_raw = float(raw_axes_val.get("reality_to_cosmos", 0.5))
        y_adj = 0.7 * y_lyric + 0.3 * y_raw
        x_lyric = adjust_warning_to_hope(0.5, text)
        x_raw = float(raw_axes_val.get("warning_to_hope", 0.5))
        x_adj = 0.7 * x_lyric + 0.3 * x_raw
        # only include non-manual values in scaling min/max
        if sid not in manual_warning:
            raw_axes["x"].append(x_adj)
        if sid not in manual_reality:
            raw_axes["y"].append(y_adj)
        # depthは元アノテーションをスケールに利用
        depth_raw = float(raw_axes_val.get("material_to_spiritual", 0.5))
        raw_axes["depth"].append(depth_raw)

    def scale(val, arr):
        lo = min(arr)
        hi = max(arr)
        if hi - lo == 0:
            return 0.5
        return (val - lo) / (hi - lo)

    def jitter(base, sid, amount=0.02):
        # deterministic tiny jitter based on id to avoid perfect overlap
        rnd = (hash(sid) % 1000) / 1000.0  # 0-0.999
        return max(0.0, min(1.0, base + (rnd - 0.5) * amount))

    songs_out = []
    q_counter = Counter()

    for sid in lyric_ids:
        ann = ann_by_id.get(sid)
        if not ann:
            continue
        text = lyrics_map.get(sid, "")
        raw_axes_val = ann.get("axes", {})
        x_lyric = adjust_warning_to_hope(0.5, text)
        y_lyric = adjust_reality(0.5, text)
        x_raw_val = float(raw_axes_val.get("warning_to_hope", 0.5))
        y_raw_val = float(raw_axes_val.get("reality_to_cosmos", 0.5))
        x_raw = 0.7 * x_lyric + 0.3 * x_raw_val
        y_raw = 0.7 * y_lyric + 0.3 * y_raw_val
        if sid in manual_reality:
            y = manual_reality[sid]
        else:
            y = jitter(scale(y_raw, raw_axes["y"]), sid)
        if sid in manual_warning:
            x = manual_warning[sid]
        else:
            x = jitter(scale(x_raw, raw_axes["x"]), sid)
        depth_raw = float(raw_axes_val.get("material_to_spiritual", 0.5))
        depth = scale(depth_raw, raw_axes["depth"])
        quad = quadrant(x, y)
        q_counter[quad["id"]] += 1
        songs_out.append(
            {
                "id": ann.get("id"),
                "title": ann.get("title"),
                "album": album_map.get(ann.get("id")),
                "x": x,
                "y": y,
                "depth": depth,
                "quadrant_id": quad["id"],
                "quadrant_name": quad["name"],
                "axes": {
                    "reality_to_cosmos": y,
                    "material_to_spiritual": depth,
                    "warning_to_hope": x,
                },
                "tags": ann.get("tags", []),
                "summary": ann.get("summary", ""),
            }
        )

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_songs": len(songs_out),
            "axes_definition": {
                "x": "warning_to_hope (0=警鐘,1=希望)",
                "y": "reality_to_cosmos (0=日常,1=宇宙)",
                "depth": "material_to_spiritual (0=物質,1=霊性)",
            },
            "quadrant_counts": dict(q_counter),
            "scaling": {
                "warning_to_hope_raw_minmax": [min(raw_axes["x"]), max(raw_axes["x"])],
                "reality_to_cosmos_raw_minmax": [min(raw_axes["y"]), max(raw_axes["y"])],
                "material_to_spiritual_raw_minmax": [min(raw_axes["depth"]), max(raw_axes["depth"])],
                "jitter": "±0.004 (deterministic by id)",
            },
        },
        "songs": songs_out,
    }
    return payload


def main():
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[output] saved {OUTPUT_PATH}")
    print(f"[meta] n_songs={payload['meta']['n_songs']} quadrant_counts={payload['meta']['quadrant_counts']}")


if __name__ == "__main__":
    main()
