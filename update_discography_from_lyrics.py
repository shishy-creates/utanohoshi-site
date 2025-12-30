#!/usr/bin/env python3
import json
from pathlib import Path

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def youtube_to_embed(url: str) -> str | None:
    if not url or not isinstance(url, str):
        return None
    if "youtube.com/embed/" in url:
        return url
    try:
        from urllib.parse import urlparse, parse_qs

        u = urlparse(url)
        vid = None
        if u.netloc == "youtu.be":
            vid = u.path.lstrip("/")
        elif "youtube.com" in u.netloc:
            if u.path == "/watch":
                qs = parse_qs(u.query)
                vid = (qs.get("v") or [None])[0]
            elif u.path.startswith("/embed/"):
                parts = u.path.split("/")
                if len(parts) >= 3:
                    vid = parts[2]
        if not vid:
            return None
        qs = parse_qs(u.query)
        start = (qs.get("t") or qs.get("start") or [None])[0]
        suffix = f"?start={start}" if start else ""
        return f"https://www.youtube.com/embed/{vid}{suffix}"
    except Exception:
        return None


def load_lyrics_map(root: Path):
    songs = {}
    for path in sorted(root.glob("utahoshi_lyrics_*.json")):
        data = load_json(path)
        for song in data.get("songs", []):
            song_id = song.get("id")
            if song_id is not None:
                songs[str(song_id)] = song
    return songs


def build_track(song: dict, track_meta: dict):
    youtube = song.get("youtube", "")
    return {
        "track_no": track_meta["track_no"],
        "title_ja": song.get("title", ""),
        "title_en": song.get("title_en", ""),
        "lyric_id": song.get("id"),
        "audio_embed": youtube_to_embed(youtube) or "",
        "analysis": "",
        "omikuji_message": "",
        "links": {
            "youtube": youtube or "",
            "spotify": "",
        },
    }


def main():
    root = Path(__file__).resolve().parent
    sources_path = root / "album_sources.json"
    discog_path = root / "discography.json"

    sources = load_json(sources_path)
    discog = load_json(discog_path)
    lyrics_map = load_lyrics_map(root)

    existing_keys = {a.get("key") for a in discog.get("albums", [])}
    new_albums = []

    for album in sources.get("albums", []):
        key = album.get("key")
        if key in existing_keys:
            print(f"[skip] album already exists: {key}")
            continue

        tracks = []
        for track_meta in album.get("tracks", []):
            lyric_id = track_meta.get("lyric_id")
            song = lyrics_map.get(str(lyric_id))
            if not song:
                print(f"[warn] lyric_id not found: {lyric_id}")
                continue
            if not song.get("title_en"):
                print(f"[warn] title_en missing for lyric_id: {lyric_id}")
            if not song.get("youtube"):
                print(f"[warn] youtube missing for lyric_id: {lyric_id}")
            tracks.append(build_track(song, track_meta))

        new_albums.append(
            {
                "key": album.get("key", ""),
                "title_ja": album.get("title_ja", ""),
                "title_en": album.get("title_en", ""),
                "year": album.get("year", ""),
                "type": album.get("type", "Album"),
                "album_jacket": album.get("album_jacket", ""),
                "description": album.get("description", ""),
                "tracks": tracks,
            }
        )

    if not new_albums:
        print("[info] no new albums to add")
        return

    discog["albums"] = new_albums + discog.get("albums", [])
    save_json(discog_path, discog)
    print(f"[ok] added {len(new_albums)} album(s)")


if __name__ == "__main__":
    main()
