// クエリパラメータ取得
function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

// discography.json 読み込み
async function loadDiscography() {
  const res = await fetch("discography.json");
  if (!res.ok) {
    throw new Error("discography.json を読み込めませんでした");
  }
  return res.json();
}

// 全曲をアルバム情報付きで平坦化
function flattenTracksWithAlbum(disc) {
  const out = [];
  (disc.albums || []).forEach((album) => {
    (album.tracks || []).forEach((track) => {
      out.push({ album, track });
    });
  });
  return out;
}

// lyric_id から歌詞ファイル名を決める
function getLyricsFileForId(id) {
  const n = Number(id);
  if (n >= 1 && n <= 10) return "utahoshi_lyrics_001-010.json";
  if (n >= 11 && n <= 20) return "utahoshi_lyrics_011-020.json";
  if (n >= 21 && n <= 30) return "utahoshi_lyrics_021-030.json";
  if (n >= 31 && n <= 40) return "utahoshi_lyrics_031-040.json";
  if (n >= 41 && n <= 50) return "utahoshi_lyrics_041-050.json";
  if (n >= 51 && n <= 60) return "utahoshi_lyrics_051-060.json";
  if (n >= 61 && n <= 70) return "utahoshi_lyrics_061-070.json";
  if (n >= 71 && n <= 80) return "utahoshi_lyrics_071-080.json";
  if (n >= 81 && n <= 90) return "utahoshi_lyrics_081-090.json";
  if (n >= 91 && n <= 100) return "utahoshi_lyrics_091-100.json";
  if (n >= 101 && n <= 110) return "utahoshi_lyrics_101-110.json";
  if (n >= 111 && n <= 120) return "utahoshi_lyrics_111-120.json";
  if (n >= 121 && n <= 130) return "utahoshi_lyrics_121-130.json";
  if (n >= 991 && n <= 1000) return "utahoshi_lyrics_991-1000.json";
  return null;
}

// lyric_id から歌詞データを取得
async function fetchSongByLyricId(lyricId) {
  const file = getLyricsFileForId(lyricId);
  if (!file) throw new Error("対応する歌詞ファイルがありません: " + lyricId);

  const res = await fetch(file);
  if (!res.ok) throw new Error(file + " を読み込めませんでした");

  const data = await res.json();
  const song = data.songs.find((s) => String(s.id) === String(lyricId));
  if (!song) throw new Error("歌詞ファイル内に該当曲がありません: " + lyricId);

  return song;
}

// 歌詞から冒頭 N 行を note 用に取り出す
function extractIntroFromLyrics(lyrics, lineCount = 2) {
  if (!lyrics) return "";
  const allLines = lyrics.split(/\r?\n/).map((l) => l.trim());
  const nonEmpty = allLines.filter((l) => l.length > 0);
  const picked = nonEmpty.slice(0, lineCount);
  return picked.join(" / ");
}

// おみくじ結果の描画
function renderOmikujiResult(target, pick, message) {
  if (!pick) {
    target.innerHTML =
      '<p class="omikuji-placeholder">曲データが見つかりませんでした。</p>';
    return;
  }

  const { album, track } = pick;
  const songUrl =
    track && track.track_no != null
      ? `song.html?album=${encodeURIComponent(
          album.key
        )}&track=${encodeURIComponent(track.track_no)}`
      : null;
  const youtube = track.links && track.links.youtube;
  const spotify = track.links && track.links.spotify;

  target.innerHTML = `
    <div class="omikuji-track">
      <h3 class="omikuji-track-title">
        ${track.title_ja}
        ${track.title_en ? `<span>${track.title_en}</span>` : ""}
      </h3>
      <p class="omikuji-track-meta">
        アルバム：${album.title_ja || ""}${album.title_en ? ` / ${album.title_en}` : ""} ${
    track.track_no ? ` / トラック ${track.track_no}` : ""
  }
      </p>
      ${
        message
          ? `<p class="omikuji-message">おみくじ：${message}</p>`
          : ""
      }
      <div class="omikuji-links">
        ${
          songUrl
            ? `<a class="omikuji-link-btn" href="${songUrl}">曲ページへ</a>`
            : ""
        }
        ${
          youtube
            ? `<a class="omikuji-link-btn" href="${youtube}" target="_blank" rel="noopener">YouTube</a>`
            : ""
        }
        ${
          spotify
            ? `<a class="omikuji-link-btn" href="${spotify}" target="_blank" rel="noopener">Spotify</a>`
            : ""
        }
      </div>
    </div>
  `;
}

// おみくじメッセージ（前向きな短文）
function getOmikujiMessages() {
  return [
    "風が背中を押しています。",
    "静かな祈りが届く日。",
    "あたたかな光に包まれます。",
    "小さな一歩が、大きな道を開きます。",
    "心の鼓動に素直でいてください。",
    "今日の空は味方をしています。",
    "澄んだ風が、新しい始まりを告げます。",
    "あなたの声が誰かを照らしています。",
    "やさしい時間が、すぐそばにあります。",
    "想いは遠くまで届いています。",
    "深呼吸の先に、ひらめきが待っています。",
    "静けさの中に、力強いリズムが響いています。"
  ];
}

// 曲に紐づくおみくじメッセージ（あれば優先）
function pickMessageForTrack(track) {
  if (track && track.omikuji_message && track.omikuji_message.trim() !== "") {
    return track.omikuji_message;
  }
  const messages = getOmikujiMessages();
  return messages.length
    ? messages[Math.floor(Math.random() * messages.length)]
    : "";
}

/* =========================
   おみくじ永続化 (1日1回／5時リセット)
   ========================= */

const OMIKUJI_STORAGE_KEY = "omikuji_result_v2";

function getOmikujiDayKey(now = new Date()) {
  const d = new Date(now);
  if (d.getHours() < 5) {
    d.setDate(d.getDate() - 1);
  }
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function normalizeStored(raw) {
  if (!raw) return null;
  // v2: { dayKey, draws: [{albumKey, trackNo, message}] }
  if (raw.dayKey && Array.isArray(raw.draws)) return raw;
  // v1互換: { dayKey, albumKey, trackNo, message }
  if (raw.dayKey && raw.albumKey && raw.trackNo != null) {
    return {
      dayKey: raw.dayKey,
      draws: [
        {
          albumKey: raw.albumKey,
          trackNo: raw.trackNo,
          message: raw.message || "",
        },
      ],
    };
  }
  return null;
}

function loadStoredOmikuji() {
  try {
    const raw = localStorage.getItem(OMIKUJI_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return normalizeStored(parsed);
  } catch (_e) {
    return null;
  }
}

function saveStoredOmikuji(data) {
  try {
    localStorage.setItem(OMIKUJI_STORAGE_KEY, JSON.stringify(data));
  } catch (_e) {
    // noop
  }
}

function clearStoredOmikuji() {
  try {
    localStorage.removeItem(OMIKUJI_STORAGE_KEY);
  } catch (_e) {
    // noop
  }
}

function findTrackByAlbumAndNo(disc, albumKey, trackNo) {
  const album = (disc.albums || []).find((a) => a.key === albumKey);
  if (!album) return null;
  const track = (album.tracks || []).find(
    (t) => Number(t.track_no) === Number(trackNo)
  );
  if (!track) return null;
  return { album, track };
}

/* =========================
   index.html アルバム一覧
   ========================= */

async function initIndexPage() {
  const listEl = document.getElementById("album-list");
  if (!listEl) return;

  try {
    const disc = await loadDiscography();
    // おみくじはここで disc を共有して使う
    initOmikuji(disc);

    const albums = disc.albums || [];

    if (!albums.length) {
      listEl.textContent = "アルバム情報がまだ登録されていません。";
      return;
    }

    albums.forEach((album) => {
      const card = document.createElement("a");
      card.href = `album.html?album=${encodeURIComponent(album.key)}`;
      card.className = "album-index-card";

      if (album.album_jacket) {
        const img = document.createElement("img");
        img.src = album.album_jacket;
        img.alt = `${album.title_ja} ジャケット`;
        img.className = "album-card-image";
        card.appendChild(img);
      }

      const titleDiv = document.createElement("div");
      titleDiv.className = "album-card-title";
      titleDiv.innerHTML = `
        ${album.title_ja}
        ${album.title_en ? `<span>${album.title_en}</span>` : ""}
      `;

      const metaDiv = document.createElement("div");
      metaDiv.className = "album-card-meta";
      metaDiv.textContent = `${album.year || ""} • ${album.type || "Album"}`;

      const descDiv = document.createElement("div");
      descDiv.className = "album-card-desc";
      descDiv.textContent = album.description || "";

      card.appendChild(titleDiv);
      card.appendChild(metaDiv);
      card.appendChild(descDiv);

      listEl.appendChild(card);
    });
  } catch (e) {
    console.error(e);
    listEl.textContent = "ディスコグラフィー情報の読み込みに失敗しました。";
  }
}

/* =========================
   index.html 今日の一曲おみくじ
   ========================= */

function initOmikuji(disc) {
  const button = document.getElementById("omikuji-button");
  const result = document.getElementById("omikuji-result");
  if (!button || !result) return;

  const todayKey = getOmikujiDayKey();
  const pool = flattenTracksWithAlbum(disc).filter(
    (item) => item.track && item.album
  );
  const pickRandom = () =>
    pool.length ? pool[Math.floor(Math.random() * pool.length)] : null;

  // 既存の結果が当日かどうかチェック
  const stored = loadStoredOmikuji();
  if (stored) {
    if (stored.dayKey === todayKey) {
      const last = stored.draws && stored.draws[stored.draws.length - 1];
      if (last) {
        const found = findTrackByAlbumAndNo(
          disc,
          last.albumKey,
          last.trackNo
        );
        if (found) {
          renderOmikujiResult(result, found, last.message);
          if (stored.draws.length >= 3) {
            button.textContent = "また明日";
            button.disabled = true;
            return;
          } else {
            button.textContent = `もう一度ひく (残り ${3 - stored.draws.length}回)`;
          }
        } else {
          clearStoredOmikuji();
        }
      }
    } else {
      clearStoredOmikuji();
    }
  }

  button.addEventListener("click", () => {
    const latest = loadStoredOmikuji();
    let draws = [];
    if (latest && latest.dayKey === todayKey && Array.isArray(latest.draws)) {
      draws = latest.draws;
      if (draws.length >= 3) {
        button.textContent = "また明日";
        button.disabled = true;
        return;
      }
    }

    const pick = pickRandom();
    const message = pickMessageForTrack(pick ? pick.track : null);
    renderOmikujiResult(result, pick, message);

    if (pick) {
      const newDraws = [
        ...draws,
        {
          albumKey: pick.album.key,
          trackNo: pick.track.track_no,
          message,
        },
      ].slice(-3); // 念のため上限を守る

      saveStoredOmikuji({
        dayKey: todayKey,
        draws: newDraws,
      });

      if (newDraws.length >= 3) {
        button.textContent = "また明日";
        button.disabled = true;
      } else {
        button.textContent = `もう一度ひく (残り ${3 - newDraws.length}回)`;
      }
    }
  });
}

/* =========================
   album.html アルバム詳細
   ========================= */

async function initAlbumPage() {
  const container = document.getElementById("album-detail");
  if (!container) return;

  const albumKey = getQueryParam("album");
  if (!albumKey) {
    container.textContent = "アルバムが指定されていません。";
    return;
  }

  try {
    const disc = await loadDiscography();
    const album = (disc.albums || []).find((a) => a.key === albumKey);

    if (!album) {
      container.textContent = "指定されたアルバムが見つかりません。";
      return;
    }

    // ★ ミニヘッダーにアルバム名を表示
    const pageTitleSpan = document.getElementById("album-page-title");
    if (pageTitleSpan) {
      pageTitleSpan.textContent = album.title_ja;
    }

    // --- アルバムカード（上部に固定したい部分） ---
    const headerCard = document.createElement("section");
    headerCard.className = "album album-header-card";

    const headerLayout = document.createElement("div");
    headerLayout.className = "album-header-layout";

    if (album.album_jacket) {
      const img = document.createElement("img");
      img.src = album.album_jacket;
      img.alt = `${album.title_ja} ジャケット`;
      img.className = "album-jacket-large";
      headerLayout.appendChild(img);
    }

    const headerText = document.createElement("div");
    headerText.className = "album-header";
    headerText.innerHTML = `
      <div class="album-title">
        ${album.title_ja}
        ${album.title_en ? `<span>${album.title_en}</span>` : ""}
      </div>
      <div class="album-meta">
        ${album.year || ""} • ${album.type || "Album"}
      </div>
      <p class="album-desc">${album.description || ""}</p>
    `;

    headerLayout.appendChild(headerText);
    headerCard.appendChild(headerLayout);

    // --- 曲リスト（スクロールさせたい部分） ---
    const tracksDiv = document.createElement("div");
    tracksDiv.className = "tracks";

    (album.tracks || []).forEach((track) => {
      const card = document.createElement("article");
      card.className = "track-card";

      // カード全体クリックで曲ページへ
      if (track.lyric_id) {
        card.classList.add("track-card--clickable");
        card.addEventListener("click", () => {
          window.location.href =
            `song.html?album=${encodeURIComponent(album.key)}&track=${encodeURIComponent(track.track_no)}`;
        });
      }

      // 左の番号バッジ
      const badge = document.createElement("div");
      badge.className = "track-badge";
      badge.textContent = track.track_no;

      // 右側の本文
      const main = document.createElement("div");
      main.className = "track-main";

      const titleH3 = document.createElement("h3");
      titleH3.className = "track-title";
      titleH3.innerHTML = `
        ${track.title_ja}
        ${track.title_en ? `<span>${track.title_en}</span>` : ""}
      `;

      const noteP = document.createElement("p");
      noteP.className = "track-note";
      noteP.textContent = "";
      if (track.lyric_id) {
        noteP.dataset.lyricId = String(track.lyric_id);
      }

      main.appendChild(titleH3);
      main.appendChild(noteP);

      card.appendChild(badge);
      card.appendChild(main);

      tracksDiv.appendChild(card);
    });

    // ★ ここが大事：wrapper は使わずに、直接 container に追加
    container.innerHTML = "";            // 念のため一度クリア
    container.appendChild(headerCard);   // 固定したいアルバムカード
    container.appendChild(tracksDiv);    // その下に曲リスト

    // 歌詞から note を自動生成
    await autoFillIntroSnippets(tracksDiv);

  } catch (e) {
    console.error(e);
    container.textContent = "アルバム情報の読み込みに失敗しました。";
  }
}
// note 自動埋め
async function autoFillIntroSnippets(container) {
  const noteEls = container.querySelectorAll(".track-note[data-lyric-id]");
  for (const el of noteEls) {
    const id = el.dataset.lyricId;
    try {
      const songData = await fetchSongByLyricId(id);
      const intro = extractIntroFromLyrics(songData.lyrics, 2);
      el.textContent = intro;
    } catch (e) {
      console.error(e);
      el.textContent = "";
    }
  }
}

/* =========================
   song.html 曲詳細
   ========================= */

/* =========================
   song.html 曲詳細
   ========================= */

async function initSongPage() {
  const container = document.getElementById("song-page");
  if (!container) return;

  // 新方式：album + track
  const albumKey = getQueryParam("album");
  const trackNoParam = getQueryParam("track");

  // 旧方式との互換用（?id=xxx が来たとき用）
  const legacyId = getQueryParam("id");

  try {
    const disc = await loadDiscography();

    let foundTrack = null;
    let foundAlbum = null;
    let lyricId = null;

    if (albumKey && trackNoParam) {
      // --- 新ルーティング: /song.html?album=xxx&track=yy ---
      const trackNo = Number(trackNoParam);

      foundAlbum = (disc.albums || []).find((a) => a.key === albumKey);
      if (!foundAlbum) {
        throw new Error("指定されたアルバムが見つかりません: " + albumKey);
      }

      foundTrack = (foundAlbum.tracks || []).find(
        (t) => Number(t.track_no) === trackNo
      );
      if (!foundTrack) {
        throw new Error("指定されたトラックが見つかりません: " + trackNo);
      }

      lyricId = foundTrack.lyric_id;
    } else if (legacyId) {
      // --- 互換用: /song.html?id=xxx の古いリンク ---
      lyricId = legacyId;

      for (const album of disc.albums || []) {
        for (const track of album.tracks || []) {
          if (String(track.lyric_id) === String(legacyId)) {
            foundAlbum = album;
            foundTrack = track;
            break;
          }
        }
        if (foundTrack) break;
      }
    } else {
      container.textContent = "曲が指定されていません。";
      return;
    }

    // 歌詞データはこれまで通り lyric_id から取得
    const songData = await fetchSongByLyricId(lyricId);

    const headerDiv = document.createElement("div");
    headerDiv.className = "song-header";
    headerDiv.innerHTML = `
      <h2 class="song-title">
        ${foundTrack ? foundTrack.title_ja : songData.title}
        ${foundTrack && foundTrack.title_en
        ? `<span>${foundTrack.title_en}</span>`
        : ""
      }
      </h2>
      <div class="song-meta">
        アルバム：${foundAlbum ? foundAlbum.title_ja : (songData.album || "")}
        ${songData.date ? ` / 公開日：${songData.date}` : ""}
      </div>
      <a class="back-link" href="${foundAlbum
        ? `album.html?album=${encodeURIComponent(foundAlbum.key)}`
        : "index.html"
      }">← アルバムへ戻る</a>
    `;

    container.appendChild(headerDiv);

    // プレイヤー
    if (foundTrack && foundTrack.audio_embed) {
      const playerDiv = document.createElement("div");
      playerDiv.className = "song-player";

      const iframe = document.createElement("iframe");
      iframe.src = foundTrack.audio_embed;
      iframe.allow =
        "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
      iframe.allowFullscreen = true;

      playerDiv.appendChild(iframe);
      container.appendChild(playerDiv);
    }

    // Spotifyリンクボタン
    if (
      foundTrack &&
      foundTrack.links &&
      foundTrack.links.spotify &&
      foundTrack.links.spotify.trim() !== ""
    ) {
      const spotifyDiv = document.createElement("div");
      spotifyDiv.className = "song-spotify-link";
      const spBtn = document.createElement("a");
      spBtn.href = foundTrack.links.spotify;
      spBtn.target = "_blank";
      spBtn.rel = "noopener";
      spBtn.className = "track-go-button spotify";
      spBtn.textContent = "Spotifyで聴く";
      spotifyDiv.appendChild(spBtn);
      container.appendChild(spotifyDiv);
    }

    // タブ構造
    const tabsWrapper = document.createElement("div");
    tabsWrapper.className = "song-tabs";

    const tabHeader = document.createElement("div");
    tabHeader.className = "song-tab-header";

    const lyricsBtn = document.createElement("button");
    lyricsBtn.type = "button";
    lyricsBtn.className = "song-tab-button";
    lyricsBtn.textContent = "歌詞";

    const analysisBtn = document.createElement("button");
    analysisBtn.type = "button";
    analysisBtn.className = "song-tab-button";
    analysisBtn.textContent = "AIマルコの歌詞解説";

    tabHeader.appendChild(lyricsBtn);
    tabHeader.appendChild(analysisBtn);

    const panels = document.createElement("div");
    panels.className = "song-tab-panels";

    const lyricsPanel = document.createElement("div");
    lyricsPanel.className = "song-tab-panel";

    const lyricsDiv = document.createElement("div");
    lyricsDiv.className = "lyrics-block";
    lyricsDiv.textContent = songData.lyrics || "";

    const noticeP = document.createElement("p");
    noticeP.className = "notice";
    noticeP.textContent = "※ 歌詞は公式JSONアーカイブからの引用です。";

    lyricsPanel.appendChild(lyricsDiv);
    lyricsPanel.appendChild(noticeP);

    const analysisPanel = document.createElement("div");
    analysisPanel.className = "song-tab-panel";

    const analysisBlock = document.createElement("div");
    analysisBlock.className = "analysis-block";

    const analysisTitle = document.createElement("h3");
    analysisTitle.textContent = "――以下、マルコによる解説――";

    const analysisText = document.createElement("p");
    if (foundTrack && foundTrack.analysis && foundTrack.analysis.trim() !== "") {
      analysisText.textContent = foundTrack.analysis;
    } else {
      analysisText.textContent =
        "この曲のAIマルコによる解説は、順次追加予定です。歌詞と共に、あなた自身の物語も感じてみてください。";
    }

    analysisBlock.appendChild(analysisTitle);
    analysisBlock.appendChild(analysisText);
    analysisPanel.appendChild(analysisBlock);

    panels.appendChild(lyricsPanel);
    panels.appendChild(analysisPanel);

    tabsWrapper.appendChild(tabHeader);
    tabsWrapper.appendChild(panels);
    container.appendChild(tabsWrapper);

    // タブ切り替え
    function activateTab(tabName) {
      if (tabName === "analysis") {
        lyricsBtn.classList.remove("active");
        analysisBtn.classList.add("active");
        lyricsPanel.classList.remove("active");
        analysisPanel.classList.add("active");
      } else {
        analysisBtn.classList.remove("active");
        lyricsBtn.classList.add("active");
        analysisPanel.classList.remove("active");
        lyricsPanel.classList.add("active");
      }
    }

    lyricsBtn.addEventListener("click", () => activateTab("lyrics"));
    analysisBtn.addEventListener("click", () => activateTab("analysis"));

    // 初期タブ
    activateTab("lyrics");
  } catch (e) {
    console.error(e);
    container.textContent = "歌詞の読み込みに失敗しました。";
  }
}

/* =========================
   起動
   ========================= */

document.addEventListener("DOMContentLoaded", () => {
  initIndexPage();
  initAlbumPage();
  initSongPage();
});
