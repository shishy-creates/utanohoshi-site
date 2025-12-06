(() => {
  const DATA_PATH = "nine_cell_map.json";
  const canvas = document.getElementById("constellationCanvas");
  const tooltip = document.getElementById("constellationTooltip");
  const metaEl = document.getElementById("metaInfo");
  const legendQuadrants = document.getElementById("legendQuadrants");
  let legendTooltip = null;
  let tooltipLocked = false;
  let selectedStarId = null;
  const albumSelect = document.getElementById("albumSelect");
  const allowedAlbums = new Set([
    "イブキ",
    "イノチ",
    "イノリ",
    "ストーリー",
    "歌の星",
    "メッセージ",
    "未生",
    "ベガ",
    "コイカハルカ",
    "悲しみは影を知ること",
    "スタートライン",
    "うたのほし The 1st",
    "うたのほしThe 1st",
    "歌の癒しシリーズ",
  ]);
  const albumOrder = [
    "イブキ",
    "イノチ",
    "イノリ",
    "ストーリー",
    "歌の星",
    "メッセージ",
    "未生",
    "ベガ",
    "コイカハルカ",
    "悲しみは影を知ること",
    "スタートライン",
    "うたのほし The 1st",
    "うたのほしThe 1st",
    "歌の癒しシリーズ",
  ];

  const quadrantColors = {
    1: "#7fb4ff", // 希望×宇宙
    2: "#f19ac2", // 警鐘×宇宙
    3: "#f5d27f", // 警鐘×日常
    4: "#7fd8b1", // 希望×日常
  };

  function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function mixWithWhite(hex, t) {
    const [r, g, b] = hexToRgb(hex);
    const m = (c) => Math.round(c * (1 - t) + 255 * t);
    return `rgb(${m(r)},${m(g)},${m(b)})`;
  }

  let state = {
    stars: [],
    extent: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
    hoverId: null,
    albumOptions: [],
    selectedAlbum: "none",
    showAlbumLinks: false,
  };

  function fetchData() {
    const url = new URL(DATA_PATH, window.location.href).href;
    fetch(url, { cache: "no-cache" })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        ingestData(data);
      })
      .catch((err) => {
        console.warn("fetch failed, trying inline fallback", err);
        const inline = document.getElementById("constellations-json");
        const inlineText = inline?.textContent?.trim();
        if (inlineText) {
          try {
            ingestData(JSON.parse(inlineText));
            return;
          } catch (e) {
            console.error("inline parse failed", e);
          }
        }
        metaEl.textContent = `データ読み込みに失敗しました: ${err}. file:// で開いている場合は、プロジェクトルートで「python3 -m http.server 8000」を実行し、http://localhost:8000/constellations.html を開いてください。`;
      });
  }

  function ingestData(data) {
    const { songs = [], meta = {} } = data || {};
    state.stars = flattenStars(songs);
    state.stars = applyRepulsion(state.stars);
    state.stars = assignQuadrant(state.stars);
    state.extent = computeExtent(state.stars);
    buildAlbumOptions(state.stars);
    renderMeta(meta);
    renderQuadrantLegend(state.stars);
    resizeCanvas();
  }

  function flattenStars(songs) {
    const stars = [];
    songs.forEach((s) => {
      if (isFinite(s.x) && isFinite(s.y)) {
        const depth = typeof s.depth === "number" ? s.depth : 0.5;
        stars.push({
          ...s,
          depth,
          // quadrant/color will be assigned after repulsion to reflect final coords
        });
      }
    });
    return stars;
  }

  function computeExtent(stars) {
    const xs = stars.map((s) => s.x);
    const ys = stars.map((s) => s.y);
    return {
      minX: 0,
      maxX: 1,
      minY: 0,
      maxY: 1,
    };
  }

  function renderMeta(meta) {
    metaEl.textContent = `生成: ${meta.created_at || "N/A"} / 曲: ${meta.n_songs || state.stars.length}`;
  }

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    const width = parent.clientWidth;
    const height = Math.max(360, Math.round(width * 0.55));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    draw();
  }

  function project(x, y) {
    const padding = 24;
    const { minX, maxX, minY, maxY } = state.extent;
    const w = canvas.width;
    const h = canvas.height;
    const nx = (x - minX) / ((maxX - minX) || 1);
    const ny = (y - minY) / ((maxY - minY) || 1);
    const cx = padding + nx * (w - padding * 2);
    const cy = padding + (1 - ny) * (h - padding * 2);
    return { cx, cy };
  }

  function filteredStars() {
    return state.stars;
  }

  function computeQuadrant(x, y) {
    const xRight = x >= 0.5;
    const yUp = y >= 0.5;
    if (xRight && yUp) return 1; // 希望×宇宙
    if (!xRight && yUp) return 2; // 警鐘×宇宙
    if (!xRight && !yUp) return 3; // 警鐘×日常
    return 4; // 希望×日常
  }

  function quadrantLabel(qid) {
    return { 1: "星光", 2: "夢幻", 3: "済世", 4: "希灯" }[qid] || "";
  }

  function assignQuadrant(stars) {
    return stars.map((s) => {
      const qid = computeQuadrant(s.x, s.y);
      return {
        ...s,
        constId: qid,
        constName: quadrantLabel(qid),
        color: quadrantColors[qid] || quadrantColors[1],
      };
    });
  }

  function buildAlbumOptions(stars) {
    const present = new Set(
      stars
        .map((s) => s.album && s.album.trim())
        .filter((a) => a && allowedAlbums.has(a))
    );
    const albums = albumOrder.filter((a) => present.has(a));
    state.albumOptions = albums;
    if (!albumSelect) return;
    albumSelect.innerHTML = `<option value="none">選択しない</option>`;
    albums.forEach((a) => {
      const opt = document.createElement("option");
      opt.value = a;
      opt.textContent = a;
      albumSelect.appendChild(opt);
    });
    albumSelect.value = state.selectedAlbum || "none";
  }

  // simple repulsion to prevent stars from sitting exactly on top of each other
  function applyRepulsion(stars) {
    const radius = 0.024; // normalized distance threshold (larger)
    const strength = 0.7;
    const iterations = 12;
    const eps = 1e-6;
    const arr = stars.map((s) => ({ ...s, _ox: s.x, _oy: s.y }));
    for (let iter = 0; iter < iterations; iter++) {
      for (let i = 0; i < arr.length; i++) {
        for (let j = i + 1; j < arr.length; j++) {
          const a = arr[i];
          const b = arr[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          const dist = Math.hypot(dx, dy) + eps;
          if (dist < radius) {
            const push = (radius - dist) / radius * strength;
            const nx = dx / dist;
            const ny = dy / dist;
            a.x = Math.min(1, Math.max(0, a.x + nx * push * 0.5));
            a.y = Math.min(1, Math.max(0, a.y + ny * push * 0.5));
            b.x = Math.min(1, Math.max(0, b.x - nx * push * 0.5));
            b.y = Math.min(1, Math.max(0, b.y - ny * push * 0.5));
          }
        }
      }
    }
    // limit total displacement so points don't fly too far (max 0.05 from original)
    const maxDisp = 0.05;
    return arr.map((s) => {
      const dx = s.x - s._ox;
      const dy = s.y - s._oy;
      const dist = Math.hypot(dx, dy);
      if (dist > maxDisp && dist > 0) {
        const scale = maxDisp / dist;
        s.x = s._ox + dx * scale;
        s.y = s._oy + dy * scale;
      }
      delete s._ox;
      delete s._oy;
      return s;
    });
  }

  function draw() {
    const ctx = canvas.getContext("2d");
    const stars = filteredStars();
    const dpr = window.devicePixelRatio || 1;
    const now = performance.now() * 0.001;
    const w = canvas.width;
    const h = canvas.height;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // background gradient (light alpha so CSS background shows through)
    ctx.save();
    ctx.globalAlpha = 0.4;
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, "#0c132c");
    bg.addColorStop(1, "#050812");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
    const nebula = ctx.createRadialGradient(w * 0.2, h * 0.25, 0, w * 0.2, h * 0.25, Math.max(w, h) * 0.5);
    nebula.addColorStop(0, "rgba(120,150,255,0.12)");
    nebula.addColorStop(1, "rgba(120,150,255,0)");
    ctx.fillStyle = nebula;
    ctx.fillRect(0, 0, w, h);
    const nebula2 = ctx.createRadialGradient(w * 0.8, h * 0.2, 0, w * 0.8, h * 0.2, Math.max(w, h) * 0.4);
    nebula2.addColorStop(0, "rgba(220,150,220,0.08)");
    nebula2.addColorStop(1, "rgba(220,150,220,0)");
    ctx.fillStyle = nebula2;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();

    // draw album links if enabled (connect nearest using MST)
    if (state.showAlbumLinks && state.selectedAlbum && state.selectedAlbum !== "none") {
      const group = stars.filter((s) => s.album === state.selectedAlbum);
      if (group.length > 1) {
        const edges = computeMST(group);
        ctx.save();
        ctx.lineWidth = 1 * dpr;
        ctx.strokeStyle = "rgba(255,255,255,0.38)";
        ctx.beginPath();
        edges.forEach(([aIdx, bIdx]) => {
          const a = group[aIdx];
          const b = group[bIdx];
          const { cx: ax, cy: ay } = project(a.x, a.y);
          const { cx: bx, cy: by } = project(b.x, b.y);
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
        });
        ctx.stroke();
        ctx.restore();
      }
    }

    const activeAlbum = state.showAlbumLinks && state.selectedAlbum !== "none" ? state.selectedAlbum : null;

    stars.forEach((s) => {
      const { cx, cy } = project(s.x, s.y);
      s._cx = cx;
      s._cy = cy;
      const depth = Math.max(0, Math.min(1, s.depth || 0.5));
      const base = 3.5 * dpr * 0.8; // shrink base size ~80%
      const pulse = Math.sin(now * 1.2 + depth * 3.14) * 0.6 * dpr * 0.8;
      const radius = base + depth * 5 * dpr * 0.8 + pulse + (state.hoverId === s.id ? 1.6 * dpr : 0);
      const distCenter = Math.hypot(s.x - 0.5, s.y - 0.5);
      const lighten = Math.max(0, Math.min(1, 1 - distCenter / 0.7)); // closer to center -> whiter
      const active = !activeAlbum || s.album === activeAlbum;
      const baseColor = active ? s.color : "#ffffff";
      const starColor = mixWithWhite(baseColor, lighten * 0.4);
      ctx.beginPath();
      ctx.fillStyle = starColor;
      const alphaBase = 0.45 + depth * 0.55;
      ctx.globalAlpha = alphaBase * (active ? 1.0 : 0.35);
      ctx.shadowColor = starColor;
      ctx.shadowBlur = 12 * depth * dpr + 6;
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    ctx.restore();
  }

  function handleMouseMove(evt) {
    if (tooltipLocked) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const x = (evt.clientX - rect.left) * dpr;
    const y = (evt.clientY - rect.top) * dpr;
    const stars = filteredStars();
    let nearest = null;
    let minDist = 16 * dpr; // widen hover hit box for small stars
    stars.forEach((s) => {
      if (!s._cx) return;
      const dx = s._cx - x;
      const dy = s._cy - y;
      const dist = Math.hypot(dx, dy);
      if (dist < minDist) {
        minDist = dist;
        nearest = s;
      }
    });
    if (nearest) {
      state.hoverId = nearest.id;
      showTooltip(nearest, evt.clientX - rect.left, evt.clientY - rect.top);
    } else {
      state.hoverId = null;
      hideTooltip();
    }
    draw();
  }

  function showTooltip(star, offsetX, offsetY, showAction = false) {
    tooltip.style.display = "block";
    tooltip.innerHTML = `
      <div class="tt-title">${star.title}</div>
      <div class="tt-meta">${star.album || "Unknown"}</div>
      <div class="tt-const">${star.constName}${star.cell ? ` / ${star.cell}` : ""}</div>
      <div class="tt-coord">x: ${star.x.toFixed(3)}, y: ${star.y.toFixed(3)}, depth: ${ (star.depth ?? 0).toFixed(3)}</div>
      ${showAction ? `<button class="tt-link" data-id="${star.id}">曲を聴く</button>` : ""}
    `;
    // Position with viewport bounds check
    const pad = 12;
    const rect = canvas.getBoundingClientRect();
    const ttRect = tooltip.getBoundingClientRect();
    let left = offsetX + 16;
    let top = offsetY + 12;
    const maxLeft = rect.width - ttRect.width - pad;
    const maxTop = rect.height - ttRect.height - pad;
    // if overflowing right, flip to the left side of the cursor
    if (left > maxLeft) {
      left = offsetX - ttRect.width - 16;
      if (left < pad) left = pad;
    }
    // if overflowing bottom, flip above cursor
    if (top > maxTop) {
      top = offsetY - ttRect.height - 16;
      if (top < pad) top = pad;
    }
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    if (showAction) {
      const btn = tooltip.querySelector(".tt-link");
      if (btn) {
        btn.onclick = (e) => {
          e.stopPropagation();
          window.location.href = `song.html?id=${star.id}`;
        };
      }
    }
  }

  function hideTooltip() {
    tooltip.style.display = "none";
  }

  function setupCanvasEvents() {
    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseleave", () => {
      if (!tooltipLocked) {
        state.hoverId = null;
        hideTooltip();
        draw();
      }
    });
    if (albumSelect) {
      albumSelect.addEventListener("change", (e) => {
        state.selectedAlbum = e.target.value;
        state.showAlbumLinks = state.selectedAlbum !== "none";
        draw();
      });
    }
    // album selection auto-enables links
    state.showAlbumLinks = state.selectedAlbum !== "none";
    canvas.addEventListener("click", handleCanvasClick);
    document.addEventListener("click", (e) => {
      if (e.target.closest(".constellation-tooltip")) return;
      if (e.target === canvas) return;
      tooltipLocked = false;
      selectedStarId = null;
      hideTooltip();
      draw();
    });
  }

  function handleCanvasClick(evt) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const x = (evt.clientX - rect.left) * dpr;
    const y = (evt.clientY - rect.top) * dpr;
    const stars = filteredStars();
    let nearest = null;
    let minDist = 16 * dpr; // match hover sensitivity
    stars.forEach((s) => {
      if (!s._cx) return;
      const dx = s._cx - x;
      const dy = s._cy - y;
      const dist = Math.hypot(dx, dy);
      if (dist < minDist) {
        minDist = dist;
        nearest = s;
      }
    });
    if (nearest) {
      tooltipLocked = true;
      selectedStarId = nearest.id;
      state.hoverId = nearest.id;
      showTooltip(nearest, evt.clientX - rect.left, evt.clientY - rect.top, true);
      draw();
    } else {
      tooltipLocked = false;
      selectedStarId = null;
      hideTooltip();
      draw();
    }
  }

  // Minimum Spanning Tree (Prim) on normalized coords to connect nearest neighbors
  function computeMST(nodes) {
    const n = nodes.length;
    if (n <= 1) return [];
    const inTree = Array(n).fill(false);
    const dist = Array(n).fill(Infinity);
    const parent = Array(n).fill(-1);
    dist[0] = 0;
    for (let _ = 0; _ < n; _++) {
      let u = -1;
      let best = Infinity;
      for (let i = 0; i < n; i++) {
        if (!inTree[i] && dist[i] < best) {
          best = dist[i];
          u = i;
        }
      }
      if (u === -1) break;
      inTree[u] = true;
      for (let v = 0; v < n; v++) {
        if (inTree[v]) continue;
        const dx = nodes[u].x - nodes[v].x;
        const dy = nodes[u].y - nodes[v].y;
        const d = dx * dx + dy * dy;
        if (d < dist[v]) {
          dist[v] = d;
          parent[v] = u;
        }
      }
    }
    const edges = [];
    for (let i = 1; i < n; i++) {
      if (parent[i] !== -1) edges.push([i, parent[i]]);
    }
    return edges;
  }

  function renderQuadrantLegend(stars) {
    if (!legendQuadrants) return;
    const counts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    const desc = {
      1: "星光　宇宙で光る希求の星雲",
      2: "夢幻　宇宙に響く夢と啓示の星雲",
      3: "済世　日常の痛みをとかす再生の星雲",
      4: "希灯　日常に希望を灯す星雲",
    };
    stars.forEach((s) => {
      const qid = computeQuadrant(s.x, s.y);
      if (counts[qid] !== undefined) counts[qid] += 1;
    });
    legendQuadrants.innerHTML = "";
    [1, 2, 3, 4].forEach((qid) => {
      const item = document.createElement("div");
      item.className = "legend-item";
      const color = quadrantColors[qid] || quadrantColors[1];
      item.innerHTML = `
        <span class="legend-dot" style="background:${color};"></span>
        <span class="legend-name">${quadrantLabel(qid)}</span>
        <span class="legend-keywords">${counts[qid] || 0} 曲</span>
      `;
      item.addEventListener("mouseenter", (e) => {
        showLegendTooltip(desc[qid] || "", e);
      });
      item.addEventListener("mousemove", (e) => {
        showLegendTooltip(desc[qid] || "", e);
      });
      item.addEventListener("mouseleave", hideLegendTooltip);
      legendQuadrants.appendChild(item);
    });
  }

  function ensureLegendTooltip() {
    if (legendTooltip) return legendTooltip;
    legendTooltip = document.createElement("div");
    legendTooltip.className = "legend-tooltip";
    document.body.appendChild(legendTooltip);
    return legendTooltip;
  }

  function showLegendTooltip(text, evt) {
    const tt = ensureLegendTooltip();
    tt.textContent = text;
    tt.style.display = "block";
    const pad = 12;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const ttRect = tt.getBoundingClientRect();
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    if (x + ttRect.width + pad > vw) x = vw - ttRect.width - pad;
    if (y + ttRect.height + pad > vh) y = vh - ttRect.height - pad;
    tt.style.left = `${x}px`;
    tt.style.top = `${y}px`;
  }

  function hideLegendTooltip() {
    if (legendTooltip) legendTooltip.style.display = "none";
  }

  window.addEventListener("resize", resizeCanvas);

  document.addEventListener("DOMContentLoaded", () => {
    setupCanvasEvents();
    fetchData();
    const loop = () => {
      draw();
      requestAnimationFrame(loop);
    };
    loop();
  });
})();
