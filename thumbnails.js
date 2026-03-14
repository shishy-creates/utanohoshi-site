(() => {
  const gridEl = document.getElementById("secretGrid");
  const emptyEl = document.getElementById("secretEmpty");
  const statusEl = document.getElementById("secretStatus");
  const copyInviteBtn = document.getElementById("copyInviteUrl");

  const COLLECTION = [
    {
      type: "image",
      title: "「使徒、アトラスよ」初音ミクver",
      note: "タイトル有り",
      src: "images/library/atlus_miku-01.jpg",
      filename: "atlus_miku-01.jpg",
    },
    {
      type: "image",
      title: "「使徒、アトラスよ」初音ミクver",
      note: "タイトル無し",
      src: "images/library/atlus_miku-02.jpg",
      filename: "atlus_miku-02.jpg",
    },
    {
      type: "audio",
      title: "CODE:GIZA",
      note: "Audio",
      src: "audio/library/code_GIZA_miku_432.mp3",
      filename: "code_GIZA_miku_432.mp3",
    },
  ];

  function setStatus(message, tone = "idle") {
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.dataset.tone = tone;
  }

  function ensureGridVisibility(hasItems) {
    if (emptyEl) emptyEl.style.display = hasItems ? "none" : "block";
  }

  async function downloadFile(src, filename, buttonEl) {
    if (!src) return;
    const originalText = buttonEl ? buttonEl.textContent : "";
    if (buttonEl) {
      buttonEl.disabled = true;
      buttonEl.textContent = "Preparing...";
    }
    try {
      const res = await fetch(src);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename || "download";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setStatus("", "idle");
    } catch (err) {
      setStatus("Download failed.", "warn");
    } finally {
      if (buttonEl) {
        buttonEl.disabled = false;
        buttonEl.textContent = originalText;
      }
    }
  }

  function renderPreview(item) {
    if (item.type === "audio") {
      const cover = item.cover
        ? `<img class="thumb-image thumb-audio-cover" src="${item.cover}" alt="${item.title}" />`
        : `<div class="thumb-audio-icon">AUDIO</div>`;
      return `
        <div class="thumb-image-wrap thumb-audio-wrap">
          ${cover}
        </div>
        <div class="thumb-audio-player">
          <audio controls preload="none" src="${item.src}"></audio>
        </div>
      `;
    }

    return `
      <div class="thumb-image-wrap">
        <img class="thumb-image" src="${item.src}" alt="${item.title}" />
      </div>
    `;
  }

  function renderCard(item) {
    const card = document.createElement("article");
    card.className = "secret-card thumb-card";
    card.innerHTML = `
      ${renderPreview(item)}
      <div class="thumb-meta">
        <div class="thumb-id">
          <strong>${item.title || "Untitled"}</strong>
        </div>
        ${item.note ? `<div class="thumb-note">${item.note}</div>` : ""}
        <div class="thumb-kind">${item.type === "audio" ? "Audio" : "Image"}</div>
        <div class="thumb-links">
          <a class="secret-btn ghost sm" href="${item.src}" target="_blank" rel="noreferrer">Open</a>
          <button class="secret-btn primary sm thumb-download" type="button">Downroad</button>
        </div>
      </div>
    `;

    const downloadBtn = card.querySelector(".thumb-download");
    downloadBtn?.addEventListener("click", () => {
      downloadFile(item.src, item.filename || item.title || "download", downloadBtn);
    });
    return card;
  }

  function render() {
    if (!gridEl) return;
    gridEl.innerHTML = "";
    COLLECTION.forEach((item) => {
      gridEl.appendChild(renderCard(item));
    });
    ensureGridVisibility(COLLECTION.length > 0);
    setStatus("", "idle");
  }

  function copyInviteUrl() {
    if (!navigator.clipboard) {
      setStatus("Clipboard unsupported.", "warn");
      return;
    }
    navigator.clipboard
      .writeText(window.location.href)
      .then(() => setStatus("URL copied.", "success"))
      .catch(() => setStatus("Copy failed.", "warn"));
  }

  copyInviteBtn?.addEventListener("click", copyInviteUrl);

  render();
})();
