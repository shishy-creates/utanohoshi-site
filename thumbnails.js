(() => {
  const gridEl = document.getElementById("secretGrid");
  const emptyEl = document.getElementById("secretEmpty");
  const statusEl = document.getElementById("secretStatus");
  const copyInviteBtn = document.getElementById("copyInviteUrl");

  // ここを書き換えてコレクションを更新してください。
  // src は `images/` 配下のファイル名（または完全パス）を指定。
  const COLLECTION = [
    {
      title: "「使徒、アトラスよ」初音ミクver",
      note: "タイトル有り",
      src: "images/library/atlus_miku-01.jpg",
      filename: "atlus_miku-01.jpg",
    },
    {
      title: "「使徒、アトラスよ」初音ミクver",
      note: "タイトル無し",
      src: "images/library/atlus_miku-02.jpg",
      filename: "atlus_miku-02.jpg",
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

  function downloadImage(src, filename, buttonEl) {
    if (!src) return;
    if (buttonEl) {
      buttonEl.disabled = true;
      buttonEl.textContent = "準備中...";
    }
    fetch(src)
      .then((res) => res.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename || "image.jpg";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        setStatus("画像をダウンロード用に用意しました。", "success");
      })
      .catch(() => setStatus("ダウンロードに失敗しました。ファイルパスを確認してください。", "warn"))
      .finally(() => {
        if (buttonEl) {
          buttonEl.disabled = false;
          buttonEl.textContent = "画像を保存";
        }
      });
  }

  function render() {
    if (!gridEl) return;
    gridEl.innerHTML = "";
    COLLECTION.forEach((item) => {
      const card = document.createElement("article");
      card.className = "secret-card thumb-card";
      card.innerHTML = `
        <div class="thumb-image-wrap">
          <img class="thumb-image" src="${item.src}" alt="${item.title}" />
        </div>
        <div class="thumb-meta">
          <div class="thumb-id">
            <strong>${item.title || "タイトルなし"}</strong>
          </div>
          ${item.note ? `<div class="thumb-note">${item.note}</div>` : ""}
          <div class="thumb-links">
            <a class="secret-btn ghost sm" href="${item.src}" target="_blank" rel="noreferrer">Open</a>
            <button class="secret-btn primary sm thumb-download" type="button">Downroad</button>
          </div>
        </div>
      `;
      const downloadBtn = card.querySelector(".thumb-download");
      downloadBtn?.addEventListener("click", () => downloadImage(item.src, item.filename || `${item.title || "image"}.jpg`, downloadBtn));
      gridEl.appendChild(card);
    });
    ensureGridVisibility(COLLECTION.length > 0);
    setStatus("", "idle");
  }

  function copyInviteUrl() {
    if (!navigator.clipboard) {
      setStatus("クリップボードに対応していません。手動でコピーしてください。", "warn");
      return;
    }
    navigator.clipboard
      .writeText(window.location.href)
      .then(() => setStatus("このページのURLをコピーしました。", "success"))
      .catch(() => setStatus("コピーに失敗しました。手動で選択してください。", "warn"));
  }

  copyInviteBtn?.addEventListener("click", copyInviteUrl);

  render();
})(); 
