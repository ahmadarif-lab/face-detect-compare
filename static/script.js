// ---------- Tabs ----------
document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".pane").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.target).classList.add("active");
    });
});

// ---------- Helpers ----------
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function bindDropzone(zoneEl, inputEl, previewEl) {
    inputEl.addEventListener("change", () => {
        const file = inputEl.files && inputEl.files[0];
        if (!file) {
            zoneEl.classList.remove("has-file");
            previewEl.removeAttribute("src");
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            previewEl.src = e.target.result;
            zoneEl.classList.add("has-file");
        };
        reader.readAsDataURL(file);
    });

    ["dragenter", "dragover"].forEach((ev) => {
        zoneEl.addEventListener(ev, (e) => {
            e.preventDefault();
            zoneEl.classList.add("dragover");
        });
    });
    ["dragleave", "drop"].forEach((ev) => {
        zoneEl.addEventListener(ev, (e) => {
            e.preventDefault();
            zoneEl.classList.remove("dragover");
        });
    });
    zoneEl.addEventListener("drop", (e) => {
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) {
            const dt = new DataTransfer();
            dt.items.add(file);
            inputEl.files = dt.files;
            inputEl.dispatchEvent(new Event("change"));
        }
    });
}

// ---------- Detection ----------
const detectForm = document.getElementById("detectForm");
const detectFile = document.getElementById("detectFile");
const detectDrop = document.getElementById("detectDrop");
const detectPreview = document.getElementById("detectPreview");
const detectBtn = document.getElementById("detectBtn");
const detectLoader = document.getElementById("detectLoader");
const detectError = document.getElementById("detectError");
const detectEmpty = document.getElementById("detectEmpty");
const detectResult = document.getElementById("detectResult");
const detectCount = document.getElementById("detectCount");
const detectImg = document.getElementById("detectImg");
const detectOverlay = document.getElementById("detectOverlay");
const detectActiveFace = document.getElementById("detectActiveFace");
const detectFacesLabel = document.getElementById("detectFacesLabel");
const detectFaceNav = document.getElementById("detectFaceNav");
const detectPrev = document.getElementById("detectPrev");
const detectNext = document.getElementById("detectNext");
const detectSelectedIdx = document.getElementById("detectSelectedIdx");
const detectTotal = document.getElementById("detectTotal");

let detectState = null;

bindDropzone(detectDrop, detectFile, detectPreview);

detectForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!detectFile.files[0]) return;
    hide(detectError);
    hide(detectResult);
    hide(detectEmpty);
    show(detectLoader);
    detectBtn.disabled = true;

    const fd = new FormData(detectForm);
    try {
        const res = await fetch("/api/detect", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");

        detectCount.textContent = data.face_count;
        detectImg.src = data.image;
        if (data.faces.length === 0) {
            detectFacesLabel.classList.add("hidden");
            detectFaceNav.classList.add("hidden");
            detectActiveFace.classList.add("hidden");
            detectOverlay.innerHTML = "";
            detectState = null;
        } else {
            detectFacesLabel.classList.remove("hidden");
            detectState = { data, selectedIdx: 0 };
            renderDetectSelection(0);
        }
        show(detectResult);
    } catch (err) {
        detectError.textContent = err.message;
        show(detectError);
        show(detectEmpty);
    } finally {
        hide(detectLoader);
        detectBtn.disabled = false;
    }
});

// ---------- Compare ----------
const compareForm = document.getElementById("compareForm");
const compareFile1 = document.getElementById("compareFile1");
const compareFile2 = document.getElementById("compareFile2");
const compareDrop1 = document.getElementById("compareDrop1");
const compareDrop2 = document.getElementById("compareDrop2");
const comparePreview1 = document.getElementById("comparePreview1");
const comparePreview2 = document.getElementById("comparePreview2");
const compareBtn = document.getElementById("compareBtn");
const compareLoader = document.getElementById("compareLoader");
const compareError = document.getElementById("compareError");
const compareResult = document.getElementById("compareResult");
const compareVerdict = document.getElementById("compareVerdict");
const cmpSimilarity = document.getElementById("cmpSimilarity");
const metricsGrid = document.getElementById("metricsGrid");
const cmpCount1 = document.getElementById("cmpCount1");
const cmpCount2 = document.getElementById("cmpCount2");
const cmpImg1 = document.getElementById("cmpImg1");
const cmpImg2 = document.getElementById("cmpImg2");
const cmpOverlay1 = document.getElementById("cmpOverlay1");
const cmpOverlay2 = document.getElementById("cmpOverlay2");
const cmpCrop1 = document.getElementById("cmpCrop1");
const cmpCrop2 = document.getElementById("cmpCrop2");
const cmpIdx1 = document.getElementById("cmpIdx1");
const cmpIdx2 = document.getElementById("cmpIdx2");
const cmpFaceNav = document.getElementById("cmpFaceNav");
const cmpPrev = document.getElementById("cmpPrev");
const cmpNext = document.getElementById("cmpNext");
const cmpSelectedIdx = document.getElementById("cmpSelectedIdx");
const cmpTotal = document.getElementById("cmpTotal");
let cmpState = null;

const SVG_NS = "http://www.w3.org/2000/svg";

function renderOverlay(svgEl, w, h, boxes, activeIdx, withDataIdx) {
    svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const baseStroke = Math.max(2, Math.min(w, h) * 0.005);
    const labelH = Math.max(20, Math.min(w, h) * 0.035);
    const labelW = labelH * 1.6;
    const fontSize = labelH * 0.62;

    svgEl.innerHTML = boxes.map((b, i) => {
        const [x1, y1, x2, y2] = b;
        const bw = x2 - x1, bh = y2 - y1;
        const active = i === activeIdx;
        const color = active ? "#22c55e" : "#ef4444";
        const lblY = Math.max(0, y1 - labelH - 2);
        const dataAttr = withDataIdx ? `data-face-idx="${i}"` : "";
        return `
            <g ${dataAttr}>
                <rect x="${x1}" y="${y1}" width="${bw}" height="${bh}"
                      fill="transparent" stroke="${color}"
                      stroke-width="${active ? baseStroke * 1.8 : baseStroke}"
                      rx="${baseStroke}" />
                <rect x="${x1}" y="${lblY}" width="${labelW}" height="${labelH}"
                      fill="${color}" rx="${labelH * 0.18}"/>
                <text x="${x1 + labelW / 2}" y="${lblY + labelH * 0.72}"
                      fill="#fff" font-weight="700"
                      text-anchor="middle" font-size="${fontSize}"
                      font-family="JetBrains Mono, monospace">${i + 1}</text>
            </g>`;
    }).join("");
}

function renderSelection(idx) {
    if (!cmpState) return;
    const d = cmpState.data;
    const total = d.face_count_2;
    const sel = Math.max(0, Math.min(total - 1, idx));
    cmpState.selectedIdx2 = sel;

    const m = d.matches_2to1[sel];
    const targetIdx1 = m.match_index;
    const totalPairs = d.face_count_1 * d.face_count_2;

    compareVerdict.classList.remove("success", "fail");
    const subline = `WAJAH #${sel + 1} (G2) ↔ WAJAH #${targetIdx1 + 1} (G1) · ${totalPairs} TOTAL PASANGAN`;
    if (m.verified) {
        compareVerdict.classList.add("success");
        compareVerdict.innerHTML = `${CHECK_ICON}<div><div class="verdict-title">Wajah cocok — kemungkinan orang yang sama</div><div class="verdict-sub">${subline}</div></div>`;
    } else {
        compareVerdict.classList.add("fail");
        compareVerdict.innerHTML = `${X_ICON}<div><div class="verdict-title">Wajah tidak cocok — kemungkinan orang berbeda</div><div class="verdict-sub">${subline}</div></div>`;
    }

    renderOverlay(cmpOverlay1, d.image1_w, d.image1_h, d.boxes_1, targetIdx1, true);
    renderOverlay(cmpOverlay2, d.image2_w, d.image2_h, d.boxes_2, sel, true);

    cmpCrop1.src = d.crops_1[targetIdx1];
    cmpCrop2.src = d.crops_2[sel];
    cmpIdx1.textContent = targetIdx1 + 1;
    cmpIdx2.textContent = sel + 1;

    cmpSimilarity.textContent = m.similarity_percent.toFixed(2) + "%";
    renderMetrics(m.metrics || []);

    cmpSelectedIdx.textContent = sel + 1;
    cmpTotal.textContent = total;
    if (total > 1) {
        cmpFaceNav.classList.remove("hidden");
    } else {
        cmpFaceNav.classList.add("hidden");
    }
}

cmpPrev.addEventListener("click", () => {
    if (!cmpState) return;
    const total = cmpState.data.face_count_2;
    renderSelection((cmpState.selectedIdx2 - 1 + total) % total);
});
cmpNext.addEventListener("click", () => {
    if (!cmpState) return;
    const total = cmpState.data.face_count_2;
    renderSelection((cmpState.selectedIdx2 + 1) % total);
});
cmpOverlay2.addEventListener("click", (e) => {
    const g = e.target.closest("[data-face-idx]");
    if (!g) return;
    const idx = parseInt(g.getAttribute("data-face-idx"), 10);
    if (!Number.isNaN(idx)) renderSelection(idx);
});
cmpOverlay1.addEventListener("click", (e) => {
    const g = e.target.closest("[data-face-idx]");
    if (!g || !cmpState) return;
    const idx1 = parseInt(g.getAttribute("data-face-idx"), 10);
    if (Number.isNaN(idx1)) return;
    const m = cmpState.data.matches_1to2;
    if (m && m[idx1]) renderSelection(m[idx1].match_index);
});

function renderDetectSelection(idx) {
    if (!detectState) return;
    const d = detectState.data;
    const total = d.face_count;
    if (total === 0) return;
    const sel = Math.max(0, Math.min(total - 1, idx));
    detectState.selectedIdx = sel;

    const f = d.faces[sel];
    const boxes = d.faces.map((face) => face.facial_area);
    renderOverlay(detectOverlay, d.image_w, d.image_h, boxes, sel, true);

    detectActiveFace.classList.remove("hidden");
    detectActiveFace.innerHTML = `
        <img src="${f.crop}" alt="face #${sel + 1}">
        <div class="active-face-info">
            <div class="active-face-num">#${sel + 1}</div>
            <div class="active-face-stats">
                <div><span class="stat-key">Score</span><span class="stat-val">${f.score.toFixed(4)}</span></div>
                <div><span class="stat-key">Bounding Box</span><span class="stat-val">[${f.facial_area.join(", ")}]</span></div>
            </div>
        </div>`;

    detectSelectedIdx.textContent = sel + 1;
    detectTotal.textContent = total;
    if (total > 1) {
        detectFaceNav.classList.remove("hidden");
    } else {
        detectFaceNav.classList.add("hidden");
    }
}

detectPrev.addEventListener("click", () => {
    if (!detectState) return;
    const total = detectState.data.face_count;
    renderDetectSelection((detectState.selectedIdx - 1 + total) % total);
});
detectNext.addEventListener("click", () => {
    if (!detectState) return;
    const total = detectState.data.face_count;
    renderDetectSelection((detectState.selectedIdx + 1) % total);
});
detectOverlay.addEventListener("click", (e) => {
    const g = e.target.closest("[data-face-idx]");
    if (!g) return;
    const idx = parseInt(g.getAttribute("data-face-idx"), 10);
    if (!Number.isNaN(idx)) renderDetectSelection(idx);
});

bindDropzone(compareDrop1, compareFile1, comparePreview1);
bindDropzone(compareDrop2, compareFile2, comparePreview2);

const CHECK_ICON = '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>';
const X_ICON = '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>';

function renderMetrics(metrics) {
    metricsGrid.innerHTML = metrics.map((m, i) => {
        const cls = m.verified === true ? "ok" : m.verified === false ? "no" : "unset";
        const stateLabel = m.verified === true ? "MATCH" : m.verified === false ? "NO MATCH" : "INFO";
        const thrText = m.threshold != null ? `thr ${m.threshold.toFixed(4)}` : "no threshold";
        const simLine = m.similarity_percent != null ? `<div class="metric-card-sim">~ ${m.similarity_percent.toFixed(2)}% similar</div>` : "";
        return `
            <div class="metric-card ${cls}" style="animation-delay: ${i * 60}ms">
                <div class="metric-card-label">${m.label}</div>
                <div class="metric-card-value">${m.distance.toFixed(4)}</div>
                ${simLine}
                <div class="metric-card-thr">${thrText}</div>
                <span class="metric-card-state">${stateLabel}</span>
            </div>`;
    }).join("");
}

compareForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!compareFile1.files[0] || !compareFile2.files[0]) return;
    hide(compareError);
    hide(compareResult);
    show(compareLoader);
    compareBtn.disabled = true;

    const fd = new FormData(compareForm);
    try {
        const res = await fetch("/api/compare", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");

        cmpCount1.textContent = data.face_count_1;
        cmpCount2.textContent = data.face_count_2;
        cmpImg1.src = data.image1;
        cmpImg2.src = data.image2;
        cmpState = { data, selectedIdx2: data.best_match_index_2 };
        renderSelection(data.best_match_index_2);
        show(compareResult);
    } catch (err) {
        compareError.textContent = err.message;
        show(compareError);
    } finally {
        hide(compareLoader);
        compareBtn.disabled = false;
    }
});
