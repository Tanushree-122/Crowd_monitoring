/**
 * script.js
 * ---------
 * All dashboard numbers displayed here come directly from the JSON
 * response returned by the Flask backend's /api/analyze endpoint, which
 * itself is derived from real CSRNet inference (see backend/inference.py
 * and backend/crowd_analyzer.py). Nothing in this file uses Math.random()
 * or any other fabricated/simulated data source.
 */

// Change this if your Flask backend runs on a different host/port.
const API_BASE = "http://127.0.0.1:5000";

// ---------------------------------------------------------------------------
// Element references
// ---------------------------------------------------------------------------
const uploadZone = document.getElementById("uploadZone");
const videoInput = document.getElementById("videoInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const videoPreview = document.getElementById("videoPreview");
const fileMeta = document.getElementById("fileMeta");
const fileNameEl = document.getElementById("fileName");

const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");

const progressWrap = document.getElementById("progressWrap");
const progressFill = document.getElementById("progressFill");
const progressPct = document.getElementById("progressPct");
const progressLabel = document.getElementById("progressLabel");

const processingStatusTag = document.getElementById("processingStatusTag");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const statCurrent = document.getElementById("statCurrent");
const statPeak = document.getElementById("statPeak");
const statAverage = document.getElementById("statAverage");
const statDensity = document.getElementById("statDensity");
const statSafety = document.getElementById("statSafety");

const infoVideoName = document.getElementById("infoVideoName");
const infoDuration = document.getElementById("infoDuration");
const infoFrames = document.getElementById("infoFrames");
const infoFps = document.getElementById("infoFps");
const infoStatus = document.getElementById("infoStatus");

const densityMapImg = document.getElementById("densityMapImg");
const densityPlaceholder = document.getElementById("densityPlaceholder");

const alertsList = document.getElementById("alertsList");
const datetimeEl = document.getElementById("datetime");
const crowdChartCanvas = document.getElementById("crowdChart");

let selectedFile = null;

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
function updateClock() {
  const now = new Date();
  datetimeEl.textContent = now.toLocaleString(undefined, {
    weekday: "short", year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}
setInterval(updateClock, 1000);
updateClock();

// ---------------------------------------------------------------------------
// Canvas chart
// ---------------------------------------------------------------------------
let chartData = { counts: [], peak: 0, average: 0 };

function drawCrowdChart() {
  const canvas = crowdChartCanvas;
  const width = canvas.clientWidth;
  const height = Number(canvas.getAttribute("height")) || 220;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);

  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const pad = { top: 28, right: 12, bottom: 26, left: 42 };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  const maximum = Math.max(1, chartData.peak, ...chartData.counts) * 1.1;

  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (plotHeight * i) / 4;
    const value = Math.round(maximum * (1 - i / 4));
    ctx.strokeStyle = "#e3ebe7";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = "#718589";
    ctx.fillText(String(value), pad.left - 7, y);
  }

  const legend = [
    ["Crowd Count", "#12876F"],
    ["Peak", "#C34450"],
    ["Average", "#A96813"],
  ];
  let legendX = pad.left;
  ctx.textAlign = "left";
  legend.forEach(([label, color]) => {
    ctx.fillStyle = color;
    ctx.fillRect(legendX, 10, 12, 2);
    ctx.fillStyle = "#647981";
    ctx.fillText(label, legendX + 17, 11);
    legendX += ctx.measureText(label).width + 34;
  });

  const xAt = (index) => pad.left + (chartData.counts.length < 2
    ? plotWidth / 2
    : (plotWidth * index) / (chartData.counts.length - 1));
  const yAt = (value) => pad.top + plotHeight * (1 - value / maximum);

  if (chartData.counts.length > 1) {
    const area = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotHeight);
    area.addColorStop(0, "rgba(18, 135, 111, 0.18)");
    area.addColorStop(1, "rgba(18, 135, 111, 0.01)");
    ctx.beginPath();
    chartData.counts.forEach((value, index) => {
      const x = xAt(index);
      const y = yAt(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(xAt(chartData.counts.length - 1), pad.top + plotHeight);
    ctx.lineTo(xAt(0), pad.top + plotHeight);
    ctx.closePath();
    ctx.fillStyle = area;
    ctx.fill();
  }

  [
    { values: chartData.counts, color: "#12876F", dash: [] },
    { values: chartData.counts.map(() => chartData.peak), color: "#C34450", dash: [4, 4] },
    { values: chartData.counts.map(() => chartData.average), color: "#A96813", dash: [2, 3] },
  ].forEach(({ values, color, dash }) => {
    if (values.length === 0) return;
    ctx.beginPath();
    ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = dash.length ? 1.5 : 2.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    values.forEach((value, index) => {
      const x = xAt(index);
      const y = yAt(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);

  if (chartData.counts.length > 0) {
    const lastIndex = chartData.counts.length - 1;
    ctx.beginPath();
    ctx.arc(xAt(lastIndex), yAt(chartData.counts[lastIndex]), 4, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#12876F";
    ctx.stroke();
  }
}
drawCrowdChart();
window.addEventListener("resize", drawCrowdChart);

// ---------------------------------------------------------------------------
// Upload handling
// ---------------------------------------------------------------------------
videoInput.addEventListener("change", function () {
  if (!this.files || this.files.length === 0) {
    return;
  }

  const file = this.files[0];
  const maxSize = 200 * 1024 * 1024;
  if (file.size > maxSize) {
    alert("Video is larger than 200MB.");
    this.value = "";
    return;
  }

  selectedFile = file;
  videoPreview.src = URL.createObjectURL(selectedFile);
  videoPreview.hidden = false;
  uploadPrompt.hidden = true;
  fileMeta.hidden = false;
  fileNameEl.textContent =
    `${selectedFile.name} (${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)`;
  infoVideoName.textContent = selectedFile.name;
  infoStatus.textContent = "READY";
  startBtn.disabled = false;
  processingStatusTag.textContent = "Video ready";
});
// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------
startBtn.addEventListener("click", runAnalysis);
resetBtn.addEventListener("click", resetDashboard);

let statusPollTimer = null;

async function runAnalysis() {
  if (!selectedFile) return;

  setSystemStatus("processing", "Analyzing video…");
  const formData = new FormData();
  formData.append("video", selectedFile);
  progressWrap.hidden = false;
  progressLabel.textContent = "Processing frames…";
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";

  // Poll /api/status while the (blocking) /api/analyze call runs, so the
  // progress bar reflects real frame-processing progress reported by the
  // backend, not a simulated timer.
  statusPollTimer = setInterval(pollStatus, 700);

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    clearInterval(statusPollTimer);

    if (!response.ok) {
      throw new Error(data.error || "Analysis failed.");
    }

    renderResults(data);
    setSystemStatus("done", "Analysis complete");
    processingStatusTag.textContent = "Analysis complete";
    infoStatus.textContent = "DONE";
    progressLabel.textContent = "Complete";
    progressFill.style.width = "100%";
    progressPct.textContent = "100%";
  } catch (err) {
    clearInterval(statusPollTimer);
    setSystemStatus("error", "Error");
    processingStatusTag.textContent = "Error";
    infoStatus.textContent = "ERROR";
    progressLabel.textContent = "Failed";
    alert(`Analysis failed: ${err.message}`);
  } finally {
    startBtn.disabled = false;
  }
}

async function pollStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();
    if (data.status === "PROCESSING" && data.progress) {
      // We don't know the total frame count until the video finishes, so
      // show a monotonically growing indicator based on frames processed
      // relative to the backend's MAX_FRAMES_TO_PROCESS ceiling.
      const estimatedMax = 150; // mirrors backend config.MAX_FRAMES_TO_PROCESS default
      const pct = Math.min(95, Math.round((data.progress / estimatedMax) * 100));
      progressFill.style.width = `${pct}%`;
      progressPct.textContent = `${pct}%`;
      progressLabel.textContent = `Processed ${data.progress} frame(s)…`;
    }
  } catch (_err) {
    // Non-fatal: status polling is best-effort UI feedback only.
  }
}

function renderResults(data) {
  // ---- Stat cards ----
  statCurrent.textContent = data.current_count;
  statPeak.textContent = data.peak_count;
  statAverage.textContent = data.average_count;

  statDensity.textContent = data.density;
  statDensity.className = "stat-value stat-value--tag tag-" + data.density.toLowerCase();

  statSafety.textContent = data.status;
  statSafety.className = "stat-value stat-value--tag tag-" + data.status.toLowerCase();

  // ---- Monitoring info ----
  infoVideoName.textContent = data.video_name || selectedFile.name;
  infoDuration.textContent = `${data.video_duration_sec}s`;
  infoFrames.textContent = data.frames_processed;
  infoFps.textContent = data.processing_fps;

  // ---- Density map ----
  if (data.density_map_path) {
    const filename = data.density_map_path.split(/[\\/]/).pop();
    densityMapImg.src = `${API_BASE}/outputs/density_maps/${filename}`;
    densityMapImg.hidden = false;
    densityPlaceholder.hidden = true;
  }

  // ---- Chart ----
  chartData = {
    counts: data.frame_counts,
    peak: data.peak_count,
    average: data.average_count,
  };
  drawCrowdChart();

  // ---- Alerts ----
  alertsList.innerHTML = "";
  if (!data.alerts || data.alerts.length === 0) {
    alertsList.innerHTML = '<li class="alert-empty">No density-level changes detected.</li>';
  } else {
    data.alerts.forEach((a) => {
      const li = document.createElement("li");
      li.className = `alert-item severity-${a.severity}`;
      li.textContent = a.message;
      alertsList.appendChild(li);
    });
  }
}

function setSystemStatus(kind, label) {
  statusText.textContent = label;
  statusDot.className = "dot dot-" + kind;
}

async function resetDashboard() {
  try {
    await fetch(`${API_BASE}/api/reset`, { method: "POST" });
  } catch (_err) {
    // Backend may be offline; still reset the UI locally.
  }

  clearInterval(statusPollTimer);
  selectedFile = null;
  videoInput.value = "";

  videoPreview.hidden = true;
  videoPreview.removeAttribute("src");
  uploadPrompt.hidden = false;
  fileMeta.hidden = true;

  startBtn.disabled = true;
  progressWrap.hidden = true;
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";

  statCurrent.textContent = "0";
  statPeak.textContent = "0";
  statAverage.textContent = "0";
  statDensity.textContent = "—";
  statDensity.className = "stat-value stat-value--tag";
  statSafety.textContent = "—";
  statSafety.className = "stat-value stat-value--tag";

  infoVideoName.textContent = "—";
  infoDuration.textContent = "—";
  infoFrames.textContent = "—";
  infoFps.textContent = "—";
  infoStatus.textContent = "Idle";

  densityMapImg.hidden = true;
  densityMapImg.removeAttribute("src");
  densityPlaceholder.hidden = false;

  chartData = { counts: [], peak: 0, average: 0 };
  drawCrowdChart();

  alertsList.innerHTML = '<li class="alert-empty">No alerts yet. Run an analysis to populate this feed.</li>';

  processingStatusTag.textContent = "No video loaded";
  setSystemStatus("idle", "System Idle");
}

// ---------------------------------------------------------------------------
// Sidebar navigation
// ---------------------------------------------------------------------------
const navViews = {
  dashboard: { panels: null, showStats: true },
  surveillance: { panels: ["surveillance", "settings"], showStats: true },
  analytics: { panels: ["densityAnalysis", "analytics"], showStats: true },
  alerts: { panels: ["alerts"], showStats: true },
  settings: { panels: ["settings"], showStats: false },
};

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", (e) => {
    e.preventDefault();
    const view = navViews[item.dataset.section];
    if (!view) return;

    document.querySelectorAll(".nav-item").forEach((i) => i.classList.remove("active"));
    item.classList.add("active");

    document.querySelectorAll(".grid-2col .panel").forEach((panel) => {
      const visible = view.panels === null || view.panels.includes(panel.id);
      panel.classList.toggle("nav-view-hidden", !visible);
    });
    document.querySelectorAll(".grid-2col .col").forEach((column) => {
      const hasVisiblePanel = Array.from(column.querySelectorAll(".panel"))
        .some((panel) => !panel.classList.contains("nav-view-hidden"));
      column.classList.toggle("nav-view-hidden", !hasVisiblePanel);
    });
    document.getElementById("statsGrid").classList.toggle("nav-view-hidden", !view.showStats);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

// Initial health check against the backend (non-blocking, informational).
fetch(`${API_BASE}/api/health`)
  .then((r) => r.json())
  .then((d) => {
    if (d.model_status !== "loaded") {
      console.warn("CSRNet model not loaded:", d.model_status);
    }
  })
  .catch(() => console.warn("Backend not reachable at", API_BASE));
