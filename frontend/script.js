/**
 * VISORA frontend: displays only results returned by the local Flask/CSRNet API.
 */

const API_BASE = "http://127.0.0.1:5000";
const MAX_FILE_BYTES = 200 * 1024 * 1024;
const allowedExtensions = new Set(["mp4", "avi", "mov", "mkv", "webm"]);
const $ = (id) => document.getElementById(id);

const app = $("app");
const sidebar = $("sidebar");
const videoInput = $("videoInput");
const uploadZone = $("uploadZone");
const uploadPrompt = $("uploadPrompt");
const videoPreview = $("videoPreview");
const densityMapImg = $("densityMapImg");
const densityEmptyState = $("densityEmptyState");
const fileMeta = $("fileMeta");
const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const resetBtn = $("resetBtn");
const progressWrap = $("progressWrap");
const progressFill = $("progressFill");
const progressPct = $("progressPct");
const progressLabel = $("progressLabel");
const processingStatusTag = $("processingStatusTag");
const chartCanvas = $("crowdChart");
const chartTooltip = $("chartTooltip");
const timeFilter = $("timeFilter");
const customRange = $("customRange");
const toastRegion = $("toastRegion");
const navAlertCount = $("navAlertCount");

let selectedFile = null;
let previewUrl = null;
let activeView = "source";
let isProcessing = false;
let cancelRequested = false;
let statusPollTimer = null;
let healthTimer = null;
let analysisData = null;
let chartData = [];
let currentAnalysisTimestamp = null;
let analysisHistory = [];
let acknowledgedAlerts = new Set();
let activeChartRange = null;
let hoverPoint = null;

const sectionTitles = {
  dashboard: "Overview",
  surveillance: "Surveillance",
  analytics: "Analytics",
  densityAnalysis: "Density maps",
  alerts: "Alerts",
  reports: "Reports",
  systemInfo: "System information",
  settings: "Settings",
};

function showToast(message, kind = "success") {
  const toast = document.createElement("div");
  toast.className = `toast toast--${kind}`;
  const text = document.createElement("span");
  text.textContent = message;
  toast.appendChild(text);
  toastRegion.appendChild(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function setStatus(kind, label) {
  $("statusText").textContent = label;
  const dot = $("statusDot");
  dot.className = `status-dot${kind === "idle" ? " status-dot--muted" : ""}${kind === "error" ? " status-dot--error" : ""}${kind === "processing" ? " status-dot--warning status-dot--pulse" : ""}`;
}

function updateClock() {
  $("datetime").textContent = new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
}
updateClock();
window.setInterval(updateClock, 30_000);

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const whole = Math.floor(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const remainder = whole % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatCount(value) {
  if (value === null || value === undefined || value === "" || !Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);
}

function formatFileSize(bytes) {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function formatTimeline(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  if (value >= 3600) {
    return `${Math.floor(value / 3600)}:${String(Math.floor((value % 3600) / 60)).padStart(2, "0")}`;
  }
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}

function setAnalysisStatus(label, kind = "muted") {
  const status = $("infoStatus");
  status.replaceChildren();
  const dot = document.createElement("i");
  dot.className = `status-dot${kind === "muted" ? " status-dot--muted" : ""}${kind === "warning" ? " status-dot--warning status-dot--pulse" : ""}${kind === "error" ? " status-dot--error" : ""}`;
  status.append(dot, document.createTextNode(` ${label.toUpperCase()}`));
  $("processHealth").textContent = label;
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Health check returned HTTP ${response.status}`);
    const data = await response.json();
    $("backendHealth").textContent = "Connected";
    $("backendHealth").className = "health-state is-online";
    const modelLoaded = data.model_status === "loaded";
    $("modelHealth").textContent = modelLoaded ? "Available" : "Unavailable";
    $("modelHealth").className = `health-state ${modelLoaded ? "is-online" : "is-warning"}`;
    $("modelDevice").textContent = modelLoaded
      ? `Inference device · ${data.device || "available"}`
      : String(data.model_status || "Model not loaded").replace(/^missing_weights:\s*/, "");
    $("engineStatus").textContent = modelLoaded ? "MODEL READY" : "WEIGHTS REQUIRED";
    $("engineDot").className = `status-dot${modelLoaded ? "" : " status-dot--warning"}`;
    $("healthBadge").innerHTML = `<i class="status-dot${modelLoaded ? "" : " status-dot--warning"}"></i> ${modelLoaded ? "OPERATIONAL" : "MODEL UNAVAILABLE"}`;
    setStatus(isProcessing ? "processing" : "idle", isProcessing ? "Analyzing video" : "API connected");
  } catch (error) {
    $("backendHealth").textContent = "Offline";
    $("backendHealth").className = "health-state is-error";
    $("modelHealth").textContent = "Unknown";
    $("modelHealth").className = "health-state is-error";
    $("modelDevice").textContent = "Backend unreachable";
    $("engineStatus").textContent = "API OFFLINE";
    $("engineDot").className = "status-dot status-dot--error";
    $("healthBadge").innerHTML = '<i class="status-dot status-dot--error"></i> OFFLINE';
    setStatus("error", "Backend unavailable");
    console.warn("VISORA backend health check failed:", error);
  }
}

checkHealth();
healthTimer = window.setInterval(checkHealth, 30_000);

function setSelectedFile(file) {
  if (!file) return;
  const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
  if (!allowedExtensions.has(extension)) {
    showToast("Unsupported format. Choose MP4, AVI, MOV, MKV, or WEBM.", "error");
    videoInput.value = "";
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showToast("This video exceeds the 200 MB upload limit.", "error");
    videoInput.value = "";
    return;
  }
  if (file.size === 0) {
    showToast("This video file is empty.", "error");
    videoInput.value = "";
    return;
  }
  if (isProcessing) {
    showToast("Stop the current analysis before choosing another video.", "warning");
    return;
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  uploadZone.classList.remove("is-empty");
  clearCurrentResult();
  previewUrl = URL.createObjectURL(file);
  videoPreview.src = previewUrl;
  setView(activeView);
  densityMapImg.hidden = true;
  densityEmptyState.hidden = true;
  $("fileName").textContent = file.name;
  $("fileDetails").textContent = `${formatFileSize(file.size)} · Video file`;
  fileMeta.hidden = false;
  startBtn.disabled = false;
  processingStatusTag.textContent = "Ready to analyze";
  $("infoVideoName").textContent = file.name;
  setAnalysisStatus("Ready");
  showToast("Video ready for local analysis.");
}

function clearCurrentResult() {
  analysisData = null;
  currentAnalysisTimestamp = null;
  chartData = [];
  acknowledgedAlerts = new Set();
  activeChartRange = null;
  densityMapImg.hidden = true;
  densityMapImg.removeAttribute("src");
  $("stageWatermark").hidden = true;
  ["statCurrent", "statPeak", "statAverage"].forEach((id) => { $(id).textContent = "—"; });
  $("statDensity").textContent = "AWAITING";
  $("statDensity").className = "stat-value stat-value--tag";
  $("statSafety").textContent = "PENDING";
  $("statSafety").className = "stat-value stat-value--tag";
  $("visualCount").textContent = "—";
  $("visualDensity").textContent = "NO RESULT";
  $("visualDensity").className = "density-pill";
  $("infoDuration").textContent = "—";
  $("infoTotalFrames").textContent = "—";
  $("infoFrames").textContent = "—";
  $("infoFps").textContent = "—";
  $("infoModel").textContent = "CSRNet";
  $("infoTimestamp").textContent = "—";
  $("currentTrend").textContent = "LIVE RESULT";
  $("alertsList").innerHTML = '<li class="alert-empty"><span class="empty-mark"><svg class="icon"><use href="#i-bell"/></svg></span><strong>No alerts to display</strong><span>Alert events will appear here after analysis.</span></li>';
  $("alertCounter").textContent = "0 EVENTS";
  navAlertCount.hidden = true;
  $("notificationDot").hidden = true;
  $("exportCsvBtn").disabled = true;
  $("exportPdfBtn").disabled = true;
  progressWrap.hidden = true;
  $("chartFootnote").textContent = "Run an analysis to view real frame estimates. Historical video records are not stored by the backend.";
  updateChartRange();
}

videoInput.addEventListener("change", () => {
  if (videoInput.files && videoInput.files.length) setSelectedFile(videoInput.files[0]);
});
$("chooseVideoBtn").addEventListener("click", (event) => {
  event.stopPropagation();
  if (!isProcessing) videoInput.click();
});
uploadZone.addEventListener("click", (event) => {
  if (selectedFile || event.target.closest("button, video, input")) return;
  if (!isProcessing) videoInput.click();
});
uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.classList.add("is-dragging");
});
uploadZone.addEventListener("dragleave", (event) => {
  if (!uploadZone.contains(event.relatedTarget)) uploadZone.classList.remove("is-dragging");
});
uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.classList.remove("is-dragging");
  if (event.dataTransfer.files.length) setSelectedFile(event.dataTransfer.files[0]);
});

function setView(view) {
  activeView = view;
  document.querySelectorAll(".view-option").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  $("viewNote").textContent = view === "source" ? "SOURCE FOOTAGE" : "CSRNET DENSITY MAP";
  if (view === "source") {
    uploadPrompt.hidden = Boolean(selectedFile);
    videoPreview.hidden = !selectedFile;
    densityMapImg.hidden = true;
    densityEmptyState.hidden = true;
    $("stageWatermark").hidden = true;
    return;
  }
  videoPreview.hidden = true;
  uploadPrompt.hidden = true;
  if (analysisData && analysisData.density_map_path) {
    densityMapImg.hidden = false;
    densityEmptyState.hidden = true;
    $("stageWatermark").hidden = false;
  } else {
    densityMapImg.hidden = true;
    densityEmptyState.hidden = false;
    $("stageWatermark").hidden = true;
  }
}
document.querySelectorAll(".view-option").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

function setBusy(busy) {
  isProcessing = busy;
  startBtn.hidden = busy;
  startBtn.disabled = busy || !selectedFile;
  stopBtn.hidden = !busy;
  resetBtn.disabled = busy;
  videoInput.disabled = busy;
  $("processHealth").textContent = busy ? "Processing" : "Idle";
}

startBtn.addEventListener("click", runAnalysis);
stopBtn.addEventListener("click", stopAnalysis);
resetBtn.addEventListener("click", resetDashboard);

async function runAnalysis() {
  if (!selectedFile || isProcessing) return;
  clearCurrentResult();
  setView(activeView);
  setBusy(true);
  cancelRequested = false;
  acknowledgedAlerts = new Set();
  startBtn.hidden = true;
  progressWrap.hidden = false;
  progressFill.classList.add("is-indeterminate");
  progressFill.style.width = "";
  progressPct.textContent = "LIVE";
  progressLabel.textContent = "Waiting for CSRNet inference…";
  processingStatusTag.textContent = "Analysis in progress";
  setStatus("processing", "Analyzing video");
  setAnalysisStatus("Processing", "warning");

  const body = new FormData();
  body.append("video", selectedFile);
  statusPollTimer = window.setInterval(pollStatus, 700);
  try {
    const response = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body });
    let data;
    try {
      data = await response.json();
    } catch (error) {
      const message = response.ok
        ? "The backend returned an invalid analysis response."
        : `Analysis failed (HTTP ${response.status}); the backend response was not valid JSON.`;
      throw new Error(message, { cause: error });
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new Error("The backend returned an invalid analysis response.");
    }
    if (!response.ok) {
      if (cancelRequested && response.status === 409) {
        finishCancelled();
        return;
      }
      throw new Error(data.error || `Analysis failed (HTTP ${response.status}).`);
    }

    analysisData = data;
    currentAnalysisTimestamp = new Date();
    renderResults(data);
    addHistoryRecord(data, currentAnalysisTimestamp);
    setBusy(false);
    progressFill.classList.remove("is-indeterminate");
    progressFill.style.width = "100%";
    progressPct.textContent = "100%";
    progressLabel.textContent = `${data.frames_processed} sampled frames analyzed`;
    processingStatusTag.textContent = "Analysis complete";
    setAnalysisStatus("Complete");
    setStatus("idle", "Analysis complete");
    showToast("CSRNet analysis completed.");
  } catch (error) {
    setBusy(false);
    progressFill.classList.remove("is-indeterminate");
    progressLabel.textContent = "Analysis failed";
    progressPct.textContent = "—";
    processingStatusTag.textContent = "Analysis failed";
    setAnalysisStatus("Error", "error");
    setStatus("error", "Analysis failed");
    showToast(error.message || "Analysis failed. Check the backend service.", "error");
    console.error("VISORA analysis failed:", error);
  } finally {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
    if (!cancelRequested) startBtn.disabled = !selectedFile;
  }
}

async function stopAnalysis() {
  if (!isProcessing || cancelRequested) return;
  cancelRequested = true;
  stopBtn.disabled = true;
  processingStatusTag.textContent = "Stopping analysis…";
  progressLabel.textContent = "Stopping after the current inference step…";
  setAnalysisStatus("Stopping", "warning");
  setStatus("processing", "Stopping analysis");
  try {
    const response = await fetch(`${API_BASE}/api/stop`, { method: "POST" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "The backend could not stop the analysis.");
    showToast("Stop requested. The current frame will finish first.", "warning");
  } catch (error) {
    cancelRequested = false;
    if (isProcessing) {
      stopBtn.disabled = false;
      processingStatusTag.textContent = "Analysis in progress";
      progressLabel.textContent = "CSRNet is processing sampled frames…";
      setAnalysisStatus("Processing", "warning");
      setStatus("processing", "Analyzing video");
    }
    showToast(error.message || "Could not stop the analysis.", "error");
    console.error("VISORA stop request failed:", error);
  }
}

function finishCancelled() {
  setBusy(false);
  progressFill.classList.remove("is-indeterminate");
  progressFill.style.width = "0%";
  progressPct.textContent = "—";
  progressLabel.textContent = "Analysis stopped";
  processingStatusTag.textContent = "Analysis stopped";
  setAnalysisStatus("Stopped");
  setStatus("idle", "Analysis stopped");
  showToast("Analysis stopped. No partial results were saved.", "warning");
}

async function pollStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/status`, { cache: "no-store" });
    if (!response.ok) return;
    const data = await response.json();
    if (data.status === "PROCESSING") {
      progressLabel.textContent = data.progress
        ? `CSRNet processed ${data.progress} sampled frame${data.progress === 1 ? "" : "s"}…`
        : "Preparing CSRNet inference…";
    } else if (data.status === "CANCELLED" && cancelRequested) {
      progressLabel.textContent = "Finishing cancellation…";
    }
  } catch (error) {
    console.warn("VISORA status polling failed:", error);
  }
}

function renderResults(data) {
  $("statCurrent").textContent = formatCount(data.current_count);
  $("statPeak").textContent = formatCount(data.peak_count);
  $("statAverage").textContent = formatCount(data.average_count);
  $("statDensity").textContent = data.density || "—";
  $("statDensity").className = `stat-value stat-value--tag ${data.density ? `tag-${data.density.toLowerCase()}` : ""}`;
  $("statSafety").textContent = data.status || "—";
  $("statSafety").className = `stat-value stat-value--tag ${data.status ? `tag-${data.status.toLowerCase()}` : ""}`;
  $("visualCount").textContent = formatCount(data.current_count);
  $("visualDensity").textContent = data.density || "NO RESULT";
  $("visualDensity").className = `density-pill ${data.density ? `tag-${data.density.toLowerCase()}` : ""}`;

  $("infoVideoName").textContent = data.video_name || selectedFile?.name || "—";
  $("infoDuration").textContent = formatDuration(data.video_duration_sec);
  $("infoTotalFrames").textContent = Number.isFinite(data.total_frames_in_video)
    ? new Intl.NumberFormat().format(data.total_frames_in_video)
    : "—";
  $("infoFrames").textContent = Number.isFinite(data.frames_processed)
    ? new Intl.NumberFormat().format(data.frames_processed)
    : "—";
  $("infoFps").textContent = Number.isFinite(data.processing_fps) ? `${data.processing_fps} fps` : "—";
  $("infoModel").textContent = data.model_used || "CSRNet";
  $("infoTimestamp").textContent = currentAnalysisTimestamp.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
  $("currentTrend").textContent = "LATEST FRAME";

  if (data.density_map_path) {
    const filename = String(data.density_map_path).split(/[\\/]/).pop();
    densityMapImg.src = `${API_BASE}/outputs/density_maps/${encodeURIComponent(filename)}`;
    densityMapImg.onerror = () => showToast("The backend could not load the generated density map.", "error");
  }
  if (activeView === "density") setView("density");

  const rawCounts = Array.isArray(data.frame_counts) ? data.frame_counts : [];
  const duration = Number(data.video_duration_sec) || 0;
  const videoFps = Number(data.video_fps) || 0;
  const interval = Number(data.frame_sample_interval) || 1;
  chartData = rawCounts.map((count, index) => ({
    count: Number(count),
    index,
    seconds: videoFps > 0 ? (index * interval) / videoFps : (rawCounts.length > 1 ? duration * index / (rawCounts.length - 1) : 0),
  })).filter((sample) => Number.isFinite(sample.count));
  activeChartRange = null;
  $("chartFootnote").textContent = chartData.length
    ? `${chartData.length} real sampled-frame estimates from this video. The backend does not persist historical sessions.`
    : "No sampled-frame data was returned by the backend.";
  updateChartRange();
  renderAlerts(Array.isArray(data.alerts) ? data.alerts : [], data);
  $("exportCsvBtn").disabled = false;
  $("exportPdfBtn").disabled = false;
}

function selectedChartPoints() {
  if (!analysisData || !chartData.length) return [];
  const videoDuration = Number(analysisData.video_duration_sec) || chartData[chartData.length - 1].seconds || 0;
  if (activeChartRange && activeChartRange.type === "custom") {
    return chartData.filter((point) => point.seconds >= activeChartRange.start && point.seconds <= activeChartRange.end);
  }
  if (activeChartRange && activeChartRange.seconds) {
    const boundary = Math.max(0, videoDuration - activeChartRange.seconds);
    return chartData.filter((point) => point.seconds >= boundary);
  }
  return chartData;
}

function updateChartRange() {
  if (!analysisData) {
    activeChartRange = null;
  } else if (!activeChartRange || activeChartRange.type !== "custom") {
    const rangeSeconds = { "1h": 3600, "24h": 86400, "7d": 604800 }[timeFilter.value];
    activeChartRange = rangeSeconds ? { seconds: rangeSeconds } : null;
  }
  const points = selectedChartPoints();
  const videoDuration = Number(analysisData?.video_duration_sec) || 0;
  const timeRange = timeFilter.value === "custom" ? "Custom video range" : {
    "1h": "Last 1 hour", "24h": "Last 24 hours", "7d": "Last 7 days",
  }[timeFilter.value];
  $("chartContext").textContent = analysisData
    ? `${points.length} / ${chartData.length} samples · ${timeRange}`
    : "No analysis data";
  if (timeFilter.value === "custom") {
    customRange.hidden = false;
    $("rangeEnd").placeholder = (videoDuration / 60).toFixed(1);
    if (!activeChartRange || activeChartRange.type !== "custom") {
      $("rangeStart").value = "0";
      $("rangeEnd").value = (videoDuration / 60).toFixed(1);
    }
  } else {
    customRange.hidden = true;
  }
  drawChart();
}

timeFilter.addEventListener("change", () => {
  activeChartRange = null;
  updateChartRange();
});
$("applyRange").addEventListener("click", () => {
  const start = Number($("rangeStart").value);
  const end = Number($("rangeEnd").value || (Number(analysisData?.video_duration_sec) / 60));
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start) {
    showToast("Enter a valid start and end time in minutes.", "error");
    return;
  }
  activeChartRange = { type: "custom", start: start * 60, end: end * 60 };
  updateChartRange();
});

function drawChart() {
  if (!chartCanvas) return;
  const rect = chartCanvas.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const ratio = window.devicePixelRatio || 1;
  chartCanvas.width = Math.round(width * ratio);
  chartCanvas.height = Math.round(height * ratio);
  const context = chartCanvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const pad = { top: 17, right: 12, bottom: 28, left: 39 };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  const points = selectedChartPoints();
  const values = points.map((point) => point.count);
  const average = values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
  const maxValue = Math.max(1, ...values, Number(analysisData?.peak_count) || 0) * 1.15;
  const plotStart = points.length ? points[0].seconds : 0;
  const plotEnd = points.length > 1 ? points[points.length - 1].seconds : plotStart + 1;
  const x = (index) => pad.left + (points.length <= 1 ? plotWidth / 2 : (plotWidth * index) / (points.length - 1));
  const y = (value) => pad.top + plotHeight * (1 - value / maxValue);

  context.font = "9px 'DM Sans', sans-serif";
  context.textBaseline = "middle";
  context.textAlign = "right";
  for (let tick = 0; tick <= 4; tick += 1) {
    const tickY = pad.top + (plotHeight * tick) / 4;
    context.beginPath();
    context.strokeStyle = "rgba(151, 174, 193, .105)";
    context.lineWidth = 1;
    context.moveTo(pad.left, tickY);
    context.lineTo(width - pad.right, tickY);
    context.stroke();
    context.fillStyle = "#738797";
    context.fillText(String(Math.round(maxValue * (1 - tick / 4))), pad.left - 7, tickY);
  }

  context.textAlign = "left";
  context.fillStyle = "#738797";
  context.fillText(formatTimeline(plotStart), pad.left, height - 10);
  const endLabel = formatTimeline(plotEnd);
  context.textAlign = "right";
  context.fillText(endLabel, width - pad.right, height - 10);
  if (points.length > 2) {
    context.textAlign = "center";
    context.fillText(formatTimeline((plotStart + plotEnd) / 2), pad.left + plotWidth / 2, height - 10);
  }

  if (!points.length) {
    context.textAlign = "center";
    context.fillStyle = "#718495";
    context.fillText(analysisData ? "No samples in this range" : "Analysis data will appear here", width / 2, pad.top + plotHeight / 2);
    hoverPoint = null;
    return;
  }

  const averageY = y(average);
  context.beginPath();
  context.setLineDash([4, 5]);
  context.strokeStyle = "rgba(240, 189, 105, .68)";
  context.lineWidth = 1;
  context.moveTo(pad.left, averageY);
  context.lineTo(width - pad.right, averageY);
  context.stroke();
  context.setLineDash([]);

  if (points.length > 1) {
    const gradient = context.createLinearGradient(0, pad.top, 0, pad.top + plotHeight);
    gradient.addColorStop(0, "rgba(84, 223, 208, .19)");
    gradient.addColorStop(1, "rgba(84, 223, 208, .005)");
    context.beginPath();
    points.forEach((point, index) => {
      if (index === 0) context.moveTo(x(index), y(point.count));
      else context.lineTo(x(index), y(point.count));
    });
    context.lineTo(x(points.length - 1), pad.top + plotHeight);
    context.lineTo(x(0), pad.top + plotHeight);
    context.closePath();
    context.fillStyle = gradient;
    context.fill();
  }

  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) context.moveTo(x(index), y(point.count));
    else context.lineTo(x(index), y(point.count));
  });
  context.strokeStyle = "#55dfd0";
  context.lineWidth = 2;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.stroke();
  if (points.length === 1) {
    context.beginPath();
    context.arc(x(0), y(points[0].count), 3.5, 0, Math.PI * 2);
    context.fillStyle = "#55dfd0";
    context.fill();
  }
  if (hoverPoint !== null && hoverPoint < points.length) {
    context.beginPath();
    context.arc(x(hoverPoint), y(points[hoverPoint].count), 4, 0, Math.PI * 2);
    context.fillStyle = "#e9ffff";
    context.fill();
    context.lineWidth = 2;
    context.strokeStyle = "#44cdbc";
    context.stroke();
  }
  chartCanvas._chartCoordinates = points.map((point, index) => ({ x: x(index), y: y(point.count), point }));
}

chartCanvas.addEventListener("pointermove", (event) => {
  const rect = chartCanvas.getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const coordinates = chartCanvas._chartCoordinates || [];
  if (!coordinates.length) return;
  let nearest = 0;
  coordinates.forEach((coordinate, index) => {
    if (Math.abs(coordinate.x - localX) < Math.abs(coordinates[nearest].x - localX)) nearest = index;
  });
  hoverPoint = nearest;
  drawChart();
  const point = coordinates[nearest].point;
  chartTooltip.textContent = `${formatCount(point.count)} people · ${formatTimeline(point.seconds)}`;
  chartTooltip.hidden = false;
  chartTooltip.style.left = `${Math.max(4, Math.min(rect.width - 130, coordinates[nearest].x + 8))}px`;
  chartTooltip.style.top = `${Math.max(3, coordinates[nearest].y - 31)}px`;
});
chartCanvas.addEventListener("pointerleave", () => {
  hoverPoint = null;
  chartTooltip.hidden = true;
  drawChart();
});
window.addEventListener("resize", drawChart);
drawChart();

function renderAlerts(alerts, data) {
  const list = $("alertsList");
  list.replaceChildren();
  navAlertCount.hidden = alerts.length === 0;
  navAlertCount.textContent = String(alerts.length);
  $("notificationDot").hidden = alerts.length === 0;
  $("alertCounter").textContent = `${alerts.length} ${alerts.length === 1 ? "EVENT" : "EVENTS"}`;
  if (!alerts.length) {
    const empty = document.createElement("li");
    empty.className = "alert-empty";
    empty.innerHTML = '<span class="empty-mark"><svg class="icon"><use href="#i-bell"/></svg></span><strong>No density transitions detected</strong><span>CSRNet did not report a threshold change in the sampled frames.</span>';
    list.appendChild(empty);
    return;
  }

  const interval = Number(data.frame_sample_interval) || 1;
  const fps = Number(data.video_fps) || 0;
  alerts.forEach((alert, alertIndex) => {
    const item = document.createElement("li");
    item.className = `alert-item severity-${alert.severity || "info"}`;
    const dot = document.createElement("i");
    dot.className = "alert-level-dot";
    const content = document.createElement("div");
    content.className = "alert-content";
    const topline = document.createElement("div");
    topline.className = "alert-topline";
    const level = document.createElement("strong");
    level.className = "alert-density";
    level.textContent = alert.density_level || "DENSITY";
    const frame = document.createElement("span");
    frame.className = "alert-frame";
    frame.textContent = `· ${formatCount(alert.count)} people`;
    topline.append(level, frame);
    const message = document.createElement("p");
    message.className = "alert-message";
    message.textContent = alert.message || "Density level changed.";
    const elapsed = fps > 0 ? ((Number(alert.frame_index) || 0) * interval) / fps : null;
    const timestamp = document.createElement("time");
    timestamp.className = "alert-time";
    timestamp.textContent = elapsed === null ? `SAMPLE ${Number(alert.frame_index) + 1 || alertIndex + 1}` : `${formatTimeline(elapsed)} INTO VIDEO`;
    content.append(topline, message, timestamp);

    const key = `${alert.frame_index}-${alert.density_level}-${alert.count}`;
    const acknowledge = document.createElement("button");
    acknowledge.className = "alert-ack";
    acknowledge.type = "button";
    acknowledge.textContent = acknowledgedAlerts.has(key) ? "Acknowledged" : "Acknowledge";
    acknowledge.setAttribute("aria-pressed", String(acknowledgedAlerts.has(key)));
    acknowledge.addEventListener("click", () => {
      if (acknowledgedAlerts.has(key)) acknowledgedAlerts.delete(key);
      else acknowledgedAlerts.add(key);
      acknowledge.setAttribute("aria-pressed", String(acknowledgedAlerts.has(key)));
      acknowledge.textContent = acknowledgedAlerts.has(key) ? "Acknowledged" : "Acknowledge";
    });
    item.append(dot, content, acknowledge);
    list.appendChild(item);
  });
}

function addHistoryRecord(data, timestamp) {
  analysisHistory.unshift({ data, timestamp });
  const historyList = $("historyList");
  historyList.replaceChildren();
  analysisHistory.forEach((record) => {
    const row = document.createElement("div");
    row.className = "history-row";
    const name = document.createElement("span");
    name.textContent = record.data.video_name || selectedFile?.name || "Uploaded video";
    const peak = document.createElement("span");
    peak.textContent = `${formatCount(record.data.peak_count)} people`;
    const density = document.createElement("span");
    density.className = "history-density";
    density.textContent = record.data.density || "—";
    const status = document.createElement("span");
    status.className = "history-state";
    status.innerHTML = '<i class="status-dot"></i> Complete';
    const analyzed = document.createElement("span");
    analyzed.textContent = record.timestamp.toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
    row.append(name, peak, density, status, analyzed);
    historyList.appendChild(row);
  });
}

function csvEscape(value) {
  const text = String(value ?? "");
  return `"${text.replace(/"/g, '""')}"`;
}

$("exportCsvBtn").addEventListener("click", () => {
  if (!analysisData) return;
  const columns = ["sample", "video_time_seconds", "estimated_people"];
  const rows = chartData.map((point) => [point.index + 1, point.seconds, point.count]);
  const csv = [columns, ...rows].map((row) => row.map(csvEscape).join(",")).join("\r\n");
  downloadBlob(csv, "text/csv;charset=utf-8", `visora-crowd-analysis-${Date.now()}.csv`);
  showToast("Crowd statistics exported as CSV.");
});

$("exportPdfBtn").addEventListener("click", () => {
  if (!analysisData) return;
  showToast("Choose “Save as PDF” in the print dialog.");
  window.print();
});

function downloadBlob(content, type, filename) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function resetDashboard() {
  if (isProcessing) {
    showToast("Stop the current analysis before resetting.", "warning");
    return;
  }
  try {
    const response = await fetch(`${API_BASE}/api/reset`, { method: "POST" });
    if (!response.ok) throw new Error(`Reset returned HTTP ${response.status}`);
  } catch (error) {
    showToast("The backend did not confirm reset; clearing this dashboard only.", "warning");
    console.warn("VISORA backend reset failed:", error);
  }

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  analysisData = null;
  currentAnalysisTimestamp = null;
  chartData = [];
  acknowledgedAlerts = new Set();
  activeChartRange = null;
  videoInput.value = "";
  videoInput.disabled = false;
  uploadZone.classList.add("is-empty");
  videoPreview.pause();
  videoPreview.removeAttribute("src");
  videoPreview.load();
  videoPreview.hidden = true;
  densityMapImg.hidden = true;
  densityMapImg.removeAttribute("src");
  uploadPrompt.hidden = false;
  densityEmptyState.hidden = true;
  fileMeta.hidden = true;
  setView("source");

  ["statCurrent", "statPeak", "statAverage"].forEach((id) => { $(id).textContent = "—"; });
  $("statDensity").textContent = "AWAITING";
  $("statDensity").className = "stat-value stat-value--tag";
  $("statSafety").textContent = "PENDING";
  $("statSafety").className = "stat-value stat-value--tag";
  $("visualCount").textContent = "—";
  $("visualDensity").textContent = "NO RESULT";
  $("visualDensity").className = "density-pill";
  $("currentTrend").textContent = "LIVE RESULT";
  $("infoVideoName").textContent = "—";
  $("infoDuration").textContent = "—";
  $("infoTotalFrames").textContent = "—";
  $("infoFrames").textContent = "—";
  $("infoFps").textContent = "—";
  $("infoModel").textContent = "CSRNet";
  $("infoTimestamp").textContent = "—";
  setAnalysisStatus("Idle");
  $("chartFootnote").textContent = "Run an analysis to view real frame estimates. Historical video records are not stored by the backend.";
  $("chartContext").textContent = "No analysis data";
  $("alertsList").innerHTML = '<li class="alert-empty"><span class="empty-mark"><svg class="icon"><use href="#i-bell"/></svg></span><strong>No alerts to display</strong><span>Alert events will appear here after analysis.</span></li>';
  $("alertCounter").textContent = "0 EVENTS";
  navAlertCount.hidden = true;
  $("notificationDot").hidden = true;
  $("historyList").innerHTML = '<div class="history-empty">Completed analysis records will appear here. Past sessions are not persisted by the backend.</div>';
  $("exportCsvBtn").disabled = true;
  $("exportPdfBtn").disabled = true;
  progressWrap.hidden = true;
  progressFill.classList.remove("is-indeterminate");
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";
  progressLabel.textContent = "Preparing analysis…";
  processingStatusTag.textContent = "No video selected";
  setBusy(false);
  setStatus("idle", "API connected");
  updateChartRange();
  showToast("Analysis workspace reset.");
}

document.querySelectorAll(".nav-item").forEach((item) => {
  const label = item.textContent.trim();
  item.setAttribute("aria-label", label);
  item.title = label;
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((link) => {
      link.classList.remove("active");
      link.removeAttribute("aria-current");
    });
    item.classList.add("active");
    item.setAttribute("aria-current", "page");
    $("pageTitle").textContent = sectionTitles[item.dataset.section] || "Overview";
    if (item.dataset.section === "densityAnalysis") setView("density");
    if (item.dataset.section === "surveillance") setView("source");
    sidebar.classList.remove("is-open");
  });
});

$("sidebarToggle").addEventListener("click", () => {
  app.classList.toggle("sidebar-collapsed");
  const collapsed = app.classList.contains("sidebar-collapsed");
  $("sidebarToggle").setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
  $("sidebarToggle").title = collapsed ? "Expand sidebar" : "Collapse sidebar";
});
$("mobileMenu").addEventListener("click", () => sidebar.classList.toggle("is-open"));
$("notificationButton").addEventListener("click", () => {
  document.querySelector('[data-section="alerts"]').click();
  $("alerts").scrollIntoView({ behavior: "smooth", block: "start" });
});

window.addEventListener("beforeunload", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  if (healthTimer) clearInterval(healthTimer);
});
