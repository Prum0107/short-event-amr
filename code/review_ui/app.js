/* Dependency-free manual review UI. Human labels are saved by qid. */

const REVIEW_FIELDS = [
  "gt_audible",
  "top1_contains_event",
  "outside_gt_occurrence",
  "gt_boundary_reasonable",
  "event_type",
  "background_complexity",
  "provisional_interpretation",
  "primary_observation",
];

const state = {
  manifest: [],
  reviews: {},
  filter: "all",
  visible: [],
  currentVisibleIndex: 0,
  currentCase: null,
  reviewStop: null,
};

const $ = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const number = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const formatSeconds = (value) => `${number(value).toFixed(1)}s`;
const formatScore = (value) => number(value).toFixed(4);
const isReviewed = (qid) => Boolean(state.reviews[qid]?.reviewed_at);

function bestIoU(interval, gtWindows) {
  const [start, end] = interval;
  return Math.max(...gtWindows.map(([gtStart, gtEnd]) => {
    const intersection = Math.max(0, Math.min(end, gtEnd) - Math.max(start, gtStart));
    const union = Math.max(end, gtEnd) - Math.min(start, gtStart);
    return union > 0 ? intersection / union : 0;
  }));
}

function caseMatchesFilter(item) {
  if (state.filter === "unreviewed") return !isReviewed(item.qid);
  if (state.filter === "reviewed") return isReviewed(item.qid);
  if (state.filter === "0-2") return item.max_gt_length < 2;
  if (state.filter === "2-5") return item.max_gt_length >= 2 && item.max_gt_length < 5;
  if (state.filter === "iou0") return item.top1_best_iou === 0;
  if (state.filter === "oracle-gap") return item.top1_best_iou < 0.5 && item.oracle10_best_iou >= 0.7;
  return true;
}

function updateVisible() {
  state.visible = state.manifest.filter(caseMatchesFilter);
  if (!state.visible.length) {
    state.currentVisibleIndex = 0;
    state.currentCase = null;
    $("caseList").innerHTML = '<p class="muted">No cases match this filter.</p>';
    $("caseView").classList.add("hidden");
    return;
  }
  state.currentVisibleIndex = clamp(state.currentVisibleIndex, 0, state.visible.length - 1);
  renderCaseList();
  loadCase(state.visible[state.currentVisibleIndex]);
}

function renderProgress() {
  const reviewed = state.manifest.filter((item) => isReviewed(item.qid)).length;
  $("progressText").textContent = `Reviewed: ${reviewed} / ${state.manifest.length} · Remaining: ${state.manifest.length - reviewed}`;
  $("progressBar").style.width = `${state.manifest.length ? (reviewed / state.manifest.length) * 100 : 0}%`;
}

function renderCaseList() {
  $("caseList").innerHTML = state.visible.map((item, index) => {
    const active = state.currentCase?.qid === item.qid ? " active" : "";
    const reviewed = isReviewed(item.qid) ? " reviewed" : "";
    const shortGroup = item.max_gt_length < 2 ? "0–2s" : "2–5s";
    return `<button class="case-item${active}${reviewed}" data-case-index="${index}" type="button">
      <span class="case-item-top"><span>${index + 1}. ${shortGroup}</span><span>IoU ${formatScore(item.top1_best_iou)}</span></span>
      <span class="case-item-query">${escapeHtml(item.query)}</span>
      <span class="case-item-meta">${formatSeconds(item.duration)} · ${escapeHtml(item.diagnostic_regime)}</span>
    </button>`;
  }).join("");
  document.querySelectorAll(".case-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.currentVisibleIndex = Number(button.dataset.caseIndex);
      loadCase(state.visible[state.currentVisibleIndex]);
    });
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"}[character]));
}

function loadCase(item) {
  state.currentCase = item;
  state.reviewStop = null;
  $("caseView").classList.remove("hidden");
  $("loading").classList.add("hidden");
  $("errorState").classList.add("hidden");
  $("caseCounter").textContent = `Case ${state.currentVisibleIndex + 1} / ${state.visible.length} (filtered)`;
  $("reviewGroup").textContent = item.review_group;
  $("caseTitle").textContent = `Case ${state.currentVisibleIndex + 1}`;
  $("qid").textContent = `qid: ${item.qid}`;
  $("vid").textContent = `vid: ${item.vid}`;
  $("queryText").textContent = `“${item.query}”`;
  $("gtGroupBadge").textContent = item.max_gt_length < 2 ? "GT 0–2 sec" : "GT 2–5 sec";
  $("audioBadge").textContent = item.audio_duration_bin;
  $("audioDuration").textContent = formatSeconds(item.duration);
  $("gtCount").textContent = String(item.num_gt_windows);
  $("gtLength").textContent = formatSeconds(item.max_gt_length);
  $("caseGroup").textContent = item.review_group;
  $("diagnosticRegime").textContent = item.diagnostic_regime;
  $("top1Window").textContent = `[${formatSeconds(item.top1_start)}, ${formatSeconds(item.top1_end)}]`;
  $("top1Score").textContent = formatScore(item.top1_score);
  $("top1Iou").textContent = formatScore(item.top1_best_iou);
  $("oracle10Iou").textContent = formatScore(item.oracle10_best_iou);
  $("top1Status").textContent = item.top1_best_iou >= 0.7 ? "PASS @ 0.7" : "FAIL @ 0.7";
  $("top1Status").style.background = item.top1_best_iou >= 0.7 ? "var(--green-soft)" : "var(--amber-soft)";
  $("top1Status").style.color = item.top1_best_iou >= 0.7 ? "var(--green)" : "var(--amber)";
  $("audioStatus").textContent = `Expected audio: ${item.audio_path}`;
  setupAudio(item);
  renderGtSummary(item);
  renderPredictions(item);
  renderReviewForm(state.reviews[item.qid] || {});
  renderCaseList();
  renderProgress();
}

function setupAudio(item) {
  const player = $("audioPlayer");
  player.pause();
  player.src = item.audio_url;
  player.load();
  $("audioMissing").classList.add("hidden");
  player.onerror = () => $("audioMissing").classList.remove("hidden");
  player.onloadedmetadata = () => {
    if (Number.isFinite(player.duration) && player.duration > 0) {
      item.browser_duration = player.duration;
      $("durationLabel").textContent = formatSeconds(player.duration);
      renderTimeline(item, player.duration);
    }
  };
  player.ontimeupdate = () => {
    updatePlayhead(item, Number.isFinite(player.duration) ? player.duration : item.duration);
    if (state.reviewStop !== null && player.currentTime >= state.reviewStop) {
      player.pause();
      state.reviewStop = null;
    }
  };
  renderTimeline(item, item.duration);
}

function renderGtSummary(item) {
  $("gtSummary").innerHTML = item.gt_windows.map((window, index) => `<div><strong>GT ${index + 1}</strong> · [${formatSeconds(window[0])}, ${formatSeconds(window[1])}] · length ${formatSeconds(window[1] - window[0])}</div>`).join("");
  $("gtButtons").innerHTML = item.gt_windows.map((window, index) => `<button class="gt-button" data-gt-index="${index}" type="button">GT ${index + 1}</button>`).join("");
  document.querySelectorAll("[data-gt-index]").forEach((button) => {
    button.addEventListener("click", () => jumpTo(item.gt_windows[Number(button.dataset.gtIndex)][0]));
  });
}

function renderTimeline(item, duration) {
  const safeDuration = Math.max(number(duration, item.duration), 0.001);
  const interval = (start, end, className, label) => `<div class="interval ${className}" title="${label}" style="left:${clamp(start / safeDuration * 100, 0, 100)}%;width:${clamp((end - start) / safeDuration * 100, 0.4, 100)}%"></div>`;
  let html = item.gt_windows.map((window, index) => interval(window[0], window[1], "gt", `GT ${index + 1}`)).join("");
  const predictions = item.pred_relevant_windows || [];
  html += predictions.slice(0, 3).map((prediction, index) => interval(Number(prediction[0]), Number(prediction[1]), index === 0 ? "top1" : "other", `Top-${index + 1}`)).join("");
  html += '<div id="playhead" class="playhead" style="left:0%"></div>';
  $("timeline").innerHTML = html;
  $("durationLabel").textContent = formatSeconds(duration);
}

function updatePlayhead(item, duration) {
  const playhead = $("playhead");
  if (playhead) playhead.style.left = `${clamp($("audioPlayer").currentTime / Math.max(duration, 0.001) * 100, 0, 100)}%`;
}

function playRange(start, end) {
  const player = $("audioPlayer");
  const duration = Number.isFinite(player.duration) ? player.duration : state.currentCase.duration;
  const from = clamp(start - 3, 0, duration);
  state.reviewStop = clamp(end + 3, 0, duration);
  player.currentTime = from;
  player.play().catch(() => {});
}

function jumpTo(time) {
  const player = $("audioPlayer");
  player.currentTime = clamp(time, 0, Number.isFinite(player.duration) ? player.duration : state.currentCase.duration);
  player.focus();
}

function renderPredictions(item) {
  const rows = (item.pred_relevant_windows || []).slice(0, 10).map((prediction, index) => {
    const start = Number(prediction[0]);
    const end = Number(prediction[1]);
    const score = Number(prediction[2]);
    const iou = bestIoU([start, end], item.gt_windows);
    return `<tr><td>${index + 1}</td><td>${formatSeconds(start)}</td><td>${formatSeconds(end)}</td><td>${formatScore(score)}</td><td>${formatScore(iou)}</td></tr>`;
  }).join("");
  $("predTable").innerHTML = rows;
}

function renderReviewForm(review) {
  REVIEW_FIELDS.forEach((field) => {
    const element = document.querySelector(`[data-review="${field}"]`);
    if (element) element.value = review[field] || "";
  });
  $("savedAt").textContent = review.reviewed_at ? `Saved ${new Date(review.reviewed_at).toLocaleString()}` : "Not yet reviewed";
  $("saveStatus").textContent = "";
}

function readReviewForm() {
  const review = {};
  REVIEW_FIELDS.forEach((field) => {
    const element = document.querySelector(`[data-review="${field}"]`);
    review[field] = element ? element.value : "";
  });
  review.reviewed_at = new Date().toISOString();
  return review;
}

async function saveCurrent() {
  if (!state.currentCase) return false;
  const review = readReviewForm();
  const response = await fetch("/api/save-review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ qid: state.currentCase.qid, review }) });
  if (!response.ok) throw new Error(`Save failed (${response.status})`);
  const payload = await response.json();
  state.reviews[state.currentCase.qid] = payload.review;
  $("savedAt").textContent = `Saved ${new Date(payload.review.reviewed_at).toLocaleString()}`;
  $("saveStatus").textContent = "Review saved.";
  renderProgress();
  renderCaseList();
  return true;
}

function move(delta) {
  if (!state.visible.length) return;
  state.currentVisibleIndex = (state.currentVisibleIndex + delta + state.visible.length) % state.visible.length;
  loadCase(state.visible[state.currentVisibleIndex]);
}

function exportReviews(extension) {
  const link = document.createElement("a");
  link.href = `/api/export?format=${extension}`;
  link.download = `manual_reviews.${extension}`;
  link.click();
}

function bindEvents() {
  document.querySelectorAll(".filter").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".filter").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.filter = button.dataset.filter;
    state.currentVisibleIndex = 0;
    updateVisible();
  }));
  $("refreshButton").addEventListener("click", async () => { await loadReviews(); updateVisible(); });
  $("playFull").addEventListener("click", () => { state.reviewStop = null; jumpTo(0); $("audioPlayer").play().catch(() => {}); });
  $("playGt").addEventListener("click", () => { const window = state.currentCase.gt_windows[0]; playRange(window[0], window[1]); });
  $("playPred").addEventListener("click", () => { const prediction = state.currentCase.pred_relevant_windows[0]; playRange(Number(prediction[0]), Number(prediction[1])); });
  $("jumpGt").addEventListener("click", () => jumpTo(state.currentCase.gt_windows[0][0]));
  $("jumpPred").addEventListener("click", () => jumpTo(Number(state.currentCase.pred_relevant_windows[0][0])));
  $("nudgeBack").addEventListener("click", () => jumpTo($("audioPlayer").currentTime - 5));
  $("nudgeForward").addEventListener("click", () => jumpTo($("audioPlayer").currentTime + 5));
  $("previousButton").addEventListener("click", () => move(-1));
  $("nextButton").addEventListener("click", () => move(1));
  $("saveButton").addEventListener("click", async () => { try { await saveCurrent(); } catch (error) { $("saveStatus").textContent = error.message; } });
  $("saveNextButton").addEventListener("click", async () => { try { if (await saveCurrent()) move(1); } catch (error) { $("saveStatus").textContent = error.message; } });
  $("exportCsvButton").addEventListener("click", () => exportReviews("csv"));
  $("exportJsonButton").addEventListener("click", () => exportReviews("json"));
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); move(-1); }
    if (event.key === "ArrowRight") { event.preventDefault(); move(1); }
    if (event.key === " ") { event.preventDefault(); const player = $("audioPlayer"); if (player.paused) player.play().catch(() => {}); else player.pause(); }
    if (event.key.toLowerCase() === "s") { event.preventDefault(); saveCurrent().catch((error) => { $("saveStatus").textContent = error.message; }); }
    if (event.key.toLowerCase() === "g") { event.preventDefault(); jumpTo(state.currentCase.gt_windows[0][0]); }
    if (event.key.toLowerCase() === "p") { event.preventDefault(); jumpTo(Number(state.currentCase.pred_relevant_windows[0][0])); }
  });
}

async function loadReviews() {
  const response = await fetch("/api/reviews");
  if (!response.ok) throw new Error("Could not load review database");
  state.reviews = await response.json();
}

async function init() {
  try {
    const response = await fetch("/api/manifest");
    if (!response.ok) throw new Error("Could not load review manifest");
    const payload = await response.json();
    state.manifest = payload.cases || [];
    if (state.manifest.length !== 30) throw new Error(`Expected 30 review cases, found ${state.manifest.length}`);
    await loadReviews();
    bindEvents();
    updateVisible();
  } catch (error) {
    $("loading").classList.add("hidden");
    $("errorState").textContent = error.message;
    $("errorState").classList.remove("hidden");
  }
}

init();
