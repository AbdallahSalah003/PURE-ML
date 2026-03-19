const API_URL      = "http://localhost:8000/predict";
const MAX_HISTORY  = 5;

const urlInput       = document.getElementById("url-input");
const checkBtn       = document.getElementById("check-btn");
const clearInputBtn  = document.getElementById("clear-input-btn");
const loadingEl      = document.getElementById("loading");
const loadingStep    = document.getElementById("loading-step");
const resultEl       = document.getElementById("result");
const errorEl        = document.getElementById("error-box");
const historySection = document.getElementById("history-section");
const historyList    = document.getElementById("history-list");

// ── Input handling ────────────────────────────────────────────────────────────

urlInput.addEventListener("input", () => {
  const hasValue    = urlInput.value.trim().length > 0;
  checkBtn.disabled = !hasValue;
  clearInputBtn.style.display = hasValue ? "block" : "none";
  hideAll();
});

urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !checkBtn.disabled) checkUrl();
});

clearInputBtn.addEventListener("click", () => {
  urlInput.value = "";
  urlInput.focus();
  checkBtn.disabled = true;
  clearInputBtn.style.display = "none";
  hideAll();
});

// ── Check URL ─────────────────────────────────────────────────────────────────

checkBtn.addEventListener("click", checkUrl);

async function checkUrl() {
  const url = urlInput.value.trim();

  if (!url) return;

  // Basic URL validation
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    showError("URL must start with http:// or https://");
    return;
  }

  hideAll();
  showLoading();
  checkBtn.disabled = true;

  // Simulate loading steps so user knows what is happening
  // (dynamic features take several seconds)
  const steps = [
    "Extracting URL features...",
    "Running DNS lookups...",
    "Checking SSL certificate...",
    "Running XGBoost model...",
  ];
  let stepIndex = 0;
  const stepInterval = setInterval(() => {
    stepIndex = (stepIndex + 1) % steps.length;
    loadingStep.textContent = steps[stepIndex];
  }, 1800);

  try {
    const response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
    });

    clearInterval(stepInterval);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await response.json();
    hideAll();
    showResult(data);
    saveToHistory(data);

  } catch (err) {
    clearInterval(stepInterval);
    hideAll();

    if (err.message.includes("Failed to fetch")) {
      showError("Cannot reach the API. Make sure the Docker container is running on port 8000.");
    } else {
      showError(`Error: ${err.message}`);
    }
  } finally {
    checkBtn.disabled = false;
  }
}

// ── Display helpers ───────────────────────────────────────────────────────────

function hideAll() {
  loadingEl.style.display = "none";
  resultEl.style.display  = "none";
  errorEl.style.display   = "none";
}

function showLoading() {
  loadingEl.style.display = "block";
  loadingStep.textContent = "Extracting URL features...";
}

function showResult(data) {
  const prob       = data.probability;
  const isSafe     = data.safe;
  const isUnclear  = prob > 0.35 && prob < 0.65;

  // Card class
  resultEl.className = "result " + (isUnclear ? "warning" : isSafe ? "safe" : "danger");

  // Icon and title
  document.getElementById("result-icon").textContent =
    isUnclear ? "⚠️" : isSafe ? "✅" : "🚨";

  document.getElementById("result-title").textContent =
    isUnclear ? "Uncertain — proceed carefully"
              : isSafe ? "Safe URL"
                       : "Phishing Detected";

  // Truncate URL for display
  const displayUrl = data.url.length > 45
    ? data.url.substring(0, 42) + "..."
    : data.url;
  document.getElementById("result-url").textContent = displayUrl;

  // Probability bar
  const pct = Math.round(prob * 100);
  document.getElementById("prob-value").textContent   = `${pct}%`;
  document.getElementById("prob-bar").style.width     = `${pct}%`;

  // Details
  document.getElementById("detail-confidence").textContent =
    data.confidence.charAt(0).toUpperCase() + data.confidence.slice(1);
  document.getElementById("detail-verdict").textContent =
    data.label.charAt(0).toUpperCase() + data.label.slice(1);

  resultEl.style.display = "block";
}

function showError(message) {
  errorEl.textContent    = `⚠ ${message}`;
  errorEl.style.display  = "block";
}

// ── History ───────────────────────────────────────────────────────────────────

function saveToHistory(data) {
  let history = getHistory();

  // Remove duplicate if same URL checked again
  history = history.filter(item => item.url !== data.url);

  // Prepend new entry
  history.unshift({
    url:         data.url,
    label:       data.label,
    probability: data.probability,
    safe:        data.safe,
  });

  // Keep only last N
  history = history.slice(0, MAX_HISTORY);

  chrome.storage.local.set({ history });
  renderHistory(history);
}

function getHistory() {
  return JSON.parse(localStorage.getItem("phishing_history") || "[]");
}

function renderHistory(history) {
  if (!history.length) {
    historySection.style.display = "none";
    return;
  }

  historySection.style.display = "block";
  historyList.innerHTML = "";

  history.forEach(item => {
    const div = document.createElement("div");
    div.className = "history-item";
    div.title     = item.url;

    const dot  = document.createElement("div");
    dot.className = `history-dot ${item.safe ? "safe" : "danger"}`;

    const urlSpan  = document.createElement("span");
    urlSpan.className   = "history-url";
    urlSpan.textContent = item.url;

    const probSpan  = document.createElement("span");
    probSpan.className   = "history-prob";
    probSpan.textContent = `${Math.round(item.probability * 100)}%`;

    div.appendChild(dot);
    div.appendChild(urlSpan);
    div.appendChild(probSpan);

    // Click to re-check
    div.addEventListener("click", () => {
      urlInput.value = item.url;
      checkBtn.disabled = false;
      clearInputBtn.style.display = "block";
      hideAll();
    });

    historyList.appendChild(div);
  });
}

// ── Clear history ─────────────────────────────────────────────────────────────

document.getElementById("clear-history-btn").addEventListener("click", () => {
  chrome.storage.local.remove("history");
  historySection.style.display = "none";
});

// ── Load history on open ──────────────────────────────────────────────────────

chrome.storage.local.get("history", (result) => {
  if (result.history && result.history.length) {
    renderHistory(result.history);
  }
});
