const API_URL     = "http://localhost:8000/predict";
const MAX_HISTORY = 5;

const urlInput       = document.getElementById("url-input");
const checkBtn       = document.getElementById("check-btn");
const clearBtn       = document.getElementById("clear-btn");
const loadingEl      = document.getElementById("loading");
const loadingStep    = document.getElementById("loading-step");
const resultEl       = document.getElementById("result");
const errorEl        = document.getElementById("error-box");
const historySection = document.getElementById("history-section");
const historyList    = document.getElementById("history-list");

// ── Input ─────────────────────────────────────────────────────

urlInput.addEventListener("input", () => {
  checkBtn.disabled = urlInput.value.trim().length === 0;
  hideAll();
});

urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !checkBtn.disabled) checkUrl();
});

clearBtn.addEventListener("click", () => {
  urlInput.value    = "";
  checkBtn.disabled = true;
  hideAll();
  urlInput.focus();
});

// ── Check ─────────────────────────────────────────────────────

checkBtn.addEventListener("click", checkUrl);

async function checkUrl() {
  const url = urlInput.value.trim();
  if (!url) return;

  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    showError("ERROR: URL must start with http:// or https://");
    return;
  }

  hideAll();
  showLoading();
  checkBtn.disabled = true;

  const steps = [
    "Extracting URL features...",
    "Running DNS lookups...",
    "Checking SSL certificate...",
    "Querying WHOIS...",
    "Running XGBoost model...",
  ];
  let stepIndex  = 0;
  const interval = setInterval(() => {
    stepIndex           = (stepIndex + 1) % steps.length;
    loadingStep.textContent = steps[stepIndex];
  }, 1600);

  try {
    const response = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
    });

    clearInterval(interval);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server returned an error");
    }

    const data = await response.json();
    hideAll();
    showResult(data);
    saveToHistory(data);

  } catch (err) {
    clearInterval(interval);
    hideAll();

    if (err.message.includes("Failed to fetch")) {
      showError(
        "ERROR: Cannot reach API.\n" +
        "Make sure Docker container is running.\n" +
        "Check: http://localhost:8000/health"
      );
    } else {
      showError("ERROR: " + err.message);
    }
  } finally {
    checkBtn.disabled = false;
  }
}

// ── Result ────────────────────────────────────────────────────

function showResult(data) {
  const prob      = data.probability;
  const pct       = Math.round(prob * 100);
  const isUnclear = prob > 0.35 && prob < 0.65;

  // Card style
  resultEl.className = "result " +
    (isUnclear ? "warning" : data.safe ? "safe" : "danger");

  // Verdict line
  let verdictText;
  if (isUnclear)   verdictText = ">> UNCERTAIN — PROCEED WITH CAUTION";
  else if (data.safe) verdictText = ">> SAFE — LEGITIMATE URL";
  else             verdictText = ">> WARNING — PHISHING DETECTED";

  document.getElementById("result-verdict").textContent = verdictText;

  // Probability bar
  document.getElementById("prob-bar").style.width = pct + "%";

  // Detail rows
  document.getElementById("detail-prob").textContent =
    pct + "% phishing";
  document.getElementById("detail-verdict").textContent =
    data.label.toUpperCase();
  document.getElementById("detail-confidence").textContent =
    data.confidence.toUpperCase();

  const displayUrl = data.url.length > 35
    ? data.url.substring(0, 32) + "..."
    : data.url;
  document.getElementById("detail-url").textContent = displayUrl;
  document.getElementById("detail-url").title       = data.url;

  resultEl.style.display = "block";
}

// ── Error ─────────────────────────────────────────────────────

function showError(msg) {
  errorEl.textContent  = msg;
  errorEl.style.display = "block";
}

// ── Loading ───────────────────────────────────────────────────

function showLoading() {
  loadingEl.style.display  = "block";
  loadingStep.textContent  = "Extracting URL features...";
}

function hideAll() {
  loadingEl.style.display = "none";
  resultEl.style.display  = "none";
  errorEl.style.display   = "none";
}

// ── History ───────────────────────────────────────────────────

function saveToHistory(data) {
  chrome.storage.local.get("history", (res) => {
    let history = res.history || [];
    history     = history.filter(item => item.url !== data.url);
    history.unshift({
      url:         data.url,
      label:       data.label,
      probability: data.probability,
      safe:        data.safe,
    });
    history = history.slice(0, MAX_HISTORY);
    chrome.storage.local.set({ history });
    renderHistory(history);
  });
}

function renderHistory(history) {
  if (!history || !history.length) {
    historySection.style.display = "none";
    return;
  }

  historySection.style.display = "block";
  historyList.innerHTML        = "";

  history.forEach(item => {
    const div = document.createElement("div");
    div.className = "history-item";
    div.title     = item.url;

    const marker       = document.createElement("span");
    marker.className   = "history-marker " + (item.safe ? "safe" : "danger");
    marker.textContent = item.safe ? "[+]" : "[!]";

    const urlSpan       = document.createElement("span");
    urlSpan.className   = "history-url";
    urlSpan.textContent = item.url;

    const probSpan       = document.createElement("span");
    probSpan.className   = "history-prob";
    probSpan.textContent = Math.round(item.probability * 100) + "%";

    div.appendChild(marker);
    div.appendChild(urlSpan);
    div.appendChild(probSpan);

    div.addEventListener("click", () => {
      urlInput.value    = item.url;
      checkBtn.disabled = false;
      hideAll();
    });

    historyList.appendChild(div);
  });
}

document.getElementById("clear-history-btn").addEventListener("click", () => {
  chrome.storage.local.remove("history");
  historySection.style.display = "none";
});

// ── Load history on open ──────────────────────────────────────

chrome.storage.local.get("history", (res) => {
  if (res.history && res.history.length) {
    renderHistory(res.history);
  }
});