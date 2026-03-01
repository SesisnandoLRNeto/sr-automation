/* ============================================================
   SR-Automation — Frontend Logic
   ============================================================ */

// ---------------------------------------------------------------------------
// Configuracao dos steps
// ---------------------------------------------------------------------------
const STEPS = [
  { id: "corpus",    name: "Corpus",       desc: "Coleta de artigos" },
  { id: "triage",    name: "Triagem",      desc: "Classificacao YES/NO" },
  { id: "extract",   name: "Extracao",     desc: "Dados estruturados" },
  { id: "summarize", name: "Resumos",      desc: "TL;DR 3 frases" },
  { id: "hallcheck", name: "Alucinacoes",  desc: "Amostra para review" },
  { id: "validate",  name: "Validacao",    desc: "Amostra de extracao" },
  { id: "gold",      name: "Gold Standard",desc: "Template p/ revisores" },
  { id: "metrics",   name: "Metricas",     desc: "Recall, Kappa, F1" },
  { id: "crossval",  name: "Cross-Val",    desc: "Validacao cruzada" },
  { id: "report",    name: "Relatorio",    desc: "Figuras e tabelas" },
];

const STEP_DEPS = {
  corpus: [],
  triage: ["corpus"],
  extract: ["triage"],
  summarize: ["extract"],
  hallcheck: ["extract"],
  validate: ["extract"],
  gold: ["corpus"],
  metrics: ["gold"],
  crossval: ["corpus"],
  report: ["metrics", "crossval"],
};

// Estado local
let currentStatus = {};
let logEventSource = null;

// ---------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------
async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showToast(msg) {
  let t = document.querySelector(".toast");
  if (!t) {
    t = document.createElement("div");
    t.className = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3000);
}

function escHtml(s) {
  if (s == null) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function truncate(s, len) {
  if (!s) return "";
  return s.length > len ? s.slice(0, len) + "..." : s;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Main tabs
  $$(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach(b => b.classList.remove("active"));
      $$(".tab-content").forEach(s => s.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-${btn.dataset.tab}`).classList.add("active");
      onTabSwitch(btn.dataset.tab);
    });
  });

  // Sub-tabs (delegated per section)
  $$("#tab-results .sub-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#tab-results .sub-tab").forEach(b => b.classList.remove("active"));
      $$("#tab-results .subtab-content").forEach(s => s.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-results #subtab-${btn.dataset.subtab}`).classList.add("active");
    });
  });
  $$("#tab-review .sub-tab").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#tab-review .sub-tab").forEach(b => b.classList.remove("active"));
      $$("#tab-review .subtab-content").forEach(s => s.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-review #subtab-${btn.dataset.subtab}`).classList.add("active");
    });
  });

  // Init
  loadConfig();
  refreshStatus();
  startLogStream();
  setInterval(refreshStatus, 5000);

  // Buttons
  $("#btn-save-config").addEventListener("click", saveConfig);
  $("#btn-export").addEventListener("click", exportResults);
  $("#btn-reset").addEventListener("click", resetPipeline);
  $("#btn-clear-log").addEventListener("click", () => { $("#log-output").textContent = ""; });
  $("#btn-save-gold").addEventListener("click", () => saveReview("gold"));
  $("#btn-save-hall").addEventListener("click", () => saveReview("hallucination"));
  $("#btn-save-val").addEventListener("click", () => saveReview("validation"));
  $("#btn-save-likert").addEventListener("click", () => saveReview("likert"));
  $("#btn-generate-likert").addEventListener("click", generateLikertSample);

  // Corpus search
  $("#corpus-search").addEventListener("input", filterCorpusTable);

  // Triage filter
  $("#triage-filter").addEventListener("change", filterTriageTable);
});

function onTabSwitch(tab) {
  if (tab === "corpus") loadCorpus();
  if (tab === "results") loadResults();
  if (tab === "review") loadReviews();
  if (tab === "metrics") loadMetrics();
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
async function loadConfig() {
  try {
    const cfg = await api("GET", "/api/config");
    $("#cfg-query").value = cfg.query || "";
    if (cfg.year_range && cfg.year_range.length === 2) {
      $("#cfg-year-start").value = cfg.year_range[0];
      $("#cfg-year-end").value = cfg.year_range[1];
    }
  } catch (e) {
    console.error("Erro ao carregar config:", e);
  }
}

async function saveConfig() {
  try {
    const body = {
      query: $("#cfg-query").value,
      year_range: [
        parseInt($("#cfg-year-start").value) || 2022,
        parseInt($("#cfg-year-end").value) || 2024,
      ],
    };
    await api("POST", "/api/config", body);
    showToast("Configuracao salva.");
  } catch (e) {
    showToast("Erro: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// Status & Step Grid
// ---------------------------------------------------------------------------
async function refreshStatus() {
  try {
    const data = await api("GET", "/api/status");
    currentStatus = data.steps;
    renderStepGrid(data);
  } catch (e) {
    console.error("Erro ao buscar status:", e);
  }
}

function renderStepGrid(data) {
  const grid = $("#step-grid");
  grid.innerHTML = "";

  STEPS.forEach(s => {
    const status = data.steps[s.id] || "pending";
    const card = document.createElement("div");
    card.className = "step-card";

    const badgeClass = `badge-${status}`;
    const badgeText = status === "done" ? "Concluido"
      : status === "running" ? '<span class="spinner"></span>Rodando'
      : "Pendente";

    // Verificar dependencias
    const deps = STEP_DEPS[s.id] || [];
    const depsOk = deps.every(d => data.steps[d] === "done");
    const canRun = depsOk && status !== "running" && !data.running;

    // Info extra (contagens)
    let info = s.desc;
    if (s.id === "corpus" && data.counts.corpus_total) {
      info = `${data.counts.corpus_total} artigos de ${data.counts.corpus_sources || "?"} fontes`;
    }
    if (s.id === "triage" && data.counts.triage_yes !== undefined) {
      info = `${data.counts.triage_yes} YES / ${data.counts.triage_no} NO`;
    }

    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="step-name">${escHtml(s.name)}</span>
        <span class="badge ${badgeClass}">${badgeText}</span>
      </div>
      <span class="step-info">${escHtml(info)}</span>
      <button class="btn btn-primary btn-sm btn-run"
        data-step="${s.id}"
        ${canRun ? "" : "disabled"}>
        Executar
      </button>
    `;

    card.querySelector(".btn-run").addEventListener("click", () => runStep(s.id));
    grid.appendChild(card);
  });
}

async function runStep(step) {
  try {
    await api("POST", `/api/run/${step}`);
    showToast(`Step "${step}" iniciado.`);
    refreshStatus();
  } catch (e) {
    showToast("Erro: " + e.message);
  }
}

async function exportResults() {
  try {
    showToast("Gerando export...");
    const resp = await fetch("/api/export", { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    // Extrair nome do arquivo do header ou usar fallback
    const disposition = resp.headers.get("content-disposition");
    const match = disposition && disposition.match(/filename="?([^"]+)"?/);
    a.download = match ? match[1] : "sr_export.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast("Export concluido!");
  } catch (e) {
    showToast("Erro no export: " + e.message);
  }
}

async function resetPipeline() {
  if (!confirm("Tem certeza? Todos os outputs serao removidos.")) return;
  try {
    const res = await api("POST", "/api/reset");
    showToast(`Reset: ${res.removed.length} arquivos removidos.`);
    refreshStatus();
  } catch (e) {
    showToast("Erro: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// Log streaming (SSE)
// ---------------------------------------------------------------------------
function startLogStream() {
  if (logEventSource) logEventSource.close();
  logEventSource = new EventSource("/api/logs");

  logEventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      const logEl = $("#log-output");
      logEl.textContent += data.msg + "\n";
      logEl.scrollTop = logEl.scrollHeight;
    } catch (e) {}
  };

  logEventSource.onerror = () => {
    // Reconectar apos 3s
    logEventSource.close();
    setTimeout(startLogStream, 3000);
  };
}

// ---------------------------------------------------------------------------
// Aba: Corpus
// ---------------------------------------------------------------------------
let corpusData = [];

async function loadCorpus() {
  try {
    corpusData = await api("GET", "/api/data/corpus");
    renderCorpusTable(corpusData);
  } catch (e) {
    console.error("Erro ao carregar corpus:", e);
  }
}

function renderCorpusTable(data) {
  const tbody = $("#corpus-table tbody");
  tbody.innerHTML = "";

  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">Nenhum artigo coletado.</td></tr>';
    $("#corpus-count").textContent = "";
    return;
  }

  // Contar fontes unicas
  const sources = new Set(data.map(r => r.source));
  $("#corpus-count").textContent = `${data.length} artigos de ${sources.size} fontes`;

  data.forEach(row => {
    const tr = document.createElement("tr");
    tr.className = "expandable";
    tr.innerHTML = `
      <td>${escHtml(row.id)}</td>
      <td title="${escHtml(row.title)}">${escHtml(truncate(row.title, 80))}</td>
      <td title="${escHtml(row.authors)}">${escHtml(truncate(row.authors, 50))}</td>
      <td>${escHtml(row.year)}</td>
      <td>${escHtml(row.source)}</td>
    `;
    tr.addEventListener("click", () => {
      const next = tr.nextElementSibling;
      if (next && next.classList.contains("detail-row")) {
        next.remove();
      } else {
        const detail = document.createElement("tr");
        detail.className = "detail-row";
        detail.innerHTML = `<td colspan="5"><strong>Abstract:</strong> ${escHtml(row.abstract || "N/A")}</td>`;
        tr.after(detail);
      }
    });
    tbody.appendChild(tr);
  });
}

function filterCorpusTable() {
  const q = $("#corpus-search").value.toLowerCase();
  if (!q) {
    renderCorpusTable(corpusData);
    return;
  }
  const filtered = corpusData.filter(r =>
    (r.title || "").toLowerCase().includes(q) ||
    (r.authors || "").toLowerCase().includes(q) ||
    (r.abstract || "").toLowerCase().includes(q)
  );
  renderCorpusTable(filtered);
}

// ---------------------------------------------------------------------------
// Aba: Resultados
// ---------------------------------------------------------------------------
let triageData = [];

async function loadResults() {
  try {
    const [triage, extraction, summaries, corpus] = await Promise.all([
      api("GET", "/api/data/triage"),
      api("GET", "/api/data/extraction"),
      api("GET", "/api/data/summaries"),
      api("GET", "/api/data/corpus"),
    ]);
    // Montar lookup do corpus se ainda nao existe
    if (!Object.keys(corpusLookup).length) {
      corpus.forEach(r => { corpusLookup[r.id] = r; });
    }
    triageData = triage;
    renderTriageTable(triage);
    renderExtractionTable(extraction);
    renderSummariesTable(summaries);
  } catch (e) {
    console.error("Erro ao carregar resultados:", e);
  }
}

function renderTriageTable(data) {
  const tbody = $("#triage-table tbody");
  tbody.innerHTML = "";
  const yes = data.filter(r => r.decision === "YES").length;
  const no = data.filter(r => r.decision === "NO").length;
  $("#triage-count").textContent = data.length ? `${data.length} artigos: ${yes} YES, ${no} NO` : "";

  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-state">Nenhuma triagem realizada.</td></tr>';
    return;
  }
  data.forEach(row => {
    const tr = document.createElement("tr");
    const cls = row.decision === "YES" ? "decision-yes" : "decision-no";
    const id = row.id || row.article_id || "";
    const c = corpusLookup[id] || {};
    const title = row.title || c.title || "";
    tr.innerHTML = `
      <td>${escHtml(id)}</td>
      <td title="${escHtml(title)}">${escHtml(truncate(title, 70))}</td>
      <td class="${cls}">${escHtml(row.decision)}</td>
      <td title="${escHtml(row.justification)}">${escHtml(truncate(row.justification, 100))}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterTriageTable() {
  const f = $("#triage-filter").value;
  const filtered = f ? triageData.filter(r => r.decision === f) : triageData;
  renderTriageTable(filtered);
}

function renderExtractionTable(data) {
  const tbody = $("#extraction-table tbody");
  tbody.innerHTML = "";
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="empty-state">Nenhuma extracao realizada.</td></tr>';
    return;
  }
  data.forEach(row => {
    const tr = document.createElement("tr");
    // Montar dados extraidos como key/value
    const fields = Object.entries(row)
      .filter(([k]) => !["id", "title", "article_id", "tokens_used", "provider", "latency_ms", "timestamp"].includes(k))
      .map(([k, v]) => `<strong>${escHtml(k)}:</strong> ${escHtml(truncate(String(v), 120))}`)
      .join("<br>");
    const id = row.id || row.article_id || "";
    const c = corpusLookup[id] || {};
    const title = row.title || c.title || "";
    tr.innerHTML = `
      <td>${escHtml(id)}</td>
      <td title="${escHtml(title)}">${escHtml(truncate(title, 60))}</td>
      <td>${fields || "—"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderSummariesTable(data) {
  const tbody = $("#summaries-table tbody");
  tbody.innerHTML = "";
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="empty-state">Nenhum resumo gerado.</td></tr>';
    return;
  }
  data.forEach(row => {
    const tr = document.createElement("tr");
    const id = row.id || row.article_id || "";
    const c = corpusLookup[id] || {};
    const title = row.title || c.title || "";
    tr.innerHTML = `
      <td>${escHtml(id)}</td>
      <td title="${escHtml(title)}">${escHtml(truncate(title, 60))}</td>
      <td>${escHtml(row.summary || row.tldr || "")}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Aba: Revisao Manual
// ---------------------------------------------------------------------------
let goldData = [];
let hallData = [];
let valData = [];
let likertData = [];

let corpusLookup = {};  // id → {authors, year, source, abstract}

async function loadReviews() {
  try {
    const [gold, hall, val, likert, corpus] = await Promise.all([
      api("GET", "/api/data/gold"),
      api("GET", "/api/data/hallucination"),
      api("GET", "/api/data/validation"),
      api("GET", "/api/data/likert"),
      api("GET", "/api/data/corpus"),
    ]);
    // Montar lookup do corpus para enriquecer gold standard
    corpusLookup = {};
    corpus.forEach(r => { corpusLookup[r.id] = r; });

    goldData = gold;
    hallData = hall;
    valData = val;
    likertData = likert;
    renderGoldTable(gold);
    renderHallTable(hall);
    renderValTable(val);
    renderLikertTable(likert);
  } catch (e) {
    console.error("Erro ao carregar revisoes:", e);
  }
}

function makeSelect(options, current) {
  const opts = options.map(o => {
    const sel = (String(current).toUpperCase() === o.toUpperCase()) ? "selected" : "";
    return `<option value="${o}" ${sel}>${o}</option>`;
  });
  return `<select><option value="">—</option>${opts.join("")}</select>`;
}

function renderGoldTable(data) {
  const tbody = $("#gold-table tbody");
  tbody.innerHTML = "";
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">Gold standard nao gerado. Execute o step "Gold Standard" primeiro.</td></tr>';
    $("#gold-progress").textContent = "";
    return;
  }

  const filled = data.filter(r => r.consensus && String(r.consensus).trim() !== "").length;
  $("#gold-progress").innerHTML = progressBar(filled, data.length, "preenchidos");

  data.forEach((row, i) => {
    const c = corpusLookup[row.id] || {};
    const tr = document.createElement("tr");
    tr.dataset.index = i;
    tr.className = "expandable";
    tr.innerHTML = `
      <td>${escHtml(row.id)}</td>
      <td title="${escHtml(row.title)}">${escHtml(truncate(row.title, 60))}</td>
      <td title="${escHtml(c.authors)}">${escHtml(truncate(c.authors || "", 30))}</td>
      <td>${escHtml(c.year || "")}</td>
      <td>${makeSelect(["INCLUDE", "EXCLUDE"], row.reviewer_a)}</td>
      <td>${makeSelect(["INCLUDE", "EXCLUDE"], row.reviewer_b)}</td>
      <td>${makeSelect(["INCLUDE", "EXCLUDE"], row.consensus)}</td>
      <td><textarea>${escHtml(row.justification || "")}</textarea></td>
    `;
    // Clicar na linha expande o abstract (sem interferir com selects/textarea)
    tr.addEventListener("click", (e) => {
      if (e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "OPTION") return;
      const next = tr.nextElementSibling;
      if (next && next.classList.contains("detail-row")) {
        next.remove();
      } else {
        const detail = document.createElement("tr");
        detail.className = "detail-row";
        detail.innerHTML = `<td colspan="8"><strong>Abstract:</strong> ${escHtml(c.abstract || "N/A")}</td>`;
        tr.after(detail);
      }
    });
    tbody.appendChild(tr);
  });
}

function renderHallTable(data) {
  const tbody = $("#hall-table tbody");
  tbody.innerHTML = "";
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-state">Amostra de alucinacoes nao gerada.</td></tr>';
    $("#hall-progress").textContent = "";
    return;
  }

  const filled = data.filter(r => r.classification && String(r.classification).trim() !== "").length;
  $("#hall-progress").innerHTML = progressBar(filled, data.length, "classificados");

  data.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.dataset.index = i;
    tr.innerHTML = `
      <td>${escHtml(row.module)}</td>
      <td title="${escHtml(row.claim)}">${escHtml(truncate(row.claim, 80))}</td>
      <td title="${escHtml(row.source_text)}">${escHtml(truncate(row.source_text, 80))}</td>
      <td>${makeSelect(["GROUNDED", "INFERRED", "HALLUCINATED"], row.classification)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderValTable(data) {
  const tbody = $("#val-table tbody");
  tbody.innerHTML = "";
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">Amostra de validacao nao gerada.</td></tr>';
    $("#val-progress").textContent = "";
    return;
  }

  const filled = data.filter(r => r.error_type && String(r.error_type).trim() !== "").length;
  $("#val-progress").innerHTML = progressBar(filled, data.length, "validados");

  data.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.dataset.index = i;
    tr.innerHTML = `
      <td>${escHtml(row.article_id)}</td>
      <td>${escHtml(row.field)}</td>
      <td title="${escHtml(row.extracted_value)}">${escHtml(truncate(row.extracted_value, 80))}</td>
      <td>${makeSelect(["CORRECT", "HALLUCINATION", "OMISSION", "IMPRECISION"], row.error_type)}</td>
      <td><textarea>${escHtml(row.notes || "")}</textarea></td>
    `;
    tbody.appendChild(tr);
  });
}

function progressBar(filled, total, label) {
  const pct = total > 0 ? Math.round((filled / total) * 100) : 0;
  return `
    <span class="progress-bar-wrap">
      <div class="progress-bar"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
      <span>${filled} de ${total} ${label} (${pct}%)</span>
    </span>`;
}

// ---------------------------------------------------------------------------
// Salvar revisoes
// ---------------------------------------------------------------------------
async function saveReview(type) {
  try {
    let rows, endpoint;

    if (type === "gold") {
      endpoint = "/api/data/gold";
      rows = collectGoldRows();
    } else if (type === "hallucination") {
      endpoint = "/api/data/hallucination";
      rows = collectHallRows();
    } else if (type === "validation") {
      endpoint = "/api/data/validation";
      rows = collectValRows();
    } else if (type === "likert") {
      endpoint = "/api/data/likert";
      rows = collectLikertRows();
    }

    await api("POST", endpoint, rows);
    showToast("Salvo com sucesso!");

    // Recarregar para atualizar progresso
    if (type === "gold") { goldData = rows; renderGoldTable(rows); }
    if (type === "hallucination") { hallData = rows; renderHallTable(rows); }
    if (type === "validation") { valData = rows; renderValTable(rows); }
    if (type === "likert") { likertData = rows; renderLikertTable(rows); }
  } catch (e) {
    showToast("Erro ao salvar: " + e.message);
  }
}

function collectGoldRows() {
  const tbody = $("#gold-table tbody");
  return Array.from(tbody.querySelectorAll("tr[data-index]")).map((tr) => {
    const i = parseInt(tr.dataset.index);
    const selects = tr.querySelectorAll("select");
    const textarea = tr.querySelector("textarea");
    return {
      id: goldData[i].id,
      title: goldData[i].title,
      reviewer_a: selects[0]?.value || "",
      reviewer_b: selects[1]?.value || "",
      consensus: selects[2]?.value || "",
      justification: textarea?.value || "",
    };
  });
}

function collectHallRows() {
  const tbody = $("#hall-table tbody");
  return Array.from(tbody.querySelectorAll("tr:not(.empty-state tr)")).map((tr, i) => {
    const select = tr.querySelector("select");
    return {
      ...hallData[i],
      classification: select?.value || "",
    };
  });
}

function collectValRows() {
  const tbody = $("#val-table tbody");
  return Array.from(tbody.querySelectorAll("tr:not(.empty-state tr)")).map((tr, i) => {
    const select = tr.querySelector("select");
    const textarea = tr.querySelector("textarea");
    return {
      ...valData[i],
      error_type: select?.value || "",
      notes: textarea?.value || "",
    };
  });
}

// ---------------------------------------------------------------------------
// Likert: render, collect, generate
// ---------------------------------------------------------------------------
function makeLikertSelect(current) {
  const opts = ["1", "2", "3", "4", "5"].map(v => {
    const sel = String(current) === v ? "selected" : "";
    return `<option value="${v}" ${sel}>${v}</option>`;
  });
  return `<select><option value="">—</option>${opts.join("")}</select>`;
}

function renderLikertTable(data) {
  const tbody = $("#likert-table tbody");
  tbody.innerHTML = "";
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">Nenhuma amostra gerada. Clique em "Gerar Amostra" para iniciar.</td></tr>';
    $("#likert-progress").textContent = "";
    return;
  }

  const filled = data.filter(r =>
    r.clareza && r.completude && r.acuracia && r.utilidade &&
    String(r.clareza).trim() !== "" && String(r.completude).trim() !== "" &&
    String(r.acuracia).trim() !== "" && String(r.utilidade).trim() !== ""
  ).length;
  $("#likert-progress").innerHTML = progressBar(filled, data.length, "avaliados");

  data.forEach((row, i) => {
    const tr = document.createElement("tr");
    tr.dataset.index = i;
    tr.innerHTML = `
      <td>${escHtml(row.article_id)}</td>
      <td title="${escHtml(row.title)}">${escHtml(truncate(row.title, 60))}</td>
      <td title="${escHtml(row.summary)}">${escHtml(truncate(row.summary, 100))}</td>
      <td>${makeLikertSelect(row.clareza)}</td>
      <td>${makeLikertSelect(row.completude)}</td>
      <td>${makeLikertSelect(row.acuracia)}</td>
      <td>${makeLikertSelect(row.utilidade)}</td>
      <td><textarea>${escHtml(row.notas || "")}</textarea></td>
    `;
    tbody.appendChild(tr);
  });
}

function collectLikertRows() {
  const tbody = $("#likert-table tbody");
  return Array.from(tbody.querySelectorAll("tr[data-index]")).map((tr) => {
    const i = parseInt(tr.dataset.index);
    const selects = tr.querySelectorAll("select");
    const textarea = tr.querySelector("textarea");
    return {
      article_id: likertData[i].article_id,
      title: likertData[i].title,
      summary: likertData[i].summary,
      clareza: selects[0]?.value || "",
      completude: selects[1]?.value || "",
      acuracia: selects[2]?.value || "",
      utilidade: selects[3]?.value || "",
      notas: textarea?.value || "",
    };
  });
}

async function generateLikertSample() {
  if (likertData.length && !confirm("Ja existe uma amostra. Gerar uma nova ira substituir. Continuar?")) return;
  try {
    showToast("Gerando amostra Likert...");
    const res = await api("POST", "/api/generate-likert-sample");
    showToast(`Amostra gerada: ${res.count} resumos.`);
    likertData = await api("GET", "/api/data/likert");
    renderLikertTable(likertData);
  } catch (e) {
    showToast("Erro: " + e.message);
  }
}

// ---------------------------------------------------------------------------
// Aba: Metricas
// ---------------------------------------------------------------------------
async function loadMetrics() {
  try {
    const [metrics, crossval] = await Promise.all([
      api("GET", "/api/data/metrics"),
      api("GET", "/api/data/crossval"),
    ]);
    renderMetricsCards(metrics);
    renderMetricsFigures();
    renderCrossval(crossval);
  } catch (e) {
    console.error("Erro ao carregar metricas:", e);
  }
}

function renderMetricsCards(data) {
  const container = $("#metrics-cards");
  container.innerHTML = "";

  if (!data || !Object.keys(data).length) {
    container.innerHTML = '<div class="empty-state">Metricas nao calculadas. Execute o step "Metricas" primeiro.</div>';
    return;
  }

  const items = [
    { key: "recall", label: "Recall", target: 0.85, fmt: "decimal" },
    { key: "precision", label: "Precision", target: null, fmt: "decimal" },
    { key: "f1_score", label: "F1-Score", target: null, fmt: "decimal" },
    { key: "workload_reduction_pct", label: "Workload Reduction", target: null, fmt: "pct" },
    { key: "kappa_gold_standard", label: "Cohen's Kappa", target: 0.7, fmt: "decimal" },
  ];

  items.forEach(item => {
    const val = data[item.key];
    if (val === undefined || val === null) return;

    const card = document.createElement("div");
    const numVal = parseFloat(val);
    let cls = "metric-card";
    if (item.target !== null) {
      cls += numVal >= item.target ? " metric-ok" : " metric-warn";
    }
    card.className = cls;

    const display = isNaN(numVal) ? val :
      (item.fmt === "pct" ? numVal.toFixed(1) + "%" : numVal.toFixed(3));

    card.innerHTML = `
      <div class="metric-value">${display}</div>
      <div class="metric-label">${item.label}</div>
    `;
    container.appendChild(card);
  });
}

function renderMetricsFigures() {
  const container = $("#metrics-figures");
  container.innerHTML = "";

  const figures = ["confusion_matrix.png", "metrics_comparison.png"];
  figures.forEach(f => {
    const img = document.createElement("img");
    img.src = `/api/figures/${f}`;
    img.alt = f;
    img.onerror = () => img.remove();
    container.appendChild(img);
  });
}

function renderCrossval(data) {
  const container = $("#crossval-content");
  container.innerHTML = "";

  if (!data || !Object.keys(data).length) {
    container.innerHTML = '<div class="empty-state">Cross-validation nao executada.</div>';
    return;
  }

  // Kappa medio
  if (data.kappa_mean !== undefined) {
    const cls = parseFloat(data.kappa_mean) >= (data.expected_kappa || 0.8) ? "metric-ok" : "metric-warn";
    const p = document.createElement("div");
    p.className = `metric-card ${cls}`;
    p.style.display = "inline-block";
    p.innerHTML = `
      <div class="metric-value">${parseFloat(data.kappa_mean).toFixed(3)}</div>
      <div class="metric-label">Kappa Medio</div>
    `;
    container.appendChild(p);
  }

  // Tabela de pares
  const pairs = [
    { label: "Run 1 vs Run 2", key: "kappa_run1_run2" },
    { label: "Run 1 vs Run 3", key: "kappa_run1_run3" },
    { label: "Run 2 vs Run 3", key: "kappa_run2_run3" },
  ];
  const hasPairs = pairs.some(p => data[p.key] !== undefined);

  if (hasPairs) {
    const table = document.createElement("table");
    table.innerHTML = `
      <thead>
        <tr><th>Par de Runs</th><th>Kappa</th></tr>
      </thead>
      <tbody>
        ${pairs.map(p => `
          <tr>
            <td>${p.label}</td>
            <td>${data[p.key] !== undefined ? parseFloat(data[p.key]).toFixed(4) : "—"}</td>
          </tr>
        `).join("")}
        <tr>
          <td><strong>Concordancia total</strong></td>
          <td><strong>${data.agreement_pct !== undefined ? parseFloat(data.agreement_pct).toFixed(1) + "%" : "—"}</strong></td>
        </tr>
      </tbody>
    `;
    container.appendChild(table);
  }
}
