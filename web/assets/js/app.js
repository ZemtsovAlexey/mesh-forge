import { MeshViewer } from "./viewer.js";

const $ = (sel) => document.querySelector(sel);

let activeProjectId = null;
let viewer = null;
let activeProjectHasMesh = false;
let uiMode = "chat"; // chat | workspace | settings
let previousUiMode = "chat";
let lastPipeline = null;
let lastNotebook = [];
let lastChatMessages = [];
let replyRef = null; // { id, preview }
let pendingImages = [];
let pendingMesh = null;
const STAGE_LABELS = {
  concept: "ComfyUI: concept",
  front: "Front",
  views: "ComfyUI: views",
  mesh: "ComfyUI: mesh",
  guided: "Guided edit",
  finalize: "Finalize",
};

const SETTINGS_HINTS = {}; // hints live in HTML

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body instanceof FormData ? undefined : { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const msg = typeof data === "object" && data?.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : text;
    throw new Error(msg || res.statusText);
  }
  return data;
}

function toast(msg, type = "ok") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 5000);
}

let progressTimer = null;
let loadingActive = false;
let loadingLabel = "";

function setProgressUI(percent, stage, elapsedSec = null) {
  const bar = $("#progress-bar");
  const meta = $("#progress-meta");
  const pct = Math.max(0, Math.min(100, Number(percent) || 0));
  const stageLabel = STAGE_LABELS[stage] || stage || loadingLabel;
  if (bar) bar.style.width = `${pct}%`;
  const time = elapsedSec != null ? ` · ${Math.round(elapsedSec)}с` : "";
  if (meta) meta.textContent = `${Math.round(pct)}%${time}${stageLabel ? ` — ${stageLabel}` : ""}`;
  if (stageLabel) {
    const loadingText = $("#loading-text");
    if (loadingText) loadingText.textContent = stageLabel;
  }
  updateChatStatusProgress(pct, stageLabel, elapsedSec);
}

function chatStatusHtml(text, percent = 2, elapsedSec = null) {
  const pct = Math.max(0, Math.min(100, Number(percent) || 0));
  const time = elapsedSec != null ? ` · ${Math.round(elapsedSec)}с` : "";
  return `
    <div class="chat-avatar" aria-hidden="true">MF</div>
    <div class="chat-msg-body">
      <div class="chat-msg-meta">MeshForge · работает</div>
      <div class="chat-status-card">
        <div class="chat-status-row">
          <span class="chat-status-spinner" aria-hidden="true"></span>
          <span class="chat-status-text">${escapeHtml(text || "Обработка…")}</span>
          <button type="button" class="btn danger chat-stop-inline" data-chat-stop title="Остановить">Стоп</button>
        </div>
        <div class="chat-status-bar"><span style="width:${pct}%"></span></div>
        <div class="chat-status-meta">${Math.round(pct)}%${time}</div>
      </div>
    </div>
  `;
}

function ensureChatStatusEl() {
  const log = $("#chat-log");
  if (!log) return null;
  let el = $("#chat-status-msg");
  if (!el) {
    el = document.createElement("div");
    el.id = "chat-status-msg";
    el.className = "chat-msg assistant status";
    log.appendChild(el);
  }
  return el;
}

function updateChatStatusProgress(percent, stageLabel, elapsedSec = null) {
  if (!loadingActive) return;
  const el = $("#chat-status-msg");
  if (!el) return;
  const label = stageLabel || loadingLabel || "Обработка…";
  el.innerHTML = chatStatusHtml(label, percent, elapsedSec);
  const log = $("#chat-log");
  if (log) log.scrollTop = log.scrollHeight;
}

function setChatStatus(on, text = "Обработка…") {
  if (!on) {
    $("#chat-status-msg")?.remove();
    return;
  }
  const el = ensureChatStatusEl();
  if (!el) return;
  el.innerHTML = chatStatusHtml(text, 2, 0);
  const log = $("#chat-log");
  if (log) log.scrollTop = log.scrollHeight;
}

function setLoading(on, text = "Обработка…", opts = {}) {
  loadingActive = !!on;
  loadingLabel = on ? text : "";
  const useOverlay = !!opts.overlay || uiMode === "workspace" || uiMode === "settings";
  const sendBtn = $("#btn-chat-send");
  const stopBtn = $("#btn-chat-stop");
  if (sendBtn) sendBtn.disabled = !!on;
  if (stopBtn) stopBtn.classList.toggle("hidden", !on);

  if (useOverlay) {
    setChatStatus(false);
    $("#loading")?.classList.toggle("hidden", !on);
    if ($("#loading-text")) $("#loading-text").textContent = text;
    if (on) setProgressUI(2, text, 0);
    else {
      setProgressUI(0, "");
      if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
      }
    }
    return;
  }

  // Chat-inline status — never block the whole window in agent mode.
  $("#loading")?.classList.add("hidden");
  if (on) {
    setChatStatus(true, text);
    setProgressUI(2, text, 0);
  } else {
    setChatStatus(false);
    setProgressUI(0, "");
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
  }
}

async function stopActiveJob() {
  if (!activeProjectId) return;
  const stopBtn = $("#btn-chat-stop");
  if (stopBtn) stopBtn.disabled = true;
  toast("Останавливаю…");
  try {
    await api(`/api/projects/${activeProjectId}/cancel`, { method: "POST" });
  } catch (e) {
    toast(e.message?.slice(0, 160) || String(e), "error");
  } finally {
    if (stopBtn) stopBtn.disabled = false;
  }
}

function startProgressPolling(projectId) {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = setInterval(async () => {
    if (!projectId || !loadingActive) return;
    try {
      const p = await api(`/api/projects/${projectId}/progress`);
      if (!p) return;
      setProgressUI(p.percent, p.stage || loadingLabel || "Обработка…", p.elapsed_sec);
    } catch {
      /* ignore poll errors while request runs */
    }
  }, 700);
}

async function runOp(label, fn, opts = {}) {
  if (!activeProjectId) { toast("Выберите проект", "error"); return null; }
  setLoading(true, label, opts);
  startProgressPolling(activeProjectId);
  try {
    const result = await fn();
    setProgressUI(100, "Готово", null);
    afterOperation(result);
    return result;
  } catch (e) {
    toast(e.message?.slice(0, 200) || String(e), "error");
    return null;
  } finally {
    setLoading(false);
  }
}

function bindRanges() {
  [["job-smooth", "job-smooth-val"], ["job-solid", "job-solid-val"]].forEach(([id, out]) => {
    const input = $(`#${id}`);
    const output = $(`#${out}`);
    input?.addEventListener("input", () => { output.textContent = input.value; });
  });
}

function bindFileDrop(inputId, dropSelector) {
  const input = $(inputId);
  const drop = dropSelector ? $(dropSelector) : input?.closest(".file-drop, .composer-icon-btn");
  if (!input || !drop) return;
  input.addEventListener("change", () => {
    if (input.id === "chat-images") {
      const added = Array.from(input.files || []);
      if (added.length) pendingImages = [...pendingImages, ...added];
      input.value = "";
      $("#chat-images-drop")?.classList.toggle("has-file", pendingImages.length > 0);
      renderAttachChips();
      return;
    }
    if (input.id === "chat-mesh") {
      pendingMesh = input.files?.[0] || null;
      input.value = "";
      $("#chat-mesh-drop")?.classList.toggle("has-file", !!pendingMesh);
      renderAttachChips();
      return;
    }
    drop.classList.toggle("has-file", !!input.files?.length);
    const span = drop.querySelector("span");
    if (!span || drop.classList.contains("composer-icon-btn")) return;
    if (!input.files?.length) {
      span.textContent = span.dataset.fallback || "Файл";
      return;
    }
    span.textContent = input.multiple
      ? `${input.files.length} file(s): ${Array.from(input.files).slice(0, 2).map((f) => f.name).join(", ")}`
      : input.files[0].name;
  });
}

function syncPendingImagesToInput() {
  $("#chat-images-drop")?.classList.toggle("has-file", pendingImages.length > 0);
  renderAttachChips();
}

function syncPendingMeshToInput() {
  $("#chat-mesh-drop")?.classList.toggle("has-file", !!pendingMesh);
  renderAttachChips();
}

function renderAttachChips() {
  const box = $("#attach-chips");
  if (!box) return;
  const chips = [];
  pendingImages.forEach((file, index) => {
    chips.push(`
      <span class="attach-chip" data-kind="image" data-index="${index}">
        <span class="attach-chip-kind">фото</span>
        <span class="attach-chip-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
        <button type="button" class="attach-chip-remove" data-remove="image" data-index="${index}" aria-label="Открепить">×</button>
      </span>
    `);
  });
  if (pendingMesh) {
    chips.push(`
      <span class="attach-chip" data-kind="mesh">
        <span class="attach-chip-kind">mesh</span>
        <span class="attach-chip-name" title="${escapeHtml(pendingMesh.name)}">${escapeHtml(pendingMesh.name)}</span>
        <button type="button" class="attach-chip-remove" data-remove="mesh" aria-label="Открепить">×</button>
      </span>
    `);
  }
  box.innerHTML = chips.join("");
  box.classList.toggle("hidden", chips.length === 0);
}

function removeAttachment(kind, index = -1) {
  if (kind === "image") {
    if (index < 0 || index >= pendingImages.length) return;
    pendingImages.splice(index, 1);
    syncPendingImagesToInput();
    return;
  }
  if (kind === "mesh") {
    pendingMesh = null;
    syncPendingMeshToInput();
  }
}

function clearChatImages() {
  pendingImages = [];
  syncPendingImagesToInput();
}

function clearChatMesh() {
  pendingMesh = null;
  syncPendingMeshToInput();
}

async function loadStatus() {
  try {
    const data = await api("/api/status");
    const pills = $("#status-pills");
    pills.innerHTML = "";
    for (const [name, ok] of Object.entries(data.services)) {
      const span = document.createElement("span");
      span.className = `pill ${ok ? "ok" : "bad"}`;
      span.textContent = name;
      pills.appendChild(span);
    }
    const gpu = data.gpu || {};
    const gpuPill = document.createElement("span");
    if (gpu.active) {
      const waiting = Array.isArray(gpu.waiting) ? gpu.waiting.length : 0;
      gpuPill.className = "pill ok";
      gpuPill.textContent = waiting
        ? `gpu: ${gpu.active.label} +${waiting}`
        : `gpu: ${gpu.active.label}`;
    } else {
      gpuPill.className = "pill";
      gpuPill.textContent = "gpu: idle";
    }
    pills.appendChild(gpuPill);
  } catch (e) {
    console.warn("status", e);
  }
}

async function loadProjects(selectId = activeProjectId) {
  const projects = await api("/api/projects");
  const list = $("#project-list");
  list.innerHTML = "";
  if (!projects.length) {
    list.innerHTML = "<li><p class='muted' style='padding:8px'>Нет проектов</p></li>";
    activeProjectId = null;
    return;
  }
  for (const p of projects) {
    const li = document.createElement("li");
    li.className = "project-row";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `project-select ${p.id === selectId ? "active" : ""}`;
    btn.innerHTML = `${escapeHtml(p.name)}<span class="sub">v${p.current_version}${p.has_mesh ? " · mesh" : ""}</span>`;
    btn.addEventListener("click", () => selectProject(p.id));

    const actions = document.createElement("div");
    actions.className = "project-actions";
    actions.innerHTML = `
      <button type="button" class="icon-btn" title="Переименовать" data-act="rename">✎</button>
      <button type="button" class="icon-btn" title="Копировать" data-act="dup">⧉</button>
      <button type="button" class="icon-btn danger" title="Удалить" data-act="del">✕</button>
    `;
    actions.querySelector('[data-act="rename"]').addEventListener("click", (e) => {
      e.stopPropagation();
      renameProject(p.id, p.name);
    });
    actions.querySelector('[data-act="dup"]').addEventListener("click", (e) => {
      e.stopPropagation();
      duplicateProject(p.id, p.name);
    });
    actions.querySelector('[data-act="del"]').addEventListener("click", (e) => {
      e.stopPropagation();
      deleteProject(p.id, p.name);
    });

    li.appendChild(btn);
    li.appendChild(actions);
    list.appendChild(li);
  }
  if (selectId && projects.some((p) => p.id === selectId)) await selectProject(selectId, false);
  else if (projects.length) await selectProject(projects[0].id, false);
}

async function renameProject(id, currentName) {
  const name = prompt("Новое имя проекта", currentName || "");
  if (name == null) return;
  const cleaned = name.trim();
  if (!cleaned) return toast("Имя пустое", "error");
  try {
    await api(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify({ name: cleaned }) });
    toast("Переименовано");
    await loadProjects(id);
  } catch (e) {
    toast(e.message, "error");
  }
}

async function duplicateProject(id, currentName) {
  const name = prompt("Имя копии", `${currentName || "проект"} (копия)`);
  if (name == null) return;
  try {
    const p = await api(`/api/projects/${id}/duplicate`, {
      method: "POST",
      body: JSON.stringify({ name: name.trim() || null }),
    });
    toast(`Копия: ${p.name}`);
    await loadProjects(p.id);
  } catch (e) {
    toast(e.message, "error");
  }
}

async function deleteProject(id, name) {
  try {
    const res = await fetch(`/api/projects/${id}`, { method: "DELETE" });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || res.statusText);
    }
    toast("Удалено");
    if (activeProjectId === id) {
      activeProjectId = null;
      viewer?.clear();
      $("#viewer-empty")?.classList.remove("hidden");
      $("#btn-download")?.classList.add("hidden");
      $("#btn-regenerate")?.classList.add("hidden");
      $("#project-meta").innerHTML = `<p class="muted">Проект не выбран</p>`;
      $("#history-list").innerHTML = "";
      $("#chat-log").innerHTML = "";
      $("#qc-log").textContent = "—";
    }
    await loadProjects(activeProjectId);
  } catch (e) {
    toast(e.message, "error");
  }
}

async function selectProject(id, refreshList = true) {
  activeProjectId = id;
  if (refreshList) await loadProjects(id);
  const project = await api(`/api/projects/${id}`);
  renderProjectMeta(project);
  renderHistory(project);
  await loadChat();
  if (project.mesh_url) {
    await loadMesh(project.mesh_url);
    $("#viewer-empty").classList.add("hidden");
    $("#btn-download").classList.remove("hidden");
    $("#btn-download").href = project.mesh_url;
  } else {
    viewer?.clear();
    $("#viewer-empty").classList.remove("hidden");
    $("#btn-download").classList.add("hidden");
    $("#qc-log").textContent = "—";
  }
  const regen = $("#btn-regenerate");
  if (regen) {
    const canRegen = (project.versions || []).some((v) => (v.instruction || "").trim());
    regen.classList.toggle("hidden", !canRegen);
  }
}

function renderProjectMeta(p) {
  activeProjectHasMesh = !!p.has_mesh;
  const useCurrent = $("#job-use-current");
  if (useCurrent) {
    useCurrent.disabled = !activeProjectHasMesh;
    if (!activeProjectHasMesh) useCurrent.checked = false;
  }
  const nameEl = $("#chat-project-name");
  if (nameEl) nameEl.textContent = p.name || "Проект";
  const hint = $("#chat-hint");
  if (hint) {
    hint.textContent = activeProjectHasMesh
      ? "Есть модель: опишите правку текстом или приложите новый mesh."
      : "Текст / фото → генерация. Можно сразу приложить mesh (.stl/.obj).";
  }
  $("#project-meta").innerHTML = `
    <p><strong>${p.name}</strong></p>
    <p class="muted">ID: ${p.id.slice(0, 8)}…</p>
    <p class="muted">Версия: v${p.current_version}</p>
    <p class="muted">${p.has_mesh ? "Модель загружена" : "Модель отсутствует"}</p>
  `;
}

function setUiMode(mode) {
  if (mode !== "settings" && uiMode !== "settings") previousUiMode = mode;
  if (mode === "settings") previousUiMode = uiMode === "settings" ? previousUiMode : uiMode;
  uiMode = mode;
  document.body.classList.remove("mode-chat", "mode-workspace", "mode-settings");
  document.body.classList.add(`mode-${mode}`);
  const settings = $("#settings-screen");
  if (settings) settings.hidden = mode !== "settings";
  if (mode === "workspace" && viewer) {
    requestAnimationFrame(() => viewer?.resize?.());
  }
}

function openLightbox(src, alt = "") {
  const box = $("#lightbox");
  const img = $("#lightbox-img");
  if (!box || !img || !src) return;
  img.src = src;
  img.alt = alt;
  box.classList.remove("hidden");
}

function closeLightbox() {
  $("#lightbox")?.classList.add("hidden");
  const img = $("#lightbox-img");
  if (img) img.removeAttribute("src");
}

function stepTitle(step, pipeline) {
  if (pipeline === "photo_gated") {
    if (step === "photo") return "Превью фото";
    if (step === "done") return "Mesh готов";
    return "Фото → mesh";
  }
  if (step === "front") return "Шаг 1 · Front";
  if (step === "views") return "Шаг 2 · Проекции";
  if (step === "done") return "Шаг 3 · Mesh готов";
  if (step === "mesh") return "Шаг 3 · Mesh";
  return "Пайплайн";
}

function renderPipelineCard(pipeline) {
  // Pipeline actions removed — agent runs steps; results live in chat messages.
  return "";
}

function renderDraftCard(state) {
  // Confirm buttons removed — agent starts generation via tools.
  return "";
}

let chatMeshViewers = [];

function disposeChatMeshViewers() {
  for (const v of chatMeshViewers) {
    try { v.dispose?.(); } catch { /* ignore */ }
  }
  chatMeshViewers = [];
}

function mountChatMeshViewers() {
  disposeChatMeshViewers();
  document.querySelectorAll("[data-chat-mesh-url]").forEach((el) => {
    const url = el.getAttribute("data-chat-mesh-url");
    if (!url) return;
    try {
      const mini = new MeshViewer(el);
      chatMeshViewers.push(mini);
      mini.loadUrl(url).catch((err) => {
        console.warn("chat mesh preview", err);
        el.innerHTML = `<div class="muted">Не удалось загрузить 3D-превью</div>`;
      });
      // Fit after layout
      requestAnimationFrame(() => mini.resize());
    } catch (err) {
      console.warn("chat mesh viewer", err);
    }
  });
}

function renderMessageArtifacts(artifacts) {
  const list = artifacts || [];
  if (!list.length) return "";
  const views = list.filter((a) => a.kind === "image");
  const previews = list.filter((a) => a.kind === "mesh_preview");
  const meshes = list.filter((a) => a.kind === "mesh");

  const viewsGrid = views.length
    ? `<div class="chat-art-section">
        <div class="chat-art-section-title">Виды</div>
        <div class="chat-art-grid">${views.map((img) => {
          const src = img.url || "";
          const label = escapeHtml(img.label || "");
          return `<figure class="chat-art-item">
            <img src="${src}?t=${Date.now()}" alt="${label}" data-lightbox="${src}" title="${label}" />
            <figcaption>${label}</figcaption>
          </figure>`;
        }).join("")}</div>
      </div>`
    : "";

  const previewBlock = (!meshes.length ? previews : []).map((img) => {
    const src = img.url || "";
    return `<div class="chat-mesh-preview-block">
      <div class="chat-art-section-title">Превью mesh</div>
      <figure class="chat-mesh-preview-still">
        <img src="${src}?t=${Date.now()}" alt="mesh preview" data-lightbox="${src}" />
      </figure>
    </div>`;
  }).join("");

  const meshBlock = meshes.map((m) => {
    const href = m.url || "#";
    return `<div class="chat-mesh-preview-block">
      <div class="chat-art-section-title">Превью mesh</div>
      <div class="chat-mesh-viewer" data-chat-mesh-url="${href}"></div>
      <div class="chat-art-actions">
        <a class="btn ghost chat-mesh-dl" href="${href}" download>Скачать STL</a>
      </div>
    </div>`;
  }).join("");

  // If we have mesh but no PNG preview yet, still show 3D + download.
  return `<div class="chat-artifacts">${viewsGrid}${previewBlock}${meshBlock}</div>`;
}

function renderChat(state) {
  const log = $("#chat-log");
  if (!log) return;
  lastPipeline = state?.pipeline || null;
  lastNotebook = state?.notebook || lastNotebook || [];
  lastChatMessages = state?.messages || [];
  const messages = state?.messages || [];
  const byId = Object.fromEntries(messages.filter((m) => m.id).map((m) => [m.id, m]));
  let html = "";
  if (!messages.length) {
    html = `<div class="chat-empty">${activeProjectHasMesh
      ? "Опишите правку — агент сам запустит шаги. «Дальше» / «переделай» — текстом."
      : "Опишите объект. Агент уточнит при необходимости и сам сгенерирует front → views → mesh."}</div>`;
  } else {
    html = messages.map((m) => {
      const isUser = m.role === "user";
      const who = isUser ? "Вы" : "MeshForge";
      const avatar = isUser ? "Вы" : "MF";
      const mid = m.id || "";
      let refHtml = "";
      if (m.ref_ids?.length) {
        const bits = m.ref_ids.map((rid) => {
          const src = byId[rid];
          const preview = src ? String(src.content || "").slice(0, 80) : rid;
          return `<div class="chat-ref-quote" data-ref-id="${escapeHtml(rid)}">↩ ${escapeHtml(preview)}</div>`;
        }).join("");
        refHtml = bits;
      }
      const kindClass = m.kind && m.kind !== "text" ? ` kind-${escapeHtml(m.kind)}` : "";
      const actions = mid
        ? `<div class="chat-msg-actions">
            <button type="button" class="chat-msg-action" data-ref-msg="${escapeHtml(mid)}" title="Сослаться">↩</button>
            <button type="button" class="chat-msg-action" data-apply-msg="${escapeHtml(mid)}" data-apply-mode="main" title="Сделать основным">★</button>
            <button type="button" class="chat-msg-action" data-apply-msg="${escapeHtml(mid)}" data-apply-mode="revise" title="Переделать на основе">↻</button>
            <button type="button" class="chat-msg-action" data-restart-msg="${escapeHtml(mid)}" title="Перезапустить чат с этого сообщения">⟲</button>
          </div>`
        : "";
      return `
      <div class="chat-msg ${isUser ? "user" : "assistant"}${kindClass}" data-msg-id="${escapeHtml(mid)}">
        <div class="chat-avatar" aria-hidden="true">${avatar}</div>
        <div class="chat-msg-body">
          <div class="chat-msg-meta">${who}${mid ? ` · <span class="chat-msg-id">${escapeHtml(mid.slice(0, 6))}</span>` : ""}${m.kind && m.kind !== "text" ? ` · ${escapeHtml(m.kind)}` : ""}</div>
          ${refHtml}
          <div class="chat-bubble-text">${escapeHtml(m.content)}</div>
          ${renderMessageArtifacts(m.artifacts)}
          ${actions}
        </div>
      </div>`;
    }).join("");
  }
  log.innerHTML = html;
  if (loadingActive) {
    const status = document.createElement("div");
    status.id = "chat-status-msg";
    status.className = "chat-msg assistant status";
    status.innerHTML = chatStatusHtml(loadingLabel || "Обработка…", 2, 0);
    log.appendChild(status);
  }
  mountChatMeshViewers();
  log.scrollTop = log.scrollHeight;
  renderNotebookList(lastNotebook);
  syncReplyChip();
}

function syncReplyChip() {
  const chip = $("#reply-chip");
  const label = $("#reply-chip-label");
  if (!chip || !label) return;
  if (!replyRef?.id) {
    chip.classList.add("hidden");
    return;
  }
  chip.classList.remove("hidden");
  label.textContent = `↩ ${replyRef.preview || replyRef.id}`;
}

function setReplyRef(id, preview) {
  replyRef = id ? { id, preview: String(preview || id).slice(0, 100) } : null;
  syncReplyChip();
}

function renderNotebookList(entries) {
  const list = $("#notebook-list");
  if (!list) return;
  const items = entries || [];
  if (!items.length) {
    list.innerHTML = `<p class="muted">Пока пусто — записи появятся после шагов пайплайна и действий агента.</p>`;
    return;
  }
  list.innerHTML = [...items].reverse().map((e) => {
    const brief = e.brief_en || e.summary || e.user_prompt || "";
    return `
      <article class="notebook-entry" data-nb-id="${escapeHtml(e.id)}">
        <div class="notebook-entry-title">${escapeHtml(e.title || e.kind)}</div>
        <div class="notebook-entry-meta">${escapeHtml(e.kind)}${e.step ? ` · ${escapeHtml(e.step)}` : ""}${e.version != null ? ` · v${e.version}` : ""}</div>
        ${brief ? `<div class="notebook-entry-body">${escapeHtml(String(brief).slice(0, 180))}</div>` : ""}
        <div class="notebook-entry-actions">
          <button type="button" class="btn ghost" data-ref-nb="${escapeHtml(e.id)}">↩ ссылка</button>
          <button type="button" class="btn ghost" data-apply-nb="${escapeHtml(e.id)}" data-apply-mode="main">★ основное</button>
          <button type="button" class="btn ghost" data-apply-nb="${escapeHtml(e.id)}" data-apply-mode="revise">↻ переделать</button>
        </div>
      </article>`;
  }).join("");
}

function toggleNotebook(open) {
  const drawer = $("#notebook-drawer");
  if (!drawer) return;
  const show = open == null ? drawer.classList.contains("hidden") : open;
  drawer.classList.toggle("hidden", !show);
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadChat() {
  if (!activeProjectId) return;
  try {
    const state = await api(`/api/projects/${activeProjectId}/chat`);
    renderChat(state);
  } catch (e) {
    console.warn("chat", e);
    renderChat({ messages: [] });
  }
}

async function sendChatMessage(extraText = "", extraRefs = null) {
  if (!activeProjectId) return toast("Выберите проект", "error");
  if (loadingActive) return;
  const text = (extraText || ($("#chat-input").value || "").trim());
  const images = pendingImages.slice();
  const meshFile = pendingMesh;
  const refs = extraRefs || (replyRef?.id ? [replyRef.id] : []);
  if (!text && !images.length && !meshFile && !refs.length) {
    return toast("Введите сообщение, приложите фото или mesh", "error");
  }

  // Mesh attach: import into project (cleanup/light), then optional chat text/photos.
  if (meshFile) {
    const fd = new FormData();
    fd.append("prompt", "");
    fd.append("mesh", meshFile);
    fd.append("use_current_mesh", false);
    fd.append("backend", "comfyui");
    fd.append("remove_bg", $("#job-rmbg")?.checked ?? true);
    fd.append("solidify_mm", $("#job-solid")?.value || "0");
    fd.append("mode", $("#job-mode")?.value || "light");
    fd.append("smooth_iters", $("#job-smooth")?.value || "1");
    const imported = await runOp("Импорт mesh…", () =>
      api(`/api/projects/${activeProjectId}/jobs`, { method: "POST", body: fd })
    );
    clearChatMesh();
    if (!imported || (!text && !images.length && !refs.length)) return;
  }

  const fd = new FormData();
  fd.append("text", text);
  if (refs.length) fd.append("ref_ids", refs.join(","));
  images.forEach((file) => fd.append("images", file));

  // Optimistic UI: clear composer and show the user bubble immediately.
  if (!extraText) $("#chat-input").value = "";
  clearChatImages();
  setReplyRef(null);
  const optimistic = {
    messages: [
      ...lastChatMessages,
      {
        id: `tmp-${Date.now()}`,
        role: "user",
        content: text || (images.length ? "(фото)" : "(сообщение)"),
        ref_ids: refs,
        kind: "text",
        artifacts: [],
      },
    ],
    notebook: lastNotebook,
    pipeline: lastPipeline,
  };
  setLoading(true, "Агент…");
  startProgressPolling(activeProjectId);
  renderChat(optimistic);

  try {
    const state = await api(`/api/projects/${activeProjectId}/chat/messages`, { method: "POST", body: fd });
    renderChat(state);
  } catch (e) {
    if (!extraText && text) $("#chat-input").value = text;
    toast(e.message?.slice(0, 200) || String(e), "error");
    await loadChat();
  } finally {
    setLoading(false);
  }
}

async function restartChatFromMessage(messageId) {
  if (!activeProjectId) return toast("Выберите проект", "error");
  if (!messageId || String(messageId).startsWith("tmp-")) {
    return toast("Дождитесь ответа агента", "error");
  }
  if (loadingActive) return;
  const msg = lastChatMessages.find((m) => m.id === messageId);
  const preview = (msg?.content || messageId).slice(0, 80);
  if (!confirm(`Перезапустить чат с этого сообщения?\n\n«${preview}»\n\nВсё после него будет удалено из истории, пайплайн сбросится.`)) {
    return;
  }

  // Optimistic truncate in UI
  const idx = lastChatMessages.findIndex((m) => m.id === messageId);
  let cut = idx;
  if (cut >= 0) {
    while (cut >= 0 && lastChatMessages[cut].role !== "user") cut -= 1;
  }
  const kept = cut >= 0 ? lastChatMessages.slice(0, cut) : lastChatMessages;
  const target = cut >= 0 ? lastChatMessages[cut] : null;
  setLoading(true, "Перезапуск…");
  startProgressPolling(activeProjectId);
  renderChat({
    messages: [
      ...kept,
      ...(target
        ? [{ ...target, id: `tmp-restart-${Date.now()}` }]
        : []),
    ],
    notebook: lastNotebook,
    pipeline: null,
  });

  try {
    const fd = new FormData();
    fd.append("message_id", messageId);
    const state = await api(`/api/projects/${activeProjectId}/chat/restart`, { method: "POST", body: fd });
    renderChat(state);
  } catch (e) {
    toast(e.message?.slice(0, 200) || String(e), "error");
    await loadChat();
  } finally {
    setLoading(false);
  }
}

async function applyRefAsPrompt(refId, mode, kind = "message") {
  const verb = mode === "revise" ? "переделай на основе" : "сделай основным";
  const prefix = kind === "notebook" ? "записи блокнота" : "сообщения";
  const text = mode === "revise"
    ? `Переделай на основании этой ${prefix} (id ${refId}).`
    : `Примени эту ${prefix} (id ${refId}) как основной промпт.`;
  await sendChatMessage(text, [refId]);
}

async function confirmChat() {
  if (!activeProjectId) return toast("Выберите проект", "error");
  const fd = new FormData();
  fd.append("solidify_mm", $("#job-solid")?.value || "0");
  fd.append("mode", $("#job-mode")?.value || "light");
  fd.append("smooth_iters", $("#job-smooth")?.value || "1");
  fd.append("remove_bg", $("#job-rmbg")?.checked ?? true);
  runOp("ComfyUI…", async () => {
    const result = await api(`/api/projects/${activeProjectId}/chat/confirm`, { method: "POST", body: fd });
    await loadChat();
    return result;
  });
}

async function regenerateProject() {
  if (!activeProjectId) return toast("Выберите проект", "error");
  const fd = new FormData();
  fd.append("solidify_mm", $("#job-solid")?.value || "0");
  runOp("Перезапуск генерации…", async () => {
    const result = await api(`/api/projects/${activeProjectId}/regenerate`, { method: "POST", body: fd });
    await loadChat();
    return result;
  });
}

function renderHistory(p) {
  const ul = $("#history-list");
  if (!p.versions?.length) {
    ul.innerHTML = "<li class='muted'>Пусто</li>";
    return;
  }
  ul.innerHTML = p.versions.map((v) => {
    const images = (v.artifacts || []).filter((a) => a.kind === "image");
    const thumbs = images.length
      ? `<div class="hist-views">${images.map((a) => {
          const name = (a.path || "").split(/[/\\]/).pop();
          if (!name) return "";
          const src = `/api/projects/${p.id}/artifacts/${v.version}/${encodeURIComponent(name)}?t=${Date.now()}`;
          return `<img class="hist-view" src="${src}" data-lightbox="${src}" alt="${a.label || name}" title="${a.label || name}" />`;
        }).join("")}</div>`
      : "";
    return `
    <li><strong>v${v.version}</strong> [${v.branch}] ${v.action}
    ${v.instruction ? `<br><span>${escapeHtml(v.instruction.slice(0, 80))}</span>` : ""}
    ${thumbs}
    ${v.artifacts?.length ? `<br><span class="muted">${v.artifacts.map((a) => a.label).join(", ")}</span>` : ""}</li>
  `;
  }).join("");
}

async function loadMesh(url) {
  const bust = `${url}?t=${Date.now()}`;
  await viewer.loadUrl(bust);
  try {
    const exp = await api(`/api/projects/${activeProjectId}/export`);
    $("#qc-log").textContent = exp.report;
  } catch { /* ignore */ }
}

function afterOperation(result) {
  if (!result) return;
  if (result.project) {
    renderProjectMeta(result.project);
    renderHistory(result.project);
  }
  if (result.qc_report) $("#qc-log").textContent = result.qc_report;
  toast(result.message || "OK");
  if (result.pipeline) {
    lastPipeline = result.pipeline;
    // Refresh chat to show step card + messages
  }
  if (result.project?.mesh_url && result.pipeline?.step === "done") {
    loadMesh(result.project.mesh_url);
    $("#viewer-empty")?.classList.add("hidden");
    $("#btn-download")?.classList.remove("hidden");
    $("#btn-download").href = result.project.mesh_url;
  } else if (result.project?.mesh_url && !result.pipeline) {
    loadMesh(result.project.mesh_url);
    $("#viewer-empty")?.classList.add("hidden");
    $("#btn-download")?.classList.remove("hidden");
    $("#btn-download").href = result.project.mesh_url;
  }
  loadProjects(activeProjectId);
  loadChat();
}

async function continuePipeline() {
  if (!activeProjectId) return toast("Выберите проект", "error");
  runOp("Далее…", () => api(`/api/projects/${activeProjectId}/pipeline/continue`, { method: "POST" }));
}

async function redoPipeline(step = "front") {
  if (!activeProjectId) return toast("Выберите проект", "error");
  runOp("Переделать…", () =>
    api(`/api/projects/${activeProjectId}/pipeline/redo`, {
      method: "POST",
      body: JSON.stringify({ step }),
    })
  );
}

async function initLLM() {
  try {
    const s = await api("/api/settings/llm");
    $("#llm-url").value = s.base_url;
    $("#llm-key").value = s.api_key;
    await refreshLLMModels();
  } catch (e) {
    $("#llm-status").textContent = String(e);
  }
}

async function refreshLLMModels() {
  const base_url = $("#llm-url").value;
  const api_key = $("#llm-key").value;
  const q = new URLSearchParams({ base_url, api_key });
  const data = await api(`/api/settings/llm/models?${q}`);
  const fill = (sel, val) => {
    const el = $(sel);
    el.innerHTML = data.models.map((m) => `<option value="${m}" ${m === val ? "selected" : ""}>${m}</option>`).join("");
  };
  fill("#llm-planner", data.planner_model);
  fill("#llm-vision", data.vision_model);
  $("#llm-status").textContent = data.status;
}

function bindActions() {
  $("#btn-refresh-projects").addEventListener("click", () => loadProjects(activeProjectId));
  $("#btn-create-project").addEventListener("click", async () => {
    const name = $("#new-project-name").value.trim();
    if (!name) return toast("Введите имя", "error");
    const p = await api("/api/projects", { method: "POST", body: JSON.stringify({ name }) });
    $("#new-project-name").value = "";
    await loadProjects(p.id);
    toast(`Создан: ${p.name}`);
  });

  $("#btn-chat-send")?.addEventListener("click", () => sendChatMessage());
  $("#btn-chat-stop")?.addEventListener("click", () => stopActiveJob());
  $("#btn-regenerate")?.addEventListener("click", () => regenerateProject());
  $("#btn-open-workspace")?.addEventListener("click", () => {
    if (!activeProjectId) return toast("Выберите проект", "error");
    setUiMode("workspace");
  });
  $("#btn-back-chat")?.addEventListener("click", () => setUiMode("chat"));
  $("#btn-open-settings")?.addEventListener("click", () => setUiMode("settings"));
  $("#btn-close-settings")?.addEventListener("click", () => setUiMode(previousUiMode || "chat"));
  $("#lightbox-close")?.addEventListener("click", () => closeLightbox());
  $("#lightbox")?.addEventListener("click", (e) => {
    if (e.target === $("#lightbox")) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeLightbox();
      toggleNotebook(false);
    }
  });
  document.addEventListener("click", (e) => {
    const removeBtn = e.target?.closest?.("[data-remove]");
    if (removeBtn) {
      e.preventDefault();
      e.stopPropagation();
      const kind = removeBtn.getAttribute("data-remove");
      const index = Number(removeBtn.getAttribute("data-index") ?? -1);
      removeAttachment(kind, index);
      return;
    }
    const confirmBtn = e.target?.closest?.('[data-chat-action="confirm"]');
    if (confirmBtn) {
      e.preventDefault();
      confirmChat();
      return;
    }
    const stopBtn = e.target?.closest?.("[data-chat-stop]");
    if (stopBtn) {
      e.preventDefault();
      stopActiveJob();
      return;
    }
    const refMsg = e.target?.closest?.("[data-ref-msg]");
    if (refMsg) {
      e.preventDefault();
      const id = refMsg.getAttribute("data-ref-msg");
      const msg = lastChatMessages.find((m) => m.id === id);
      setReplyRef(id, msg?.content || id);
      $("#chat-input")?.focus();
      return;
    }
    const applyMsg = e.target?.closest?.("[data-apply-msg]");
    if (applyMsg) {
      e.preventDefault();
      applyRefAsPrompt(
        applyMsg.getAttribute("data-apply-msg"),
        applyMsg.getAttribute("data-apply-mode") || "main",
        "message",
      );
      return;
    }
    const restartMsg = e.target?.closest?.("[data-restart-msg]");
    if (restartMsg) {
      e.preventDefault();
      restartChatFromMessage(restartMsg.getAttribute("data-restart-msg"));
      return;
    }
    const refNb = e.target?.closest?.("[data-ref-nb]");
    if (refNb) {
      e.preventDefault();
      const id = refNb.getAttribute("data-ref-nb");
      const entry = lastNotebook.find((n) => n.id === id);
      setReplyRef(id, entry?.title || entry?.brief_en || id);
      $("#chat-input")?.focus();
      return;
    }
    const applyNb = e.target?.closest?.("[data-apply-nb]");
    if (applyNb) {
      e.preventDefault();
      applyRefAsPrompt(
        applyNb.getAttribute("data-apply-nb"),
        applyNb.getAttribute("data-apply-mode") || "main",
        "notebook",
      );
      return;
    }
    const lb = e.target?.closest?.("[data-lightbox]");
    if (lb) {
      e.preventDefault();
      openLightbox(lb.getAttribute("data-lightbox"), lb.getAttribute("alt") || "");
      return;
    }
    const actionBtn = e.target?.closest?.("[data-pipe-action]");
    if (!actionBtn) return;
    const action = actionBtn.getAttribute("data-pipe-action");
    if (action === "continue") continuePipeline();
    else if (action === "redo") redoPipeline(actionBtn.getAttribute("data-redo-step") || "front");
    else if (action === "open-project") setUiMode("workspace");
  });
  $("#btn-notebook")?.addEventListener("click", () => toggleNotebook());
  $("#btn-notebook-close")?.addEventListener("click", () => toggleNotebook(false));
  $("#reply-chip-clear")?.addEventListener("click", () => setReplyRef(null));
  $("#chat-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
  $("#toggle-advanced")?.addEventListener("click", () => $("#advanced-body").classList.toggle("hidden"));
  $("#btn-run-cleanup")?.addEventListener("click", () => {
    const meshFile = $("#job-mesh").files?.[0];
    const useCurrent = $("#job-use-current").checked;
    if (!meshFile && !useCurrent) {
      return toast("Выберите mesh или текущую модель", "error");
    }
    if (useCurrent && !activeProjectHasMesh) {
      return toast("В проекте пока нет текущей модели", "error");
    }
    const fd = new FormData();
    fd.append("prompt", "");
    if (meshFile) fd.append("mesh", meshFile);
    fd.append("use_current_mesh", useCurrent);
    fd.append("backend", "comfyui");
    fd.append("remove_bg", $("#job-rmbg").checked);
    fd.append("solidify_mm", $("#job-solid").value);
    fd.append("mode", $("#job-mode").value);
    fd.append("smooth_iters", $("#job-smooth").value);
    runOp("Cleanup…", () => api(`/api/projects/${activeProjectId}/jobs`, { method: "POST", body: fd }));
  });

  $("#btn-shade").addEventListener("click", () => {
    viewer.setWireframe(false);
    $("#btn-shade").classList.add("active");
    $("#btn-wire").classList.remove("active");
  });
  $("#btn-wire").addEventListener("click", () => {
    viewer.setWireframe(true);
    $("#btn-wire").classList.add("active");
    $("#btn-shade").classList.remove("active");
  });
  $("#btn-grid").addEventListener("click", (e) => {
    const on = !e.currentTarget.classList.contains("active");
    viewer.setGridVisible(on);
    e.currentTarget.classList.toggle("active", on);
  });
  $("#btn-reset-cam").addEventListener("click", () => viewer.resetCamera());

  $("#gen-quality")?.addEventListener("change", () => {
    applyPresetToKnobs($("#gen-quality").value);
  });
  $("#btn-gen-save")?.addEventListener("click", async () => {
    const quality_preset = $("#gen-quality").value;
    const view_consistency = $("#gen-views")?.value || "img2img";
    const view_style = $("#gen-style")?.value || "clay";
    const mesh_postprocess = $("#gen-postprocess")?.checked ?? true;
    const knobs = readGenKnobs();
    const needsHeavyDl = quality_preset === "quality" || view_consistency === "zero123";
    setLoading(true, needsHeavyDl
      ? "Сохранение + загрузка checkpoints (может занять несколько минут)…"
      : "Сохранение…", { overlay: true });
    try {
      const data = await api("/api/settings/generation", {
        method: "PUT",
        body: JSON.stringify({
          quality_preset,
          view_consistency,
          view_style,
          mesh_postprocess,
          knobs,
          download_missing: true,
        }),
      });
      renderGenerationSettings(data);
      if (data.downloaded_checkpoints?.length) {
        toast(`Скачано: ${data.downloaded_checkpoints.join(", ")}`);
      } else if (data.missing_checkpoints?.length) {
        toast(`Не хватает: ${data.missing_checkpoints.join(", ")}`, "error");
      } else {
        toast("Настройки генерации сохранены");
      }
      loadStatus();
    } catch (e) {
      toast(e.message?.slice(0, 240) || String(e), "error");
      initGeneration().catch(() => {});
    } finally {
      setLoading(false);
    }
  });
  $("#btn-llm-refresh").addEventListener("click", () => refreshLLMModels().catch((e) => toast(e.message, "error")));
  $("#btn-llm-save").addEventListener("click", async () => {
    try {
      const body = {
        base_url: $("#llm-url").value,
        api_key: $("#llm-key").value,
        planner_model: $("#llm-planner").value,
        vision_model: $("#llm-vision").value,
      };
      const r = await api("/api/settings/llm", { method: "PUT", body: JSON.stringify(body) });
      $("#llm-status").textContent = r.llm_status;
      toast("Настройки LLM сохранены");
      loadStatus();
    } catch (e) {
      toast(e.message, "error");
    }
  });
}

function numVal(id, fallback) {
  const el = $(id);
  if (!el || el.value === "") return fallback;
  const n = Number(el.value);
  return Number.isFinite(n) ? n : fallback;
}

let genSettingsCache = null;

function applyPresetToKnobs(presetKey) {
  const preset = genSettingsCache?.presets?.[presetKey];
  if (!preset) return;
  const set = (id, v) => { const el = $(id); if (el && v != null) el.value = v; };
  set("#knob-ckpt", preset.checkpoint);
  set("#knob-mesh-ckpt", preset.mesh_checkpoint);
  set("#knob-image-ckpt", preset.image_checkpoint || preset.mesh_checkpoint);
  set("#knob-steps", preset.steps);
  set("#knob-cfg", preset.cfg);
  set("#knob-mesh-steps", preset.mesh_steps);
  set("#knob-mesh-cfg", preset.mesh_cfg);
  set("#knob-mesh-guidance", preset.mesh_guidance);
}

function readGenKnobs() {
  const str = (id, fallback) => (($(id)?.value || fallback) || "").trim();
  return {
    checkpoint: str("#knob-ckpt", "sd_xl_turbo_1.0_fp16.safetensors"),
    mesh_checkpoint: str("#knob-mesh-ckpt", "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"),
    image_checkpoint: str("#knob-image-ckpt", "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"),
    zero123_checkpoint: str("#knob-z-ckpt", "stable_zero123.ckpt"),
    width: numVal("#knob-width", 768),
    height: numVal("#knob-height", 768),
    steps: numVal("#knob-steps", 8),
    cfg: numVal("#knob-cfg", 1.5),
    view_denoise: numVal("#knob-denoise", 0.58),
    view_denoise_turbo: numVal("#knob-denoise-turbo", 0.72),
    view_sampler: str("#knob-sampler", "euler"),
    view_scheduler: str("#knob-scheduler", "sgm_uniform"),
    zero123_width: numVal("#knob-z-width", 256),
    zero123_height: numVal("#knob-z-height", 256),
    zero123_steps: numVal("#knob-z-steps", 20),
    zero123_cfg: numVal("#knob-z-cfg", 3.0),
    zero123_sampler: str("#knob-z-sampler", "euler"),
    zero123_scheduler: str("#knob-z-scheduler", "normal"),
    zero123_elevation: numVal("#knob-elev", 0),
    zero123_azimuth_left: numVal("#knob-az-l", -90),
    zero123_azimuth_back: numVal("#knob-az-b", 180),
    zero123_azimuth_right: numVal("#knob-az-r", 90),
    mesh_steps: numVal("#knob-mesh-steps", 20),
    mesh_cfg: numVal("#knob-mesh-cfg", 4),
    mesh_guidance: numVal("#knob-mesh-guidance", 3.5),
    mesh_resolution: numVal("#knob-mesh-res", 3072),
    mesh_octree_resolution: numVal("#knob-mesh-octree", 256),
    mesh_num_chunks: numVal("#knob-mesh-chunks", 8000),
  };
}

function fillGenKnobs(knobs) {
  if (!knobs) return;
  const set = (id, v) => { const el = $(id); if (el && v != null) el.value = v; };
  set("#knob-ckpt", knobs.checkpoint);
  set("#knob-mesh-ckpt", knobs.mesh_checkpoint);
  set("#knob-image-ckpt", knobs.image_checkpoint);
  set("#knob-z-ckpt", knobs.zero123_checkpoint);
  set("#knob-steps", knobs.steps);
  set("#knob-cfg", knobs.cfg);
  set("#knob-width", knobs.width);
  set("#knob-height", knobs.height);
  set("#knob-denoise", knobs.view_denoise);
  set("#knob-denoise-turbo", knobs.view_denoise_turbo);
  set("#knob-sampler", knobs.view_sampler);
  set("#knob-scheduler", knobs.view_scheduler);
  set("#knob-z-width", knobs.zero123_width);
  set("#knob-z-height", knobs.zero123_height);
  set("#knob-z-steps", knobs.zero123_steps);
  set("#knob-z-cfg", knobs.zero123_cfg);
  set("#knob-z-sampler", knobs.zero123_sampler);
  set("#knob-z-scheduler", knobs.zero123_scheduler);
  set("#knob-elev", knobs.zero123_elevation);
  set("#knob-az-l", knobs.zero123_azimuth_left);
  set("#knob-az-b", knobs.zero123_azimuth_back);
  set("#knob-az-r", knobs.zero123_azimuth_right);
  set("#knob-mesh-steps", knobs.mesh_steps);
  set("#knob-mesh-cfg", knobs.mesh_cfg);
  set("#knob-mesh-guidance", knobs.mesh_guidance);
  set("#knob-mesh-res", knobs.mesh_resolution);
  set("#knob-mesh-octree", knobs.mesh_octree_resolution);
  set("#knob-mesh-chunks", knobs.mesh_num_chunks);
}

function renderGenerationSettings(data) {
  genSettingsCache = data;
  const sel = $("#gen-quality");
  if (sel && data.quality_preset) sel.value = data.quality_preset;
  const views = $("#gen-views");
  if (views) {
    const modes = data.view_modes || {};
    if (Object.keys(modes).length) {
      views.innerHTML = "";
      for (const [key, info] of Object.entries(modes)) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = info.label || key;
        views.appendChild(opt);
      }
    }
    if (data.view_consistency) views.value = data.view_consistency;
  }
  const styleSel = $("#gen-style");
  if (styleSel) {
    const styles = data.view_styles || {};
    if (Object.keys(styles).length) {
      styleSel.innerHTML = "";
      for (const [key, info] of Object.entries(styles)) {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = info.label || key;
        styleSel.appendChild(opt);
      }
    }
    if (data.view_style) styleSel.value = data.view_style;
  }
  const post = $("#gen-postprocess");
  if (post) post.checked = data.mesh_postprocess !== false;
  fillGenKnobs(data.knobs || data.active?.knobs);
  const active = data.active || {};
  const knobs = data.knobs || active.knobs || {};
  const activeEl = $("#gen-active");
  if (activeEl) {
    activeEl.textContent = [
      `SDXL: ${knobs.checkpoint || active.checkpoint || "—"}`,
      `  steps=${knobs.steps ?? active.steps} cfg=${knobs.cfg ?? active.cfg} ${knobs.width}x${knobs.height}`,
      `Hunyuan: ${knobs.mesh_checkpoint || active.mesh_checkpoint || "—"}`,
      `  steps=${knobs.mesh_steps ?? active.mesh_steps} cfg=${knobs.mesh_cfg ?? active.mesh_cfg} res=${knobs.mesh_resolution} octree=${knobs.mesh_octree_resolution}`,
      `проекции: ${active.view_consistency || data.view_consistency || "img2img"}`,
      `стиль: ${active.view_style || data.view_style || "clay"}`,
      `postprocess: ${active.mesh_postprocess !== false && data.mesh_postprocess !== false ? "on" : "off"}`,
      active.view_consistency === "zero123" || data.view_consistency === "zero123"
        ? `zero123: ${knobs.zero123_checkpoint || active.zero123_checkpoint || "stable_zero123.ckpt"} ${knobs.zero123_steps}st cfg=${knobs.zero123_cfg}`
        : null,
    ].filter(Boolean).join("\n");
  }
  const miss = $("#gen-missing");
  if (miss) {
    const missing = data.missing_checkpoints || [];
    const errs = data.download_errors || [];
    if (missing.length) {
      miss.textContent = `Нет файлов: ${missing.join(", ")}. Нажмите «Сохранить» — скачаю автоматически.`;
    } else if (errs.length) {
      miss.textContent = errs.join("; ");
    } else {
      miss.textContent = data.downloaded_checkpoints?.length
        ? `Скачано: ${data.downloaded_checkpoints.join(", ")}`
        : "";
    }
  }
}

async function initGeneration() {
  try {
    const data = await api("/api/settings/generation");
    renderGenerationSettings(data);
  } catch (e) {
    console.warn("generation settings", e);
  }
}

async function init() {
  setUiMode("chat");
  viewer = new MeshViewer($("#viewer-canvas"));
  bindRanges();
  bindFileDrop("#chat-images", "#chat-images-drop");
  bindFileDrop("#chat-mesh", "#chat-mesh-drop");
  bindFileDrop("#job-mesh", "#job-mesh-drop");
  bindActions();
  await loadStatus();
  await loadProjects();
  await initLLM();
  await initGeneration();
}

init().catch((e) => toast(e.message, "error"));
