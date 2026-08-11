import { MeshViewer } from "./viewer.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let activeProjectId = null;
let viewer = null;

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

function setProgressUI(percent, stage, elapsedSec = null) {
  const bar = $("#progress-bar");
  const meta = $("#progress-meta");
  const pct = Math.max(0, Math.min(100, Number(percent) || 0));
  if (bar) bar.style.width = `${pct}%`;
  const time = elapsedSec != null ? ` · ${Math.round(elapsedSec)}с` : "";
  if (meta) meta.textContent = `${Math.round(pct)}%${time}${stage ? ` — ${stage}` : ""}`;
  if (stage) $("#loading-text").textContent = stage;
}

function setLoading(on, text = "Обработка…") {
  $("#loading").classList.toggle("hidden", !on);
  $("#loading-text").textContent = text;
  if (on) setProgressUI(2, text, 0);
  else {
    setProgressUI(0, "");
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
  }
}

function startProgressPolling(projectId) {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = setInterval(async () => {
    if (!projectId) return;
    try {
      const p = await api(`/api/projects/${projectId}/progress`);
      if (!p) return;
      setProgressUI(p.percent, p.stage || "Обработка…", p.elapsed_sec);
    } catch {
      /* ignore poll errors while request runs */
    }
  }, 700);
}

async function runOp(label, fn) {
  if (!activeProjectId) { toast("Выберите проект", "error"); return; }
  setLoading(true, label);
  startProgressPolling(activeProjectId);
  try {
    const result = await fn();
    setProgressUI(100, "Готово", null);
    afterOperation(result);
  } catch (e) {
    toast(e.message?.slice(0, 200) || String(e), "error");
  } finally {
    setLoading(false);
  }
}

function bindRanges() {
  [["photo-solid", "photo-solid-val"], ["scan-smooth", "scan-smooth-val"], ["scan-solid", "scan-solid-val"], ["edit-solid", "edit-solid-val"]].forEach(([id, out]) => {
    const input = $(`#${id}`);
    const output = $(`#${out}`);
    input?.addEventListener("input", () => { output.textContent = input.value; });
  });
}

function bindTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("active"));
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`#tab-${tab.dataset.tab}`).classList.add("active");
    });
  });
}

function bindFileDrop(inputId, dropSelector) {
  const input = $(inputId);
  const drop = dropSelector ? $(dropSelector) : input?.closest(".file-drop");
  if (!input || !drop) return;
  input.addEventListener("change", () => {
    drop.classList.toggle("has-file", !!input.files?.length);
    if (input.files?.[0]) drop.querySelector("span").textContent = input.files[0].name;
  });
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
    return;
  }
  for (const p of projects) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = p.id === selectId ? "active" : "";
    btn.innerHTML = `${p.name}<span class="sub">v${p.current_version}${p.has_mesh ? " · mesh" : ""}</span>`;
    btn.addEventListener("click", () => selectProject(p.id));
    li.appendChild(btn);
    list.appendChild(li);
  }
  if (selectId) await selectProject(selectId, false);
  else if (projects.length && !activeProjectId) await selectProject(projects[0].id, false);
}

async function selectProject(id, refreshList = true) {
  activeProjectId = id;
  if (refreshList) await loadProjects(id);
  const project = await api(`/api/projects/${id}`);
  renderProjectMeta(project);
  renderHistory(project);
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
}

function renderProjectMeta(p) {
  $("#project-meta").innerHTML = `
    <p><strong>${p.name}</strong></p>
    <p class="muted">ID: ${p.id.slice(0, 8)}…</p>
    <p class="muted">Версия: v${p.current_version}</p>
    <p class="muted">${p.has_mesh ? "Модель загружена" : "Модель отсутствует"}</p>
  `;
}

function renderHistory(p) {
  const ul = $("#history-list");
  if (!p.versions?.length) {
    ul.innerHTML = "<li class='muted'>Пусто</li>";
    return;
  }
  ul.innerHTML = p.versions.map((v) => `
    <li><strong>v${v.version}</strong> [${v.branch}] ${v.action}
    ${v.instruction ? `<br><span>${v.instruction.slice(0, 80)}</span>` : ""}</li>
  `).join("");
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
  renderProjectMeta(result.project);
  renderHistory(result.project);
  if (result.qc_report) $("#qc-log").textContent = result.qc_report;
  toast(result.message);
  if (result.project.mesh_url) {
    loadMesh(result.project.mesh_url);
    $("#viewer-empty").classList.add("hidden");
    $("#btn-download").classList.remove("hidden");
    $("#btn-download").href = result.project.mesh_url;
  }
  loadProjects(activeProjectId);
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

  $("#btn-photo").addEventListener("click", () => {
    const file = $("#photo-file").files?.[0];
    if (!file) return toast("Выберите фото", "error");
    const backend = $("#photo-backend")?.value || "hunyuan3d";
    const fd = new FormData();
    fd.append("image", file);
    fd.append("remove_bg", $("#photo-rmbg").checked);
    fd.append("solidify_mm", $("#photo-solid").value);
    fd.append("backend", backend);
    const label = backend === "triposr" ? "TripoSR: фото → 3D…" : "Hunyuan3D: фото → 3D…";
    runOp(label, () => api(`/api/projects/${activeProjectId}/photo`, { method: "POST", body: fd }));
  });

  $("#btn-scan").addEventListener("click", () => {
    const file = $("#scan-file").files?.[0];
    if (!file) return toast("Выберите файл", "error");
    const fd = new FormData();
    fd.append("scan", file);
    fd.append("mode", $("#scan-mode").value);
    fd.append("smooth_iters", $("#scan-smooth").value);
    fd.append("solidify_mm", $("#scan-solid").value);
    runOp("Обработка скана…", () => api(`/api/projects/${activeProjectId}/scan`, { method: "POST", body: fd }));
  });

  $("#btn-text").addEventListener("click", () => {
    const prompt = $("#text-prompt").value.trim();
    if (!prompt) return toast("Введите описание", "error");
    runOp("Генерация по тексту…", () => api(`/api/projects/${activeProjectId}/text`, {
      method: "POST",
      body: JSON.stringify({ prompt, mode: $("#text-mode").value }),
    }));
  });

  $("#btn-edit-text").addEventListener("click", () => {
    const instruction = $("#edit-instr").value.trim();
    if (!instruction) return toast("Введите инструкцию", "error");
    runOp("Правка…", () => api(`/api/projects/${activeProjectId}/edit/text`, {
      method: "POST",
      body: JSON.stringify({ instruction, solidify_mm: parseFloat($("#edit-solid").value) }),
    }));
  });

  $("#btn-edit-photo").addEventListener("click", () => {
    const fd = new FormData();
    fd.append("instruction", $("#edit-instr").value.trim() || "match reference");
    const ref = $("#edit-ref").files?.[0];
    if (ref) fd.append("reference", ref);
    runOp("Правка по фото…", () => api(`/api/projects/${activeProjectId}/edit/photo`, { method: "POST", body: fd }));
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

  $("#toggle-settings").addEventListener("click", () => $("#settings-body").classList.toggle("hidden"));
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

async function init() {
  viewer = new MeshViewer($("#viewer-canvas"));
  bindRanges();
  bindTabs();
  bindFileDrop("#photo-file", "#photo-drop");
  bindFileDrop("#scan-file");
  bindFileDrop("#edit-ref");
  bindActions();
  await loadStatus();
  await loadProjects();
  await initLLM();
}

init().catch((e) => toast(e.message, "error"));
