import { MeshViewer } from "./viewer.js";

const $ = (sel) => document.querySelector(sel);

let activeProjectId = null;
let viewer = null;
let activeProjectHasMesh = false;
const STAGE_LABELS = {
  concept: "ComfyUI: concept",
  views: "ComfyUI: views",
  mesh: "ComfyUI: mesh",
  finalize: "Finalize",
};

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
  const stageLabel = STAGE_LABELS[stage] || stage;
  if (bar) bar.style.width = `${pct}%`;
  const time = elapsedSec != null ? ` · ${Math.round(elapsedSec)}с` : "";
  if (meta) meta.textContent = `${Math.round(pct)}%${time}${stageLabel ? ` — ${stageLabel}` : ""}`;
  if (stageLabel) $("#loading-text").textContent = stageLabel;
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
  if (!activeProjectId) { toast("Выберите проект", "error"); return null; }
  setLoading(true, label);
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
  const drop = dropSelector ? $(dropSelector) : input?.closest(".file-drop");
  if (!input || !drop) return;
  input.addEventListener("change", () => {
    drop.classList.toggle("has-file", !!input.files?.length);
    if (!input.files?.length) {
      const fallback = drop.classList.contains("chat-attach") ? "Фото" : drop.querySelector("span")?.dataset?.fallback || "Файл";
      drop.querySelector("span").textContent = fallback;
      return;
    }
    const label = input.multiple
      ? `${input.files.length} file(s): ${Array.from(input.files).slice(0, 2).map((f) => f.name).join(", ")}`
      : input.files[0].name;
    drop.querySelector("span").textContent = label;
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
}

function renderProjectMeta(p) {
  activeProjectHasMesh = !!p.has_mesh;
  const useCurrent = $("#job-use-current");
  if (useCurrent) {
    useCurrent.disabled = !activeProjectHasMesh;
    if (!activeProjectHasMesh) useCurrent.checked = false;
  }
  const hint = $("#chat-hint");
  if (hint) {
    hint.textContent = activeProjectHasMesh
      ? "Есть модель: смысловая правка текстом, или приложите новый mesh. Фильтры — в Advanced."
      : "Текст / фото → генерация. Можно сразу приложить готовый mesh (.stl/.obj).";
  }
  $("#project-meta").innerHTML = `
    <p><strong>${p.name}</strong></p>
    <p class="muted">ID: ${p.id.slice(0, 8)}…</p>
    <p class="muted">Версия: v${p.current_version}</p>
    <p class="muted">${p.has_mesh ? "Модель загружена" : "Модель отсутствует"}</p>
  `;
}

function renderChat(state) {
  const log = $("#chat-log");
  if (!log) return;
  const messages = state?.messages || [];
  if (!messages.length) {
    log.innerHTML = `<p class="muted chat-empty">${activeProjectHasMesh
      ? "Опишите, что исправить в модели…"
      : "Опишите объект для генерации…"}</p>`;
  } else {
    log.innerHTML = messages.map((m) => `
      <div class="chat-bubble ${m.role === "user" ? "user" : "assistant"}">
        <div class="chat-bubble-role">${m.role === "user" ? "Вы" : "MeshForge"}</div>
        <div class="chat-bubble-text">${escapeHtml(m.content)}</div>
      </div>
    `).join("");
  }
  log.scrollTop = log.scrollHeight;

  const draftBox = $("#chat-draft");
  const draftText = $("#chat-draft-text");
  const draftLabel = draftBox?.querySelector(".chat-draft-label");
  const brief = (state?.edit_brief_en || "").trim();
  const draft = (state?.draft_prompt_en || "").trim();
  const show = !!(state?.ready && (brief || draft || (state?.intent === "create" && !draft && messages.length)));
  if (draftBox && draftText) {
    if (state?.ready && (brief || draft)) {
      draftBox.classList.remove("hidden");
      if (draftLabel) {
        draftLabel.textContent = brief
          ? "Brief для перегенерации (ComfyUI)"
          : "Промпт для ComfyUI";
      }
      draftText.textContent = brief || draft;
    } else if (state?.ready && state?.intent === "create" && !draft) {
      draftBox.classList.remove("hidden");
      if (draftLabel) draftLabel.textContent = "Готово к реконструкции из фото";
      draftText.textContent = state.user_prompt || "Images → mesh";
    } else {
      draftBox.classList.add("hidden");
      draftText.textContent = "";
    }
  }
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

async function sendChatMessage() {
  if (!activeProjectId) return toast("Выберите проект", "error");
  const text = ($("#chat-input").value || "").trim();
  const images = Array.from($("#chat-images").files || []);
  const meshFile = $("#chat-mesh")?.files?.[0];
  if (!text && !images.length && !meshFile) {
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
    if (!imported || (!text && !images.length)) return;
  }

  const fd = new FormData();
  fd.append("text", text);
  images.forEach((file) => fd.append("images", file));

  setLoading(true, "LLM…");
  try {
    const state = await api(`/api/projects/${activeProjectId}/chat/messages`, { method: "POST", body: fd });
    $("#chat-input").value = "";
    const imagesInput = $("#chat-images");
    if (imagesInput) imagesInput.value = "";
    const drop = $("#chat-images-drop");
    if (drop) {
      drop.classList.remove("has-file");
      drop.querySelector("span").textContent = "Фото";
    }
    renderChat(state);
  } catch (e) {
    toast(e.message?.slice(0, 200) || String(e), "error");
  } finally {
    setLoading(false);
  }
}

function clearChatMesh() {
  const meshInput = $("#chat-mesh");
  if (meshInput) meshInput.value = "";
  const drop = $("#chat-mesh-drop");
  if (drop) {
    drop.classList.remove("has-file");
    drop.querySelector("span").textContent = "Mesh";
  }
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

function renderHistory(p) {
  const ul = $("#history-list");
  if (!p.versions?.length) {
    ul.innerHTML = "<li class='muted'>Пусто</li>";
    return;
  }
  ul.innerHTML = p.versions.map((v) => `
    <li><strong>v${v.version}</strong> [${v.branch}] ${v.action}
    ${v.instruction ? `<br><span>${v.instruction.slice(0, 80)}</span>` : ""}
    ${v.artifacts?.length ? `<br><span class="muted">${v.artifacts.map((a) => a.label).join(", ")}</span>` : ""}</li>
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
  loadChat();
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
  $("#btn-chat-confirm")?.addEventListener("click", () => confirmChat());
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

  $("#toggle-settings").addEventListener("click", () => $("#settings-body").classList.toggle("hidden"));
  $("#toggle-gen-settings")?.addEventListener("click", () => $("#gen-settings-body").classList.toggle("hidden"));
  $("#btn-gen-save")?.addEventListener("click", async () => {
    const quality_preset = $("#gen-quality").value;
    setLoading(true, quality_preset === "quality"
      ? "Сохранение + загрузка checkpoints (может занять несколько минут)…"
      : "Сохранение…");
    try {
      const data = await api("/api/settings/generation", {
        method: "PUT",
        body: JSON.stringify({ quality_preset, download_missing: true }),
      });
      renderGenerationSettings(data);
      if (data.downloaded_checkpoints?.length) {
        toast(`Скачано: ${data.downloaded_checkpoints.join(", ")}`);
      } else if (data.missing_checkpoints?.length) {
        toast(`Не хватает: ${data.missing_checkpoints.join(", ")}`, "error");
      } else {
        toast(quality_preset === "quality" ? "Quality готов" : "Draft сохранён");
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

function renderGenerationSettings(data) {
  const sel = $("#gen-quality");
  if (sel && data.quality_preset) sel.value = data.quality_preset;
  const active = data.active || {};
  const activeEl = $("#gen-active");
  if (activeEl) {
    activeEl.textContent = [
      `views: ${active.checkpoint || "—"}`,
      `  steps=${active.steps} cfg=${active.cfg}`,
      `mesh: ${active.mesh_checkpoint || "—"}`,
      `  steps=${active.mesh_steps} cfg=${active.mesh_cfg}`,
    ].join("\n");
  }
  const miss = $("#gen-missing");
  if (miss) {
    const missing = data.missing_checkpoints || [];
    const errs = data.download_errors || [];
    if (missing.length) {
      miss.textContent = `Нет файлов: ${missing.join(", ")}. Нажмите «Сохранить» — скачаю автоматически (~5–10 ГБ).`;
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
