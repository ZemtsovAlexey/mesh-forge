import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, streamMessage } from "./api";
import Composer from "./components/Composer";
import Settings from "./components/Settings";
import Sidebar from "./components/Sidebar";
import StatusPills from "./components/StatusPills";
import Transcript from "./components/Transcript";
import type { Artifact, ChatDetail, ChatMessage, ChatSummary, LLMSettings, MeshTopo, ReplyTarget, SystemStatus, ToolCall } from "./types";

const REGION_RU: Record<string, string> = {
  legs: "ножки",
  seat: "сиденье",
  back: "спинка",
  left: "слева",
  right: "справа",
  top: "верх",
  bottom: "низ",
  front: "спереди",
};

function topoChip(topo: MeshTopo): string {
  const kind = topo.kind === "vertex" ? "вершина" : topo.kind === "edge" ? "ребро" : "грань";
  if (topo.kind === "edge" && topo.edge && topo.edge.length >= 2) {
    return `${kind} ${topo.edge[0]}–${topo.edge[1]}`;
  }
  if (topo.kind === "vertex") return `${kind} ${topo.vertex}`;
  return `${kind} ${topo.face}`;
}

export default function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [details, setDetails] = useState<Record<string, ChatDetail>>({});
  const [activeId, setActiveId] = useState<string | null>(null);
  const [streamingById, setStreamingById] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [settings, setSettings] = useState(false);
  const [llm, setLlm] = useState<LLMSettings | null>(null);
  const [reply, setReply] = useState<ReplyTarget | null>(null);
  const [titleEdit, setTitleEdit] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const titleInputRef = useRef<HTMLInputElement>(null);
  const titleEditRef = useRef(false);

  const abortById = useRef<Record<string, AbortController>>({});
  const detailsRef = useRef(details);
  detailsRef.current = details;
  const streamingRef = useRef(streamingById);
  streamingRef.current = streamingById;

  const active = (activeId && details[activeId]) || null;
  const streaming = Boolean(activeId && streamingById[activeId]);

  const remember = useCallback((chat: ChatDetail) => {
    setDetails((cur) => ({ ...cur, [chat.id]: chat }));
  }, []);

  const patchChat = useCallback((chatId: string, fn: (chat: ChatDetail) => ChatDetail) => {
    setDetails((cur) => {
      const prev = cur[chatId];
      if (!prev) return cur;
      return { ...cur, [chatId]: fn(prev) };
    });
  }, []);

  const refreshChats = useCallback(async () => {
    const list = await api.listChats();
    setChats(list);
    return list;
  }, []);

  useEffect(() => {
    refreshChats()
      .then(async (list) => {
        if (list[0]) {
          const chat = await api.getChat(list[0].id);
          remember(chat);
          setActiveId(chat.id);
        }
      })
      .catch(() => undefined);
  }, [refreshChats, remember]);

  useEffect(() => {
    api.getLlm().then(setLlm).catch(() => undefined);
  }, []);

  useEffect(() => {
    const tick = () => api.status().then(setStatus).catch(() => undefined);
    tick();
    const id = window.setInterval(tick, 4000);
    return () => window.clearInterval(id);
  }, []);

  const select = async (id: string) => {
    if (id !== activeId) setReply(null);
    const cached = detailsRef.current[id];
    if (cached) setActiveId(id);
    if (cached && streamingRef.current[id]) return;
    const chat = await api.getChat(id);
    setDetails((cur) => {
      if (streamingRef.current[id] && cur[id]) return cur;
      return { ...cur, [id]: chat };
    });
    setActiveId(id);
  };

  const create = async () => {
    setReply(null);
    const chat = await api.createChat();
    await refreshChats();
    remember(chat);
    setActiveId(chat.id);
  };

  const rename = async (id: string, title: string) => {
    const next = title.trim();
    if (!next) return;
    try {
      const chat = await api.renameChat(id, next);
      remember(chat);
      setChats((cur) => cur.map((item) => (item.id === id ? { ...item, title: chat.title } : item)));
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err));
    }
  };

  const remove = async (id: string) => {
    const wasActive = activeId === id;
    try {
      if (streamingRef.current[id]) {
        await api.stopChat(id).catch(() => undefined);
      }
      await api.deleteChat(id);
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err));
      return;
    }
    setStreamingById((cur) => {
      const next = { ...cur };
      delete next[id];
      return next;
    });
    setDetails((cur) => {
      const next = { ...cur };
      delete next[id];
      return next;
    });
    if (wasActive) setReply(null);
    const list = await refreshChats();
    if (!wasActive) return;
    const fallback = list[0];
    if (!fallback) {
      setActiveId(null);
      return;
    }
    const chat = await api.getChat(fallback.id);
    remember(chat);
    setActiveId(chat.id);
  };

  const stop = async () => {
    if (!activeId) return;
    abortById.current[activeId]?.abort();
    await api.stopChat(activeId).catch(() => undefined);
  };

  const send = async (text: string, files: File[]) => {
    const citing = reply;
    setReply(null);
    let chat = active;
    if (!chat) {
      chat = await api.createChat(text.slice(0, 40) || "Новый чат");
      remember(chat);
      setActiveId(chat.id);
      await refreshChats();
    }
    const chatId = chat.id;
    abortById.current[chatId]?.abort();
    await api.stopChat(chatId).catch(() => undefined);
    const ac = new AbortController();
    abortById.current[chatId] = ac;
    setStreamingById((cur) => ({ ...cur, [chatId]: true }));
    let assistantId = `tmp-${Date.now()}`;
    const pendingPick = Array.isArray(chat.mesh_pick) ? chat.mesh_pick : [];
    const pendingRegion = chat.mesh_region || "";
    const user: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      attachments: [],
      tools: [],
      artifacts: [],
      reply_to: citing?.messageId || "",
      reply_artifact_ids: citing?.artifactIds || [],
      mesh_region: pendingRegion,
      mesh_pick: pendingPick,
      mesh_topo: chat.mesh_topo || {},
    };
    const assistant: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      created_at: new Date().toISOString(),
      attachments: [],
      tools: [],
      artifacts: [],
      blocks: [],
    };
    patchChat(chatId, (cur) => ({
      ...cur,
      mesh_region: "",
      mesh_pick: [],
      mesh_topo: {},
      messages: [...(Array.isArray(cur.messages) ? cur.messages : []), user, assistant],
    }));
    try {
      await streamMessage(
        chatId,
        text,
        files,
        (event, data) => {
          if (event === "assistant_start" && data.id) {
            assistantId = String(data.id);
          }
          setDetails((cur) => {
            const prev = cur[chatId];
            if (!prev) return cur;
            return { ...cur, [chatId]: applyEvent(prev, assistantId, event, data) };
          });
        },
        ac.signal,
        citing ? { messageId: citing.messageId, artifactIds: citing.artifactIds } : null,
      );
      await refreshChats();
      remember(await api.getChat(chatId));
    } catch (err) {
      if (ac.signal.aborted || (err instanceof DOMException && err.name === "AbortError")) {
        try {
          remember(await api.getChat(chatId));
        } catch {
          /* still unlocking composer */
        }
        return;
      }
      patchChat(chatId, (cur) => ({
        ...cur,
        messages: (Array.isArray(cur.messages) ? cur.messages : []).map((m) =>
          m.id === assistantId ? { ...m, content: m.content || String(err) } : m,
        ),
      }));
    } finally {
      if (abortById.current[chatId] === ac) {
        delete abortById.current[chatId];
      }
      setStreamingById((cur) => {
        const next = { ...cur };
        delete next[chatId];
        return next;
      });
    }
  };

  const title = useMemo(() => active?.title || "MeshForge", [active]);

  useEffect(() => {
    titleEditRef.current = false;
    setTitleEdit(false);
  }, [activeId]);

  useEffect(() => {
    if (titleEdit) titleInputRef.current?.select();
  }, [titleEdit]);

  const startTitleEdit = () => {
    if (!active) return;
    setTitleDraft(active.title);
    titleEditRef.current = true;
    setTitleEdit(true);
  };

  const commitTitleEdit = async () => {
    if (!active || !titleEditRef.current) return;
    titleEditRef.current = false;
    setTitleEdit(false);
    const next = titleDraft.trim();
    if (next && next !== active.title) await rename(active.id, next);
  };

  const confirmDelete = async () => {
    if (!active) return;
    if (!window.confirm(`Удалить чат «${active.title}»?`)) return;
    await remove(active.id);
  };

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        activeId={activeId}
        onSelect={select}
        onCreate={create}
        onRename={rename}
        onDelete={remove}
        onSettings={() => setSettings(true)}
      />
      <main className="main">
        <header className="topbar">
          {active && titleEdit ? (
            <input
              ref={titleInputRef}
              className="topbar-rename"
              value={titleDraft}
              maxLength={120}
              aria-label="Название чата"
              onChange={(e) => setTitleDraft(e.target.value)}
              onBlur={() => void commitTitleEdit()}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void commitTitleEdit();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  titleEditRef.current = false;
                  setTitleEdit(false);
                }
              }}
            />
          ) : (
            <h2>{title}</h2>
          )}
          {active ? (
            <div className="topbar-actions">
              <button type="button" title="Переименовать" onClick={startTitleEdit}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden>
                  <path
                    d="M11.2 2.6 13.4 4.8 5.9 12.3 3.5 12.5 3.7 10.1 11.2 2.6Z"
                    stroke="currentColor"
                    strokeWidth="1.3"
                    strokeLinejoin="round"
                  />
                  <path d="M9.9 3.9 12.1 6.1" stroke="currentColor" strokeWidth="1.3" />
                </svg>
              </button>
              <button type="button" className="danger" title="Удалить чат" onClick={() => void confirmDelete()}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden>
                  <path d="M3.5 4.5h9M6 4.5V3h4v1.5M5.2 4.5 5.7 13h4.6l.5-8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          ) : null}
          <StatusPills status={status} />
        </header>
        <Transcript
          messages={Array.isArray(active?.messages) ? active.messages : []}
          streaming={streaming}
          onReply={setReply}
          pick={
            active?.mesh_pick && active.mesh_pick.length >= 3
              ? {
                  x: active.mesh_pick[0],
                  y: active.mesh_pick[1],
                  z: active.mesh_pick[2],
                  kind: (active.mesh_topo?.kind as "vertex" | "edge" | "face") || "face",
                  mesh: active.mesh_topo?.mesh,
                }
              : null
          }
          onPick={
            activeId
              ? async (p) => {
                  try {
                    const out = await api.setMeshPick(activeId, p);
                    patchChat(activeId, (chat) => ({
                      ...chat,
                      mesh_region: out.region,
                      mesh_pick: out.pick,
                      mesh_topo: out.topo || {},
                      look_view: undefined,
                    }));
                  } catch (err) {
                    window.alert(err instanceof Error ? err.message : String(err));
                  }
                }
              : undefined
          }
          onViewportAim={
            activeId
              ? async (aim) => {
                  try {
                    const out = await api.setViewportAim(activeId, {
                      x: aim.x,
                      y: aim.y,
                      views: aim.views,
                      zoom: 1.5,
                    });
                    patchChat(activeId, (chat) => ({
                      ...chat,
                      look_view: out.look_view as ChatDetail["look_view"],
                      mesh_region: "",
                      mesh_pick: [],
                      mesh_topo: {},
                    }));
                  } catch (err) {
                    window.alert(err instanceof Error ? err.message : String(err));
                  }
                }
              : undefined
          }
        />
        <Composer
          disabled={false}
          streaming={streaming}
          reply={reply}
          onClearReply={() => setReply(null)}
          regionLabel={
            active?.look_view?.aim_x != null && active?.look_view?.aim_y != null
              ? `кадр · ${active.look_view.views || "view"} (${Number(active.look_view.aim_x).toFixed(2)}, ${Number(active.look_view.aim_y).toFixed(2)})`
              : active?.mesh_topo && (active.mesh_topo.face >= 0 || active.mesh_topo.vertex >= 0)
              ? topoChip(active.mesh_topo)
              : active?.mesh_pick?.length
              ? `клик · ${REGION_RU[active.mesh_region || ""] || active.mesh_region || "точка"}`
              : active?.mesh_region
                ? REGION_RU[active.mesh_region] || active.mesh_region
                : ""
          }
          onClearRegion={
            activeId
              ? async () => {
                  try {
                    await api.clearMeshPick(activeId);
                    patchChat(activeId, (chat) => ({
                      ...chat,
                      mesh_region: "",
                      mesh_pick: [],
                      mesh_topo: {},
                      look_view: chat.look_view
                        ? { ...chat.look_view, aim_x: undefined, aim_y: undefined }
                        : undefined,
                    }));
                  } catch (err) {
                    window.alert(err instanceof Error ? err.message : String(err));
                  }
                }
              : undefined
          }
          onSend={send}
          onStop={stop}
          effort={llm?.reasoning_effort || "medium"}
          onEffort={async (next) => {
            const current = llm || (await api.getLlm().catch(() => null));
            if (!current) return;
            const body = { ...current, reasoning_effort: next };
            setLlm(body);
            try {
              await api.saveLlm(body);
            } catch (err) {
              setLlm(current);
              window.alert(err instanceof Error ? err.message : String(err));
            }
          }}
        />
      </main>
      {settings ? (
        <Settings
          onClose={() => {
            setSettings(false);
            api.status().then(setStatus).catch(() => undefined);
            api.getLlm().then(setLlm).catch(() => undefined);
          }}
        />
      ) : null}
    </div>
  );
}

function findAssistantIndex(messages: ChatMessage[], assistantId: string, event: string, data: Record<string, unknown>): number {
  const messageId = event === "assistant_start" ? String(data.id || "") : "";
  const idx = messages.findIndex(
    (m) => m.role === "assistant" && (m.id === assistantId || (messageId && m.id === messageId)),
  );
  if (idx >= 0) return idx;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") return i;
  }
  return -1;
}

function applyEvent(
  cur: ChatDetail,
  assistantId: string,
  event: string,
  data: Record<string, unknown>,
): ChatDetail {
  const messages = Array.isArray(cur.messages) ? [...cur.messages] : [];
  let target = findAssistantIndex(messages, assistantId, event, data);
  if (target < 0) {
    messages.push({
      id: assistantId,
      role: "assistant",
      content: "",
      created_at: new Date().toISOString(),
      attachments: [],
      tools: [],
      artifacts: [],
      blocks: [],
    });
    target = messages.length - 1;
  }
  const prev = messages[target];
  const prevBlocks = Array.isArray(prev.blocks) ? prev.blocks : [];
  const msg = {
    ...prev,
    tools: Array.isArray(prev.tools) ? [...prev.tools] : [],
    artifacts: Array.isArray(prev.artifacts) ? [...prev.artifacts] : [],
    blocks: [...prevBlocks],
  };

  if (event === "user" && data.message) {
    const incoming = data.message as Partial<ChatMessage>;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        messages[i] = {
          ...messages[i],
          id: String(incoming.id || messages[i].id),
          content: String(incoming.content ?? messages[i].content),
          mesh_region: String(incoming.mesh_region || messages[i].mesh_region || ""),
          mesh_pick: Array.isArray(incoming.mesh_pick)
            ? incoming.mesh_pick.map(Number).filter((n) => Number.isFinite(n))
            : messages[i].mesh_pick || [],
          mesh_topo: incoming.mesh_topo || messages[i].mesh_topo || {},
          reply_to: String(incoming.reply_to || messages[i].reply_to || ""),
          reply_artifact_ids: Array.isArray(incoming.reply_artifact_ids)
            ? incoming.reply_artifact_ids.map(String)
            : messages[i].reply_artifact_ids || [],
        };
        break;
      }
    }
    return { ...cur, messages, mesh_region: "", mesh_pick: [], mesh_topo: {} };
  } else if (event === "assistant_start" && data.id) {
    msg.id = String(data.id);
  } else if (event === "text_delta") {
    const delta = String(data.delta || "");
    msg.content += delta;
    const last = msg.blocks[msg.blocks.length - 1];
    if (last && last.kind === "text") {
      msg.blocks[msg.blocks.length - 1] = { ...last, text: (last.text || "") + delta };
    } else if (delta) {
      msg.blocks = [...msg.blocks, { kind: "text", text: delta }];
    }
  } else if (event === "thinking_delta") {
    const delta = String(data.delta || "");
    const last = msg.blocks[msg.blocks.length - 1];
    if (last && last.kind === "thinking") {
      msg.blocks[msg.blocks.length - 1] = { ...last, text: (last.text || "") + delta };
    } else if (delta) {
      msg.blocks = [...msg.blocks, { kind: "thinking", text: delta }];
    }
  } else if (event === "tool_start") {
    const tool: ToolCall = {
      id: String(data.id || `t-${Date.now()}`),
      name: String(data.name || "tool"),
      title: String(data.title || ""),
      status: "running",
      args: (data.args as Record<string, unknown>) || {},
      knobs: {},
      summary: "",
      progress: 0,
      stage: "",
      thinking: "",
      artifacts: [],
    };
    msg.tools = [...msg.tools, tool];
    msg.blocks = [...msg.blocks, { kind: "tool", tool_id: tool.id }];
  } else if (event === "tool_end") {
    const id = String(data.id || "");
    msg.tools = msg.tools.map((t, i, arr) => {
      const match = (id && t.id === id) || (!id && i === arr.length - 1);
      if (!match) return t;
      const incoming = String(data.summary || "");
      return {
        ...t,
        status: data.ok === false ? "error" : "ok",
        summary: incoming.length > (t.summary || "").length ? incoming : t.summary || incoming,
        knobs:
          data.knobs && typeof data.knobs === "object" && !Array.isArray(data.knobs)
            ? (data.knobs as Record<string, unknown>)
            : t.knobs,
      };
    });
  } else if (event === "tool_progress") {
    msg.tools = msg.tools.map((t, i, arr) =>
      i === arr.length - 1
        ? { ...t, progress: Number(data.percent || t.progress), stage: String(data.stage || t.stage) }
        : t,
    );
  } else if (event === "tool_thinking_delta") {
    const delta = String(data.delta || "");
    const idx = msg.tools.map((t) => t.status).lastIndexOf("running");
    if (idx >= 0 && delta) {
      msg.tools = msg.tools.map((t, i) => (i === idx ? { ...t, thinking: (t.thinking || "") + delta } : t));
    }
  } else if (event === "tool_text_delta") {
    const delta = String(data.delta || "");
    const idx = msg.tools.map((t) => t.status).lastIndexOf("running");
    if (idx >= 0 && delta) {
      msg.tools = msg.tools.map((t, i) => (i === idx ? { ...t, summary: (t.summary || "") + delta } : t));
    }
  } else if (event === "tool_text") {
    const text = String(data.text || "");
    const idx = msg.tools.map((t) => t.status).lastIndexOf("running");
    if (idx >= 0) {
      msg.tools = msg.tools.map((t, i) => (i === idx ? { ...t, summary: text } : t));
    }
  } else if (event === "artifact" && data.artifact) {
    const art = data.artifact as Artifact;
    if (msg.tools.length) {
      const last = msg.tools.length - 1;
      msg.tools = msg.tools.map((t, i) => (i === last ? { ...t, artifacts: [...t.artifacts, art] } : t));
    } else {
      msg.artifacts = [...msg.artifacts, art];
    }
  } else if (event === "error") {
    msg.content = msg.content || String(data.message || "Ошибка");
  }
  messages[target] = msg;
  return { ...cur, messages };
}
