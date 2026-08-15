import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, streamMessage } from "./api";
import Composer from "./components/Composer";
import Settings from "./components/Settings";
import Sidebar from "./components/Sidebar";
import StatusPills from "./components/StatusPills";
import Transcript from "./components/Transcript";
import type { Artifact, ChatDetail, ChatMessage, ChatSummary, SystemStatus, ToolCall } from "./types";

export default function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [details, setDetails] = useState<Record<string, ChatDetail>>({});
  const [activeId, setActiveId] = useState<string | null>(null);
  const [streamingById, setStreamingById] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [settings, setSettings] = useState(false);

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
    const tick = () => api.status().then(setStatus).catch(() => undefined);
    tick();
    const id = window.setInterval(tick, 4000);
    return () => window.clearInterval(id);
  }, []);

  const select = async (id: string) => {
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
    const chat = await api.createChat();
    await refreshChats();
    remember(chat);
    setActiveId(chat.id);
  };

  const stop = async () => {
    if (!activeId) return;
    await api.stopChat(activeId);
  };

  const send = async (text: string, files: File[]) => {
    let chat = active;
    if (!chat) {
      chat = await api.createChat(text.slice(0, 40) || "Новый чат");
      remember(chat);
      setActiveId(chat.id);
      await refreshChats();
    }
    const chatId = chat.id;
    setStreamingById((cur) => ({ ...cur, [chatId]: true }));
    let assistantId = `tmp-${Date.now()}`;
    const user: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      attachments: [],
      tools: [],
      artifacts: [],
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
      messages: [...(Array.isArray(cur.messages) ? cur.messages : []), user, assistant],
    }));
    try {
      await streamMessage(chatId, text, files, (event, data) => {
        if (event === "assistant_start" && data.id) {
          assistantId = String(data.id);
        }
        setDetails((cur) => {
          const prev = cur[chatId];
          if (!prev) return cur;
          return { ...cur, [chatId]: applyEvent(prev, assistantId, event, data) };
        });
      });
      await refreshChats();
      remember(await api.getChat(chatId));
    } catch (err) {
      patchChat(chatId, (cur) => ({
        ...cur,
        messages: (Array.isArray(cur.messages) ? cur.messages : []).map((m) =>
          m.id === assistantId ? { ...m, content: m.content || String(err) } : m,
        ),
      }));
    } finally {
      setStreamingById((cur) => {
        const next = { ...cur };
        delete next[chatId];
        return next;
      });
    }
  };

  const title = useMemo(() => active?.title || "MeshForge", [active]);

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        activeId={activeId}
        onSelect={select}
        onCreate={create}
        onSettings={() => setSettings(true)}
      />
      <main className="main">
        <header className="topbar">
          <h2>{title}</h2>
          <StatusPills status={status} />
        </header>
        <Transcript messages={Array.isArray(active?.messages) ? active.messages : []} streaming={streaming} />
        <Composer disabled={false} streaming={streaming} onSend={send} onStop={stop} />
      </main>
      {settings ? <Settings onClose={() => setSettings(false)} /> : null}
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
  const msg = {
    ...messages[target],
    tools: Array.isArray(messages[target].tools) ? [...messages[target].tools] : [],
    artifacts: Array.isArray(messages[target].artifacts) ? [...messages[target].artifacts] : [],
    blocks: Array.isArray(messages[target].blocks) ? [...messages[target].blocks] : [],
  };

  if (event === "user" && data.message) {
    /* already inserted locally */
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
      artifacts: [],
    };
    msg.tools = [...msg.tools, tool];
    msg.blocks = [...msg.blocks, { kind: "tool", tool_id: tool.id }];
  } else if (event === "tool_end") {
    const id = String(data.id || "");
    msg.tools = msg.tools.map((t, i, arr) => {
      const match = (id && t.id === id) || (!id && i === arr.length - 1);
      if (!match) return t;
      return {
        ...t,
        status: data.ok === false ? "error" : "ok",
        summary: String(data.summary || t.summary),
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
