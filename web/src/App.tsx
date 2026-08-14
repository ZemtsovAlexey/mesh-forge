import { useCallback, useEffect, useMemo, useState } from "react";
import { api, streamMessage } from "./api";
import Composer from "./components/Composer";
import Settings from "./components/Settings";
import Sidebar from "./components/Sidebar";
import StatusPills from "./components/StatusPills";
import Transcript from "./components/Transcript";
import type { Artifact, ChatDetail, ChatMessage, ChatSummary, SystemStatus, ToolCall } from "./types";

export default function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [active, setActive] = useState<ChatDetail | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [settings, setSettings] = useState(false);

  const refreshChats = useCallback(async () => {
    const list = await api.listChats();
    setChats(list);
    return list;
  }, []);

  useEffect(() => {
    refreshChats()
      .then(async (list) => {
        if (list[0]) {
          setActive(await api.getChat(list[0].id));
        }
      })
      .catch(() => undefined);
  }, [refreshChats]);

  useEffect(() => {
    const tick = () => api.status().then(setStatus).catch(() => undefined);
    tick();
    const id = window.setInterval(tick, 4000);
    return () => window.clearInterval(id);
  }, []);

  const select = async (id: string) => {
    setActive(await api.getChat(id));
  };

  const create = async () => {
    const chat = await api.createChat();
    await refreshChats();
    setActive(chat);
  };

  const stop = async () => {
    if (!active) return;
    await api.stopChat(active.id);
  };

  const send = async (text: string, files: File[]) => {
    let chat = active;
    if (!chat) {
      chat = await api.createChat(text.slice(0, 40) || "Новый чат");
      setActive(chat);
      await refreshChats();
    }
    const chatId = chat.id;
    setStreaming(true);
    const assistantId = `tmp-${Date.now()}`;
    setActive((cur) => {
      if (!cur || cur.id !== chatId) return cur;
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
      };
      return { ...cur, messages: [...(Array.isArray(cur.messages) ? cur.messages : []), user, assistant] };
    });
    try {
      await streamMessage(chatId, text, files, (event, data) => {
        setActive((cur) => applyEvent(cur, chatId, assistantId, event, data));
      });
      await refreshChats();
      const fresh = await api.getChat(chatId);
      setActive(fresh);
    } catch (err) {
      setActive((cur) => {
        if (!cur) return cur;
        const messages = Array.isArray(cur.messages) ? cur.messages : [];
        return {
          ...cur,
          messages: messages.map((m) =>
            m.id === assistantId ? { ...m, content: m.content || String(err) } : m,
          ),
        };
      });
    } finally {
      setStreaming(false);
    }
  };

  const title = useMemo(() => active?.title || "MeshForge", [active]);

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        activeId={active?.id ?? null}
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

function applyEvent(
  cur: ChatDetail | null,
  chatId: string,
  assistantId: string,
  event: string,
  data: Record<string, unknown>,
): ChatDetail | null {
  if (!cur || cur.id !== chatId) return cur;
  const messages = Array.isArray(cur.messages) ? [...cur.messages] : [];
  const idx = messages.findIndex((m) => m.role === "assistant" && (m.id === assistantId || m.id === data.id));
  const target = idx >= 0 ? idx : messages.length - 1;
  if (target < 0) return cur;
  const msg = {
    ...messages[target],
    tools: Array.isArray(messages[target].tools) ? [...messages[target].tools] : [],
    artifacts: Array.isArray(messages[target].artifacts) ? [...messages[target].artifacts] : [],
  };

  if (event === "user" && data.message) {
    /* already inserted locally */
  } else if (event === "text_delta") {
    msg.content += String(data.delta || "");
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
  } else if (event === "tool_end") {
    const id = String(data.id || "");
    msg.tools = msg.tools.map((t, i, arr) => {
      const match = (id && t.id === id) || (!id && i === arr.length - 1);
      if (!match) return t;
      return {
        ...t,
        status: data.ok === false ? "error" : "ok",
        summary: String(data.summary || t.summary),
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
      msg.tools = msg.tools.map((t, i) =>
        i === last ? { ...t, artifacts: [...t.artifacts, art] } : t,
      );
    } else {
      msg.artifacts = [...msg.artifacts, art];
    }
  } else if (event === "error") {
    msg.content = msg.content || String(data.message || "Ошибка");
  }
  messages[target] = msg;
  return { ...cur, messages };
}
