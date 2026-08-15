import type { ChatDetail, ChatMessage, ChatSummary, ComfyUISettings, LLMSettings, SystemStatus, ToolCall } from "./types";

function errorMessage(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) return detail.map((item) => JSON.stringify(item)).join("; ");
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return "Request failed";
}

async function json<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      if (!res.ok) throw new Error(text.slice(0, 200) || res.statusText);
      throw new Error("Server returned non-JSON");
    }
  }
  if (!res.ok) {
    const detail = body && typeof body === "object" && "detail" in body ? (body as { detail: unknown }).detail : body;
    throw new Error(errorMessage(detail) || res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return body as T;
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function normalizeTool(raw: Partial<ToolCall> | null | undefined): ToolCall {
  return {
    id: String(raw?.id || ""),
    name: String(raw?.name || "tool"),
    title: String(raw?.title || ""),
    status: raw?.status === "ok" || raw?.status === "error" ? raw.status : "running",
    args: raw?.args && typeof raw.args === "object" && !Array.isArray(raw.args) ? raw.args : {},
    knobs: raw?.knobs && typeof raw.knobs === "object" && !Array.isArray(raw.knobs) ? raw.knobs : {},
    summary: String(raw?.summary || ""),
    progress: Number(raw?.progress || 0),
    stage: String(raw?.stage || ""),
    artifacts: asArray(raw?.artifacts),
  };
}

function normalizeMessage(raw: Partial<ChatMessage> | null | undefined): ChatMessage {
  return {
    id: String(raw?.id || ""),
    role: raw?.role === "user" ? "user" : "assistant",
    content: String(raw?.content || ""),
    created_at: String(raw?.created_at || ""),
    attachments: asArray(raw?.attachments),
    tools: asArray<Partial<ToolCall>>(raw?.tools).map(normalizeTool),
    artifacts: asArray(raw?.artifacts),
    blocks: asArray(raw?.blocks),
    reply_to: String(raw?.reply_to || ""),
    reply_artifact_ids: asArray<string>(raw?.reply_artifact_ids).map(String).filter(Boolean),
  };
}

export function normalizeChat(raw: Partial<ChatDetail> | null | undefined): ChatDetail {
  return {
    id: String(raw?.id || ""),
    title: String(raw?.title || "Чат"),
    created_at: String(raw?.created_at || ""),
    updated_at: String(raw?.updated_at || ""),
    current_mesh: String(raw?.current_mesh || ""),
    messages: asArray<Partial<ChatMessage>>(raw?.messages).map(normalizeMessage),
  };
}

export const api = {
  listChats: async () => asArray<ChatSummary>(await json(await fetch("/api/chats"))),
  createChat: async (title = "Новый чат") =>
    normalizeChat(
      await json(
        await fetch("/api/chats", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        }),
      ),
    ),
  getChat: async (id: string) => normalizeChat(await json(await fetch(`/api/chats/${id}`))),
  renameChat: async (id: string, title: string) =>
    normalizeChat(
      await json(
        await fetch(`/api/chats/${id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        }),
      ),
    ),
  deleteChat: (id: string) => fetch(`/api/chats/${id}`, { method: "DELETE" }).then((r) => json<void>(r)),
  stopChat: (id: string) => fetch(`/api/chats/${id}/stop`, { method: "POST" }).then((r) => json<{ ok: boolean }>(r)),
  status: () => fetch("/api/status").then((r) => json<SystemStatus>(r)),
  getLlm: () => fetch("/api/settings/llm").then((r) => json<LLMSettings>(r)),
  saveLlm: (body: LLMSettings) =>
    fetch("/api/settings/llm", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<{ ok: boolean }>(r)),
  llmModels: async (baseUrl: string, apiKey: string) => {
    const q = new URLSearchParams({ base_url: baseUrl, api_key: apiKey });
    const raw = await json<{ models: string[]; status: string; planner_model?: string; vision_model?: string }>(
      await fetch(`/api/settings/llm/models?${q}`),
    );
    return { ...raw, models: asArray<string>(raw?.models) };
  },
  getComfyui: () => fetch("/api/settings/comfyui").then((r) => json<ComfyUISettings>(r)),
  saveComfyui: (body: ComfyUISettings) =>
    fetch("/api/settings/comfyui", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<{ ok: boolean }>(r)),
  probeComfyui: (baseUrl: string) => {
    const q = new URLSearchParams({ base_url: baseUrl });
    return fetch(`/api/settings/comfyui/health?${q}`).then((r) =>
      json<{ ok: boolean; base_url: string; status: string }>(r),
    );
  },
};

export type SseHandler = (event: string, data: Record<string, unknown>) => void;

export async function streamMessage(
  chatId: string,
  text: string,
  files: File[],
  onEvent: SseHandler,
  signal?: AbortSignal,
  reply?: { messageId: string; artifactIds: string[] } | null,
): Promise<void> {
  const body = new FormData();
  body.append("text", text);
  for (const file of files) body.append("files", file);
  if (reply?.messageId) {
    body.append("reply_to", reply.messageId);
    if (reply.artifactIds.length) body.append("reply_artifacts", reply.artifactIds.join(","));
  }
  const res = await fetch(`/api/chats/${chatId}/messages`, { method: "POST", body, signal });
  const ctype = res.headers.get("content-type") || "";
  if (!res.ok || !res.body) {
    throw new Error(await res.text());
  }
  if (ctype.includes("text/html") || ctype.includes("application/json")) {
    const preview = await res.clone().text();
    throw new Error(preview.slice(0, 200) || "Chat stream is not SSE");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    buf = buf.replace(/\r\n/g, "\n");
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;
      try {
        onEvent(eventName, JSON.parse(dataLines.join("\n")));
      } catch {
        /* ignore malformed */
      }
    }
  }
}
