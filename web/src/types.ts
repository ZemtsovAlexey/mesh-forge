export type ArtifactKind = "image" | "mesh" | "mesh_preview" | "file";

export type Artifact = {
  id: string;
  kind: ArtifactKind;
  name: string;
  label: string;
  url: string;
  view?: string;
};

export type ToolCall = {
  id: string;
  name: string;
  title: string;
  status: "running" | "ok" | "error";
  args: Record<string, unknown>;
  knobs: Record<string, unknown>;
  summary: string;
  progress: number;
  stage: string;
  thinking?: string;
  artifacts: Artifact[];
};

export type MessageBlock = {
  kind: "text" | "tool" | "thinking";
  text?: string;
  tool_id?: string;
};

export type MeshElem = "vertex" | "edge" | "face";

export type MeshTopo = {
  kind?: MeshElem;
  vertex?: number;
  face?: number;
  edge?: number[];
  mesh?: string;
  hops?: number;
  faces?: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  attachments: Artifact[];
  tools: ToolCall[];
  artifacts: Artifact[];
  blocks?: MessageBlock[];
  reply_to?: string;
  reply_artifact_ids?: string[];
  mesh_region?: string;
  mesh_pick?: number[];
  mesh_topo?: MeshTopo;
};

export type ReplyTarget = {
  messageId: string;
  preview: string;
  artifactIds: string[];
};

export type ChatSummary = {
  id: string;
  title: string;
  updated_at: string;
  has_mesh: boolean;
};

export type LookView = {
  views?: string;
  aim_x?: number;
  aim_y?: number;
  zoom?: number;
};

export type ChatDetail = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  current_mesh: string;
  mesh_region?: string;
  mesh_pick?: number[];
  mesh_topo?: MeshTopo;
  look_view?: LookView;
  messages: ChatMessage[];
};

export type GpuQueueEntry = {
  kind: string;
  label: string;
  project_id?: string | null;
  position: number;
};

export type SystemStatus = {
  services: Record<string, boolean>;
  status_text: string;
  llm_provider?: string;
  gpu: {
    active: GpuQueueEntry | null;
    waiting: GpuQueueEntry[];
    shared?: boolean;
    actives?: GpuQueueEntry[];
    llm_host?: string;
    comfy_host?: string;
  };
};

export type LlmProvider = "lmstudio" | "openai";

export type ReasoningEffort = "low" | "medium" | "high" | "xhigh";

export type LLMSettings = {
  provider: LlmProvider;
  base_url: string;
  api_key: string;
  planner_model: string;
  vision_model: string;
  reasoning_effort: ReasoningEffort;
};

export const LLM_PROVIDERS: { id: LlmProvider; label: string }[] = [
  { id: "lmstudio", label: "LM Studio" },
  { id: "openai", label: "OpenAI API" },
];

export const AITUNNEL_BASE_URL = "https://api.aitunnel.ru/v1";

export const REASONING_EFFORTS: { id: ReasoningEffort; label: string }[] = [
  { id: "low", label: "Низкий" },
  { id: "medium", label: "Средний" },
  { id: "high", label: "Высокий" },
  { id: "xhigh", label: "Максимум" },
];

export type ComfyUISettings = {
  enabled: boolean;
  base_url: string;
};
