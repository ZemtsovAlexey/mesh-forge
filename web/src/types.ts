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
  artifacts: Artifact[];
};

export type MessageBlock = {
  kind: "text" | "tool";
  text?: string;
  tool_id?: string;
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
};

export type ChatSummary = {
  id: string;
  title: string;
  updated_at: string;
  has_mesh: boolean;
};

export type ChatDetail = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  current_mesh: string;
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
  gpu: {
    active: GpuQueueEntry | null;
    waiting: GpuQueueEntry[];
    shared?: boolean;
    actives?: GpuQueueEntry[];
    llm_host?: string;
    comfy_host?: string;
  };
};

export type LLMSettings = {
  base_url: string;
  api_key: string;
  planner_model: string;
  vision_model: string;
};
