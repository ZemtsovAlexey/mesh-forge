import type { SystemStatus } from "../types";

export default function StatusPills({ status }: { status: SystemStatus | null }) {
  if (!status) return null;
  const llm = status.services?.lmstudio;
  const comfy = status.services?.comfyui;
  const gpu = status.gpu?.active;
  return (
    <div className="status-pills">
      <span className={`dot ${llm ? "ok" : "bad"}`} title={llm ? "LM Studio" : "LM Studio недоступен"} />
      <span className={`dot ${comfy ? "ok" : "bad"}`} title={comfy ? "ComfyUI" : "ComfyUI недоступен"} />
      <span className={`dot ${gpu ? "busy" : "ok"}`} title={gpu ? gpu.label : "GPU свободен"} />
    </div>
  );
}
