import type { SystemStatus } from "../types";

export default function StatusPills({ status }: { status: SystemStatus | null }) {
  if (!status) return null;
  const llm = status.services?.lmstudio;
  const comfy = status.services?.comfyui;
  const gpu = status.gpu;
  const actives = gpu?.actives?.length ? gpu.actives : gpu?.active ? [gpu.active] : [];
  const split = gpu?.shared === false;
  const hosts =
    split && gpu?.llm_host && gpu?.comfy_host ? ` · ${gpu.llm_host} != ${gpu.comfy_host}` : "";
  const gpuTitle = actives.length
    ? `${actives.map((item) => item.label).join(" · ")}${split ? " (раздельные GPU)" : ""}${hosts}`
    : split
      ? `GPU свободны (раздельные очереди)${hosts}`
      : "GPU свободен";
  return (
    <div className="status-pills">
      <span className={`dot ${llm ? "ok" : "bad"}`} title={llm ? "LM Studio" : "LM Studio недоступен"} />
      <span className={`dot ${comfy ? "ok" : "bad"}`} title={comfy ? "ComfyUI" : "ComfyUI недоступен"} />
      <span className={`dot ${actives.length ? "busy" : "ok"}`} title={gpuTitle} />
    </div>
  );
}
