import { useState } from "react";
import type { Artifact, ToolCall } from "../types";
import type { MeshPick } from "./MeshViewer";
import ArtifactBlock from "./ArtifactBlock";
import ThinkingBlock from "./ThinkingBlock";

function cleanSummary(value: string): string {
  const text = value.trim();
  if (!text) return "";
  if (/^(none|null|undefined|ok|done)$/i.test(text)) return "";
  return text.replace(/\s*knobs=\{[^}]*\}\s*$/, "").trim();
}

function promptFromArgs(args: Record<string, unknown> | undefined): string {
  const prompt = args?.prompt;
  return typeof prompt === "string" ? prompt.trim() : "";
}

function formatArgValue(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (item && typeof item === "object" && "ref" in item) {
          const ref = String((item as { ref?: string }).ref || "");
          const view = String((item as { view?: string }).view || "");
          return view ? `${view}:${ref}` : ref;
        }
        return typeof item === "string" ? item : JSON.stringify(item);
      })
      .filter(Boolean)
      .join(", ");
  }
  return JSON.stringify(value);
}

function extraKnobs(tool: ToolCall): [string, string][] {
  const skip = new Set(["prompt", "raw"]);
  const fromArgs = Object.entries(tool.args || {})
    .filter(([key, value]) => !skip.has(key) && value != null && value !== "")
    .map(([key, value]) => [key, formatArgValue(value)] as [string, string])
    .filter(([, value]) => Boolean(value));
  const fromEcho = Object.entries(tool.knobs || {})
    .filter(([key]) => !fromArgs.some(([existing]) => existing === key))
    .map(([key, value]) => [key, String(value)] as [string, string]);
  return [...fromArgs, ...fromEcho];
}

export default function ToolCard({
  tool,
  onReply,
  pick,
  onPick,
  onViewportAim,
}: {
  tool: ToolCall;
  onReply?: (art: Artifact) => void;
  pick?: MeshPick | null;
  onPick?: (pick: MeshPick) => void;
  onViewportAim?: (aim: { x: number; y: number; views: string }) => void;
}) {
  const [open, setOpen] = useState(false);
  const knobs = extraKnobs(tool);
  const running = tool.status === "running";
  const percent = Math.min(100, Math.max(0, Math.round(tool.progress || 0)));
  const prompt = promptFromArgs(tool.args);
  const summary = cleanSummary(tool.summary || "");
  const thinking = (tool.thinking || "").trim();
  const preview = prompt || summary;
  const hasDetails = knobs.length > 0 || Boolean(preview);
  const longSummary = Boolean(prompt && summary && summary !== prompt);
  const statusText = running
    ? percent > 0
      ? `${percent}%`
      : ""
    : tool.status === "error"
      ? "ошибка"
      : "";
  return (
    <div className={`step ${tool.status}`}>
      <button type="button" className="step-row" onClick={() => setOpen((v) => !v)}>
        {running ? (
          <span className="step-spin" aria-hidden />
        ) : (
          <span className={`step-mark ${tool.status}`} aria-hidden>
            {tool.status === "ok" ? "✓" : "!"}
          </span>
        )}
        <span className="step-title">{tool.title || tool.name}</span>
        {running && tool.stage ? <span className="step-sub">{tool.stage}</span> : null}
        {statusText ? <span className={`step-status ${tool.status}`}>{statusText}</span> : null}
      </button>
      {running ? (
        <div className="progress">
          <span style={{ width: `${percent > 0 ? percent : 14}%` }} />
        </div>
      ) : null}
      {thinking ? <ThinkingBlock text={thinking} live={running && !summary} /> : null}
      {hasDetails ? (
        <div className="step-body">
          {prompt ? <div className="step-prompt">{prompt}</div> : null}
          {knobs.length ? (
            <div className="knobs">
              {knobs.map(([k, v]) => (
                <span className="knob" key={k}>
                  {k}={v}
                </span>
              ))}
            </div>
          ) : null}
          {!prompt && summary ? <div className="step-note">{summary}</div> : null}
          {open && longSummary ? <div className="step-note">{summary}</div> : null}
        </div>
      ) : null}
      <ArtifactBlock artifacts={tool.artifacts} onReply={onReply} pick={pick} onPick={onPick} onViewportAim={onViewportAim} />
    </div>
  );
}
