import { useState } from "react";
import type { ToolCall } from "../types";
import ArtifactBlock from "./ArtifactBlock";

function cleanSummary(value: string): string {
  const text = value.trim();
  if (!text) return "";
  if (/^(none|null|undefined|ok|done)$/i.test(text)) return "";
  return text;
}

export default function ToolCard({ tool }: { tool: ToolCall }) {
  const [open, setOpen] = useState(false);
  const knobs =
    tool.knobs && typeof tool.knobs === "object" && !Array.isArray(tool.knobs)
      ? Object.entries(tool.knobs)
      : [];
  const running = tool.status === "running";
  const percent = Math.min(100, Math.max(0, Math.round(tool.progress || 0)));
  const summary = cleanSummary(tool.summary || "");
  const hasDetails = knobs.length > 0 || Boolean(summary);
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
      {open && hasDetails ? (
        <div className="step-body">
          {knobs.length ? (
            <div className="knobs">
              {knobs.map(([k, v]) => (
                <span className="knob" key={k}>
                  {k}={String(v)}
                </span>
              ))}
            </div>
          ) : null}
          {summary ? <div>{summary}</div> : null}
        </div>
      ) : null}
      <ArtifactBlock artifacts={tool.artifacts} />
    </div>
  );
}
