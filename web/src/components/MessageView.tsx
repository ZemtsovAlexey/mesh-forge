import type { Artifact, ChatMessage, ReplyTarget, ToolCall } from "../types";
import type { MeshPick } from "./MeshViewer";
import { makeReplyTarget } from "../reply";
import ArtifactBlock from "./ArtifactBlock";
import ThinkingBlock from "./ThinkingBlock";
import ToolCard from "./ToolCard";

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

function meshPlaceLabel(region?: string, pick?: number[], topo?: { kind?: string; vertex?: number; face?: number; edge?: number[] }): string {
  if (topo && (Number(topo.face) >= 0 || Number(topo.vertex) >= 0)) {
    const kind = topo.kind === "vertex" ? "вершина" : topo.kind === "edge" ? "ребро" : "грань";
    if (topo.kind === "edge" && topo.edge && topo.edge.length >= 2) return `${kind} ${topo.edge[0]}–${topo.edge[1]}`;
    if (topo.kind === "vertex") return `${kind} ${topo.vertex}`;
    return `${kind} ${topo.face}`;
  }
  if (pick && pick.length >= 3) {
    return `клик · ${REGION_RU[region || ""] || region || "точка"}`;
  }
  if (region) return REGION_RU[region] || region;
  return "";
}

type TimelineItem =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string }
  | { kind: "tool"; tool: ToolCall };

function timeline(message: ChatMessage): TimelineItem[] {
  const tools = Array.isArray(message.tools) ? message.tools : [];
  const byId = new Map(tools.map((tool) => [tool.id, tool]));
  const blocks = Array.isArray(message.blocks) ? message.blocks : [];
  if (blocks.length) {
    const items: TimelineItem[] = [];
    const used = new Set<string>();
    for (const block of blocks) {
      if (block.kind === "tool") {
        const tool = byId.get(block.tool_id || "");
        if (tool) {
          items.push({ kind: "tool", tool });
          used.add(tool.id);
        }
        continue;
      }
      if (block.kind === "thinking") {
        if (block.text) items.push({ kind: "thinking", text: block.text });
        continue;
      }
      if (block.text) items.push({ kind: "text", text: block.text });
    }
    for (const tool of tools) {
      if (!used.has(tool.id)) items.push({ kind: "tool", tool });
    }
    return items;
  }
  const items: TimelineItem[] = [];
  if (message.content) items.push({ kind: "text", text: message.content });
  for (const tool of tools) items.push({ kind: "tool", tool });
  return items;
}

export default function MessageView({
  message,
  quoted,
  pending = false,
  onReply,
  pick,
  onPick,
  onViewportAim,
}: {
  message: ChatMessage;
  quoted?: string;
  pending?: boolean;
  onReply?: (target: ReplyTarget) => void;
  pick?: MeshPick | null;
  onPick?: (pick: MeshPick) => void;
  onViewportAim?: (aim: { x: number; y: number; views: string }) => void;
}) {
  const items = timeline(message);
  const place = meshPlaceLabel(message.mesh_region, message.mesh_pick, message.mesh_topo);
  const hasBody = Boolean(
    items.length ||
      message.attachments?.length ||
      message.artifacts?.length ||
      place,
  );
  if (!hasBody && !pending) return null;

  const waiting = pending && !items.length;
  const canReply =
    Boolean(onReply) &&
    Boolean(message.id) &&
    !message.id.startsWith("tmp-") &&
    !message.id.startsWith("u-");
  const replyArt = (art: Artifact) => onReply?.(makeReplyTarget(message, art));
  const replyMsg = () => onReply?.(makeReplyTarget(message));
  const replyBar = canReply ? (
    <div className="msg-bar">
      <button type="button" className="reply-link" onClick={replyMsg}>
        Ответить
      </button>
    </div>
  ) : null;

  if (message.role === "user") {
    return (
      <article className="msg user">
        <div className="user-col">
          {quoted ? <div className="user-quote">Ответ на {quoted}</div> : null}
          {place ? <div className="user-quote">Место: {place}</div> : null}
          {message.attachments?.length ? (
            <ArtifactBlock artifacts={message.attachments} onReply={canReply ? replyArt : undefined} pick={pick} onPick={onPick} onViewportAim={onViewportAim} />
          ) : null}
          {message.content ? <div className="user-text">{message.content}</div> : null}
          {replyBar}
        </div>
      </article>
    );
  }

  return (
    <article className="msg assistant">
      <div className="mark" aria-hidden />
      <div className="bubble">
        {message.attachments?.length ? (
          <ArtifactBlock artifacts={message.attachments} onReply={canReply ? replyArt : undefined} pick={pick} onPick={onPick} onViewportAim={onViewportAim} />
        ) : null}
        {waiting ? (
          <div className="thinking">
            <span className="thinking-dots" aria-hidden>
              <i />
              <i />
              <i />
            </span>
            Думаю…
          </div>
        ) : null}
        {items.map((item, index) => {
          if (item.kind === "text") {
            return (
              <div className="bubble-text" key={`text-${index}`}>
                {item.text}
              </div>
            );
          }
          if (item.kind === "thinking") {
            const later = items.slice(index + 1);
            const live = pending && !later.some((next) => next.kind !== "thinking");
            return <ThinkingBlock key={`think-${index}`} text={item.text} live={live} />;
          }
          return (
            <div className="steps" key={item.tool.id}>
              <ToolCard tool={item.tool} onReply={canReply ? replyArt : undefined} pick={pick} onPick={onPick} onViewportAim={onViewportAim} />
            </div>
          );
        })}
        {message.artifacts?.length ? (
          <ArtifactBlock artifacts={message.artifacts} onReply={canReply ? replyArt : undefined} pick={pick} onPick={onPick} onViewportAim={onViewportAim} />
        ) : null}
        {replyBar}
      </div>
    </article>
  );
}
