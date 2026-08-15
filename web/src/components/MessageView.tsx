import type { Artifact, ChatMessage, ReplyTarget, ToolCall } from "../types";
import { makeReplyTarget } from "../reply";
import ArtifactBlock from "./ArtifactBlock";
import ToolCard from "./ToolCard";

function timeline(message: ChatMessage): Array<{ kind: "text"; text: string } | { kind: "tool"; tool: ToolCall }> {
  const tools = Array.isArray(message.tools) ? message.tools : [];
  const byId = new Map(tools.map((tool) => [tool.id, tool]));
  const blocks = Array.isArray(message.blocks) ? message.blocks : [];
  if (blocks.length) {
    const items: Array<{ kind: "text"; text: string } | { kind: "tool"; tool: ToolCall }> = [];
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
      if (block.text) items.push({ kind: "text", text: block.text });
    }
    for (const tool of tools) {
      if (!used.has(tool.id)) items.push({ kind: "tool", tool });
    }
    return items;
  }
  const items: Array<{ kind: "text"; text: string } | { kind: "tool"; tool: ToolCall }> = [];
  if (message.content) items.push({ kind: "text", text: message.content });
  for (const tool of tools) items.push({ kind: "tool", tool });
  return items;
}

export default function MessageView({
  message,
  quoted,
  pending = false,
  onReply,
}: {
  message: ChatMessage;
  quoted?: string;
  pending?: boolean;
  onReply?: (target: ReplyTarget) => void;
}) {
  const items = timeline(message);
  const hasBody = Boolean(
    items.length ||
      message.attachments?.length ||
      message.artifacts?.length,
  );
  if (!hasBody && !pending) return null;

  const thinking = pending && !items.length;
  const canReply = Boolean(onReply) && !pending && Boolean(message.id) && !message.id.startsWith("tmp-") && !message.id.startsWith("u-");
  const replyArt = (art: Artifact) => onReply?.(makeReplyTarget(message, art));
  const replyMsg = () => onReply?.(makeReplyTarget(message));

  if (message.role === "user") {
    return (
      <article className="msg user">
        <div className="user-col">
          {quoted ? <div className="user-quote">Ответ на {quoted}</div> : null}
          {message.attachments?.length ? <ArtifactBlock artifacts={message.attachments} /> : null}
          {message.content ? <div className="user-text">{message.content}</div> : null}
        </div>
      </article>
    );
  }

  return (
    <article className="msg assistant">
      <div className="mark" aria-hidden />
      <div className="bubble">
        {message.attachments?.length ? (
          <ArtifactBlock artifacts={message.attachments} onReply={canReply ? replyArt : undefined} />
        ) : null}
        {thinking ? (
          <div className="thinking">
            <span className="thinking-dots" aria-hidden>
              <i />
              <i />
              <i />
            </span>
            Думаю…
          </div>
        ) : null}
        {items.map((item, index) =>
          item.kind === "text" ? (
            <div className="bubble-text" key={`text-${index}`}>
              {item.text}
            </div>
          ) : (
            <div className="steps" key={item.tool.id}>
              <ToolCard tool={item.tool} onReply={canReply ? replyArt : undefined} />
            </div>
          ),
        )}
        {message.artifacts?.length ? (
          <ArtifactBlock artifacts={message.artifacts} onReply={canReply ? replyArt : undefined} />
        ) : null}
        {canReply ? (
          <div className="msg-bar">
            <button type="button" className="reply-link" onClick={replyMsg}>
              Ответить
            </button>
          </div>
        ) : null}
      </div>
    </article>
  );
}
