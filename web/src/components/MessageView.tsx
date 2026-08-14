import type { ChatMessage } from "../types";
import ArtifactBlock from "./ArtifactBlock";
import ToolCard from "./ToolCard";

export default function MessageView({
  message,
  pending = false,
}: {
  message: ChatMessage;
  pending?: boolean;
}) {
  const tools = Array.isArray(message.tools) ? message.tools : [];
  const hasBody = Boolean(
    message.content ||
      tools.length ||
      message.attachments?.length ||
      message.artifacts?.length,
  );
  if (!hasBody && !pending) return null;

  const thinking = pending && !message.content && !tools.length;
  if (message.role === "user") {
    return (
      <article className="msg user">
        <div className="user-col">
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
        {message.attachments?.length ? <ArtifactBlock artifacts={message.attachments} /> : null}
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
        {message.content ? <div className="bubble-text">{message.content}</div> : null}
        {tools.length ? (
          <div className="steps">
            {tools.map((tool) => (
              <ToolCard key={tool.id} tool={tool} />
            ))}
          </div>
        ) : null}
        {message.artifacts?.length ? <ArtifactBlock artifacts={message.artifacts} /> : null}
      </div>
    </article>
  );
}
