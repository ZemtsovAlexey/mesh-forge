import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import MessageView from "./MessageView";

export default function Transcript({
  messages,
  streaming,
}: {
  messages: ChatMessage[];
  streaming: boolean;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const onScroll = () => {
      stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = scroller.current;
    if (!el || !stick.current) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  const list = Array.isArray(messages) ? messages : [];
  return (
    <div className="transcript" ref={scroller}>
      <div className="transcript-inner">
        {!list.length ? (
          <div className="chat-empty">
            <svg className="empty-mark" viewBox="0 0 32 32" fill="none" aria-hidden>
              <path d="M16 3.5 28 10.5 16 17.5 4 10.5 16 3.5Z" fill="#e0b07a" />
              <path d="M4 10.5 16 17.5v11L4 21.5v-11Z" fill="#b88955" />
              <path d="M16 17.5 28 10.5v11L16 28.5v-11Z" fill="#8f6a3e" />
            </svg>
            <h3>Что слепим?</h3>
            <p>Опишите фигурку или прикрепите фото — картинки и 3D появятся в чате.</p>
          </div>
        ) : (
          list.map((m, i) => {
            const last = i === list.length - 1;
            const pending = streaming && last && m.role === "assistant";
            return <MessageView key={m.id} message={m} pending={pending} />;
          })
        )}
      </div>
    </div>
  );
}
