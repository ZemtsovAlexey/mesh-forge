import { useEffect, useRef, useState } from "react";
import type { ReplyTarget } from "../types";

export default function Composer({
  disabled,
  streaming,
  reply,
  onClearReply,
  onSend,
  onStop,
}: {
  disabled?: boolean;
  streaming: boolean;
  reply?: ReplyTarget | null;
  onClearReply?: () => void;
  onSend: (text: string, files: File[]) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const imageRef = useRef<HTMLInputElement>(null);
  const meshRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (reply) inputRef.current?.focus();
  }, [reply]);

  const submit = () => {
    const value = text.trim();
    if (!value && !files.length) return;
    onSend(value, files);
    setText("");
    setFiles([]);
  };

  return (
    <div className="composer">
      <div className="composer-inner">
        {reply ? (
          <div className="reply-chip">
            <span className="reply-chip-label">Ответ на</span>
            <span className="reply-chip-text">{reply.preview}</span>
            <button type="button" className="btn ghost" onClick={onClearReply} aria-label="Снять ответ">
              ×
            </button>
          </div>
        ) : null}
        {files.length ? (
          <div className="chips">
            {files.map((f, i) => (
              <span className="chip" key={`${f.name}-${i}`}>
                {f.name}
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => setFiles((cur) => cur.filter((_, idx) => idx !== i))}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        ) : null}
        <textarea
          ref={inputRef}
          rows={2}
          placeholder={reply ? "Что сделать с этим?" : "Опишите объект…"}
          value={text}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape" && reply) {
              e.preventDefault();
              onClearReply?.();
              return;
            }
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!streaming) submit();
            }
          }}
        />
        <div className="composer-bar">
          <button type="button" className="attach-btn" onClick={() => imageRef.current?.click()}>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
              <rect x="2" y="3.5" width="12" height="9" rx="1.6" stroke="currentColor" strokeWidth="1.4" />
              <circle cx="5.5" cy="6.6" r="1.1" fill="currentColor" />
              <path d="M2.8 11.2 6.2 8.4l2.2 1.8 2.1-2.4 2.7 3.4" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
            </svg>
            фото
          </button>
          <button type="button" className="attach-btn" onClick={() => meshRef.current?.click()}>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden>
              <path d="M8 2.2 13.5 5.4v5.2L8 13.8 2.5 10.6V5.4L8 2.2Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
              <path d="M8 13.8V8M2.5 5.4 8 8l5.5-2.6" stroke="currentColor" strokeWidth="1.3" />
            </svg>
            mesh
          </button>
          <input
            ref={imageRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => setFiles((cur) => [...cur, ...Array.from(e.target.files || [])])}
          />
          <input
            ref={meshRef}
            type="file"
            accept=".stl,.obj,.glb"
            hidden
            onChange={(e) => setFiles((cur) => [...cur, ...Array.from(e.target.files || [])])}
          />
          <span className="spacer" />
          {streaming ? (
            <button type="button" className="btn danger" onClick={onStop}>
              Стоп
            </button>
          ) : (
            <button
              type="button"
              className="send-btn"
              disabled={disabled}
              onClick={submit}
              title="Отправить"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
                <path d="M3 8h9M8.5 4.5 12.5 8l-4 3.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
