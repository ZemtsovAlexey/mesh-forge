import { useEffect, useRef, useState } from "react";
import type { ReasoningEffort, ReplyTarget } from "../types";
import { REASONING_EFFORTS } from "../types";

export default function Composer({
  disabled,
  streaming,
  reply,
  onClearReply,
  regionLabel,
  onClearRegion,
  effort,
  onEffort,
  onSend,
  onStop,
}: {
  disabled?: boolean;
  streaming: boolean;
  reply?: ReplyTarget | null;
  onClearReply?: () => void;
  regionLabel?: string;
  onClearRegion?: () => void;
  effort: ReasoningEffort;
  onEffort: (value: ReasoningEffort) => void;
  onSend: (text: string, files: File[]) => void;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [effortOpen, setEffortOpen] = useState(false);
  const imageRef = useRef<HTMLInputElement>(null);
  const meshRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const effortRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (reply) inputRef.current?.focus();
  }, [reply]);

  useEffect(() => {
    if (!effortOpen) return;
    const onDown = (e: MouseEvent) => {
      if (!effortRef.current?.contains(e.target as Node)) setEffortOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [effortOpen]);

  const submit = () => {
    const value = text.trim();
    if (!value && !files.length) return;
    onSend(value, files);
    setText("");
    setFiles([]);
  };

  const effortLabel = REASONING_EFFORTS.find((item) => item.id === effort)?.label || "Средний";

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
        {regionLabel ? (
          <div className="reply-chip">
            <span className="reply-chip-label">Место</span>
            <span className="reply-chip-text">{regionLabel}</span>
            <button type="button" className="btn ghost" onClick={onClearRegion} aria-label="Снять место">
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
          placeholder={
            reply ? "Что сделать с этим?" : regionLabel ? "Что сделать с этим местом?" : "Опишите объект…"
          }
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
              submit();
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
          <div className="effort-wrap" ref={effortRef}>
            <button
              type="button"
              className="effort-btn"
              title="Уровень размышления"
              onClick={() => setEffortOpen((cur) => !cur)}
            >
              {effortLabel}
              <span className="picker-caret" aria-hidden>
                {effortOpen ? "▴" : "▾"}
              </span>
            </button>
            {effortOpen ? (
              <div className="effort-menu">
                <p className="effort-menu-title">Effort</p>
                {REASONING_EFFORTS.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    className={`picker-item${item.id === effort ? " active" : ""}`}
                    onClick={() => {
                      onEffort(item.id);
                      setEffortOpen(false);
                    }}
                  >
                    {item.label}
                    {item.id === effort ? <span className="effort-check">✓</span> : null}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
          {streaming ? (
            <button type="button" className="btn danger" onClick={onStop}>
              Стоп
            </button>
          ) : null}
          <button
            type="button"
            className="send-btn"
            disabled={disabled}
            onClick={submit}
            title={streaming ? "Прервать и отправить" : "Отправить"}
          >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
                <path d="M3 8h9M8.5 4.5 12.5 8l-4 3.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
        </div>
      </div>
    </div>
  );
}
