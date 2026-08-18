import { useEffect, useRef, useState, type MouseEvent } from "react";

function nearBottom(el: HTMLElement) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 48;
}

export default function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(true);
  const [follow, setFollow] = useState(true);
  const bodyRef = useRef<HTMLPreElement>(null);
  const followRef = useRef(true);

  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const onScroll = () => {
      const pinned = nearBottom(el);
      followRef.current = pinned;
      setFollow(pinned);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const el = bodyRef.current;
    if (!live || !el || !followRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [text, live]);

  const jumpLatest = (event: MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    const el = bodyRef.current;
    followRef.current = true;
    setFollow(true);
    if (el) el.scrollTop = el.scrollHeight;
  };

  return (
    <details
      className={`think-block${live ? " live" : ""}`}
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary>
        {live ? (
          <span className="thinking-dots" aria-hidden>
            <i />
            <i />
            <i />
          </span>
        ) : null}
        {live ? "Думаю…" : "Рассуждение"}
      </summary>
      <div className="think-wrap">
        <pre ref={bodyRef} className="think-body">
          {text}
        </pre>
        {live && !follow ? (
          <button type="button" className="think-latest" onClick={jumpLatest}>
            К новым
          </button>
        ) : null}
      </div>
    </details>
  );
}
