import { useEffect, useRef, useState } from "react";
import type { ChatSummary } from "../types";

function Cube({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none" aria-hidden>
      <path d="M16 3.5 28 10.5 16 17.5 4 10.5 16 3.5Z" fill="#e0b07a" />
      <path d="M4 10.5 16 17.5v11L4 21.5v-11Z" fill="#b88955" />
      <path d="M16 17.5 28 10.5v11L16 28.5v-11Z" fill="#8f6a3e" />
    </svg>
  );
}

export default function Sidebar({
  chats,
  activeId,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onSettings,
}: {
  chats: ChatSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, title: string) => Promise<void> | void;
  onDelete: (id: string) => Promise<void> | void;
  onSettings: () => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const editingIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (editingId) inputRef.current?.select();
  }, [editingId]);

  const beginRename = (chat: ChatSummary) => {
    setPendingDelete(null);
    editingIdRef.current = chat.id;
    setEditingId(chat.id);
    setDraft(chat.title);
  };

  const cancelRename = () => {
    editingIdRef.current = null;
    setEditingId(null);
    setDraft("");
  };

  const commitRename = async () => {
    const id = editingIdRef.current;
    if (!id) return;
    editingIdRef.current = null;
    const title = draft.trim();
    const current = chats.find((c) => c.id === id);
    setEditingId(null);
    setDraft("");
    if (!title || !current || title === current.title) return;
    await onRename(id, title);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <Cube className="brand-mark" />
        <div className="brand">
          MeshForge
          <span>3D agent</span>
        </div>
        <button type="button" className="icon-btn" onClick={onSettings} title="Настройки">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
            <circle cx="8" cy="8" r="2.2" stroke="currentColor" strokeWidth="1.4" />
            <path
              d="M8 2.2v1.4M8 12.4v1.4M2.2 8h1.4M12.4 8h1.4M4 4l1 1M11 11l1 1M12 4l-1 1M5 11l-1 1"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>
      <div className="sidebar-actions">
        <button type="button" className="new-chat" onClick={onCreate}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
            <path d="M7 2v10M2 7h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          Новый чат
        </button>
      </div>
      <div className="chat-list">
        {Array.isArray(chats)
          ? chats.map((chat) => {
              const editing = chat.id === editingId;
              const confirming = chat.id === pendingDelete;
              return (
                <div
                  key={chat.id}
                  className={`chat-item${chat.id === activeId ? " active" : ""}${editing ? " editing" : ""}`}
                >
                  {editing ? (
                    <input
                      ref={inputRef}
                      className="chat-rename"
                      value={draft}
                      maxLength={120}
                      aria-label="Название чата"
                      onChange={(e) => setDraft(e.target.value)}
                      onBlur={() => void commitRename()}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          void commitRename();
                        } else if (e.key === "Escape") {
                          e.preventDefault();
                          cancelRename();
                        }
                      }}
                    />
                  ) : (
                    <button
                      type="button"
                      className="chat-item-main"
                      onClick={() => {
                        setPendingDelete(null);
                        onSelect(chat.id);
                      }}
                      onDoubleClick={() => beginRename(chat)}
                    >
                      <span className="title">{chat.title}</span>
                      {chat.has_mesh ? <span className="mesh-dot" title="есть mesh" /> : null}
                    </button>
                  )}
                  {confirming ? (
                    <div className="chat-item-confirm">
                      <span>Удалить?</span>
                      <button
                        type="button"
                        className="danger"
                        title="Удалить"
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => {
                          setPendingDelete(null);
                          void onDelete(chat.id);
                        }}
                      >
                        Да
                      </button>
                      <button
                        type="button"
                        title="Отмена"
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => setPendingDelete(null)}
                      >
                        Нет
                      </button>
                    </div>
                  ) : (
                    <div className="chat-item-actions">
                      <button
                        type="button"
                        title="Переименовать"
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => beginRename(chat)}
                      >
                        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden>
                          <path
                            d="M11.2 2.6 13.4 4.8 5.9 12.3 3.5 12.5 3.7 10.1 11.2 2.6Z"
                            stroke="currentColor"
                            strokeWidth="1.3"
                            strokeLinejoin="round"
                          />
                          <path d="M9.9 3.9 12.1 6.1" stroke="currentColor" strokeWidth="1.3" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        className="danger"
                        title="Удалить"
                        onMouseDown={(e) => e.preventDefault()}
                        onClick={() => {
                          cancelRename();
                          setPendingDelete(chat.id);
                        }}
                      >
                        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden>
                          <path d="M3.5 4.5h9M6 4.5V3h4v1.5M5.2 4.5 5.7 13h4.6l.5-8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </button>
                    </div>
                  )}
                </div>
              );
            })
          : null}
      </div>
    </aside>
  );
}
