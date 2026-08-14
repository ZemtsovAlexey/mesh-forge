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
  onSettings,
}: {
  chats: ChatSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onSettings: () => void;
}) {
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
          ? chats.map((chat) => (
              <button
                type="button"
                key={chat.id}
                className={`chat-item${chat.id === activeId ? " active" : ""}`}
                onClick={() => onSelect(chat.id)}
              >
                <span className="title">{chat.title}</span>
                {chat.has_mesh ? <span className="mesh-dot" title="есть mesh" /> : null}
              </button>
            ))
          : null}
      </div>
    </aside>
  );
}
