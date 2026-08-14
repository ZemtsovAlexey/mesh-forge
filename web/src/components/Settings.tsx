import { useEffect, useState } from "react";
import { api } from "../api";
import type { LLMSettings } from "../types";

const EMPTY: LLMSettings = {
  base_url: "http://127.0.0.1:1234/v1",
  api_key: "lm-studio",
  planner_model: "",
  vision_model: "",
};

export default function Settings({ onClose }: { onClose: () => void }) {
  const [form, setForm] = useState<LLMSettings>(EMPTY);
  const [models, setModels] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [hint, setHint] = useState("");
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [openField, setOpenField] = useState<"planner" | "vision" | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getLlm()
      .then((data) => {
        if (cancelled) return;
        setForm({ ...EMPTY, ...data });
        setLoading(false);
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setError(err.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const loadModels = async () => {
    setFetching(true);
    setError("");
    setHint("Запрашиваю LM Studio…");
    try {
      const res = await api.llmModels(form.base_url, form.api_key);
      const list = res.models;
      setModels(list);
      if (!list.length) {
        setHint("");
        setError(res.status || "Модели не найдены");
        return;
      }
      setHint(`${list.length} моделей`);
      setForm((cur) => ({
        ...cur,
        planner_model: list.includes(cur.planner_model) ? cur.planner_model : list[0],
        vision_model: list.includes(cur.vision_model) ? cur.vision_model : list[0],
      }));
    } catch (err) {
      setHint("");
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setFetching(false);
    }
  };

  const save = async () => {
    try {
      await api.saveLlm(form);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="modal-back" onClick={onClose} role="presentation">
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Настройки</h3>
        <div className="modal-body">
          {loading ? <p className="modal-hint">Загрузка…</p> : null}
          {hint ? <p className="modal-hint">{hint}</p> : null}
          {error ? <p className="modal-error">{error}</p> : null}
          <label className="field">
            LM Studio URL
            <input value={form.base_url} onChange={(e) => setForm({ ...form, base_url: e.target.value })} />
          </label>
          <label className="field">
            API key
            <input value={form.api_key} onChange={(e) => setForm({ ...form, api_key: e.target.value })} />
          </label>
          <ModelField
            label="Planner / chat model"
            value={form.planner_model}
            models={models}
            open={openField === "planner"}
            onToggle={() => setOpenField((cur) => (cur === "planner" ? null : "planner"))}
            onChange={(planner_model) => {
              setForm({ ...form, planner_model });
              setOpenField(null);
            }}
          />
          <ModelField
            label="Vision model"
            value={form.vision_model}
            models={models}
            open={openField === "vision"}
            onToggle={() => setOpenField((cur) => (cur === "vision" ? null : "vision"))}
            onChange={(vision_model) => {
              setForm({ ...form, vision_model });
              setOpenField(null);
            }}
          />
        </div>
        <div className="modal-actions">
          <button type="button" className="btn ghost" onClick={loadModels} disabled={fetching || loading}>
            {fetching ? "Загрузка…" : "Модели"}
          </button>
          <span className="spacer" />
          <button type="button" className="btn ghost" onClick={onClose}>
            Отмена
          </button>
          <button type="button" className="btn primary" onClick={save} disabled={loading}>
            Сохранить
          </button>
        </div>
      </div>
    </div>
  );
}

function ModelField({
  label,
  value,
  models,
  open,
  onToggle,
  onChange,
}: {
  label: string;
  value: string;
  models: string[];
  open: boolean;
  onToggle: () => void;
  onChange: (value: string) => void;
}) {
  const options = models.length && value && !models.includes(value) ? [value, ...models] : models;
  return (
    <div className="field">
      {label}
      {options.length ? (
        <div className="picker">
          <button type="button" className="picker-value" onClick={onToggle}>
            <span>{value || "Выберите модель"}</span>
            <span className="picker-caret" aria-hidden>
              {open ? "▴" : "▾"}
            </span>
          </button>
          {open ? (
            <div className="picker-menu">
              {options.map((m) => (
                <button
                  type="button"
                  key={m}
                  className={`picker-item${m === value ? " active" : ""}`}
                  onClick={() => onChange(m)}
                >
                  {m}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ) : (
        <input value={value} onChange={(e) => onChange(e.target.value)} />
      )}
    </div>
  );
}
