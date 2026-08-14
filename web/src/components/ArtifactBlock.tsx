import { useState } from "react";
import type { Artifact } from "../types";
import MeshViewer from "./MeshViewer";
import Lightbox from "./Lightbox";

const LABELS: Record<string, string> = {
  front: "спереди",
  left: "слева",
  right: "справа",
  back: "сзади",
  preview: "превью",
};

function caption(art: Artifact): string {
  const key = (art.label || art.view || "").toLowerCase();
  return LABELS[key] || art.label || "";
}

export default function ArtifactBlock({ artifacts }: { artifacts: Artifact[] }) {
  const [light, setLight] = useState<string | null>(null);
  const [fullMesh, setFullMesh] = useState<Artifact | null>(null);
  const list = Array.isArray(artifacts) ? artifacts : [];
  if (!list.length) return null;
  const images = list.filter((a) => a.kind === "image" || a.kind === "mesh_preview");
  const meshes = list.filter((a) => a.kind === "mesh");
  const count = Math.min(4, Math.max(1, images.length));
  return (
    <>
      {images.length ? (
        <div className={`art-grid count-${count}`}>
          {images.map((img) => (
            <figure key={img.id}>
              <img
                src={img.url}
                alt={caption(img) || img.name}
                onClick={() => setLight(img.url)}
              />
              {caption(img) ? <figcaption>{caption(img)}</figcaption> : null}
            </figure>
          ))}
        </div>
      ) : null}
      {meshes.map((mesh) => (
        <MeshViewer
          key={mesh.id}
          url={mesh.url}
          downloadUrl={mesh.url}
          fullscreen={fullMesh?.id === mesh.id}
          onToggleFullscreen={() => setFullMesh((cur) => (cur?.id === mesh.id ? null : mesh))}
        />
      ))}
      {light ? <Lightbox src={light} onClose={() => setLight(null)} /> : null}
    </>
  );
}
