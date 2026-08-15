import { useState } from "react";
import type { Artifact } from "../types";
import { artifactCaption } from "../reply";
import MeshViewer from "./MeshViewer";
import Lightbox from "./Lightbox";

export default function ArtifactBlock({
  artifacts,
  onReply,
}: {
  artifacts: Artifact[];
  onReply?: (art: Artifact) => void;
}) {
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
                alt={artifactCaption(img) || img.name}
                onClick={() => setLight(img.url)}
              />
              {onReply ? (
                <button
                  type="button"
                  className="art-reply"
                  onClick={(e) => {
                    e.stopPropagation();
                    onReply(img);
                  }}
                >
                  ответить
                </button>
              ) : null}
              {artifactCaption(img) ? <figcaption>{artifactCaption(img)}</figcaption> : null}
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
          onReply={onReply ? () => onReply(mesh) : undefined}
        />
      ))}
      {light ? <Lightbox src={light} onClose={() => setLight(null)} /> : null}
    </>
  );
}
