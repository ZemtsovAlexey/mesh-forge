import { useState } from "react";
import type { Artifact } from "../types";
import type { MeshPick } from "./MeshViewer";
import { artifactCaption } from "../reply";
import MeshViewer from "./MeshViewer";
import Lightbox from "./Lightbox";

function meshPosterUrl(url: string): string {
  const [path, query] = url.split("?");
  return query ? `${path}/preview?${query}` : `${path}/preview`;
}

export default function ArtifactBlock({
  artifacts,
  onReply,
  pick,
  onPick,
  onViewportAim,
}: {
  artifacts: Artifact[];
  onReply?: (art: Artifact) => void;
  pick?: MeshPick | null;
  onPick?: (pick: MeshPick) => void;
  onViewportAim?: (aim: { x: number; y: number; views: string }) => void;
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
                className={onViewportAim ? "art-aimable" : undefined}
                onClick={(e) => {
                  if (onViewportAim) {
                    const rect = e.currentTarget.getBoundingClientRect();
                    onViewportAim({
                      x: (e.clientX - rect.left) / Math.max(rect.width, 1),
                      y: (e.clientY - rect.top) / Math.max(rect.height, 1),
                      views: img.view || img.label || "",
                    });
                    return;
                  }
                  setLight(img.url);
                }}
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
        <figure key={mesh.id} className="mesh-artifact">
          <MeshViewer
            url={mesh.url}
            previewUrl={meshPosterUrl(mesh.url)}
            downloadUrl={mesh.url}
            fullscreen={fullMesh?.id === mesh.id}
            onToggleFullscreen={() => setFullMesh((cur) => (cur?.id === mesh.id ? null : mesh))}
            onReply={onReply ? () => onReply(mesh) : undefined}
            pick={
              pick && (!pick.mesh || pick.mesh === mesh.id || pick.mesh === mesh.name) ? pick : null
            }
            onPick={
              onPick
                ? (p) => onPick({ ...p, mesh: mesh.name || mesh.id })
                : undefined
            }
          />
          {artifactCaption(mesh) ? <figcaption>{artifactCaption(mesh)}</figcaption> : null}
        </figure>
      ))}
      {light ? <Lightbox src={light} onClose={() => setLight(null)} /> : null}
    </>
  );
}
