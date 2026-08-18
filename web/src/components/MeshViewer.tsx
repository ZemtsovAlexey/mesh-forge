import { useEffect, useId, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

import type { MeshElem } from "../types";

export type MeshPick = { x: number; y: number; z: number; kind?: MeshElem; mesh?: string };

type Props = {
  url: string;
  previewUrl?: string;
  fullscreen?: boolean;
  onToggleFullscreen?: () => void;
  downloadUrl?: string;
  onReply?: () => void;
  pick?: MeshPick | null;
  onPick?: (pick: MeshPick) => void;
};

type Slot = {
  id: string;
  priority: number;
  evict: () => void;
};

const MAX_LIVE = 1;
const slots: Slot[] = [];
const waiters = new Set<() => void>();

function notifyWaiters() {
  for (const waiter of [...waiters]) waiter();
}

function acquireSlot(id: string, priority: number, evict: () => void): boolean {
  const existing = slots.find((slot) => slot.id === id);
  if (existing) {
    existing.priority = priority;
    existing.evict = evict;
    return true;
  }
  if (slots.length < MAX_LIVE) {
    slots.push({ id, priority, evict });
    return true;
  }
  let weakest = 0;
  for (let i = 1; i < slots.length; i += 1) {
    if (slots[i].priority < slots[weakest].priority) weakest = i;
  }
  if (priority <= slots[weakest].priority) return false;
  const victim = slots[weakest];
  slots.splice(weakest, 1);
  slots.push({ id, priority, evict });
  victim.evict();
  return true;
}

function releaseSlot(id: string) {
  const index = slots.findIndex((slot) => slot.id === id);
  if (index < 0) return;
  slots.splice(index, 1);
  notifyWaiters();
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh || child instanceof THREE.LineSegments)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) material.dispose();
  });
}

export default function MeshViewer({
  url,
  previewUrl,
  fullscreen,
  onToggleFullscreen,
  downloadUrl,
  onReply,
  pick,
  onPick,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const slotId = useId();
  const pickRef = useRef(pick);
  pickRef.current = pick;
  const onPickRef = useRef(onPick);
  onPickRef.current = onPick;
  const kindRef = useRef<MeshElem>(pick?.kind || "face");
  const [kind, setKind] = useState<MeshElem>(pick?.kind || "face");
  kindRef.current = kind;
  const placeRef = useRef<(norm: MeshPick | null | undefined) => void>(() => undefined);
  const [visible, setVisible] = useState(false);
  const [opened, setOpened] = useState(false);
  const [active, setActive] = useState(false);
  const [shot, setShot] = useState("");
  const [broken, setBroken] = useState(false);
  const poster = shot || (broken ? "" : previewUrl || "");
  const wantLive = Boolean(fullscreen || (opened && visible));

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const root = el.closest(".transcript");
    const io = new IntersectionObserver(
      ([entry]) => setVisible(Boolean(entry?.isIntersecting)),
      { root: root instanceof Element ? root : null, rootMargin: "40px", threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (pick?.kind) setKind(pick.kind);
  }, [pick?.kind]);

  useEffect(() => {
    setBroken(false);
    setShot("");
  }, [url, previewUrl]);

  useEffect(() => {
    if (fullscreen) setOpened(true);
  }, [fullscreen]);

  useEffect(() => {
    if (!visible && !fullscreen) setOpened(false);
  }, [visible, fullscreen]);

  useEffect(() => {
    if (!wantLive) {
      releaseSlot(slotId);
      setActive(false);
      return;
    }
    const priority = fullscreen ? 2 : 1;
    const evict = () => {
      setOpened(false);
      setActive(false);
    };
    const tryClaim = () => setActive(acquireSlot(slotId, priority, evict));
    tryClaim();
    waiters.add(tryClaim);
    return () => {
      waiters.delete(tryClaim);
      releaseSlot(slotId);
      setActive(false);
    };
  }, [wantLive, fullscreen, slotId]);

  useEffect(() => {
    const el = host.current;
    if (!el || !active) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x100f0d);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    camera.position.set(2.2, 1.6, 2.2);
    const renderer = new THREE.WebGLRenderer({
      antialias: Boolean(fullscreen),
      powerPreference: "low-power",
      alpha: false,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, fullscreen ? 2 : 1.25));
    el.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xf0e6d4, 0x2a241c, 0.7));
    const key = new THREE.DirectionalLight(0xfff4e5, 1.05);
    key.position.set(4, 6, 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x8aa4c8, 0.25);
    fill.position.set(-3, 2, -2);
    scene.add(fill);
    const grid = new THREE.GridHelper(6, 18, 0x3a342c, 0x221e18);
    scene.add(grid);
    const group = new THREE.Group();
    scene.add(group);
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xe0b07a }),
    );
    marker.visible = false;
    scene.add(marker);
    const material = new THREE.MeshStandardMaterial({
      color: 0xc4a574,
      metalness: 0.08,
      roughness: 0.52,
      side: THREE.DoubleSide,
    });
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let down: { x: number; y: number } | null = null;
    let worldBox = new THREE.Box3();
    let raf = 0;
    let frames = 0;
    let cancelled = false;

    const invalidate = (count = 2) => {
      frames = Math.max(frames, count);
      if (!raf) raf = requestAnimationFrame(tick);
    };
    const tick = () => {
      raf = 0;
      if (frames <= 0) return;
      frames -= 1;
      controls.update();
      renderer.render(scene, camera);
      if (frames > 0) raf = requestAnimationFrame(tick);
    };

    const placeMarker = (norm: MeshPick | null | undefined) => {
      if (!norm || worldBox.isEmpty()) {
        marker.visible = false;
        invalidate();
        return;
      }
      const size = worldBox.getSize(new THREE.Vector3());
      marker.position.set(
        worldBox.min.x + norm.x * size.x,
        worldBox.min.y + norm.y * size.y,
        worldBox.min.z + norm.z * size.z,
      );
      marker.scale.setScalar(Math.max(size.x, size.y, size.z, 0.001) * 0.018);
      marker.visible = true;
      invalidate();
    };
    placeRef.current = placeMarker;

    const resize = () => {
      const w = el.clientWidth || 400;
      const h = el.clientHeight || 280;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      invalidate();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    controls.addEventListener("change", () => invalidate(24));

    const fit = (object: THREE.Object3D) => {
      object.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(object);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      object.position.sub(center);
      object.position.y += size.y / 2;
      object.updateMatrixWorld(true);
      worldBox = new THREE.Box3().setFromObject(object);
      const dist = maxDim * 2.2;
      controls.target.set(0, size.y * 0.45, 0);
      camera.position.set(dist * 0.85, dist * 0.55, dist * 0.85);
      camera.near = Math.max(maxDim / 1000, 0.01);
      camera.far = Math.max(maxDim * 50, 500);
      camera.updateProjectionMatrix();
      placeMarker(pickRef.current);
    };

    const applyMaterial = (child: THREE.Mesh) => {
      const colors =
        child.geometry.getAttribute("color") ?? child.geometry.getAttribute("COLOR_0");
      if (colors) {
        if (!child.geometry.getAttribute("color")) {
          child.geometry.setAttribute("color", colors);
        }
        child.material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          color: 0xffffff,
          metalness: 0.05,
          roughness: 0.58,
          side: THREE.DoubleSide,
        });
        return;
      }
      child.material = material.clone();
    };

    const onObj = (object: THREE.Object3D) => {
      if (cancelled) {
        disposeObject(object);
        return;
      }
      object.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.computeVertexNormals();
          applyMaterial(child);
        }
      });
      group.add(object);
      fit(object);
    };
    const ext = url.split("?")[0].split(".").pop()?.toLowerCase() || "stl";
    if (ext === "glb" || ext === "gltf") {
      new GLTFLoader().load(
        url,
        (gltf) => {
          gltf.scene.updateMatrixWorld(true);
          onObj(gltf.scene);
        },
        undefined,
        () => setBroken(true),
      );
    } else if (ext === "obj") {
      new OBJLoader().load(url, onObj, undefined, () => undefined);
    } else {
      new STLLoader().load(
        url,
        (geometry) => {
          if (cancelled) {
            geometry.dispose();
            return;
          }
          geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(geometry, material.clone());
          group.add(mesh);
          fit(mesh);
        },
        undefined,
        () => undefined,
      );
    }

    const canvas = renderer.domElement;
    canvas.style.cursor = onPickRef.current ? "crosshair" : "grab";

    const onDown = (ev: PointerEvent) => {
      down = { x: ev.clientX, y: ev.clientY };
    };
    const onUp = (ev: PointerEvent) => {
      if (!down || !onPickRef.current) return;
      const dx = ev.clientX - down.x;
      const dy = ev.clientY - down.y;
      down = null;
      if (dx * dx + dy * dy > 25) return;
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(group, true);
      if (!hits.length || worldBox.isEmpty()) return;
      const p = hits[0].point;
      const size = worldBox.getSize(new THREE.Vector3());
      const nx = (p.x - worldBox.min.x) / Math.max(size.x, 1e-9);
      const ny = (p.y - worldBox.min.y) / Math.max(size.y, 1e-9);
      const nz = (p.z - worldBox.min.z) / Math.max(size.z, 1e-9);
      const next = {
        x: Math.min(1, Math.max(0, nx)),
        y: Math.min(1, Math.max(0, ny)),
        z: Math.min(1, Math.max(0, nz)),
        kind: kindRef.current,
      };
      placeMarker(next);
      onPickRef.current(next);
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);

    return () => {
      cancelled = true;
      try {
        renderer.render(scene, camera);
        const frame = renderer.domElement.toDataURL("image/jpeg", 0.72);
        if (frame && frame.length > 32) setShot(frame);
      } catch {
        /* ignore capture failures */
      }
      cancelAnimationFrame(raf);
      raf = 0;
      frames = 0;
      ro.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      controls.dispose();
      disposeObject(group);
      disposeObject(marker);
      disposeObject(grid);
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, [url, active, fullscreen]);

  useEffect(() => {
    if (!active) return;
    placeRef.current(pick);
  }, [pick, active]);

  return (
    <div className={`mesh-wrap${fullscreen ? " fullscreen" : ""}`} ref={wrap}>
      <div className={`mesh-canvas${active ? "" : " mesh-sleep"}`} ref={host}>
        {active ? null : (
          <button type="button" className="mesh-poster" onClick={() => setOpened(true)}>
            {poster ? (
              <img
                src={poster}
                alt=""
                loading="lazy"
                decoding="async"
                onError={() => setBroken(true)}
              />
            ) : (
              <span className="mesh-poster-empty">mesh</span>
            )}
            <span className="mesh-poster-hint">Клик — крутить модель</span>
          </button>
        )}
      </div>
      <div className="mesh-actions">
        {onPick ? (
          <span className="mesh-hint">
            Клик —{" "}
            {(["face", "edge", "vertex"] as MeshElem[]).map((item) => (
              <button
                key={item}
                type="button"
                className={`mesh-elem${kind === item ? " active" : ""}`}
                onClick={() => {
                  setKind(item);
                  const cur = pickRef.current;
                  if (cur) onPickRef.current?.({ ...cur, kind: item });
                }}
              >
                {item === "face" ? "грань" : item === "edge" ? "ребро" : "точка"}
              </button>
            ))}
          </span>
        ) : null}
        {onReply ? (
          <button type="button" className="btn ghost" onClick={onReply}>
            Ответить
          </button>
        ) : null}
        {downloadUrl ? (
          <a className="btn ghost" href={downloadUrl} download>
            Скачать
          </a>
        ) : null}
        {onToggleFullscreen ? (
          <button
            type="button"
            className="btn ghost"
            onClick={() => {
              setOpened(true);
              onToggleFullscreen();
            }}
          >
            {fullscreen ? "Закрыть" : "На весь экран"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
