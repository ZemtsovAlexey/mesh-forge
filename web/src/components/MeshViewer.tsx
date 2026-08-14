import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";

type Props = {
  url: string;
  fullscreen?: boolean;
  onToggleFullscreen?: () => void;
  downloadUrl?: string;
};

export default function MeshViewer({ url, fullscreen, onToggleFullscreen, downloadUrl }: Props) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x100f0d);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    camera.position.set(2.2, 1.6, 2.2);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
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
    const material = new THREE.MeshStandardMaterial({
      color: 0xc4a574,
      metalness: 0.08,
      roughness: 0.52,
      side: THREE.DoubleSide,
    });

    const resize = () => {
      const w = el.clientWidth || 400;
      const h = el.clientHeight || 280;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(el);

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    const fit = (object: THREE.Object3D) => {
      object.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(object);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      object.position.sub(center);
      object.position.y += size.y / 2;
      const dist = maxDim * 2.2;
      controls.target.set(0, size.y * 0.45, 0);
      camera.position.set(dist * 0.85, dist * 0.55, dist * 0.85);
      camera.near = Math.max(maxDim / 1000, 0.01);
      camera.far = Math.max(maxDim * 50, 500);
      camera.updateProjectionMatrix();
    };

    const ext = url.split("?")[0].split(".").pop()?.toLowerCase() || "stl";
    const onObj = (object: THREE.Object3D) => {
      object.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.material = material.clone();
        }
      });
      group.add(object);
      fit(object);
    };
    if (ext === "obj") {
      new OBJLoader().load(url, onObj, undefined, () => undefined);
    } else {
      new STLLoader().load(
        url,
        (geometry) => {
          geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(geometry, material.clone());
          group.add(mesh);
          fit(mesh);
        },
        undefined,
        () => undefined,
      );
    }

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url, fullscreen]);

  return (
    <div className={`mesh-wrap${fullscreen ? " fullscreen" : ""}`}>
      <div className="mesh-canvas" ref={host} />
      <div className="mesh-actions">
        {downloadUrl ? (
          <a className="btn ghost" href={downloadUrl} download>
            Скачать
          </a>
        ) : null}
        {onToggleFullscreen ? (
          <button type="button" className="btn ghost" onClick={onToggleFullscreen}>
            {fullscreen ? "Закрыть" : "На весь экран"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
