import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

export class MeshViewer {
  constructor(container) {
    this.container = container;
    this.wireframe = false;
    this.showGrid = true;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0c10);
    // Fog distances are updated per-mesh in _fitCamera (photo meshes are ~100–200 mm)
    this.scene.fog = new THREE.Fog(0x0a0c10, 200, 800);

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 5000);
    this.camera.position.set(2.2, 1.6, 2.2);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.target.set(0, 0.4, 0);

    const hemi = new THREE.HemisphereLight(0xc8e8ff, 0x1a2030, 0.55);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(4, 6, 3);
    key.castShadow = true;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x3dd6c6, 0.35);
    fill.position.set(-3, 2, -2);
    this.scene.add(fill);

    this.grid = new THREE.GridHelper(6, 24, 0x2a3142, 0x1a1f2a);
    this.grid.position.y = 0;
    this.scene.add(this.grid);

    const planeGeo = new THREE.PlaneGeometry(8, 8);
    const planeMat = new THREE.ShadowMaterial({ opacity: 0.15 });
    this.ground = new THREE.Mesh(planeGeo, planeMat);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);

    this.meshGroup = new THREE.Group();
    this.scene.add(this.meshGroup);

    this.material = new THREE.MeshStandardMaterial({
      color: 0x3dd6c6,
      metalness: 0.12,
      roughness: 0.5,
      flatShading: false,
      side: THREE.DoubleSide,
    });

    this._boundResize = () => this.resize();
    window.addEventListener("resize", this._boundResize);
    this._animate();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  clear() {
    while (this.meshGroup.children.length) {
      const child = this.meshGroup.children[0];
      this.meshGroup.remove(child);
      child.traverse?.((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
    }
  }

  _resizeHelpers(maxDim) {
    const gridSize = Math.max(6, maxDim * 3);
    const divisions = 24;

    this.scene.remove(this.grid);
    this.grid.geometry?.dispose?.();
    this.grid.material?.dispose?.();
    this.grid = new THREE.GridHelper(gridSize, divisions, 0x2a3142, 0x1a1f2a);
    this.grid.visible = this.showGrid;
    this.scene.add(this.grid);

    this.ground.geometry?.dispose?.();
    this.ground.geometry = new THREE.PlaneGeometry(gridSize * 1.2, gridSize * 1.2);
    this.ground.visible = this.showGrid;

    // Lights follow model scale
    this.scene.traverse((obj) => {
      if (obj.isDirectionalLight) {
        const dir = obj.position.clone().normalize();
        obj.position.copy(dir.multiplyScalar(maxDim * 4));
      }
    });
  }

  _fitCamera(object) {
    // World-space box before we re-center the object
    object.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.001);

    object.position.sub(center);
    object.position.y += size.y / 2;

    const dist = maxDim * 2.2;
    this.controls.target.set(0, size.y * 0.45, 0);
    this.camera.position.set(dist * 0.85, dist * 0.55, dist * 0.85);

    this.camera.near = Math.max(maxDim / 1000, 0.01);
    this.camera.far = Math.max(maxDim * 50, 500);
    this.camera.updateProjectionMatrix();

    if (this.scene.fog) {
      this.scene.fog.near = maxDim * 4;
      this.scene.fog.far = maxDim * 12;
    }

    this._resizeHelpers(maxDim);
    this.controls.update();
  }

  _applyMaterial(mesh) {
    mesh.traverse((child) => {
      if (child.isMesh) {
        child.material = this.material.clone();
        child.material.wireframe = this.wireframe;
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
  }

  async loadUrl(url) {
    this.clear();
    const clean = url.split("?")[0];
    // API serves STL without extension in path (/mesh) — default to STL
    const ext = clean.includes(".") ? (clean.split(".").pop()?.toLowerCase() || "stl") : "stl";

    return new Promise((resolve, reject) => {
      const onLoaded = (object) => {
        this._applyMaterial(object);
        this.meshGroup.add(object);
        this._fitCamera(object);
        resolve(object);
      };

      if (ext === "obj") {
        new OBJLoader().load(url, onLoaded, undefined, reject);
      } else {
        new STLLoader().load(
          url,
          (geometry) => {
            geometry.computeVertexNormals();
            geometry.computeBoundingSphere();
            const mesh = new THREE.Mesh(geometry, this.material.clone());
            mesh.material.side = THREE.DoubleSide;
            mesh.material.wireframe = this.wireframe;
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            this.meshGroup.add(mesh);
            this._fitCamera(mesh);
            resolve(mesh);
          },
          undefined,
          reject,
        );
      }
    });
  }

  setWireframe(on) {
    this.wireframe = on;
    this.meshGroup.traverse((child) => {
      if (child.isMesh && child.material) child.material.wireframe = on;
    });
  }

  setGridVisible(on) {
    this.showGrid = on;
    this.grid.visible = on;
    this.ground.visible = on;
  }

  resetCamera() {
    if (this.meshGroup.children.length) {
      this._fitCamera(this.meshGroup);
    } else {
      this.controls.target.set(0, 0.4, 0);
      this.camera.position.set(2.2, 1.6, 2.2);
      this.controls.update();
    }
  }

  dispose() {
    window.removeEventListener("resize", this._boundResize);
    this.clear();
    this.renderer.dispose();
  }
}
