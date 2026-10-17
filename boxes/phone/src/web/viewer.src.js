// The 3D viewer for `show` results (binary glTF). Bundled with three.js at
// image build time into assets/vendor/viewer.js (see docker/Dockerfile), so the
// phone never fetches a library from a third party. If the bundle is missing
// the page offers the model for download instead.
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export function mount(container, url) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.append(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x555566, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(2, 3, 4);
  scene.add(sun);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
  camera.position.set(0, 0, 3);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  function resize() {
    const w = container.clientWidth || 300, h = container.clientHeight || 300;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  new GLTFLoader().load(url, (gltf) => {
    scene.add(gltf.scene);
    gltf.scene.traverse((o) => {                       // point clouds: fixed-size dots
      if (o.isPoints && o.material) { o.material.sizeAttenuation = false; o.material.size = 2; }
    });
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const centre = box.getCenter(new THREE.Vector3());
    controls.target.copy(centre);
    camera.near = size / 1000; camera.far = size * 20;
    camera.position.copy(centre).add(new THREE.Vector3(0, size * 0.2, size * 0.9));
    camera.updateProjectionMatrix();
    controls.update();
  }, undefined, (err) => {
    container.replaceChildren(Object.assign(document.createElement('span'), { textContent: 'Could not read the 3D model: ' + (err && err.message || err) }));
  });

  renderer.setAnimationLoop(() => {
    if (!container.isConnected) { dispose(); return; }
    controls.update();
    renderer.render(scene, camera);
  });

  function dispose() {
    renderer.setAnimationLoop(null);
    ro.disconnect();
    controls.dispose();
    renderer.dispose();
  }
  return { dispose };
}
