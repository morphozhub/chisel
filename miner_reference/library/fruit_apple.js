// Example: organic/fruit form: yellow-green apple with red blush (lathe profile with top/bottom dimples, subtle ribbing, speckle DataTexture), woody stem and a leaf
export default function generate(THREE) {
  const root = new THREE.Group();

  const skinMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, map: appleSkin(THREE, 64) });
  const stemMat = new THREE.MeshStandardMaterial({ color: 0x5c4033, roughness: 0.85 });
  const leafMat = new THREE.MeshStandardMaterial({ color: 0x4f8a3a, roughness: 0.7, side: THREE.DoubleSide });

  // ---- body: profile (radius, height) from the bottom dimple up to the top dimple, smoothed by a spline ----
  const profile = new THREE.SplineCurve([
    new THREE.Vector2(0.0, 0.1),     // bottom dimple center
    new THREE.Vector2(0.14, 0.03),
    new THREE.Vector2(0.32, 0.05),
    new THREE.Vector2(0.45, 0.25),
    new THREE.Vector2(0.47, 0.5),    // widest, just below the middle
    new THREE.Vector2(0.42, 0.75),
    new THREE.Vector2(0.28, 0.9),
    new THREE.Vector2(0.12, 0.93),
    new THREE.Vector2(0.0, 0.84),    // top dimple where the stem sits
  ]).getSpacedPoints(40);
  const bodyGeom = new THREE.LatheGeometry(profile, 56);
  // Subtle 5-lobe ribbing, strongest at the top shoulder (deterministic vertex displacement).
  const pos = bodyGeom.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const a = Math.atan2(v.z, v.x);
    const k = 1 + 0.02 * Math.cos(5 * a) * Math.max(0, (v.y - 0.45) / 0.5);
    pos.setXYZ(i, v.x * k, v.y, v.z * k);
  }
  bodyGeom.computeVertexNormals();
  const body = new THREE.Mesh(bodyGeom, skinMat);
  body.name = 'body';
  root.add(body);

  // ---- stem: short curved tube from the top dimple ----
  const stemPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.8, 0),
    new THREE.Vector3(0.01, 0.95, 0.01),
    new THREE.Vector3(0.05, 1.06, 0.03),
  ]);
  const stem = new THREE.Mesh(new THREE.TubeGeometry(stemPath, 12, 0.018, 8, false), stemMat);
  stem.name = 'stem';
  root.add(stem);

  // ---- leaf: flat ShapeGeometry (two quadratic curves), hung from the stem ----
  const leafShape = new THREE.Shape();
  leafShape.moveTo(0, 0);
  leafShape.quadraticCurveTo(0.1, 0.09, 0.24, 0.0);
  leafShape.quadraticCurveTo(0.1, -0.09, 0, 0);
  const leaf = new THREE.Mesh(new THREE.ShapeGeometry(leafShape, 12), leafMat);
  leaf.name = 'leaf';
  leaf.position.set(0.04, 0.98, 0.02);
  leaf.rotation.set(-Math.PI / 2 + 0.5, 0, 0.3); // lies mostly flat, tilted up and to the side
  root.add(leaf);

  fitToUnitCube(THREE, root);
  return root;
}

// Skin texture: yellow-green base, red blush on one side (u), sparse lenticel specks. No Math.random.
function appleSkin(THREE, n) {
  const data = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const u = x / n, vv = y / n;
      const blush = Math.max(0, Math.cos((u - 0.25) * Math.PI * 2)) * (0.4 + 0.6 * vv); // one side, stronger near the top
      const h = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
      const f = h - Math.floor(h);
      const speck = f > 0.94 ? 0.75 : 1;
      const r = (235 + (205 - 235) * blush) * speck;
      const g = (215 + (70 - 215) * blush) * speck;
      const b = (70 + (45 - 70) * blush) * speck;
      const i = (y * n + x) * 4;
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, n, n);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
