# Procedural patterns (allowlisted THREE only, deterministic)

Checklist: named sub-group per part, shared geometry/material for repeats, pairs via `for (const s of [-1, 1])`, natural proportions first, `fitToUnitCube` last, faces +Z.

**fitToUnitCube (uniform scale keeps proportions)**
```js
function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root), size = new THREE.Vector3(), c = new THREE.Vector3();
  box.getSize(size); box.getCenter(c);
  const k = 0.95 / (Math.max(size.x, size.y, size.z) || 1);
  root.scale.setScalar(k); root.position.set(-c.x * k, -c.y * k, -c.z * k);
}
```

**Rounded box = rounded-rect Shape + ExtrudeGeometry with bevel** (cushions, cases, gadgets)
```js
function roundedRect(THREE, w, h, r) {
  const s = new THREE.Shape(), x = -w / 2, y = -h / 2;
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y); return s;
}
const geo = new THREE.ExtrudeGeometry(roundedRect(THREE, 0.4, 0.3, 0.05), { depth: 0.1, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.02, bevelSegments: 4 });
mesh.rotation.x = -Math.PI / 2; // shape XY -> XZ, depth -> +Y (lift by bevelThickness)
```

**Lathe profile (mugs, bottles, vases, bowls, turned legs)** — `Vector2(radius, height)` bottom to top; for hollow vessels come back down the inside; smooth with `SplineCurve`, use `DoubleSide`.
```js
const smooth = new THREE.SplineCurve([new THREE.Vector2(0.18, 0), new THREE.Vector2(0.29, 0.16), new THREE.Vector2(0.25, 0.34)]).getSpacedPoints(24);
const profile = [new THREE.Vector2(0, 0)].concat(smooth, [new THREE.Vector2(0.28, 0.39), new THREE.Vector2(0.22, 0.34), new THREE.Vector2(0, 0.03)]);
const body = new THREE.Mesh(new THREE.LatheGeometry(profile, 48), new THREE.MeshStandardMaterial({ color: 0xc46a48, roughness: 0.85, side: THREE.DoubleSide }));
```

**Spout / handle / stem = TubeGeometry on a CatmullRomCurve3** (start inside the wall to hide the joint; cap ends with spheres)
```js
const path = new THREE.CatmullRomCurve3([new THREE.Vector3(0.19, 0.5, 0), new THREE.Vector3(0.42, 0.45, 0), new THREE.Vector3(0.26, 0.15, 0)]);
root.add(new THREE.Mesh(new THREE.TubeGeometry(path, 28, 0.024, 12, false), mat));
const cap = new THREE.Mesh(new THREE.SphereGeometry(0.036, 14, 10), mat); cap.position.copy(path.getPoint(1)); root.add(cap);
```

**Ribs / lobes by deterministic vertex displacement** (pumpkin, pepper, fluted vase)
```js
const pos = geo.attributes.position, v = new THREE.Vector3();
for (let i = 0; i < pos.count; i++) { v.fromBufferAttribute(pos, i); const a = Math.atan2(v.z, v.x);
  const k = 1 + 0.08 * Math.cos(8 * a); pos.setXYZ(i, v.x * k, v.y * 0.85, v.z * k); }
geo.computeVertexNormals();
```

**Blade / flat elongated part = outline Shape (length, height) extruded thin**; `geo.translate(...)` puts the heel at the origin, `mesh.rotation.y = -Math.PI / 2` runs the length along +Z; handle = lathe with `rotation.x = -Math.PI / 2` (extends to -Z), `scale.z = 0.8` for an oval grip.

**Frame / legs / struts from cylinders** (chairs, tables, ladders, railings)
```js
function strut(THREE, parent, a, b, r, mat) {       // a, b: Vector3 endpoints
  parent.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.LineCurve3(a, b), 1, r, 8, false), mat));
}
// Splayed leg, pivot at the top: translate geometry, tilt the group
const legGeo = new THREE.CylinderGeometry(0.028, 0.038, L, 12); legGeo.translate(0, -L / 2, 0);
leg.position.set(sx * d, topY, sz * d); leg.rotation.set(-sz * splay, 0, sx * splay); // foot moves outward
```

**Repeated elements with InstancedMesh** (keys, rivets, bricks, links) — 1 draw call.
```js
const im = new THREE.InstancedMesh(geo, mat, n), d = new THREE.Object3D(), col = new THREE.Color();
for (let i = 0; i < n; i++) {
  d.position.set(x0 + (i % cols) * pitch, y, z0 + Math.floor(i / cols) * pitch); d.updateMatrix(); im.setMatrixAt(i, d.matrix);
  col.setHSL(0.0, 0.7, 0.3 + 0.05 * ((i * 7) % 5)); im.setColorAt(i, col);   // index-based variation
}
```

**Symmetric mirroring** — build paired parts once per side, never hand-copy left/right:
`for (const s of [-1, 1]) addBox(body, 0.06, 0.03, 0.01, mat, s * 0.2, 0.1, 0.45).rotation.y = s * 0.2;`

**Wheel** = torus tire + cylinder hub facing X (`tire.rotation.y = Math.PI / 2; hub.rotation.z = Math.PI / 2`), axle at y = R.

**Small procedural UV texture** (primitive/lathe/extrude geometries already have UVs)
```js
function tex(THREE, n) {
  const data = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const h = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453, f = h - Math.floor(h); // deterministic hash
    const v = f > 0.93 ? 150 : 235, i = (y * n + x) * 4; data[i] = data[i + 1] = data[i + 2] = v; data[i + 3] = 255;
  }
  const t = new THREE.DataTexture(data, n, n); t.colorSpace = THREE.SRGBColorSpace; t.needsUpdate = true; return t;
}
```
