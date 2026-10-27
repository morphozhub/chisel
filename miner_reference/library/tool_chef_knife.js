// Example: hand tool / elongated object: chef's knife with curved steel blade (extruded Shape), bolster, lathe-turned green handle with rivets and butt cap
export default function generate(THREE) {
  const root = new THREE.Group();

  const steelMat = new THREE.MeshStandardMaterial({ color: 0xd4d4d4, metalness: 0.6, roughness: 0.2 });
  const handleMat = new THREE.MeshStandardMaterial({ color: 0x3fa048, roughness: 0.6 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x222222, metalness: 0.3, roughness: 0.5 });

  // Layout along Z: tip at +Z (forward), handle extends to -Z. Blade heel at z = 0.
  const bladeLen = 0.55, bladeH = 0.065, axisY = 0.02; // handle axis height; spine sits a little above it

  // ---- blade: outline drawn as (length, height) in XY, extruded very thin, rotated so length -> +Z ----
  const blade = new THREE.Group(); blade.name = 'blade'; root.add(blade);
  const outline = new THREE.Shape();
  outline.moveTo(0, -0.01);                                           // heel, cutting edge
  outline.lineTo(0, bladeH);                                          // heel, spine
  outline.lineTo(bladeLen * 0.72, bladeH);                            // straight spine
  outline.quadraticCurveTo(bladeLen * 0.92, bladeH * 0.95, bladeLen, bladeH * 0.45); // drop to the tip
  outline.quadraticCurveTo(bladeLen * 0.55, -0.03, 0, -0.01);         // belly: edge curves down and back to the heel
  const bladeGeom = new THREE.ExtrudeGeometry(outline, { depth: 0.004, bevelEnabled: true, bevelThickness: 0.002, bevelSize: 0.002, bevelSegments: 2 });
  bladeGeom.translate(0, axisY + 0.03 - bladeH, -0.002); // spine level with the handle top, blade hangs below; thickness centered
  const bladeMesh = new THREE.Mesh(bladeGeom, steelMat);
  bladeMesh.rotation.y = -Math.PI / 2; // shape x -> +Z, extrude depth -> X
  blade.add(bladeMesh);

  // ---- handle: lathe profile (radius, distance from bolster), axis Y rotated to -Z, flattened vertically ----
  const handle = new THREE.Group(); handle.name = 'handle'; root.add(handle);
  handle.position.y = axisY;
  const gripLen = 0.4;
  const grip = new THREE.SplineCurve([
    new THREE.Vector2(0.03, 0.0),
    new THREE.Vector2(0.028, 0.06),  // finger dip behind the bolster
    new THREE.Vector2(0.036, 0.2),   // palm swell
    new THREE.Vector2(0.032, 0.33),
    new THREE.Vector2(0.027, gripLen),
  ]).getSpacedPoints(20);
  const gripProfile = grip.concat([new THREE.Vector2(0, gripLen)]);
  const gripMesh = new THREE.Mesh(new THREE.LatheGeometry(gripProfile, 24), handleMat);
  gripMesh.rotation.x = -Math.PI / 2; // lathe axis Y -> -Z
  gripMesh.scale.z = 0.8;             // after the rotation local z is world Y: slightly oval grip
  handle.add(gripMesh);
  const bolster = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.032, 0.03, 24), steelMat);
  bolster.rotation.x = Math.PI / 2;   // cylinder axis -> Z; local z is now world Y
  bolster.scale.z = 0.8;              // same oval section as the grip
  handle.add(bolster);
  const butt = new THREE.Mesh(new THREE.CylinderGeometry(0.027, 0.024, 0.02, 24), darkMat);
  butt.rotation.x = Math.PI / 2;
  butt.scale.z = 0.8;
  butt.position.z = -gripLen - 0.005;
  handle.add(butt);
  const rivetGeom = new THREE.CylinderGeometry(0.008, 0.008, 0.074, 12); // shared geometry, through the grip along X, heads just proud of the surface
  for (const z of [-0.1, -0.22, -0.34]) {
    const r = new THREE.Mesh(rivetGeom, steelMat);
    r.rotation.z = Math.PI / 2;
    r.position.set(0, 0, z);
    handle.add(r);
  }

  fitToUnitCube(THREE, root);
  return root;
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
