// Example: furniture/stool: square upholstered stool with rounded cushion, four splayed tapered wooden legs, apron and foot stretchers, brass foot caps
export default function generate(THREE) {
  const root = new THREE.Group();

  const fabricMat = new THREE.MeshStandardMaterial({ color: 0xb8a08e, roughness: 0.95 });
  const woodMat = new THREE.MeshStandardMaterial({ color: 0x8b6f47, roughness: 0.6 });
  const brassMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.6, roughness: 0.25 });

  // Proportions in natural units; fitToUnitCube rescales uniformly at the end.
  const seatW = 0.34, seatH = 0.05, cushionH = 0.05;
  const legH = 0.42, legIn = 0.13, splay = 0.08;  // legIn = half spacing of leg tops, splay = outward tilt (rad)
  const legTopY = legH, stretcherY = 0.14, railT = 0.028;

  // ---- seat: flat base slab + rounded cushion (rounded-rect Shape extruded with bevel) ----
  const seat = new THREE.Group(); seat.name = 'seat'; root.add(seat);
  const base = new THREE.Mesh(new THREE.BoxGeometry(seatW * 0.94, seatH, seatW * 0.94), fabricMat);
  base.position.y = legTopY + seatH / 2;
  seat.add(base);
  const cushionGeom = new THREE.ExtrudeGeometry(roundedRect(THREE, seatW, seatW, 0.06), {
    depth: cushionH, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.02, bevelSegments: 5,
  });
  const cushion = new THREE.Mesh(cushionGeom, fabricMat);
  cushion.rotation.x = -Math.PI / 2;          // shape XY -> XZ, extrude depth -> +Y
  cushion.position.y = legTopY + seatH + 0.02; // lift by bevelThickness so it rests on the base
  seat.add(cushion);

  // ---- legs: one Group per leg, pivot at the top, tilted so the foot moves outward ----
  const legs = new THREE.Group(); legs.name = 'legs'; root.add(legs);
  const legGeom = new THREE.CylinderGeometry(0.028, 0.038, legH, 12); // tapered: thin at top
  legGeom.translate(0, -legH / 2, 0);                                  // top of leg at group origin
  const capGeom = new THREE.CylinderGeometry(0.04, 0.036, 0.02, 16);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const leg = new THREE.Group();
    leg.position.set(sx * legIn, legTopY, sz * legIn);
    leg.rotation.set(-sz * splay, 0, sx * splay); // +z rotation moves the foot toward +x, -x rotation toward +z
    leg.add(new THREE.Mesh(legGeom, woodMat));
    const cap = new THREE.Mesh(capGeom, brassMat);
    cap.position.y = -legH + 0.01;
    leg.add(cap);
    legs.add(leg);
  }

  // ---- rails: apron right under the seat and a stretcher ring near the floor ----
  const frame = new THREE.Group(); frame.name = 'rails'; root.add(frame);
  function railRing(y, half, t, h) {
    const geo = new THREE.BoxGeometry(half * 2 + t, h, t);
    for (const [x, z, ry] of [[0, half, 0], [0, -half, 0], [half, 0, Math.PI / 2], [-half, 0, Math.PI / 2]]) {
      const rail = new THREE.Mesh(geo, woodMat);
      rail.rotation.y = ry;
      rail.position.set(x, y, z);
      frame.add(rail);
    }
  }
  railRing(legTopY - 0.03, legIn, railT, 0.05);
  railRing(stretcherY, legIn + (legTopY - stretcherY) * Math.tan(splay), railT, railT); // follow the leg splay

  fitToUnitCube(THREE, root);
  return root;
}

// Rounded rectangle centered on the origin, for ExtrudeGeometry / ShapeGeometry.
function roundedRect(THREE, w, h, r) {
  const s = new THREE.Shape();
  const x = -w / 2, y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
  return s;
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
