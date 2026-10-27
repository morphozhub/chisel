// Example: lathe vessel with handle: brushed-copper teapot with bellied body, domed lid with knob, curved spout and arched handle (lathe profile + tube spout/handle)
export default function generate(THREE) {
  const root = new THREE.Group();

  const copperMat = new THREE.MeshStandardMaterial({ color: 0xb87333, metalness: 0.6, roughness: 0.4, side: THREE.DoubleSide });
  const darkCopperMat = new THREE.MeshStandardMaterial({ color: 0x7a4a24, metalness: 0.5, roughness: 0.5 });

  // ---- body: profile as (radius, height), bottom center -> outside up -> rim -> inside down ----
  const outside = new THREE.SplineCurve([
    new THREE.Vector2(0.17, 0.0),
    new THREE.Vector2(0.25, 0.05),
    new THREE.Vector2(0.28, 0.18),   // belly, widest low on the body
    new THREE.Vector2(0.25, 0.36),
    new THREE.Vector2(0.19, 0.5),    // shoulder tapering to the neck
  ]).getSpacedPoints(26);
  const profile = [new THREE.Vector2(0, 0)].concat(outside, [
    new THREE.Vector2(0.21, 0.55),   // flared rim
    new THREE.Vector2(0.19, 0.56),
    new THREE.Vector2(0.17, 0.5),    // inner wall
    new THREE.Vector2(0.22, 0.18),
    new THREE.Vector2(0.12, 0.03),
    new THREE.Vector2(0.0, 0.03),    // hollow inner bottom
  ]);
  const body = new THREE.Mesh(new THREE.LatheGeometry(profile, 48), copperMat);
  body.name = 'body';
  root.add(body);
  const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.175, 0.16, 0.02, 48), darkCopperMat);
  foot.position.y = 0.01;
  root.add(foot);

  // ---- lid: shallow dome (second lathe) sitting in the rim, with a knob on a stalk ----
  const lid = new THREE.Group(); lid.name = 'lid'; root.add(lid);
  const lidProfile = [
    new THREE.Vector2(0.0, 0.0), new THREE.Vector2(0.17, 0.0), new THREE.Vector2(0.2, 0.025),
    new THREE.Vector2(0.17, 0.05), new THREE.Vector2(0.1, 0.085), new THREE.Vector2(0.0, 0.1),
  ];
  const lidDome = new THREE.Mesh(new THREE.LatheGeometry(lidProfile, 48), copperMat);
  lidDome.position.y = 0.55;
  lid.add(lidDome);
  const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.025, 0.04, 16), darkCopperMat);
  stalk.position.y = 0.66;
  lid.add(stalk);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.035, 20, 14), copperMat);
  knob.position.y = 0.7;
  lid.add(knob);

  // ---- spout on -X: tube on a Catmull-Rom curve, starts inside the wall, rises past the rim ----
  const spoutPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.2, 0.22, 0),     // buried in the belly
    new THREE.Vector3(-0.34, 0.3, 0),
    new THREE.Vector3(-0.44, 0.45, 0),
    new THREE.Vector3(-0.5, 0.58, 0),     // tip just above rim height
  ]);
  const spout = new THREE.Mesh(new THREE.TubeGeometry(spoutPath, 24, 0.035, 14, false), copperMat);
  spout.name = 'spout';
  root.add(spout);
  const spoutLip = new THREE.Mesh(new THREE.TorusGeometry(0.034, 0.008, 8, 24), copperMat);
  spoutLip.position.copy(spoutPath.getPoint(1));
  spoutLip.lookAt(spoutPath.getPoint(0.9)); // torus faces along the tube direction
  root.add(spoutLip);
  const spoutRoot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), copperMat);
  spoutRoot.position.set(-0.27, 0.24, 0);
  root.add(spoutRoot);

  // ---- handle on +X: arched tube from shoulder to belly, with a ferrule at each end ----
  const handle = new THREE.Group(); handle.name = 'handle'; root.add(handle);
  const handlePath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.19, 0.5, 0),      // top attachment at the shoulder
    new THREE.Vector3(0.33, 0.6, 0),
    new THREE.Vector3(0.42, 0.45, 0),     // widest point of the arch
    new THREE.Vector3(0.38, 0.25, 0),
    new THREE.Vector3(0.26, 0.15, 0),     // bottom attachment on the belly
  ]);
  handle.add(new THREE.Mesh(new THREE.TubeGeometry(handlePath, 28, 0.024, 12, false), darkCopperMat));
  const ferruleGeom = new THREE.SphereGeometry(0.036, 14, 10);
  for (const t of [0, 1]) {
    const f = new THREE.Mesh(ferruleGeom, copperMat);
    f.position.copy(handlePath.getPoint(t));
    handle.add(f);
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
  const scale = 0.95 / maxDim; // uniform: proportions are preserved
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
