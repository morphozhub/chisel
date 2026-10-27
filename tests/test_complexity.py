"""Static heaviness pre-check: flags transparent instanced particle clouds, passes normal models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miner_reference import complexity, pipeline  # noqa: E402
from renderer_client import RenderRejected  # noqa: E402
from test_pipeline_stub import IMAGE, JS, SEED, StubCalls, StubRenderer, choice, response  # noqa: E402

HEAVY = """export default function generate(THREE) {
  const root = new THREE.Group();
  const geo = new THREE.SphereGeometry(1, 7, 5);
  const mat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 });
  const N = 9000;
  const bubbles = new THREE.InstancedMesh(geo, mat, N);
  const d = new THREE.Object3D();
  for (let i = 0; i < N; i++) { d.position.set(i % 10, i % 7, i % 3); d.updateMatrix(); bubbles.setMatrixAt(i, d.matrix); }
  root.add(bubbles); return root;
}"""
LIGHT = HEAVY.replace("const N = 9000;", "const N = 1200;")
OPAQUE = HEAVY.replace("transparent: true, opacity: 0.4", "roughness: 0.5")


def test_flags_transparent_instance_cloud():
    assert complexity.assess(HEAVY) is not None
    assert "too_heavy_for_renderer" in complexity.rejection_json(complexity.assess(HEAVY))


def test_passes_light_and_opaque_variants():
    assert complexity.assess(LIGHT) is None
    assert complexity.assess(OPAQUE) is None  # 9000 opaque instances are fine (< 30000)


def test_passes_the_committed_library_examples():
    for js in (Path(__file__).resolve().parents[1] / "miner_reference" / "library").glob("*.js"):
        assert complexity.assess(js.read_text()) is None, js.name


def test_heavy_candidate_is_repaired_without_a_render_call():
    plan = {1: response(HEAVY), 2: response(JS + "//b"), 3: response(JS + "//c"),
            8: response(JS + "//fixed"), 4: choice("A"), 5: response(JS + "//d"), 6: response(JS + "//e"), 7: choice("A")}
    calls, renderer = StubCalls(plan), StubRenderer()
    out = pipeline.run(IMAGE, SEED, calls, renderer)
    assert out is not None
    assert (8, pipeline.SOL) in calls.log  # repair requested with the synthetic rejection
    assert all("InstancedMesh(geo, mat, N)" not in s for s in renderer.log)  # heavy source never rendered
