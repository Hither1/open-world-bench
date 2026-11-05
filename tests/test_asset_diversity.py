"""Stable appearance variation, semantic cues, and material export contracts."""
import hashlib
import json

import numpy as np
import pytest

from open_world_bench.render.meshes import MeshBatch, add_prop, add_entity, MATERIAL_NAMES
from owb.render.scene_export import SceneSnapshot, _add_batch, _presentation_batch


def fingerprint(batch):
    return hashlib.sha256(b"".join(array.tobytes() for array in batch.finish())).hexdigest()


@pytest.mark.parametrize("kind",["pine","broadleaf","sakura","bamboo","rock","boulder","bush","grass","crystal"])
def test_natural_asset_variants_are_finite_stable_and_distinct(kind):
    meshes=[]
    for x in (3.,3.,9.):
        batch=_presentation_batch()
        add_prop(batch,kind,np.array((x,2.,7.)),.4,1.2)
        vertices,colors,emission,ids=batch.finish()
        assert len(vertices)>0 and np.isfinite(vertices).all()
        assert np.isfinite(colors).all() and np.all((colors>=0)&(colors<=1))
        assert len(np.concatenate(batch.material_ids))==len(vertices)
        # Remove placement to compare actual shape variation.
        batch.verts=[v-np.array((x,2.,7.)) for v in batch.verts]
        meshes.append(fingerprint(batch))
    assert meshes[0]==meshes[1]
    assert meshes[0]!=meshes[2]


def test_prop_templates_are_not_mutated_by_placement_or_scaling():
    location=np.array((4.,0.,8.))
    batches=[]
    for scale in (1.,2.,1.):
        b=MeshBatch();add_prop(b,"broadleaf",location,.1,scale);batches.append(b)
    assert fingerprint(batches[0])==fingerprint(batches[2])
    np.testing.assert_allclose(batches[1].finish()[0]-location,2*(batches[0].finish()[0]-location),atol=2e-6)


def test_entity_identity_is_stable_when_it_moves():
    batches=[]
    for pos in (np.zeros(3),np.array((10,0,5)),np.zeros(3)):
        b=MeshBatch();add_entity(b,"npc",pos,0,False,100,identity=(7,"merchant-4"))
        b.verts=[v-pos for v in b.verts];batches.append(b)
    np.testing.assert_allclose(batches[0].finish()[0],batches[1].finish()[0],atol=1e-6)
    assert fingerprint(batches[0])==fingerprint(batches[2])
    other=MeshBatch();add_entity(other,"npc",np.zeros(3),0,False,100,identity=(7,"merchant-9"))
    assert fingerprint(other)!=fingerprint(batches[0])


def test_relay_components_and_rule_versions_have_distinct_geometry():
    seen=set()
    counts={}
    for role,version in [("source",1),("source",2),("drain",1),("vane",1),("core",1),("vent",1)]:
        b=MeshBatch()
        add_entity(b,"pillar",np.zeros(3),0,False,100,identity="same-site",
                   visual_role=role,visual_version=version)
        seen.add(fingerprint(b));counts[role,version]=len(b.finish()[0])
    assert len(seen)==6
    assert counts["source",2]>counts["source",1]


def test_assets_do_not_consume_global_random_state():
    np.random.seed(924)
    before=np.random.get_state()
    b=MeshBatch();add_prop(b,"sakura",np.zeros(3),0,1)
    add_entity(b,"npc",np.zeros(3),0,False,100,identity="resident")
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])


def test_material_groups_survive_snapshot_export(tmp_path):
    batch=_presentation_batch()
    add_prop(batch,"broadleaf",np.zeros(3),0,1)
    add_entity(batch,"pillar",np.array((4.,0.,0.)),0,False,100,identity="relay",visual_role="source")
    scene=SceneSnapshot({"version":1})
    _add_batch(scene,batch)
    assert {"stone","bark","foliage","metal"} <= {m["material"] for m in scene.meshes}
    assert sum(len(m["triangles"]) for m in scene.meshes)==len(batch.finish()[0])
    for mesh in scene.meshes:
        assert mesh["material"] in MATERIAL_NAMES
        assert np.isfinite(mesh["normals"]).all()
        normals=mesh["normals"]
        face=np.cross(mesh["triangles"][:,1]-mesh["triangles"][:,0],mesh["triangles"][:,2]-mesh["triangles"][:,0])
        assert np.all((face*normals.mean(axis=1)).sum(axis=1)>=-1e-5)
    path=scene.save(tmp_path/"scene.npz")
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data["metadata"]))
        assert {m["material"] for m in metadata["meshes"]}=={m["material"] for m in scene.meshes}
