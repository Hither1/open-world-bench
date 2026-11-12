"""Original campaign fixtures, shared by observations and Cycles export."""
from functools import lru_cache
import numpy as np
from .meshes import MeshBatch
from .assets import _pack, _rock, appearance_seed, ring, STONE, METAL, BARK


@lru_cache(maxsize=256)
def _fixture(role, seed, used, version):
    b = MeshBatch(); rng = np.random.default_rng(seed)
    b.current_material = STONE
    stone = (.46,.45,.40)
    if role in ('ice','quartz'):
        _rock(b,rng,(0,.13,0),(1.05,.35,.9),stone,detail=1)
        color = (.26,.60,.76) if role=='ice' else (.76,.17,.10)
        for i in range(5):
            angle = i*2.4
            radius = .18 if i==0 else rng.uniform(.35,.7)
            height = rng.uniform(2.1,3.9) if role=='ice' else rng.uniform(.8,1.6)
            if used and role=='ice': height *= .25
            p = (np.cos(angle)*radius,.2,np.sin(angle)*radius)
            b.current_material = METAL
            b.cone(p,.32 if role=='ice' else .22,height,color,sides=6,taper=.12,
                   emissive=.35 if role=='quartz' or used else .06)
        if role=='ice' and version>1:
            ring(b,(0,.8,0),.85,(.78,.91,.95),tube=.04)
    elif role=='campfire':
        for a in np.arange(8)*np.pi/4:
            _rock(b,rng,(np.cos(a)*.7,.15,np.sin(a)*.7),(.28,.22,.25),stone,detail=0)
        b.current_material = BARK
        for a in (.2,1.7,3.4):
            p = np.array((np.cos(a)*.5,.3,np.sin(a)*.5))
            b.tube(p,-p+(0,.6,0),.10,(.24,.13,.07),sides=7)
        b.current_material = METAL
        for i in range(3):
            b.cone((.18*(i-1),.38,0),.22,.65+.2*(i%2),(.95,.35,.06),sides=7,emissive=1.2)
    elif role in ('lens','slate','mural'):
        b.box((0,.28,0),(1.35,.55,1),stone)
        b.box((0,1.05,0),(.9,1.1,.25),(.60,.56,.44))
        b.current_material = METAL
        ring(b,(0,1.25,-.18),.28,(.76,.56,.22),vertical=True,tube=.05)
        if role=='lens':
            b.ellipsoid((0,1.25,-.2),(.21,.21,.055),(.21,.52,.56))
        else:
            for i in range(3):
                b.box((-.27+i*.27,.8,-.16),(.10,.15,.04),(.23,.46,.50),emissive=.18 if used else 0)
    elif role in ('seal','spire'):
        b.cone((0,0,0),1.1,.4,stone,sides=12,taper=.9)
        b.cone((0,.4,0),.55,2.2 if role=='spire' else .8,stone,sides=8,taper=.75)
        y = 2.9 if role=='spire' else 1.4
        b.current_material = METAL
        ring(b,(0,y,0),.65,(.70,.52,.24),vertical=True,tube=.09)
        b.cone((0,y-.35,0),.24,.85,(.25,.62,.66),sides=6,emissive=.8 if used else .15)
    else:
        raise ValueError('unknown campaign fixture: '+role)
    return _pack(b)


def add_campaign_fixture(batch, role, pos, yaw, used, version, identity, obj_id):
    vertices,colors,normals,materials,emission = _fixture(role,appearance_seed(identity),bool(used),version)
    c,s = np.cos(yaw),np.sin(yaw)
    rotation = np.array(((c,0,-s),(0,1,0),(s,0,c)))
    batch.add(vertices@rotation+pos,colors,obj_id,vertex_normals=normals@rotation,material=materials)
    batch.emissive[-1] = emission.copy()


def add_campaign_seals(batch, world):
    if not hasattr(world,'campaign'): return
    for i,gate in enumerate(world.architecture.active()):
        center = np.asarray(gate['position'])+(0,gate['size'][1]/2,0)
        batch.current_material = METAL
        batch.box(center,gate['size'],(.19,.40,.51),obj_id=30000+i,emissive=.12)
        for x in (-2,-1,0,1,2):
            batch.box(center+(x,0,-.3),(.055,gate['size'][1]-.25,.06),(.55,.84,.84),
                      obj_id=30000+i,emissive=.4)
