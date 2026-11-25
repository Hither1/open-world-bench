"""Seeded botanical and architectural meshes shared by observations and Cycles.

Appearance uses local generators and stable identities, never simulation RNG.
Templates are cached in object coordinates; placement and interaction stay native.
"""
from functools import lru_cache
import hashlib

import numpy as np

from .meshes import MeshBatch

STONE, BARK, FOLIAGE, METAL, CLOTH, SKIN = range(1, 7)


def appearance_seed(identity):
    return int.from_bytes(hashlib.blake2s(str(identity).encode(), digest_size=4).digest(), "little")


def _tag(batch, material):
    batch.current_material = material


def _color(rgb, rng, amount=.12):
    return np.clip(np.asarray(rgb) * rng.uniform(1-amount, 1+amount), .025, .95)


def _pack(batch):
    """Repair each primitive before caching it as one compound object."""
    vertices,colors,emission,_=batch.finish()
    normals=np.concatenate(batch.normals).copy()
    vertices=vertices.copy()
    start=0
    for part in batch.verts:
        end=start+len(part)
        face=np.cross(part[:,1]-part[:,0],part[:,2]-part[:,0])
        authored=normals[start:end].mean(axis=1)
        outward=part.mean(axis=1)-part.reshape(-1,3).mean(axis=0)
        reference=np.where(np.linalg.norm(authored,axis=1,keepdims=True)>0,authored,outward)
        flip=(face*reference).sum(axis=1)<-1e-7
        vertices[start:end][flip]=part[flip][:,[0,2,1]]
        normals[start:end][flip]=normals[start:end][flip][:,[0,2,1]]
        face[flip]*=-1
        face/=np.maximum(np.linalg.norm(face,axis=1,keepdims=True),1e-8)
        missing=np.linalg.norm(normals[start:end],axis=-1,keepdims=True)==0
        normals[start:end]=np.where(missing,face[:,None,:],normals[start:end])
        start=end
    return vertices,colors,normals,np.concatenate(batch.material_ids),emission


def _leaves(batch, rng, center, extent, color, count, *, narrow=False):
    """Individual folded leaves, with gaps, asymmetry, and upward-biased normals."""
    # Leaf density must not change the random stream that defines branch shape.
    rng=np.random.default_rng(int(rng.integers(0,2**32)))
    _tag(batch, FOLIAGE)
    center, extent = np.asarray(center), np.asarray(extent)
    directions = rng.normal(size=(count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    points = center + directions * extent * rng.uniform(.25, 1, (count, 1)) ** .4
    angles = rng.uniform(0, 2*np.pi, count)
    lengths = rng.uniform(.21, .40, count) * np.linalg.norm(extent) ** .35
    axis = np.stack((np.cos(angles), rng.uniform(-.55, .6, count), np.sin(angles)), axis=1)
    axis *= lengths[:, None]
    side = np.stack((-np.sin(angles), np.zeros(count), np.cos(angles)), axis=1)
    side *= lengths[:, None] * (.18 if narrow else .46)
    fold = points + np.array([0, .04, 0])
    a, b, c, d = points-axis, points+side, points+axis, points-side
    triangles = np.stack((np.stack((a,b,fold),1), np.stack((b,c,fold),1),
                          np.stack((c,d,fold),1), np.stack((d,a,fold),1)),1).reshape(-1,3,3)
    colors = np.asarray(color) * rng.uniform(.73, 1.23, (count,1))
    normals = directions * .25 + (0, 1, 0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    batch.add(triangles, np.repeat(np.clip(colors,.02,.95),4,axis=0),
              vertex_normals=np.repeat(np.repeat(normals[:,None,:],4,axis=1).reshape(-1,1,3),3,axis=1))


def _rock(batch, rng, center, radii, color, *, detail=1):
    """Fractured, asymmetric rock; no repeated stretched sphere silhouettes."""
    _tag(batch, STONE)
    sides, rings = (9, 5) if detail else (7, 4)
    latitude = np.linspace(0, np.pi, rings+1)
    longitude = np.linspace(0, 2*np.pi, sides, endpoint=False)
    unit = np.stack(np.broadcast_arrays(np.sin(latitude[:,None])*np.cos(longitude),
        np.cos(latitude[:,None])*np.ones(sides),np.sin(latitude[:,None])*np.sin(longitude)),axis=-1)
    deformation = rng.uniform(.78,1.16,(rings+1,sides,1))
    deformation[0] = deformation[0,0]; deformation[-1] = deformation[-1,0]
    vertices = unit * deformation * radii
    vertices[...,0] += vertices[...,1]*rng.uniform(-.22,.22)
    vertices[...,1] = np.maximum(vertices[...,1], -np.asarray(radii)[1]*.68)
    vertices += center
    triangles = []
    for row in range(rings):
        for col in range(sides):
            nxt=(col+1)%sides
            if row: triangles.append([vertices[row,col],vertices[row+1,col],vertices[row,nxt]])
            if row<rings-1: triangles.append([vertices[row,nxt],vertices[row+1,col],vertices[row+1,nxt]])
    triangles=np.asarray(triangles)
    colors=np.asarray(color)*rng.uniform(.86,1.13,(len(triangles),1))
    top=triangles.mean(axis=1)[:,1] > np.asarray(center)[1]+np.asarray(radii)[1]*.28
    colors[top]=colors[top]*.65+np.array((.22,.29,.12))*.35
    batch.add(triangles,np.clip(colors,0,1))


def _tree(batch, rng, kind, detail):
    species = int(rng.integers(4))
    height = rng.uniform(5.6,7.8) if kind!="pine" else rng.uniform(7.8,10)
    lean = rng.uniform(-.42,.42,2)
    trunk = (.53,.50,.43) if species==1 and kind=="broadleaf" else (.30,.23,.16)
    leaf = ((.30,.42,.16),(.39,.48,.20),(.23,.37,.18),(.36,.43,.17))[species]
    if kind=="sakura": leaf=(.83,.64,.67)
    if kind=="pine": leaf=(.19,.32,.24)
    leaf=_color(leaf,rng)
    _tag(batch,BARK)
    joints=[np.array((lean[0]*t*t,height*t,lean[1]*t*t)) for t in (0,.22,.46,.72,1)]
    for i in range(4):
        batch.tube(joints[i],joints[i+1],(.30-i*.058),_color(trunk,rng),end_radius=.23-i*.055,sides=8)
    for angle in rng.uniform(0,2*np.pi,5):
        root=np.array((np.cos(angle)*.65,.025,np.sin(angle)*.65))
        batch.tube(root,joints[1]*.35,.055,trunk,end_radius=.18,sides=6)
    if kind=="pine":
        levels=6
        for level in range(levels):
            y=height*(.21+.64*level/max(levels-1,1))
            reach=(height-y)*.34
            for arm in range(4):
                a=arm*np.pi/2+level*1.13+rng.uniform(-.3,.3)
                start=np.array((lean[0]*y/height,y,lean[1]*y/height))
                tip=start+(np.cos(a)*reach,rng.uniform(-.25,.3),np.sin(a)*reach)
                _tag(batch,BARK)
                batch.tube(start,tip,.075,trunk,end_radius=.014,sides=6)
                for t in (.45,.86):
                    _leaves(batch,rng,start*(1-t)+tip*t,(reach*.47,.30,reach*.40),
                            leaf,52 if detail else 18,narrow=True)
        _leaves(batch,rng,joints[-1],(.33,.65,.33),leaf,40 if detail else 18,narrow=True)
        return
    count=9
    width=(2.1,1.6,2.7,2.2)[species]
    branch_seeds=rng.integers(0,2**32,size=count)
    for k in range(count):
        rng=np.random.default_rng(int(branch_seeds[k]))
        a=k*2.39996+rng.uniform(-.3,.3)
        radial=width*rng.uniform(.5,1.1)
        y=height*rng.uniform(.60,.98)
        end=np.array((np.cos(a)*radial+lean[0],y,np.sin(a)*radial+lean[1]))
        fork=joints[2]+(0,rng.uniform(-.6,.5),0)
        mid=fork*.42+end*.58+(0,.2,0)
        _tag(batch,BARK)
        batch.tube(fork,mid,.11,trunk,end_radius=.065,sides=7)
        batch.tube(mid,end,.065,trunk,end_radius=.014,sides=6)
        extent=np.array((1.1,.82,.95))*rng.uniform(.8,1.15)
        if species==2: extent[1]*=1.5
        _leaves(batch,rng,end,extent,leaf,185 if detail else 45,narrow=species==2)
        if detail:
            twig=end+rng.uniform(-.5,.5,3)+(0,.45,0)
            _tag(batch,BARK);batch.tube(mid,twig,.027,trunk,end_radius=.008,sides=5)
            _leaves(batch,rng,twig,(.55,.5,.55),leaf,45,narrow=species==2)


def _grass(batch,rng,detail):
    _tag(batch,FOLIAGE)
    for _ in range(12 if detail else 5):
        a=rng.uniform(0,2*np.pi); h=rng.uniform(.19,.60)
        base=np.array((np.cos(a)*rng.uniform(0,.18),0,np.sin(a)*rng.uniform(0,.18)))
        side=np.array((-np.sin(a),0,np.cos(a)))*rng.uniform(.012,.025)
        bend=np.array((np.cos(a),0,np.sin(a)))*rng.uniform(.10,.28)
        mid=base+bend*.32+(0,h*.58,0);tip=base+bend+(0,h,0)
        batch.add([[base-side,base+side,mid+side*.55],
                   [base-side,mid+side*.55,mid-side*.55],
                   [mid-side*.55,mid+side*.55,tip]],
                  _color((.37,.46,.19),rng,.25))


@lru_cache(maxsize=256)
def _prop_template(kind,seed,detail):
    rng=np.random.default_rng(seed)
    batch=MeshBatch()
    if kind in ("broadleaf","sakura","pine"):
        _tree(batch,rng,kind,detail)
    elif kind=="bamboo":
        for stem in range(int(rng.integers(3,6))):
            a=stem*2.4;start=np.array((np.cos(a)*.30,0,np.sin(a)*.30))
            height=rng.uniform(4.2,6.3);bend=np.array((rng.uniform(-.45,.45),height,rng.uniform(-.45,.45)))
            for section in range(7):
                p=start+bend*section/7;q=start+bend*(section+1)/7
                _tag(batch,BARK);batch.tube(p,q,.075,(.39,.46,.24),end_radius=.067,sides=7)
                batch.tube(p,p+(0,.045,0),.086,(.59,.61,.36),sides=7)
                if section>3:_leaves(batch,rng,q+(np.cos(a)*.4,0,np.sin(a)*.4),(.6,.35,.6),
                                             (.30,.43,.16),24 if detail else 9,narrow=True)
    elif kind in ("rock","boulder"):
        radius=.52 if kind=="rock" else 1.35
        radii=radius*np.array((rng.uniform(.8,1.12),rng.uniform(.5,.94),rng.uniform(.68,1.1)))
        _rock(batch,rng,(0,radii[1]*.55,0),radii,(.43,.43,.39),detail=detail)
        if detail:
            for a in rng.uniform(0,2*np.pi,3):
                _rock(batch,rng,(np.cos(a)*radius,.04,np.sin(a)*radius),(.13,.10,.17),(.38,.39,.34),detail=0)
    elif kind=="bush":
        for k in range(5):
            a=k*2.4;tip=np.array((np.cos(a)*.45,rng.uniform(.35,.8),np.sin(a)*.45))
            _tag(batch,BARK);batch.tube((0,0,0),tip,.025,(.31,.27,.16),end_radius=.008,sides=5)
            _leaves(batch,rng,tip,(.4,.35,.4),(.30,.40,.16),45 if detail else 13)
    elif kind=="grass":
        _grass(batch,rng,detail)
    elif kind=="crystal":
        _tag(batch,STONE)
        _rock(batch,rng,(0,.08,0),(.45,.18,.36),(.37,.38,.40),detail=0)
        for k in range(4):
            a=k*2.4;r=.16 if k else 0;h=rng.uniform(.6,1.8)
            _tag(batch,METAL);batch.cone((np.cos(a)*r,.12,np.sin(a)*r),rng.uniform(.10,.22),h,
                                        (.41,.68,.72),sides=6,emissive=.18)
    return _pack(batch)


def add_detailed_prop(batch,kind,pos,yaw,scale,obj_id):
    seed=appearance_seed((kind,round(float(pos[0]),3),round(float(pos[2]),3),round(yaw,3)))
    detail=int(getattr(batch,"presentation_detail",False))
    vertices,colors,normals,materials,emission=_prop_template(kind,seed,detail)
    c,s=np.cos(yaw),np.sin(yaw);rotation=np.array(((c,0,-s),(0,1,0),(s,0,c)))
    # All arrays are copied/transformed; cached templates never alias live state.
    batch.add(vertices@rotation*scale+pos,colors,obj_id,vertex_normals=normals@rotation,material=materials)
    batch.emissive[-1]=emission.copy()


def ring(batch,center,radius,color,*,vertical=False,tube=.055,segments=20,obj_id=1):
    """A modeled metal or stone ring, not a camera-facing icon."""
    center=np.asarray(center);angles=np.linspace(0,2*np.pi,segments,endpoint=False)
    for a,b in zip(angles,np.roll(angles,-1)):
        p=np.array((np.cos(a)*radius,np.sin(a)*radius,0)) if vertical else np.array((np.cos(a)*radius,0,np.sin(a)*radius))
        q=np.array((np.cos(b)*radius,np.sin(b)*radius,0)) if vertical else np.array((np.cos(b)*radius,0,np.sin(b)*radius))
        batch.tube(center+p,center+q,tube,color,sides=5,obj_id=obj_id)


def _pedestal(batch,rng,height=.45,radius=.72):
    _tag(batch,STONE)
    for tier in range(2):
        batch.cone((0,tier*height*.5,0),radius*(1-tier*.18),height*.48,
                   _color((.48,.47,.41),rng),sides=8,taper=.96)


def _foundation(batch,rng,size=1.7,tiers=2):
    """Stepped square footing: every relay part is built on prepared ground."""
    _tag(batch,STONE)
    y=-.12
    for tier in range(tiers):
        s=size*(1-.17*tier);h=.26 if tier else .34
        batch.box((0,y+h/2,0),(s,h,s),_color((.52,.50,.44),rng,.06))
        y+=h
    # Corner kerbs break the silhouette of the plain slab.
    for x in (-1,1):
        for z in (-1,1):
            batch.box((x*size*.43,.25,z*size*.43),(size*.16,.16,size*.16),_color((.47,.45,.40),rng,.06))
    return y


def _crystal(batch,base,radius,height,color,emissive,sides=6):
    """A cut crystal (bipyramid), held point-up; replaces bare glowing cones."""
    lower=height*.32
    batch.cone(np.asarray(base)+(0,lower,0),radius,-lower,color,sides=sides,emissive=emissive)
    batch.cone(np.asarray(base)+(0,lower,0),radius,height-lower,color,sides=sides,emissive=emissive)


def _cradle(batch,center,radius,height,metal,prongs=4):
    _tag(batch,METAL)
    center=np.asarray(center)
    ring(batch,center,radius*.8,metal,tube=.035,segments=16)
    for a in np.arange(prongs)*2*np.pi/prongs+np.pi/prongs:
        d=np.array((np.cos(a),0,np.sin(a)))
        batch.tube(center+d*radius*.8,center+d*radius*1.05+(0,height*.55,0),.03,metal,sides=5)
        batch.tube(center+d*radius*1.05+(0,height*.55,0),center+d*radius*.55+(0,height,0),.025,metal,sides=5)


# Working-memory fixtures (open_world_bench/working_memory_probes.py). Colours name the
# item an entry stands for; they are keyed to the entity's visual_version, never to RNG.
WM_LEVER_RGB=((.78,.16,.12),(.16,.32,.78),(.18,.58,.25),(.90,.74,.15),(.92,.91,.86))
WM_PAWL_RGB=((1.00,.45,.22),(.26,.62,.95),(.62,.90,.98),(.78,.52,1.00),(.52,.93,.74),(.93,.78,.36))


def _wm_fixture(batch,rng,used,role,version):
    wood=_color((.42,.30,.19),rng,.08);iron=(.27,.27,.29);bronze=(.45,.34,.17);stone=_color((.50,.48,.42),rng,.05)
    if role=="wm_lever":
        # A start lever: stone plinth, bronze quadrant, and an enamelled knob in the lever's colour.
        colour=WM_LEVER_RGB[(version-1)%len(WM_LEVER_RGB)]
        _tag(batch,STONE);batch.box((0,.28,0),(.62,.56,.46),stone)
        batch.box((0,.60,0),(.5,.08,.38),_color((.36,.35,.31),rng,.05))
        _tag(batch,METAL);ring(batch,(0,.70,0),.24,bronze,vertical=True,tube=.03,segments=12)
        tip=np.array((0,1.55,-.42 if used else .42))
        batch.tube((0,.66,0),tip,.045,iron,sides=6)
        _tag(batch,CLOTH);batch.ellipsoid(tip,(.11,.11,.11),colour,sides=8,rings=4)
        batch.box((0,.36,.235),(.34,.22,.02),colour)
    elif role=="wm_pawl":
        # A winch pawl: timber post, iron ratchet tooth, and an element-coloured boss.
        colour=WM_PAWL_RGB[(version-1)%len(WM_PAWL_RGB)]
        _tag(batch,BARK);batch.box((0,.55,0),(.34,1.1,.34),wood)
        batch.box((0,.06,0),(.6,.12,.6),_color((.33,.24,.15),rng,.05))
        _tag(batch,METAL)
        angle=-.9 if used else .5
        batch.tube((0,1.02,.18),(0,1.02+.5*np.cos(angle),.18+.5*np.sin(angle)),.05,iron,sides=5)
        batch.box((0,1.12,0),(.4,.06,.4),iron)
        _tag(batch,CLOTH);batch.cone((0,.78,.17),.17,.04,colour,sides=10,taper=1)
        batch.box((0,.42,0),(.36,.14,.36),colour)
        batch.ellipsoid((0,1.24,0),(.16,.16,.16),colour,sides=8,rings=4)
    elif role=="wm_station":
        # The lift's test station: plinth, timber gantry, a large gear and a steam pipe.
        _tag(batch,STONE);batch.box((0,.25,0),(2.4,.5,1.4),stone)
        _tag(batch,BARK)
        for x in (-1.0,1.0):
            batch.box((x,1.45,0),(.18,1.9,.18),wood)
        batch.box((0,2.45,0),(2.3,.16,.22),wood)
        _tag(batch,METAL)
        ring(batch,(0,1.45,0),.62,bronze,vertical=True,tube=.06,segments=18)
        for a in np.arange(8)*np.pi/4:
            batch.box((np.cos(a)*.7,1.45+np.sin(a)*.7,0),(.12,.12,.12),bronze)
        batch.tube((0,1.45,-.05),(0,1.45,-.45),.08,iron,sides=8)
        batch.tube((.9,.5,-.4),(.9,2.9,-.4),.09,iron,sides=8)
        batch.ellipsoid((.9,3.0,-.4),(.14,.1,.14),iron,sides=8,rings=3)
    elif role=="wm_winch":
        # The causeway winch: a chain drum on two trestles with a crank.
        _tag(batch,BARK)
        for x in (-1.3,1.3):
            batch.tube((x,0,-.6),(x,1.5,0),.09,wood,sides=6);batch.tube((x,0,.6),(x,1.5,0),.09,wood,sides=6)
            batch.box((x,.05,0),(.3,.1,1.5),_color((.33,.24,.15),rng,.05))
        batch.tube((-1.1,1.5,0),(1.1,1.5,0),.48,wood,sides=12)
        _tag(batch,METAL)
        for x in (-1.0,-.35,.35,1.0):
            ring(batch,(x,1.5,0),.5,iron,vertical=True,tube=.035,segments=14)
        batch.tube((-1.5,1.5,0),(1.5,1.5,0),.07,iron,sides=6)
        batch.tube((1.5,1.5,0),(1.5,1.1,.4),.05,iron,sides=5)
        for k in range(6):
            ring(batch,(0,1.0-k*.17,.5),.07,iron,tube=.022,segments=8)


def _fixture(batch,rng,kind,used,role,version):
    if role.startswith("wm_"):
        _wm_fixture(batch,rng,used,role,version)
        return
    stone=(.50,.48,.42);dark=(.36,.35,.31);metal=(.43,.32,.16);glow=(.34,.70,.72)
    lit=lambda on,off: on if used else off
    if role in ("drain","vane","core","source") or kind=="pillar":
        top=_foundation(batch,rng)
    else:
        _pedestal(batch,rng);top=.45
    if role=="drain":
        # A sunken basin with a bronze grate: reads as an outflow, not a post.
        _tag(batch,STONE);ring(batch,(0,top+.14,0),.62,stone,tube=.14)
        batch.cone((0,top,0),.52,.12,(.17,.22,.22),sides=16,taper=1)
        for a in np.arange(4)*np.pi/2+np.pi/4:
            batch.box((np.cos(a)*.62,top+.32,np.sin(a)*.62),(.16,.36,.16),_color(dark,rng,.05),a)
        _tag(batch,METAL)
        for a in np.arange(6)*np.pi/3:
            batch.tube((np.cos(a)*.18,top+.16,np.sin(a)*.18),(np.cos(a)*.55,top+.16,np.sin(a)*.55),
                       .033,metal,sides=6)
        ring(batch,(0,top+.36,0),.28,metal,tube=.035)
    elif role=="vane":
        # A bronze mast and sail wheel stepped into a stone collar.
        _tag(batch,STONE);batch.cone((0,top,0),.34,.5,_color(stone,rng),sides=8,taper=.75)
        _tag(batch,METAL);batch.tube((0,top+.45,0),(0,top+2.7,0),.09,metal,end_radius=.06,sides=10)
        ring(batch,(0,top+1.8,0),.63,metal,vertical=True,tube=.045)
        for a in np.arange(4)*np.pi/2:
            direction=np.array((np.cos(a),np.sin(a),0));side=np.array((-np.sin(a),np.cos(a),0))
            center=np.array((0,top+1.8,0))
            _tag(batch,METAL);batch.tube(center,center+direction*.88,.028,metal,sides=5)
            _tag(batch,CLOTH)
            batch.quad(center+direction*.24,center+direction*.86,center+direction*.75+side*.27+(0,0,.10),
                       center+direction*.30+side*.17,(.63,.68,.48))
        _tag(batch,METAL);batch.ellipsoid((0,top+2.78,0),(.1,.1,.1),metal)
    elif role=="core":
        # A squat drum holding an armillary gimbal around a floating crystal.
        _tag(batch,STONE)
        batch.cone((0,top,0),.52,.75,_color(stone,rng),sides=12,taper=.86)
        batch.cone((0,top+.75,0),.58,.14,_color(dark,rng),sides=12,taper=.95)
        _tag(batch,METAL)
        ring(batch,(0,top+1.55,0),.50,metal,vertical=True)
        ring(batch,(0,top+1.55,0),.56,metal)
        for a in (0,np.pi):
            batch.tube((np.cos(a)*.56,top+.9,np.sin(a)*.56),(np.cos(a)*.56,top+1.55,np.sin(a)*.56),.035,metal,sides=5)
        _crystal(batch,(0,top+1.25,0),.17,.6,glow,lit(.6,.14))
    elif role=="vent":
        _tag(batch,STONE);ring(batch,(0,top+.05,0),.64,stone,tube=.11)
        _tag(batch,METAL)
        for a in np.arange(8)*np.pi/4:
            d=np.array((np.cos(a),0,np.sin(a)))
            batch.quad(d*.15+(0,top,0),d*.59+(0,top,0),d*.57+(0,top+.15,.08),d*.18+(0,top+.07,.08),metal)
    elif kind=="waypoint":
        _tag(batch,STONE)
        for side in (-1,1):
            batch.cone((side*.53,top,0),.20,2.55,stone,sides=6,taper=.7)
            batch.tube((side*.53,top+2.4,0),(0,top+2.8,0),.14,stone,sides=8)
        _tag(batch,METAL);ring(batch,(0,top+1.75,0),.47,metal,vertical=True,tube=.06)
        _crystal(batch,(0,top+1.35,0),.2,.75,glow,lit(.35,.14))
    elif kind=="beacon":
        _tag(batch,STONE);batch.cone((0,top,0),.38,1.8,stone,sides=8,taper=.75)
        batch.box((0,top+1.86,0),(.62,.14,.62),_color(dark,rng))
        _cradle(batch,(0,top+1.95,0),.36,.55,metal,prongs=5)
        _crystal(batch,(0,top+2.0,0),.2,.7,(.94,.58,.20),lit(.8,.04))
    else:
        # Source stele: a banded, gently tapering shaft with a capital that
        # holds a cut crystal in a bronze cradle. Bronze bands count the rule version.
        height=rng.uniform(1.85,2.15)
        _tag(batch,STONE)
        batch.box((0,top+.12,0),(.82,.24,.82),_color(dark,rng,.05))
        sections=3
        for level in range(sections):
            y=top+.24+level*height/sections
            r=.33-level*.03
            batch.cone((0,y,0),r,height/sections-.06,_color(stone,rng,.06),sides=4,taper=.94)
            batch.cone((0,y+height/sections-.06,0),r*.9,.06,_color(dark,rng,.05),sides=4,taper=1)
        capital=top+.24+height
        batch.box((0,capital+.09,0),(.66,.18,.66),_color(stone,rng,.05),np.pi/4)
        batch.box((0,capital+.23,0),(.5,.1,.5),_color(dark,rng,.05),np.pi/4)
        _tag(batch,METAL)
        for level in range(version if role=="source" else 1):
            ring(batch,(0,top+.62+level*.22,0),.34,metal,tube=.055)
        _cradle(batch,(0,capital+.28,0),.24,.5,metal)
        _crystal(batch,(0,capital+.3,0),.15,.55,glow,lit(.6,.15))


@lru_cache(maxsize=256)
def _fixture_template(kind,seed,used,role,version):
    batch=MeshBatch();rng=np.random.default_rng(seed)
    _fixture(batch,rng,kind,used,role,version)
    return _pack(batch)


def add_fixture(batch,kind,pos,yaw,used,obj_id,identity,role="",version=1):
    vertices,colors,normals,materials,emission=_fixture_template(kind,appearance_seed(identity),bool(used),role,version)
    c,s=np.cos(yaw),np.sin(yaw);rotation=np.array(((c,0,-s),(0,1,0),(s,0,c)))
    batch.add(vertices@rotation+pos,colors,obj_id,vertex_normals=normals@rotation,material=materials)
    batch.emissive[-1]=emission.copy()


def _glyph(batch,local,yaw,obj_id,glyph,x,y,z):
    """One carved glyph inlaid on a chest face; each glyph has its own silhouette."""
    gold=(.80,.66,.30);dark=(.20,.19,.17)
    if glyph=="anchor":
        batch.box(local(x,y,z),(.03,.16,.02),gold,yaw,obj_id);batch.box(local(x,y+.06,z),(.10,.025,.02),gold,yaw,obj_id)
        batch.box(local(x,y-.07,z),(.12,.03,.02),gold,yaw,obj_id)
    elif glyph=="bell":
        batch.cone(local(x,y-.07,z),.07,.14,gold,sides=8,obj_id=obj_id,taper=.25)
    elif glyph=="crown":
        batch.box(local(x,y-.04,z),(.14,.05,.02),gold,yaw,obj_id)
        for dx in (-.05,0,.05):
            batch.cone(local(x+dx,y-.015,z),.025,.08,gold,sides=4,obj_id=obj_id)
    elif glyph=="fish":
        batch.ellipsoid(local(x,y,z),(.08,.04,.015),gold,yaw,obj_id,sides=7,rings=3)
        batch.cone(local(x+.08,y,z),.035,.05,gold,sides=3,obj_id=obj_id)
    elif glyph=="key":
        ring(batch,local(x-.04,y,z),.03,gold,vertical=True,tube=.012,segments=8,obj_id=obj_id)
        batch.box(local(x+.03,y,z),(.1,.02,.02),gold,yaw,obj_id);batch.box(local(x+.07,y-.025,z),(.02,.04,.02),gold,yaw,obj_id)
    elif glyph=="lantern":
        batch.box(local(x,y,z),(.07,.11,.02),gold,yaw,obj_id);batch.box(local(x,y,z-.005),(.04,.07,.02),dark,yaw,obj_id)
    elif glyph=="shell":
        for k in range(5):
            a=np.pi*(.15+.7*k/4)
            batch.tube(local(x,y-.05,z),local(x+np.cos(a)*.08,y-.05+np.sin(a)*.1,z),.012,gold,obj_id,sides=4)
    else:   # star
        batch.box(local(x,y,z),(.13,.03,.02),gold,yaw,obj_id);batch.box(local(x,y,z),(.03,.13,.02),gold,yaw,obj_id)
        batch.box(local(x,y,z),(.09,.09,.02),gold,yaw+np.pi/4,obj_id)


def _marked_chest(batch,local,yaw,obj_id,code,used):
    """A sunken chest: weed-dark bands, a mooring ring, and its three-glyph mark on the front."""
    from open_world_bench.working_memory_probes import mark_from_code
    _tag(batch,STONE)
    batch.box(local(0,.36,-.42),(.78,.30,.03),(.17,.21,.19),yaw,obj_id)
    _tag(batch,METAL)
    for k,glyph in enumerate(mark_from_code(code)):
        _glyph(batch,local,yaw,obj_id,glyph,(k-1)*.24,.36,-.44)
    ring(batch,local(0,.12,-.5),.09,(.30,.28,.23),vertical=True,tube=.02,segments=10,obj_id=obj_id)
    _tag(batch,FOLIAGE)
    for x in (-.32,.05,.3):
        batch.tube(local(x,.0,.4),local(x+.05,.55,.45),.03,(.20,.33,.20),obj_id,end_radius=.01,sides=4)


def add_detailed_entity(batch,kind,pos,yaw,used,obj_id,bob=0,*,identity=None,visual_role="",visual_version=1):
    from .meshes import _humanoid
    from .style import ENTITY_ALBEDO
    if visual_role.startswith("campaign_"):
        from .campaign_assets import add_campaign_fixture
        add_campaign_fixture(batch,visual_role.removeprefix("campaign_"),pos,yaw,used,
                             visual_version,identity,obj_id)
        return
    p=np.asarray(pos,dtype=np.float32)
    identity=identity if identity is not None else (kind,tuple(np.round(p,3)))
    rng=np.random.default_rng(appearance_seed(identity))
    if kind in ("pillar","waypoint","beacon") or visual_role in ("wm_lever","wm_pawl","wm_station","wm_winch"):
        add_fixture(batch,kind,p,yaw,used,obj_id,identity,visual_role,visual_version)
        return
    c,s=np.cos(yaw),np.sin(yaw)
    def local(x,y,z):
        return p+np.array((c*x-s*z,y,s*x+c*z))
    if kind=="npc":
        clothes=((.40,.47,.31),(.40,.34,.26),(.27,.39,.46),(.56,.38,.29),(.57,.54,.40))
        hairs=((.16,.12,.09),(.36,.24,.15),(.62,.54,.38),(.26,.27,.25))
        skins=((.75,.58,.43),(.58,.40,.28),(.86,.70,.54),(.43,.29,.21))
        style=int(rng.integers(4));size=float(rng.uniform(.93,1.08))
        cloth=clothes[int(rng.integers(len(clothes)))]
        _humanoid(batch,p,yaw,obj_id,tunic=cloth,hair=hairs[int(rng.integers(4))],phase=0,
                  scale=size,skin=skins[int(rng.integers(4))],hair_style=style)
        _tag(batch,BARK)
        if visual_role=="wm_boatman":
            # The boatman leans on a long oar and wears a wide straw hat.
            batch.tube(local(.42,0,-.05),local(.42,2.2,-.05),.035,(.42,.30,.18),obj_id,sides=6)
            batch.box(local(.42,.25,-.05),(.04,.45,.17),(.42,.30,.18),yaw,obj_id)
            _tag(batch,CLOTH)
            batch.cone(local(0,1.78*size,0),.42*size,.18*size,(.76,.66,.40),sides=12,obj_id=obj_id,taper=.2)
            _tag(batch,BARK)
        if style%2:
            batch.ellipsoid(local(.24,.87,.04),(.13,.17,.08),(.32,.23,.14),yaw,obj_id)
            batch.tube(local(-.17,1.42,.15),local(.24,.84,.12),.025,(.25,.18,.11),obj_id,sides=5)
        else:
            batch.tube(local(.4,.1,-.1),local(.4,1.6,-.1),.032,(.32,.25,.15),obj_id,sides=6)
    elif kind=="chest":
        wood=np.array((.40,.29,.18))*rng.uniform(.88,1.14)
        _tag(batch,BARK)
        for i in range(5):
            batch.box(local((i-2)*.21,.35,0),(.20,.65,.73),wood*rng.uniform(.9,1.1),yaw,obj_id)
        lid_y,lid_z=(.79,0) if not used else (1.0,-.35)
        for i in range(5):
            batch.box(local((i-2)*.21,lid_y,lid_z),(.20,.20,.76),wood*1.12,yaw,obj_id)
        _tag(batch,METAL)
        for x in (-.38,.38):
            for z in (-.39,.39):
                batch.box(local(x,.42,z),(.07,.73,.035),(.30,.28,.23),yaw,obj_id)
            batch.box(local(x,lid_y+.12,lid_z),(.075,.035,.78),(.30,.28,.23),yaw,obj_id)
        batch.box(local(0,.57,-.405),(.16,.21,.05),(.52,.42,.23),yaw,obj_id)
        for x in (-.38,.38):
            for y in (.18,.60):
                batch.ellipsoid(local(x,y,-.415),(.026,.026,.018),(.54,.49,.34),yaw,obj_id,sides=5,rings=3)
        if visual_role=="wm_marked":
            _marked_chest(batch,local,yaw,obj_id,visual_version,used)
    elif kind in ("enemy","boss"):
        size=(1.0 if kind=="enemy" else 1.7)*rng.uniform(.96,1.04)
        cloth=np.asarray(ENTITY_ALBEDO[kind])*rng.uniform(.79,1.07)
        _humanoid(batch,p,yaw,obj_id,tunic=cloth,hair=(.18,.14,.12),phase=bob*1.3,
                  scale=size,hair_style=int(rng.integers(4)))
        _tag(batch,METAL)
        for side in (-1,1):
            batch.ellipsoid(local(side*.25*size,1.37*size,0),(.15*size,.11*size,.18*size),
                            (.40,.39,.33),yaw,obj_id,sides=7,rings=3)
    elif kind=="altar":
        _tag(batch,STONE)
        for tier in range(3):
            extent=2.2-tier*.32
            batch.box(local(0,.16+tier*.29,0),(extent,.27,extent),np.array((.49,.48,.41))*rng.uniform(.92,1.08),yaw,obj_id)
        _tag(batch,METAL)
        ring(batch,local(0,.91,0),.53,(.45,.36,.19),tube=.04,obj_id=obj_id)
        batch.cone(local(0,.91,0),.23,.50,(.39,.66,.67),sides=6,obj_id=obj_id,emissive=.45 if used else .1)
    elif kind=="collectible":
        _tag(batch,STONE)
        for k in range(3):
            a=k*2.4;r=.12 if k else 0
            batch.cone(local(np.cos(a)*r,.08+.03*np.sin(bob*2),np.sin(a)*r),.11,
                       .32+rng.uniform(0,.14),(.54,.73,.72),sides=6,obj_id=obj_id,emissive=.12)
    _tag(batch,0)
