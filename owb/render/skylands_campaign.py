"""Full-world campaign export: exact collision surfaces, stable botanical assets.

Coplanar voxel faces are merged losslessly. No spatial culling, LOD, or terrain
replacement is used. Static meshes and instanced ground cover are exported once;
only characters, mechanism state, gates, camera and lighting vary during replay.
"""
from __future__ import annotations
import numpy as np
from .scene_export import SceneSnapshot,_add_batch,_presentation_batch
from ..sims.voxel.render import models,palette as P
from ..sims.voxel.render.landscape import _NATURAL,_MODELED
from ..sims.voxel.core import blocks as B,noise as N
from ..sims.voxel.world.biomes import BIOMES


def rectangles(labels):
    """Disjoint constant-label rectangles, preserving every occupied pixel."""
    labels=labels.copy(); height,width=labels.shape
    for u in range(height):
        while True:
            occupied=np.flatnonzero(labels[u])
            if not len(occupied): break
            v=int(occupied[0]); value=labels[u,v]
            end=v+1
            while end<width and labels[u,end]==value: end+=1
            bottom=u+1
            while bottom<height and np.all(labels[bottom,v:end]==value): bottom+=1
            labels[u:bottom,v:end]=0
            yield u,v,bottom,end,int(value)


def exact_surfaces(vox):
    """Merge only coplanar, same-material exposed faces; retain exact bounds."""
    structural=~_MODELED[vox] & (vox!=B.AIR)
    opaque=structural | (vox==B.WATER)
    triangles=[]; blocks=[]; water_tri=[]; natural_tri=[]; natural_blocks=[]
    face_area=0
    for axis in range(3):
        other=[a for a in range(3) if a!=axis]
        for index in range(vox.shape[axis]):
            here=np.take(vox,index,axis=axis)
            own=np.take(opaque,index,axis=axis)
            for sign in (-1,1):
                neighbor=index+sign
                if 0<=neighbor<vox.shape[axis]:
                    nxt=np.take(opaque,neighbor,axis=axis)
                    # Water does not hide the shoreline from underwater views.
                    visible=own & (~nxt | ((here!=B.WATER)&(np.take(vox,neighbor,axis=axis)==B.WATER)))
                else: visible=own
                labels=np.where(visible,here,0)
                face_area+=int(visible.sum())
                for u,v,uu,vv,block in rectangles(labels):
                    q=np.zeros((4,3),np.float32); q[:,axis]=index+(sign==1)
                    q[:,other[0]]=(u,uu,uu,u); q[:,other[1]]=(v,v,vv,vv)
                    normal=np.cross(q[1]-q[0],q[2]-q[0])
                    if normal[axis]*sign<0: q=q[[0,3,2,1]]
                    tri=q[[[0,1,2],[0,2,3]]]
                    if block==B.WATER: water_tri.append(tri)
                    elif _NATURAL[block]: natural_tri.append(tri); natural_blocks.extend([block,block])
                    else: triangles.append(tri); blocks.extend([block,block])
    cat=lambda x: np.concatenate(x) if x else np.empty((0,3,3),np.float32)
    return cat(natural_tri),np.array(natural_blocks),cat(triangles),np.array(blocks),cat(water_tri),face_area


def safe_camera(env):
    """Keep the traveler in view, including after vertical gallery transitions."""
    target=env.agent.pos.astype(float)+np.array((0.,1.,0.))
    rise=7. if target[1]<40 else 10.
    offset=np.array((-12.,rise,-15.));best=None
    for angle in (0.,np.pi/2,-np.pi/2,np.pi,np.pi/4,-np.pi/4,3*np.pi/4):
        c,t=np.cos(angle),np.sin(angle)
        delta=np.array((offset[0]*c-offset[2]*t,rise,offset[0]*t+offset[2]*c))
        distance=np.linalg.norm(delta);direction=delta/distance
        hit=env.world.raycast(target,direction,distance)
        clear=max(.6,hit.dist-.6) if hit.hit else distance
        candidate=(clear,target+direction*clear)
        if best is None or clear>best[0]:best=candidate
        if clear>=distance*.9:break
    return dict(eye=best[1].tolist(),target=target.tolist(),fov_y=46.)


def camera_obscures_traveler(env,camera):
    head=env.agent.pos.astype(float)+np.array((0.,1.,0.))
    eye=np.asarray(camera['eye']);target=np.asarray(camera['target'])
    delta=eye-head;distance=np.linalg.norm(delta)
    hit=env.world.raycast(head,delta/max(distance,1e-6),distance)
    if hit.hit and hit.dist<distance-.8:return True
    forward=target-eye;forward/=max(np.linalg.norm(forward),1e-6)
    right=np.cross(forward,(0.,1.,0.));right/=max(np.linalg.norm(right),1e-6)
    up=np.cross(right,forward);relative=head-eye;depth=np.dot(relative,forward)
    if depth<=.5:return True
    halfheight=depth*np.tan(np.radians(camera['fov_y'])/2)
    return abs(np.dot(relative,up))>halfheight*.75 or abs(np.dot(relative,right))>halfheight*16/9*.8


def metadata(env,width=1280,height=720):
    camera=safe_camera(env)
    keys=P.sky_colors(env.elements.climate.time,env.elements.climate.weather)
    lights=env.presentation_lights()+[dict(pos=list(pos),color=[.28,.75,.85],power=500.,radius=1.3) for pos in env.world.meta['crystal_lights']]
    return dict(version=1,environment=getattr(env,'presentation_kind','skylands_campaign'),seed=env._seed,
        reference_task=getattr(env,'objective',env.task),
        time_of_day=float(env.elements.climate.time),coordinates='Y-up, metres',size=[width,height],
        camera=camera,
        lighting=dict(direction=keys['sun']['dir'].tolist(),color=keys['sun']['color'].tolist(),
            strength=keys['sun']['intensity'],sky=keys['zenith'].tolist(),horizon=keys['horizon'].tolist()),point_lights=lights)


def static_scene(env):
    w=env.world; vox=w.vox.copy()
    for obj in [*env.movers.values(),*env.rafts.values()]:
        for cell,value in obj['under'].items(): vox[cell]=value
    dynamic_bounds=[env.vault_gate] if env.task=='a' else [env.closure_bounds()]
    for lo,hi in dynamic_bounds: vox[tuple(slice(a,b+1) for a,b in zip(lo,hi))]=B.AIR
    scene=SceneSnapshot(metadata(env)); natural,nb,architecture,ab,water,area=exact_surfaces(vox)
    scene.add('Exact regional terrain',natural,P.COLOR_F[nb],material='terrain')
    scene.add('Exact architecture and regional trails',architecture,P.COLOR_F[ab],material='stone')
    scene.add('All waterways',water,(.16,.61,.66),water=True)
    scene.metadata['geometry_contract']=dict(method='lossless coplanar voxel face merging',exposed_face_area=area,
        world_shape=list(vox.shape),spatial_culling=False,tree_detail=2,ground_cover_repeats=4)
    batch=models.ModelBatch(); logs=vox==B.LOG; below=np.zeros_like(logs); below[:,1:]=logs[:,:-1]
    trees=[]
    for x,y,z in np.argwhere(logs & ~below):
        top=int(y)
        while top<w.size_y and logs[x,top,z]: top+=1
        if top-y<2:
            batch.box((x+.5,y+.5,z+.5),(1,1,1),P.COLOR_F[B.LOG]);continue
        kind=BIOMES[int(w.biome[x,z])].tree_kind
        if y>env.levels[1]+6: kind='golden'
        leaves=vox[max(0,x-3):x+4,max(y,top-4):top+4,max(0,z-3):z+4]==B.LEAVES
        if not leaves.any(): kind='dead'
        pos=(x+.5,float(y),z+.5); v=float(N.hash_unit(np.array([x]),np.array([z]),83)[0])
        models.tree(batch,pos,top-y,kind,v,2); trees.append(pos)
    for x,y,z in np.argwhere(np.isin(vox,[B.BUSH,B.COOKPOT,B.CAMPFIRE,B.CAMPFIRE_LIT,B.TORCH_WALL,B.CRYSTAL,B.VINE,B.GLASS])):
        models.prop(batch,int(vox[x,y,z]),(x+.5,y,z+.5),0)
    _add_batch(scene,batch)
    if hasattr(env, 'static_presentation_meshes'):
        architecture_batch = _presentation_batch()
        env.static_presentation_meshes(architecture_batch)
        _add_batch(scene, architecture_batch)
        scene.metadata['geometry_contract']['architecture'] = env.architecture
    # Match high-tier native ground-cover density, but keep the entire world.
    # Instance geometry in Blender instead of replicating millions of blades.
    natural_height=np.empty(w.height.shape,np.int16)
    ys=np.arange(w.size_y,dtype=np.int16)[None,:,None]
    for start in range(0,w.size_x,32):
        natural_height[start:start+32]=np.where(_NATURAL[vox[start:start+32]],ys,-1).max(axis=1)
    xx,zz=np.indices(w.height.shape); yy=np.maximum(natural_height,0)
    gx,gz=np.nonzero((vox[xx,yy,zz]==B.GRASS)&(N.hash_unit(xx,zz,731)>.12))
    points=np.stack((gx+.5,yy[gx,gz]+1.,gz+.5),axis=1).astype(np.float32)
    points=np.repeat(points,4,axis=0); i=np.arange(len(points))
    points[:,0]+=(N.hash_unit(i,i,84)-.5)*.8;points[:,2]+=(N.hash_unit(i,i,32)-.5)*.8
    variants=(N.hash_unit(i,i,12)*12).astype(np.int32).clip(0,11)
    scene.metadata['geometry_contract'].update(trees=len(trees),grass_instances=len(points))
    return scene,points,variants


def dynamic_scene(env):
    scene=SceneSnapshot(metadata(env)); batch=_presentation_batch()
    env.presentation_meshes(batch)
    for obj in [*env.movers.values(),*env.rafts.values()]:
        if obj['under']: batch.box(obj['pos']+(.5,-.5,.5),(3,1,3),P.COLOR_F[B.PLANK])
    bounds=[env.vault_gate] if env.task=='a' else [env.closure_bounds()]
    for lo,hi in bounds:
        for x in range(lo[0],hi[0]+1):
            for y in range(lo[1],hi[1]+1):
                for z in range(lo[2],hi[2]+1):
                    block=env.world.get(x,y,z)
                    if B.SOLID[block]: batch.box((x+.5,y+.5,z+.5),(1,1,1),P.COLOR_F[block])
    _add_batch(scene,batch)
    character=models.ModelBatch()
    models.traveler(character,env.agent.pos,env.agent.yaw,phase=env.agent.distance_travelled*2.5,gliding=env.agent.gliding)
    _add_batch(scene,character)
    return scene


def grass_prototypes():
    scene=SceneSnapshot(dict(version=1))
    for k in range(12):
        batch=models.ModelBatch(); models.grass(batch,(0,0,0),(k+.5)/12,0,detail=2)
        before=len(scene.meshes);_add_batch(scene,batch)
        for mesh in scene.meshes[before:]: mesh['name']=f'Grass variant {k:02}'
    return scene
