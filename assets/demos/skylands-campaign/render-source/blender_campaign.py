"""Run inside Blender: persistent, complete-world GPU Cycles campaign renderer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,os.environ.get('OWB_CAMPAIGN_RENDER_HELPERS',str(Path(__file__).resolve().parent)))
import blender_scene
from blender_scene import setup_scene,add_mesh,material,z_up,linear,build_materials,configure_haze,mesh_material,DEVICES,SKY_STRENGTH


def device_label(device):
    return 'CPU' if device=='CPU' else device+' GPU'


def load_scene(path,materials):
    objects=[]
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        for i,spec in enumerate(meta['meshes']):objects.append(add_mesh(data,i,spec,materials))
    return meta,objects


def load_dynamic(path,materials,objects):
    """Retain flat-triangle topology and object identities between replay states."""
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if not objects:
            for i,spec in enumerate(meta['meshes']):
                mesh=bpy.data.meshes.new(f'Replay mesh {i}')
                obj=bpy.data.objects.new(f'Replay object {i}',mesh);bpy.context.collection.objects.link(obj);objects.append(obj)
                mesh.materials.append(mesh_material(materials,spec))
                if not spec['water']:mesh.materials.append(materials['emission:occluder' if spec.get('occluder') else 'emission'])
        if len(objects)!=len(meta['meshes']):raise RuntimeError('Dynamic mesh slot count changed')
        for i,(obj,spec) in enumerate(zip(objects,meta['meshes'])):
            prefix=f'mesh_{i}_';triangles=z_up(data[prefix+'triangles'])
            points=np.ascontiguousarray(np.round(triangles.reshape(-1,3),5));points[points==0]=0
            mesh=obj.data;count=len(triangles)
            if len(mesh.polygons)!=count:
                mesh.clear_geometry();mesh.vertices.add(count*3);mesh.loops.add(count*3);mesh.polygons.add(count)
                mesh.loops.foreach_set('vertex_index',np.arange(count*3,dtype=np.int32))
                mesh.polygons.foreach_set('loop_start',np.arange(0,count*3,3,dtype=np.int32))
                mesh.polygons.foreach_set('loop_total',np.full(count,3,dtype=np.int32))
            mesh.vertices.foreach_set('co',points.ravel());mesh.update(calc_edges=True)
            colors=linear(np.repeat(data[prefix+'colors'],3,axis=0))
            rgba=np.concatenate((colors,np.ones((len(colors),1))),axis=1).astype(np.float32)
            attr=mesh.color_attributes.get('Color') or mesh.color_attributes.new(name='Color',type='FLOAT_COLOR',domain='CORNER')
            attr.data.foreach_set('color',rgba.ravel());mesh.color_attributes.active_color=attr
            if not spec['water']:mesh.polygons.foreach_set('material_index',(data[prefix+'emission']>.1).astype(np.int32))
            normals=data[prefix+'normals']
            if spec['smooth'] and not np.any(normals):raise RuntimeError('Unwelded smooth mesh needs authored normals')
            mesh.polygons.foreach_set('use_smooth',np.full(count,spec['smooth'],dtype=bool))
            if np.any(normals):
                normals=z_up(normals);face=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                face/=np.maximum(np.linalg.norm(face,axis=1,keepdims=True),1e-8)
                normals=np.where(np.linalg.norm(normals,axis=-1,keepdims=True)>0,normals,face[:,None])
                mesh.polygons.foreach_set('use_smooth',np.ones(count,dtype=bool))
                mesh.normals_split_custom_set(normals.reshape(-1,3).tolist())
            mesh.update()
        return meta


def ground_cover(directory,materials):
    collection=bpy.data.collections.new('Stable grass prototypes')
    _,objects=load_scene(directory/'grass-prototypes.npz',materials)
    for obj in objects:
        for c in list(obj.users_collection):c.objects.unlink(obj)
        collection.objects.link(obj)
    with np.load(directory/'ground-cover.npz',allow_pickle=False) as data:
        points=z_up(data['points']);variants=data['variants']
    mesh=bpy.data.meshes.new('All ground cover roots');mesh.vertices.add(len(points));mesh.vertices.foreach_set('co',points.ravel())
    attr=mesh.attributes.new('variant','INT','POINT');attr.data.foreach_set('value',variants)
    obj=bpy.data.objects.new('Full-world ground cover',mesh);bpy.context.collection.objects.link(obj)
    group=bpy.data.node_groups.new('Full botanical instancing','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes,links=group.nodes,group.links
    source=nodes.new('NodeGroupInput');result=nodes.new('NodeGroupOutput')
    prototypes=nodes.new('GeometryNodeCollectionInfo');prototypes.inputs['Collection'].default_value=collection
    prototypes.inputs['Separate Children'].default_value=True;prototypes.inputs['Reset Children'].default_value=True
    attribute=nodes.new('GeometryNodeInputNamedAttribute');attribute.data_type='INT';attribute.inputs['Name'].default_value='variant'
    instances=nodes.new('GeometryNodeInstanceOnPoints');instances.inputs['Pick Instance'].default_value=True
    links.new(source.outputs['Geometry'],instances.inputs['Points']);links.new(prototypes.outputs['Instances'],instances.inputs['Instance'])
    links.new(attribute.outputs['Attribute'],instances.inputs['Instance Index']);links.new(instances.outputs['Instances'],result.inputs['Geometry'])
    modifier=obj.modifiers.new('All grass, no distance culling','NODES');modifier.node_group=group
    print('Full ground cover:',len(points),'instances',flush=True)


def update_scene(scene,metadata):
    camera=scene.camera;eye=Vector(z_up(metadata['camera']['eye']));target=Vector(z_up(metadata['camera']['target']))
    camera.location=eye;camera.rotation_euler=(target-eye).to_track_quat('-Z','Y').to_euler()
    camera.data.lens=camera.data.sensor_height/(2*np.tan(np.radians(metadata['camera']['fov_y'])/2))
    grade=metadata.get('night_grade') or {}
    lighting=metadata['lighting'];sun=bpy.data.objects['Sun'];sun.data.energy=max(0,lighting['strength'])*3*grade.get('moon',1.);sun.data.color=lighting['color']
    background=next(n for n in scene.world.node_tree.nodes if n.type=='BACKGROUND')
    background.inputs['Strength'].default_value=SKY_STRENGTH*grade.get('sky',1.)
    scene.view_settings.exposure=grade.get('exposure',0.)
    sun.rotation_euler=Vector(-z_up(lighting['direction'])).to_track_quat('-Z','Y').to_euler()
    ramp=next(n for n in scene.world.node_tree.nodes if n.type=='VALTORGB').color_ramp
    ramp.elements[0].color=(*linear(np.array(lighting['horizon'])),1);ramp.elements[1].color=(*linear(np.array(lighting['sky'])),1)
    configure_haze(metadata.get('haze'),lighting['horizon'],SKY_STRENGTH)
    for index,spec in enumerate(metadata['point_lights']):
        name=f'Task lamp {index}';obj=bpy.data.objects.get(name)
        if obj is None:
            light=bpy.data.lights.new(name,'POINT');obj=bpy.data.objects.new(name,light);bpy.context.collection.objects.link(obj)
        obj.location=z_up(spec['pos']);obj.data.energy=spec['power'];obj.data.color=spec['color'];obj.data.shadow_soft_size=spec['radius']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int);p.add_argument('--samples',type=int,default=64);p.add_argument('--threads',type=int,default=8);p.add_argument('--device',choices=DEVICES,default='CUDA')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);data=json.loads((a.directory/'frames.json').read_text())
    # Exported NPZ scenes are detached immutable inputs. Validate those bytes;
    # unrelated edits to the simulator/exporter cannot alter these snapshots.
    for path,expected in data['artifact_hashes'].items():
        if hashlib.sha256((a.directory/path).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Scene artifact changed: {path}')
    render_sources={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
                    (Path(__file__),Path(blender_scene.__file__))}
    wrapper=Path(sys.argv[sys.argv.index('--python')+1]) if '--python' in sys.argv else None
    if wrapper and wrapper.exists():render_sources[wrapper.name]=hashlib.sha256(wrapper.read_bytes()).hexdigest()
    if a.samples!=data['renderer']['samples']:raise RuntimeError('Sample count differs from render contract')
    a.output.mkdir(parents=True,exist_ok=True);bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    with np.load(a.directory/'static.npz',allow_pickle=False) as d:metadata=json.loads(str(d['metadata']))
    report=a.output/'renderer-report.json'
    if report.exists() and json.loads(report.read_text()).get('render_sources')!=render_sources:raise RuntimeError('Cached renderer source changed')
    scene=setup_scene(metadata,a.samples,a.threads,a.device);scene.render.use_persistent_data=True
    preferences=bpy.context.preferences.addons['cycles'].preferences
    (a.output/'renderer-report.json').write_text(json.dumps(dict(blender_version=bpy.app.version_string,
        accelerators=[item.name for item in preferences.devices if item.type==a.device and item.use],
        device=device_label(a.device),samples=a.samples,width=1280,height=720,denoising=True,max_bounces=6,render_sources=render_sources)))
    with np.load(a.directory/'static.npz',allow_pickle=False) as d:materials=build_materials(metadata,d)
    load_scene(a.directory/'static.npz',materials);ground_cover(a.directory,materials)
    dynamic=[];started=time.monotonic();loaded=[None]
    selected=data['entries'][a.first:a.last]
    pending=[]
    for entry in selected:
        target=a.output/f'{entry["index"]:06}.png';record=target.with_suffix('.json')
        if target.exists() and record.exists():
            cached=json.loads(record.read_text())
            if cached['digest']!=entry['digest'] or cached['trace_sha256']!=data['trace_sha256'] or cached.get('image_sha256')!=hashlib.sha256(target.read_bytes()).hexdigest():raise RuntimeError('Cached frame mismatch')
        else:pending.append(entry['index'])
    if not pending:print('RENDER COMPLETE (cached)',flush=True);return
    entries={entry['index']:entry for entry in selected}
    def prepare(current_scene,depsgraph=None):
        frame=current_scene.frame_current
        if frame not in entries or frame==loaded[0]:return
        try:
            entry=entries[frame]
            if hashlib.sha256((a.directory/entry['scene']).read_bytes()).hexdigest()!=entry['scene_sha256']:raise RuntimeError('Dynamic snapshot hash changed')
            metadata=load_dynamic(a.directory/entry['scene'],materials,dynamic)
            update_scene(current_scene,metadata);loaded[0]=frame
        except Exception:
            traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(2)
    def record_frame(current_scene):
        frame=current_scene.frame_current
        try:
            if loaded[0]!=frame:raise RuntimeError('Rendered state was not prepared')
            entry=entries[frame];target=a.output/f'{frame:06}.png'
            target.with_suffix('.json').write_text(json.dumps(dict(index=frame,step=entry['step'],digest=entry['digest'],trace_sha256=data['trace_sha256'],
                image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),renderer='Blender Cycles',device=device_label(a.device),samples=a.samples,size=[1280,720],seconds=time.monotonic()-started)))
            print('FRAME COMPLETE',frame,'/',len(data['entries']),'step',entry['step'],flush=True)
        except Exception:
            traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(2)
    bpy.app.handlers.frame_change_pre.append(prepare);bpy.app.handlers.render_write.append(record_frame)
    scene.render.filepath=str(a.output.resolve()/'######');scene.render.use_file_extension=True
    scene.render.use_lock_interface=True
    try:
        # One persistent render graph per contiguous interval retains static BVHs.
        groups=[]
        for frame in pending:
            if not groups or frame!=groups[-1][-1]+1:groups.append([frame])
            else:groups[-1].append(frame)
        for group in groups:
            scene.frame_start=group[0];scene.frame_end=group[-1];scene.frame_step=1
            scene.frame_set(group[0]);bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.frame_change_pre.remove(prepare);bpy.app.handlers.render_write.remove(record_frame)
    print('RENDER COMPLETE',flush=True)
if __name__=='__main__':main()
