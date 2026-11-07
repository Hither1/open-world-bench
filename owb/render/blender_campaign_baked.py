"""Bake every static grass blade into native meshes once; no culling or LOD."""
import json
from pathlib import Path
import sys
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import blender_campaign as campaign


def canonical(triangle):
    t=tuple(map(int,triangle));return min(t,t[1:]+t[:1],t[2:]+t[:2])


def compact_faces(vertices,triangles,colors):
    faces=[];face_colors=[]
    for i in range(0,len(triangles),2):
        a,b=triangles[i:i+2];shared=set(a)&set(b);union=set(a)|set(b)
        if len(shared)==2 and len(union)==4 and np.array_equal(colors[i],colors[i+1]):
            unique=next(k for k,v in enumerate(a) if v not in shared)
            q=[int(a[(unique-1)%3]),int(a[unique]),int(a[(unique+1)%3]),int(next(v for v in b if v not in shared))]
            trial=bpy.data.meshes.new('Blade triangulation check');trial.from_pydata(vertices.tolist(),[],[q]);trial.calc_loop_triangles()
            got={canonical(t.vertices) for t in trial.loop_triangles};expected={canonical(a),canonical(b)}
            bpy.data.meshes.remove(trial)
            if got==expected:
                faces.append(q);face_colors.append(colors[i]);continue
        faces.extend([a.tolist(),b.tolist()]);face_colors.extend(colors[i:i+2])
    return faces,np.asarray(face_colors,np.float32)


def grass_material(base):
    mat=base.copy();mat.name='Full-detail blade colors';nodes,links=mat.node_tree.nodes,mat.node_tree.links
    old=next(n for n in nodes if n.bl_idname=='ShaderNodeVertexColor')
    destinations=[link.to_socket for link in list(old.outputs['Color'].links)]
    color=nodes.new('ShaderNodeAttribute');color.attribute_name='BladeColor'
    for dest in destinations:links.new(color.outputs['Color'],dest)
    nodes.remove(old);return mat


def bake_ground_cover(directory,materials):
    with np.load(directory/'ground-cover.npz',allow_pickle=False) as data:
        points=campaign.z_up(data['points']);variants=data['variants']
    collection=bpy.data.collections.new('Static ground-cover geometry')
    triangles_total=0;vertices_total=0;stored_faces=0
    with np.load(directory/'grass-prototypes.npz',allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata']))
        for k,spec in enumerate(metadata['meshes']):
            key=f'mesh_{k}_';triangles=campaign.z_up(data[key+'triangles']);normal=campaign.z_up(data[key+'normals'])
            assert np.allclose(normal,(0,0,1)), 'Preserve the original constant blade normals'
            vertices,inverse=np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True)
            original=inverse.reshape(-1,3);color=campaign.linear(data[key+'colors'])
            rgba=np.concatenate((color,np.ones((len(color),1),np.float32)),axis=1)
            faces,face_colors=compact_faces(vertices,original,rgba)
            lengths=np.array([len(face) for face in faces],np.int32);starts=np.r_[0,np.cumsum(lengths)[:-1]].astype(np.int32)
            indices=np.array([index for face in faces for index in face],np.int32)
            roots=points[variants==k];n=len(roots);nv=len(vertices);nf=len(faces);nl=len(indices)
            if not n:continue
            print('Baking grass variant',k,':',n,'clumps;',len(original),'unchanged triangles per clump',flush=True)
            mesh=bpy.data.meshes.new(f'Static grass {k:02}')
            mesh.vertices.add(n*nv);mesh.vertices.foreach_set('co',(roots[:,None,:]+vertices[None,:,:]).ravel())
            offsets=np.arange(n,dtype=np.int32)[:,None]
            mesh.loops.add(n*nl);mesh.loops.foreach_set('vertex_index',(offsets*nv+indices[None,:]).ravel())
            mesh.polygons.add(n*nf);mesh.polygons.foreach_set('loop_start',(offsets*nl+starts[None,:]).ravel())
            mesh.polygons.foreach_set('loop_total',np.tile(lengths,n));mesh.update(calc_edges=True)
            attr=mesh.attributes.new('BladeColor','FLOAT_COLOR','FACE');attr.data.foreach_set('color',np.tile(face_colors,(n,1)).ravel())
            mesh.polygons.foreach_set('use_smooth',np.ones(n*nf,dtype=bool))
            # One shared immutable tuple avoids enormous Python tuple duplication.
            mesh.normals_split_custom_set([(0.,0.,1.)]*(n*nl))
            obj=bpy.data.objects.new(f'Full-world baked grass {k:02}',mesh);collection.objects.link(obj)
            mesh.materials.append(grass_material(materials.get(spec['material'],materials['surface'])))
            triangles_total+=n*len(original);vertices_total+=n*nv;stored_faces+=n*nf
    instance=bpy.data.objects.new('Full-world static grass',None)
    instance.instance_type='COLLECTION';instance.instance_collection=collection
    bpy.context.collection.objects.link(instance)
    report=Path(sys.argv[sys.argv.index('--output')+1])/'renderer-report.json'
    state=json.loads(report.read_text());state['ground_cover']=dict(method='fixed native meshes in one collection instance',source_instances=len(points),
        triangles=triangles_total,vertices=vertices_total,stored_faces=stored_faces,
        prototype_triangulation_verified=True,full_geometry=True,distance_culling=False)
    report.write_text(json.dumps(state,indent=2))
    print('Baked full ground cover:',len(points),'clumps;',triangles_total,'triangles',flush=True)

campaign.ground_cover=bake_ground_cover
if __name__=='__main__':campaign.main()
