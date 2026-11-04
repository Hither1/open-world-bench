"""Run inside Blender; consumes scene_export's numeric NPZ without importing OWB.

blender --background --factory-startup --python blender_scene.py -- \
    --scene scene.npz --output render.png --blend scene.blend --glb scene.glb
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def z_up(points):
    """A proper rotation: (x, y, z) -> (x, -z, y), preserving winding."""
    points = np.asarray(points, dtype=np.float32)
    return points[..., [0, 2, 1]] * np.array((1, -1, 1), np.float32)


def linear(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def material(name, *, emission=False, water=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    colors = nodes.new("ShaderNodeVertexColor")
    colors.layer_name = "Color"
    links.new(colors.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 0.82
    shader.inputs["Specular IOR Level"].default_value = 0.25
    if emission:
        links.new(colors.outputs["Color"], shader.inputs["Emission Color"])
        shader.inputs["Emission Strength"].default_value = 2.0
    if water:
        shader.inputs["Roughness"].default_value = 0.17
        shader.inputs["Metallic"].default_value = 0.22
        shader.inputs["Coat Weight"].default_value = 0.4
        # Subtle geometric-scale ripples. The GLB receives the portable base
        # material; Blender-specific procedural nodes stay in the .blend file.
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 1.8
        noise.inputs["Detail"].default_value = 2
        coordinates = nodes.new("ShaderNodeTexCoord")
        links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.23
        bump.inputs["Distance"].default_value = 0.12
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return mat


def add_mesh(data, index, spec, materials):
    print(f"Building {spec['name']}", flush=True)
    prefix = f"mesh_{index}_"
    triangles = z_up(data[prefix + "triangles"])
    # Weld matching points for smooth terrain normals; the separate corner
    # color attribute preserves per-face albedo and authored model normals.
    points = np.ascontiguousarray(np.round(triangles.reshape(-1, 3), 5))
    points[points == 0] = 0  # canonicalize signed zero before byte-wise keys
    keys = points.view(np.dtype((np.void, points.dtype.itemsize * 3))).ravel()
    # Byte-wise sorting avoids NumPy's much slower lexicographic structured
    # comparisons on multi-million-vertex cave meshes. Vertex order is immaterial.
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    vertices = points[first]
    print(f"  {len(triangles):,} triangles, {len(vertices):,} welded vertices", flush=True)
    mesh = bpy.data.meshes.new(spec["name"])
    # Bulk transfer avoids millions of temporary Python lists on voxel worlds.
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.ravel())
    mesh.loops.add(len(inverse))
    mesh.loops.foreach_set("vertex_index", inverse.astype(np.int32))
    mesh.polygons.add(len(triangles))
    mesh.polygons.foreach_set("loop_start", np.arange(0, len(inverse), 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(len(triangles), 3, dtype=np.int32))
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(spec["name"], mesh)
    bpy.context.collection.objects.link(obj)
    colors = linear(np.repeat(data[prefix + "colors"], 3, axis=0))
    rgba = np.concatenate((colors, np.ones((len(colors), 1))), axis=1).astype(np.float32)
    attr = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
    attr.data.foreach_set("color", rgba.ravel())
    mesh.color_attributes.active_color = attr
    if spec["water"]:
        mesh.materials.append(materials["water"])
    else:
        mesh.materials.append(materials["surface"])
        mesh.materials.append(materials["emission"])
        mesh.polygons.foreach_set("material_index", (data[prefix + "emission"] > 0.1).astype(np.int32))
    mesh.polygons.foreach_set("use_smooth", np.full(len(mesh.polygons), spec["smooth"], dtype=bool))
    normals = data[prefix + "normals"]
    if np.any(normals):
        normals = z_up(normals)
        face = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-8)
        normals = np.where(np.linalg.norm(normals, axis=-1, keepdims=True) > 0, normals, face[:, None])
        mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
        mesh.normals_split_custom_set(normals.reshape(-1, 3).tolist())
    return obj


def setup_scene(metadata, samples, threads):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = int(metadata["seed"])
    scene.cycles.max_bounces = 6
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.resolution_x, scene.render.resolution_y = metadata["size"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.5
    scene.render.film_transparent = False

    camera = bpy.data.cameras.new("Agent camera")
    obj = bpy.data.objects.new("Agent camera", camera)
    bpy.context.collection.objects.link(obj)
    eye = Vector(z_up(metadata["camera"]["eye"]))
    target = Vector(z_up(metadata["camera"]["target"]))
    obj.location = eye
    obj.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
    camera.sensor_fit = "VERTICAL"
    camera.sensor_height = 24
    camera.lens = camera.sensor_height / (2 * np.tan(np.radians(metadata["camera"]["fov_y"]) / 2))
    camera.clip_start, camera.clip_end = 0.05, 10000
    scene.camera = obj

    lighting = metadata["lighting"]
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = max(0, lighting["strength"]) * 3
    sun.color = lighting["color"]
    sun.angle = np.radians(4)
    lamp = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(lamp)
    lamp.rotation_euler = Vector(-z_up(lighting["direction"])).to_track_quat("-Z", "Y").to_euler()

    world = bpy.data.worlds.new("Painted sky")
    world.use_nodes = True
    scene.world = world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = nodes.get("Background")
    background.inputs["Strength"].default_value = 0.8
    tex = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["Normal"], separate.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0
    ramp.color_ramp.elements[0].color = (*linear(np.array(lighting["horizon"])), 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (*linear(np.array(lighting["sky"])), 1)
    # World texture normals point inward. Elevation magnitude gives a colored
    # zenith above the horizon and a matching lower-hemisphere ambient fill.
    elevation = nodes.new("ShaderNodeMath")
    elevation.operation = "ABSOLUTE"
    links.new(separate.outputs["Z"], elevation.inputs[0])
    links.new(elevation.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], background.inputs["Color"])
    scene["owb_snapshot"] = json.dumps(metadata)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--glb", type=Path)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.samples <= 0 or args.threads <= 0:
        parser.error("samples and threads must be positive")
    for path in (args.output, args.blend, args.glb):
        if path is not None:
            if path.exists():
                parser.error(f"output already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    with np.load(args.scene, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        print(f"Loaded {metadata['environment']} snapshot", flush=True)
        if metadata.get("version") != 1:
            raise ValueError("unsupported scene snapshot version")
        scene = setup_scene(metadata, args.samples, args.threads)
        materials = dict(surface=material("Painted surfaces"),
                         emission=material("Elemental glow", emission=True),
                         water=material("Water", water=True))
        for i, spec in enumerate(metadata["meshes"]):
            add_mesh(data, i, spec, materials)
    scene.render.filepath = str(args.output.resolve())
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    if args.glb:
        bpy.ops.export_scene.gltf(filepath=str(args.glb.resolve()), export_format="GLB",
                                  export_cameras=True, export_lights=True, export_yup=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
