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


HAZE_GROUP = "Aerial haze"


def haze_group(haze=None, color=(0.7, 0.75, 0.8), strength=0.8):
    """Distance haze toward the horizon colour, so far ridges read as layers.

    Camera distance drives a mix toward a flat horizon-coloured emission. It is
    a presentation-only atmosphere; nearby subjects (< start metres) are unaffected.
    """
    group = bpy.data.node_groups.get(HAZE_GROUP)
    if group is None:
        group = bpy.data.node_groups.new(HAZE_GROUP, "ShaderNodeTree")
        group.interface.new_socket(name="Shader", in_out="INPUT", socket_type="NodeSocketShader")
        group.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
        nodes, links = group.nodes, group.links
        source, result = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
        camera = nodes.new("ShaderNodeCameraData")
        def math(op, a=None, b=None, name=None):
            node = nodes.new("ShaderNodeMath"); node.operation = op
            if name: node.name = node.label = name
            for i, v in enumerate((a, b)):
                if isinstance(v, (int, float)): node.inputs[i].default_value = v
                elif v is not None: links.new(v, node.inputs[i])
            return node.outputs[0]
        beyond = math("MAXIMUM", math("SUBTRACT", camera.outputs["View Distance"], 70.0, "Haze start"), 0.0)
        falloff = math("EXPONENT", math("MULTIPLY", math("DIVIDE", beyond, 1100.0, "Haze distance"), -1.0))
        amount = math("MULTIPLY", math("SUBTRACT", 1.0, falloff), 0.85, "Haze amount")
        glow = nodes.new("ShaderNodeEmission"); glow.name = "Haze color"
        mix = nodes.new("ShaderNodeMixShader")
        links.new(amount, mix.inputs[0]); links.new(source.outputs[0], mix.inputs[1])
        links.new(glow.outputs[0], mix.inputs[2]); links.new(mix.outputs[0], result.inputs[0])
    configure_haze(haze, color, strength)
    return group


def configure_haze(haze=None, color=(0.7, 0.75, 0.8), strength=0.8):
    group = bpy.data.node_groups.get(HAZE_GROUP)
    if group is None:
        return
    haze = dict(dict(start=70.0, distance=1100.0, amount=0.85), **(haze or {}))
    nodes = group.nodes
    nodes["Haze start"].inputs[1].default_value = haze["start"]
    nodes["Haze distance"].inputs[1].default_value = haze["distance"]
    nodes["Haze amount"].inputs[1].default_value = haze["amount"]
    nodes["Haze color"].inputs["Color"].default_value = (*linear(np.asarray(color)), 1)
    nodes["Haze color"].inputs["Strength"].default_value = strength


def _route_output(mat, insert):
    """Insert ``insert(nodes, links, socket) -> socket`` before the surface output."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    link = output.inputs["Surface"].links[0]
    socket = link.from_socket
    links.remove(link)
    links.new(insert(nodes, links, socket), output.inputs["Surface"])


def add_haze(mat):
    if bpy.data.node_groups.get(HAZE_GROUP) is None:
        return mat
    def insert(nodes, links, socket):
        node = nodes.new("ShaderNodeGroup"); node.node_tree = bpy.data.node_groups[HAZE_GROUP]
        links.new(socket, node.inputs[0])
        return node.outputs[0]
    _route_output(mat, insert)
    return mat


WET_GROUP = "Rain wetness"


def wetness_group():
    """Darker albedo and glossier surfaces in rain, with sheen pooling on
    upward-facing ground. The amount is one shared value set per frame; at
    zero the group passes colour and roughness through unchanged."""
    group = bpy.data.node_groups.get(WET_GROUP)
    if group is not None:
        return group
    group = bpy.data.node_groups.new(WET_GROUP, "ShaderNodeTree")
    for name, kind in (("Color", "NodeSocketColor"), ("Roughness", "NodeSocketFloat")):
        group.interface.new_socket(name=name, in_out="INPUT", socket_type=kind)
        group.interface.new_socket(name=name, in_out="OUTPUT", socket_type=kind)
    nodes, links = group.nodes, group.links
    source, result = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    amount = nodes.new("ShaderNodeValue"); amount.name = amount.label = "Wetness"
    amount.outputs[0].default_value = 0.0
    def math(op, a, b):
        node = nodes.new("ShaderNodeMath"); node.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)): node.inputs[i].default_value = v
            else: links.new(v, node.inputs[i])
        return node.outputs[0]
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Normal"], separate.inputs["Vector"])
    upward = _range(nodes, links, separate.outputs["Z"], .55, .95)
    puddles = _range(nodes, links, _noise(nodes, links, geometry.outputs["Position"], .35, 3.0), .5, .62)
    pooling = math("MULTIPLY", upward, puddles)
    darken = nodes.new("ShaderNodeMixRGB"); darken.blend_type = "MULTIPLY"
    links.new(math("MULTIPLY", amount.outputs[0], .8), darken.inputs[0])
    links.new(source.outputs["Color"], darken.inputs[1])
    darken.inputs[2].default_value = (.58, .58, .6, 1)
    links.new(darken.outputs["Color"], result.inputs["Color"])
    gloss = math("MULTIPLY", amount.outputs[0], math("ADD", .5, math("MULTIPLY", pooling, .45)))
    links.new(math("MULTIPLY", source.outputs["Roughness"], math("SUBTRACT", 1.0, gloss)), result.inputs["Roughness"])
    return group


def add_wetness(mat):
    """Route the Principled base colour and roughness through the wetness group."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    node = nodes.new("ShaderNodeGroup"); node.node_tree = wetness_group()
    color = shader.inputs["Base Color"]
    if color.links:
        links.new(color.links[0].from_socket, node.inputs["Color"])
    else:
        node.inputs["Color"].default_value = color.default_value
    roughness = shader.inputs["Roughness"]
    if roughness.links:
        links.new(roughness.links[0].from_socket, node.inputs["Roughness"])
    else:
        node.inputs["Roughness"].default_value = roughness.default_value
    links.new(node.outputs["Color"], color)
    links.new(node.outputs["Roughness"], roughness)
    return mat


# Presentation of the simulated weather. Streak positions are hashed from world
# cells, the scene seed and simulated time, never drawn from the simulation RNG,
# so a drop keeps its identity as the camera moves and between video frames.
PRECIPITATION = dict(
    rain=dict(cell=1.5, spacing=2.6, speed=9.0, length=.6, width=.010, alpha=.30, slant=.10, near=3.0),
    storm=dict(cell=1.1, spacing=2.2, speed=11.5, length=.7, width=.010, alpha=.30, slant=.28, near=3.5),
    snow=dict(cell=1.3, spacing=2.0, speed=1.1, length=.0, width=.026, alpha=.7, slant=.0, sway=.45, near=2.0))
WEATHER_LOOK = dict(  # sun, sky, wetness, haze start and distance factors, haze amount floor
    clear=(1., 1., 0., 1., 1., 0.), cloud=(.72, .92, 0., .9, .8, 0.),
    rain=(.4, .88, 1., .45, .3, .88), storm=(.24, .7, 1., .32, .22, .92),
    snow=(.4, .95, .25, .4, .26, .9), fog=(.45, .88, .15, .12, .08, .95))
PRECIPITATION_RADIUS, PRECIPITATION_BELOW, PRECIPITATION_ABOVE = 26.0, 8.0, 16.0


def _hash_unit(*keys):
    """splitmix64 of integer keys -> floats in [0, 1)."""
    h = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for key in keys:
            h = (h ^ np.asarray(key).astype(np.int64).astype(np.uint64)) * np.uint64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> np.uint64(31))) * np.uint64(0x94D049BB133111EB)
            h = h ^ (h >> np.uint64(29))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def precipitation_triangles(weather, eye):
    """Camera-facing streaks (rain) or flakes (snow) around ``eye``, Y-up metres."""
    spec = PRECIPITATION.get(weather.get("kind"))
    if spec is None:
        return np.empty((0, 3, 3), np.float32)
    eye = np.asarray(eye, float)
    c, s, t, seed = spec["cell"], spec["spacing"], float(weather["seconds"]), int(weather["seed"])
    span = np.arange(-int(PRECIPITATION_RADIUS / c) - 1, int(PRECIPITATION_RADIUS / c) + 2)
    i, j, k = np.meshgrid(np.floor(eye[0] / c) + span,
                          np.arange(np.floor((eye[1] - PRECIPITATION_BELOW) / s), np.ceil((eye[1] + PRECIPITATION_ABOVE) / s)),
                          np.floor(eye[2] / c) + span, indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    wind = np.asarray(weather.get("wind", (0, 0, 0)), float)
    lean = wind[[0, 2]] * .12
    if spec["slant"] and not np.any(lean):
        angle = 2 * np.pi * _hash_unit(seed, 17)
        lean = spec["slant"] * np.array((np.cos(angle), np.sin(angle)))
    fall = np.array((lean[0], -1.0, lean[1])); fall /= np.linalg.norm(fall)
    x = (i + _hash_unit(seed, i, j, k, 1)) * c
    z = (k + _hash_unit(seed, i, j, k, 2)) * c
    drift = spec["speed"] * t / s
    y = (j + np.mod(_hash_unit(seed, i, j, k, 3) - drift * (.8 + .4 * _hash_unit(seed, i, j, k, 4)), 1.0)) * s
    centers = np.stack((x, y, z), axis=1)
    centers[:, [0, 2]] += lean[None] * (y - eye[1])[:, None]
    if spec.get("sway"):
        phase = 2 * np.pi * _hash_unit(seed, i, j, k, 5)
        centers[:, 0] += spec["sway"] * np.sin(t * 1.3 + phase)
        centers[:, 2] += spec["sway"] * np.cos(t * 1.1 + phase)
    offset = centers - eye
    distance = np.linalg.norm(offset, axis=1)
    keep = (distance > spec["near"]) & (np.linalg.norm(offset[:, [0, 2]], axis=1) < PRECIPITATION_RADIUS)
    centers, offset = centers[keep], offset[keep]
    side = np.cross(fall[None], offset)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-8)
    size = (.7 + .6 * _hash_unit(seed, i[keep], j[keep], k[keep], 6))[:, None]
    if not spec["length"]:
        # Flakes: small hexagons facing the camera.
        facing = np.cross(side, offset)
        facing /= np.maximum(np.linalg.norm(facing, axis=1, keepdims=True), 1e-8)
        angles = np.linspace(0, 2 * np.pi, 7)
        rim = [centers + size * spec["width"] / 2 * (np.cos(a) * side + np.sin(a) * facing) for a in angles]
        return np.concatenate([np.stack((centers, rim[n], rim[n + 1]), 1) for n in range(6)]).astype(np.float32)
    length = spec["length"] * size
    up, across = -fall[None] * length / 2, side * spec["width"] / 2
    a, b, c_, d = centers + up - across, centers + up + across, centers - up + across, centers - up - across
    return np.concatenate((np.stack((a, b, c_), 1), np.stack((a, c_, d), 1))).astype(np.float32)


def apply_weather(scene, metadata):
    """Dim the sun and sky, tighten haze, wet surfaces and place precipitation
    for the snapshot's simulated weather. Call after lighting and haze are set
    for the frame; clear weather leaves the scene exactly as configured."""
    weather = metadata.get("weather") or {}
    kind = weather.get("kind", "clear")
    sun_k, sky_k, wet, start_k, distance_k, amount = WEATHER_LOOK.get(kind, WEATHER_LOOK["clear"])
    flash = float(weather.get("flash", 0.0))
    sun = bpy.data.objects.get("Sun")
    if sun is not None:
        sun.data.energy *= sun_k
    background = next(n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND")
    background.inputs["Strength"].default_value *= sky_k * (1 + 6 * flash)
    group = bpy.data.node_groups.get(HAZE_GROUP)
    if group is not None:
        group.nodes["Haze start"].inputs[1].default_value *= start_k
        group.nodes["Haze distance"].inputs[1].default_value *= distance_k
        haze = group.nodes["Haze amount"].inputs[1]
        haze.default_value = max(haze.default_value, amount)
    wetness = bpy.data.node_groups.get(WET_GROUP)
    if wetness is not None:
        wetness.nodes["Wetness"].outputs[0].default_value = wet
    triangles = z_up(precipitation_triangles(weather, metadata["camera"]["eye"]))
    obj = bpy.data.objects.get("Precipitation")
    if obj is None:
        if not len(triangles):
            return
        obj = bpy.data.objects.new("Precipitation", bpy.data.meshes.new("Precipitation"))
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(precipitation_material())
        for ray in ("diffuse", "glossy", "transmission", "volume_scatter", "shadow"):
            setattr(obj, "visible_" + ray, False)
    mesh = obj.data
    mesh.clear_geometry()
    count = len(triangles)
    mesh.vertices.add(count * 3); mesh.vertices.foreach_set("co", triangles.reshape(-1).astype(np.float32))
    mesh.loops.add(count * 3); mesh.loops.foreach_set("vertex_index", np.arange(count * 3, dtype=np.int32))
    mesh.polygons.add(count)
    mesh.polygons.foreach_set("loop_start", np.arange(0, count * 3, 3, dtype=np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(count, 3, dtype=np.int32))
    mesh.update(calc_edges=True)
    spec = PRECIPITATION.get(kind, PRECIPITATION["rain"])
    nodes = obj.data.materials[0].node_tree.nodes
    lighting, grade = metadata["lighting"], metadata.get("night_grade") or {}
    tone = np.asarray(lighting["horizon"]) if kind != "snow" else .35 * np.asarray(lighting["horizon"]) + .65
    nodes["Precipitation glow"].inputs["Color"].default_value = (*linear(tone), 1)
    nodes["Precipitation glow"].inputs["Strength"].default_value = (
        1.5 * SKY_STRENGTH * sky_k * grade.get("sky", 1.0) * (1 + 6 * flash))
    nodes["Precipitation opacity"].inputs[0].default_value = spec["alpha"]


def precipitation_material():
    """Streaks lit like the sky they refract, seen only by camera rays."""
    mat = bpy.data.materials.new("Precipitation")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.remove(nodes.get("Principled BSDF"))
    glow = nodes.new("ShaderNodeEmission"); glow.name = "Precipitation glow"
    clear = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader"); mix.name = "Precipitation opacity"
    links.new(clear.outputs[0], mix.inputs[1]); links.new(glow.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], next(n for n in nodes if n.type == "OUTPUT_MATERIAL").inputs["Surface"])
    mat.cycles.emission_sampling = "NONE"
    return mat


def occluder_variant(mat):
    """Camera rays see an occluding object faded; shadows and bounce are unchanged."""
    faded = mat.copy()
    faded.name = mat.name + " (faded occluder)"
    def insert(nodes, links, socket):
        path = nodes.new("ShaderNodeLightPath")
        weight = nodes.new("ShaderNodeMath"); weight.operation = "MULTIPLY"
        links.new(path.outputs["Is Camera Ray"], weight.inputs[0]); weight.inputs[1].default_value = 0.72
        clear = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        links.new(weight.outputs[0], mix.inputs[0]); links.new(socket, mix.inputs[1]); links.new(clear.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    _route_output(faded, insert)
    return faded


def _packed_image(name, rgb, mapping):
    rows, cols = rgb.shape[:2]
    image = bpy.data.images.new(name, cols, rows, alpha=False, float_buffer=False)
    image.colorspace_settings.name = "Non-Color"
    rgba = np.concatenate((rgb.astype(np.float32) / 255, np.ones((rows, cols, 1), np.float32)), axis=2)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
    image["owb_mapping"] = json.dumps(mapping)
    return image


def ground_detail_image(name, pixels, mapping):
    """Two opaque packed images. Alpha is never used for data: a packed PNG is
    reloaded premultiplied when geometry nodes sample it, which would erase
    the colour channels wherever the alpha data is low."""
    image = _packed_image(name, pixels[..., :3], mapping)
    _packed_image(name + " rock", np.repeat(pixels[..., 3:], 3, axis=2), mapping)
    return image


def _ground_detail(nodes, links, base, bump_normal, shader, image):
    """Paths, soil, paving and exposed rock painted in world space onto terrain."""
    mapping = json.loads(image["owb_mapping"])
    x0, z0 = mapping["origin"]; size = mapping["size"]
    coordinates = nodes.new("ShaderNodeTexCoord")
    place = nodes.new("ShaderNodeMapping")
    place.inputs["Scale"].default_value = (1 / size, -1 / size, 1)
    place.inputs["Location"].default_value = (-x0 / size, -z0 / size, 0)
    links.new(coordinates.outputs["Object"], place.inputs["Vector"])
    texture = nodes.new("ShaderNodeTexImage"); texture.image = image
    texture.interpolation = "Linear"; texture.extension = "CLIP"
    links.new(place.outputs["Vector"], texture.inputs["Vector"])
    split = nodes.new("ShaderNodeSeparateColor")
    links.new(texture.outputs["Color"], split.inputs["Color"])
    # Only upward-facing ground receives the painted detail; cliffs keep rock.
    geometry = nodes.new("ShaderNodeNewGeometry")
    normal = nodes.new("ShaderNodeSeparateXYZ"); links.new(geometry.outputs["Normal"], normal.inputs["Vector"])
    upward = nodes.new("ShaderNodeMapRange")
    upward.inputs["From Min"].default_value = .55; upward.inputs["From Max"].default_value = .82
    links.new(normal.outputs["Z"], upward.inputs["Value"])
    grain = nodes.new("ShaderNodeTexNoise"); grain.inputs["Scale"].default_value = 3.2
    grain.inputs["Detail"].default_value = 4
    links.new(coordinates.outputs["Object"], grain.inputs["Vector"])
    def weight(channel):
        # Ragged, crisp borders: the painted value is thresholded against noise.
        jitter = nodes.new("ShaderNodeMath"); jitter.operation = "MULTIPLY_ADD"
        links.new(grain.outputs["Fac"], jitter.inputs[0]); jitter.inputs[1].default_value = .55
        links.new(channel, jitter.inputs[2])
        crisp = nodes.new("ShaderNodeMapRange")
        crisp.inputs["From Min"].default_value = .55; crisp.inputs["From Max"].default_value = .78
        links.new(jitter.outputs[0], crisp.inputs["Value"])
        node = nodes.new("ShaderNodeMath"); node.operation = "MULTIPLY"
        links.new(crisp.outputs[0], node.inputs[0]); links.new(upward.outputs[0], node.inputs[1])
        return node.outputs[0]
    path, soil, paving = (weight(split.outputs[i]) for i in range(3))
    rock_texture = nodes.new("ShaderNodeTexImage"); rock_texture.image = bpy.data.images[image.name + " rock"]
    rock_texture.interpolation = "Linear"; rock_texture.extension = "CLIP"
    links.new(place.outputs["Vector"], rock_texture.inputs["Vector"])
    rock = weight(rock_texture.outputs["Color"])
    def tone(dark, light):
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*linear(np.array(dark)), 1)
        ramp.color_ramp.elements[1].color = (*linear(np.array(light)), 1)
        links.new(grain.outputs["Fac"], ramp.inputs["Fac"])
        return ramp.outputs["Color"]
    # Irregular weathered flagstones: one Voronoi cell per stone, mossy joints.
    cells = nodes.new("ShaderNodeTexVoronoi"); cells.feature = "F1"
    joints = nodes.new("ShaderNodeTexVoronoi"); joints.feature = "DISTANCE_TO_EDGE"
    for node in (cells, joints):
        node.inputs["Scale"].default_value = 1.7
        node.inputs["Randomness"].default_value = .8
        links.new(coordinates.outputs["Object"], node.inputs["Vector"])
    shade = nodes.new("ShaderNodeSeparateColor"); links.new(cells.outputs["Color"], shade.inputs["Color"])
    stone_tone = nodes.new("ShaderNodeValToRGB")
    stone_tone.color_ramp.elements[0].color = (*linear(np.array((.36, .34, .30))), 1)
    stone_tone.color_ramp.elements[1].color = (*linear(np.array((.50, .47, .41))), 1)
    links.new(shade.outputs[0], stone_tone.inputs["Fac"])
    grime = nodes.new("ShaderNodeMixRGB"); grime.blend_type = "MULTIPLY"; grime.inputs[0].default_value = .55
    links.new(stone_tone.outputs["Color"], grime.inputs[1]); links.new(tone((.62, .62, .58), (1, 1, 1)), grime.inputs[2])
    seam = nodes.new("ShaderNodeMapRange")
    seam.inputs["From Min"].default_value = .012; seam.inputs["From Max"].default_value = .04
    links.new(joints.outputs["Distance"], seam.inputs["Value"])
    flag = nodes.new("ShaderNodeMixRGB")
    links.new(seam.outputs[0], flag.inputs[0])
    flag.inputs[1].default_value = (*linear(np.array((.20, .24, .13))), 1)
    links.new(grime.outputs[0], flag.inputs[2])
    # The grass carpet uses this same world-space noise for its clumps; the
    # ground beneath a clump darkens with it so grass and soil read as one surface.
    clumps = nodes.new("ShaderNodeTexNoise"); clumps.inputs["Scale"].default_value = .22
    clumps.inputs["Detail"].default_value = 2
    links.new(coordinates.outputs["Object"], clumps.inputs["Vector"])
    under = nodes.new("ShaderNodeMapRange")
    under.inputs["From Min"].default_value = .45; under.inputs["From Max"].default_value = .7
    under.inputs["To Max"].default_value = .55
    links.new(clumps.outputs["Fac"], under.inputs["Value"])
    shaded = nodes.new("ShaderNodeMixRGB"); shaded.blend_type = "MULTIPLY"
    links.new(under.outputs[0], shaded.inputs[0]); links.new(base, shaded.inputs[1])
    shaded.inputs[2].default_value = (*linear(np.array((.62, .70, .50))), 1)
    color = shaded.outputs[0]
    for factor, layer in ((rock, tone((.34, .33, .31), (.52, .51, .47))),
                          (soil, tone((.24, .19, .13), (.37, .30, .21))),
                          (path, tone((.38, .33, .24), (.52, .46, .34))),
                          (paving, flag.outputs[0])):
        mix = nodes.new("ShaderNodeMixRGB")
        links.new(factor, mix.inputs[0]); links.new(color, mix.inputs[1]); links.new(layer, mix.inputs[2])
        color = mix.outputs[0]
    links.new(color, shader.inputs["Base Color"])
    groove = nodes.new("ShaderNodeMath"); groove.operation = "MULTIPLY"
    inverted = nodes.new("ShaderNodeMath"); inverted.operation = "SUBTRACT"; inverted.inputs[0].default_value = 1
    links.new(seam.outputs[0], inverted.inputs[1])
    links.new(inverted.outputs[0], groove.inputs[0]); links.new(paving, groove.inputs[1])
    bump = nodes.new("ShaderNodeBump")
    bump.invert = True
    bump.inputs["Strength"].default_value = .5; bump.inputs["Distance"].default_value = .03
    links.new(groove.outputs[0], bump.inputs["Height"])
    links.new(bump_normal, bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])


# ---------------------------------------------------------------------------
# Optional "anime" presentation style (metadata["style"]). Opt-in per snapshot:
# snapshots without the key render exactly as before.
# ---------------------------------------------------------------------------
STYLE = None
ANIME = dict(
    view="Standard", look="None", exposure=0.0, sun_scale=0.72, sun_color=(1.0, 0.95, 0.84), sky_strength=1.0,
    zenith=(0.11, 0.36, 0.86), horizon=(0.66, 0.86, 0.98), toon=0.85, toon_size=0.55, toon_smooth=0.16,
    haze=dict(start=140.0, distance=1500.0, amount=0.8), haze_color=(0.66, 0.83, 0.97),
    saturation=dict(foliage=1.12, cloth=1.2, stone=0.9, bark=1.05, metal=1.0, skin=1.08, surface=1.12),
    value=dict(foliage=1.18, cloth=1.1, stone=1.3, bark=1.35, metal=1.1, skin=1.05, surface=1.12),
    grass=((0.36, 0.63, 0.22), (0.48, 0.72, 0.27)), dry=(0.62, 0.74, 0.32),
    rock=((0.70, 0.64, 0.56), (0.56, 0.52, 0.50)), rock_dark=(0.34, 0.33, 0.40), rock_light=(0.86, 0.80, 0.70),
    sand=(0.90, 0.84, 0.64),
    water=(0.06, 0.55, 0.70), clouds=0.5)


def style_settings(metadata):
    """Resolved style parameters, or None for the default look."""
    spec = metadata.get("style")
    if not spec:
        return None
    if spec.get("name") != "anime":
        raise ValueError(f"unknown presentation style: {spec.get('name')}")
    return dict(ANIME, **{k: v for k, v in spec.items() if k != "name"})


def _rgb(nodes, color):
    node = nodes.new("ShaderNodeRGB")
    node.outputs[0].default_value = (*linear(np.asarray(color, float)), 1)
    return node.outputs[0]


def _mix(nodes, links, factor, a, b, blend="MIX"):
    node = nodes.new("ShaderNodeMixRGB"); node.blend_type = blend
    for socket, value in zip(node.inputs, (factor, a, b)):
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            links.new(value, socket)
    return node.outputs[0]


def _range(nodes, links, value, lo, hi, to_min=0.0, to_max=1.0):
    node = nodes.new("ShaderNodeMapRange")
    node.inputs["From Min"].default_value = lo; node.inputs["From Max"].default_value = hi
    node.inputs["To Min"].default_value = to_min; node.inputs["To Max"].default_value = to_max
    links.new(value, node.inputs["Value"])
    return node.outputs[0]


def _noise(nodes, links, vector, scale, detail=2.0):
    node = nodes.new("ShaderNodeTexNoise")
    node.inputs["Scale"].default_value = scale; node.inputs["Detail"].default_value = detail
    links.new(vector, node.inputs["Vector"])
    return node.outputs["Fac"]


def _anime_terrain(nodes, links, shader, style):
    """Grass, rock and sand chosen per pixel from slope and height, not per triangle.

    Grass covers level ground in two greens with dry patches; faces show light
    rock with horizontal strata and darker crevices; the shore is sand."""
    coordinates = nodes.new("ShaderNodeTexCoord").outputs["Object"]
    geometry = nodes.new("ShaderNodeNewGeometry")
    normal = nodes.new("ShaderNodeSeparateXYZ"); links.new(geometry.outputs["Normal"], normal.inputs["Vector"])
    position = nodes.new("ShaderNodeSeparateXYZ"); links.new(geometry.outputs["Position"], position.inputs["Vector"])
    broad, fine = _noise(nodes, links, coordinates, .018, 3), _noise(nodes, links, coordinates, .11, 3)
    grass = _mix(nodes, links, _range(nodes, links, broad, .35, .7), _rgb(nodes, style["grass"][0]), _rgb(nodes, style["grass"][1]))
    grass = _mix(nodes, links, _range(nodes, links, fine, .56, .78, 0, .55), grass, _rgb(nodes, style["dry"]))
    # Strata: bands in height, bent by noise so they do not read as ruled lines.
    bend = nodes.new("ShaderNodeMath"); bend.operation = "MULTIPLY_ADD"
    links.new(_noise(nodes, links, coordinates, .05, 2), bend.inputs[0]); bend.inputs[1].default_value = 9.0
    links.new(position.outputs["Z"], bend.inputs[2])
    bands = nodes.new("ShaderNodeMath"); bands.operation = "PINGPONG"
    links.new(bend.outputs[0], bands.inputs[0]); bands.inputs[1].default_value = 3.4
    rock = _mix(nodes, links, _range(nodes, links, bands.outputs[0], .6, 2.6), _rgb(nodes, style["rock"][0]), _rgb(nodes, style["rock"][1]))
    stretch = nodes.new("ShaderNodeVectorMath"); stretch.operation = "MULTIPLY"
    stretch.inputs[1].default_value = (.9, .9, .12); links.new(coordinates, stretch.inputs[0])
    crevice = _range(nodes, links, _noise(nodes, links, stretch.outputs["Vector"], .5, 4), .58, .72, 0, .7)
    rock = _mix(nodes, links, crevice, rock, _rgb(nodes, style["rock_dark"]))
    # Faceted rock: tall cells read as fractured columns. Each cell takes one flat tone,
    # its edges darken into cracks, and its distance field tilts the shading like a cut face.
    columns = nodes.new("ShaderNodeVectorMath"); columns.operation = "MULTIPLY"
    columns.inputs[1].default_value = (1.0, 1.0, .3); links.new(coordinates, columns.inputs[0])
    def cells(feature, scale):
        node = nodes.new("ShaderNodeTexVoronoi"); node.feature = feature
        node.inputs["Scale"].default_value = scale; node.inputs["Randomness"].default_value = .9
        links.new(columns.outputs["Vector"], node.inputs["Vector"])
        return node
    big, edge, chips = cells("F1", .085), cells("DISTANCE_TO_EDGE", .085), cells("F1", .32)
    tone = nodes.new("ShaderNodeSeparateColor"); links.new(big.outputs["Color"], tone.inputs["Color"])
    rock = _mix(nodes, links, _range(nodes, links, tone.outputs[0], 0, 1, .0, .75), rock, _rgb(nodes, style["rock_light"]), "MIX")
    rock = _mix(nodes, links, _range(nodes, links, tone.outputs[1], .5, 1, .0, .55), rock, _rgb(nodes, style["rock_dark"]))
    rock = _mix(nodes, links, _range(nodes, links, edge.outputs["Distance"], .05, .0, 0, .8), rock, _rgb(nodes, style["rock_dark"]))
    relief = nodes.new("ShaderNodeMath"); relief.operation = "MULTIPLY_ADD"
    links.new(chips.outputs["Distance"], relief.inputs[0]); relief.inputs[1].default_value = .25
    links.new(big.outputs["Distance"], relief.inputs[2])
    ragged = nodes.new("ShaderNodeMath"); ragged.operation = "MULTIPLY_ADD"
    links.new(_noise(nodes, links, coordinates, .6, 3), ragged.inputs[0]); ragged.inputs[1].default_value = .16
    links.new(normal.outputs["Z"], ragged.inputs[2])
    slope = _range(nodes, links, ragged.outputs[0], .93, .80)          # 0 on level ground, 1 on faces
    color = _mix(nodes, links, slope, grass, rock)
    shore = nodes.new("ShaderNodeMath"); shore.operation = "MULTIPLY"
    links.new(_range(nodes, links, position.outputs["Z"], 2.6, 1.2), shore.inputs[0])
    links.new(_range(nodes, links, normal.outputs["Z"], .7, .9), shore.inputs[1])
    color = _mix(nodes, links, shore.outputs[0], color, _rgb(nodes, style["sand"]))
    links.new(color, shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 1.0
    facets = nodes.new("ShaderNodeBump"); facets.inputs["Distance"].default_value = 6.0
    links.new(relief.outputs[0], facets.inputs["Height"])
    strength = nodes.new("ShaderNodeMath"); strength.operation = "MULTIPLY"; strength.inputs[1].default_value = .9
    links.new(slope, strength.inputs[0]); links.new(strength.outputs[0], facets.inputs["Strength"])
    links.new(facets.outputs["Normal"], shader.inputs["Normal"])
    return color, facets.outputs["Normal"]


def _stylize(mat, kind, style, *, water=False, emission=False, terrain_color=None, normal=None):
    """Saturated albedo and a soft toon diffuse over the physically based base."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    if water:
        shader.inputs["Base Color"].default_value = (*linear(np.asarray(style["water"])), 1)
        for link in list(shader.inputs["Base Color"].links):
            links.remove(link)
        shader.inputs["Roughness"].default_value = .08
        shader.inputs["Metallic"].default_value = .0
        shader.inputs["Coat Weight"].default_value = .6
        return
    if emission:
        return
    source = terrain_color if terrain_color is not None else shader.inputs["Base Color"].links[0].from_socket
    if terrain_color is None:
        grade = nodes.new("ShaderNodeHueSaturation")
        grade.inputs["Saturation"].default_value = style["saturation"].get(kind, 1.1)
        grade.inputs["Value"].default_value = style["value"].get(kind, 1.1)
        links.new(source, grade.inputs["Color"])
        source = grade.outputs["Color"]
        links.new(source, shader.inputs["Base Color"])
        for node in nodes:                       # foliage also feeds a translucent lobe
            if node.type == "BSDF_TRANSLUCENT":
                links.new(source, node.inputs["Color"])
    shader.inputs["Specular IOR Level"].default_value = .12 if kind == "metal" else .02
    toon = nodes.new("ShaderNodeBsdfToon"); toon.component = "DIFFUSE"
    toon.inputs["Size"].default_value = style["toon_size"]; toon.inputs["Smooth"].default_value = style["toon_smooth"]
    links.new(source, toon.inputs["Color"])
    if normal is not None:
        links.new(normal, toon.inputs["Normal"])
    def insert(nodes_, links_, socket):
        mix = nodes_.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = style["toon"] * (.55 if kind == "foliage" else 1.0)   # leaves keep soft light
        links_.new(socket, mix.inputs[1]); links_.new(toon.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    _route_output(mat, insert)


def _anime_sky(nodes, links, background, style, sun_direction):
    """Deep zenith, pale horizon and soft cumulus banks with lit tops."""
    tex = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ"); links.new(tex.outputs["Generated"], separate.inputs["Vector"])
    up = nodes.new("ShaderNodeMath"); up.operation = "ABSOLUTE"; links.new(separate.outputs["Z"], up.inputs[0])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0; ramp.color_ramp.elements[0].color = (*linear(np.array(style["horizon"])), 1)
    ramp.color_ramp.elements[1].position = .62; ramp.color_ramp.elements[1].color = (*linear(np.array(style["zenith"])), 1)
    links.new(up.outputs[0], ramp.inputs["Fac"])
    # Project the view direction onto a plane overhead so clouds recede to the horizon.
    lift = nodes.new("ShaderNodeMath"); lift.operation = "ADD"; links.new(separate.outputs["Z"], lift.inputs[0]); lift.inputs[1].default_value = .22
    plane = nodes.new("ShaderNodeVectorMath"); plane.operation = "DIVIDE"
    links.new(tex.outputs["Generated"], plane.inputs[0])
    spread = nodes.new("ShaderNodeCombineXYZ")
    for k in range(3):
        links.new(lift.outputs[0], spread.inputs[k])
    links.new(spread.outputs[0], plane.inputs[1])
    def density(offset):
        shift = nodes.new("ShaderNodeVectorMath"); shift.operation = "ADD"; shift.inputs[1].default_value = offset
        links.new(plane.outputs["Vector"], shift.inputs[0])
        big = _noise(nodes, links, shift.outputs["Vector"], 1.5, 6)
        return big
    cover = _range(nodes, links, density((0, 0, 0)), 1.0 - style["clouds"] - .02, 1.0 - style["clouds"] + .09)
    toward_sun = (float(sun_direction[0]) * .12, float(sun_direction[1]) * .12, 0.0)
    lit = _range(nodes, links, density(toward_sun), 1.0 - style["clouds"] - .06, 1.0 - style["clouds"] + .14)
    shade = _mix(nodes, links, lit, _rgb(nodes, (1.0, .99, .96)), _rgb(nodes, (.72, .78, .92)))
    above = nodes.new("ShaderNodeMath"); above.operation = "MULTIPLY"
    links.new(cover, above.inputs[0]); links.new(_range(nodes, links, separate.outputs["Z"], .015, .12), above.inputs[1])
    sky = _mix(nodes, links, above.outputs[0], ramp.outputs["Color"], shade)
    links.new(sky, background.inputs["Color"])


def outline_material(color=(0.10, 0.08, 0.14)):
    """Ink line for an outline hull: only its back faces show, and only to the camera."""
    mat = bpy.data.materials.new("Ink outline")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.remove(nodes.get("Principled BSDF"))
    ink = nodes.new("ShaderNodeEmission")
    ink.inputs["Color"].default_value = (*linear(np.asarray(color, float)), 1); ink.inputs["Strength"].default_value = 1.0
    clear = nodes.new("ShaderNodeBsdfTransparent")
    geometry, path = nodes.new("ShaderNodeNewGeometry"), nodes.new("ShaderNodeLightPath")
    show = nodes.new("ShaderNodeMath"); show.operation = "MULTIPLY"
    links.new(geometry.outputs["Backfacing"], show.inputs[0]); links.new(path.outputs["Is Camera Ray"], show.inputs[1])
    mix = nodes.new("ShaderNodeMixShader")
    links.new(show.outputs[0], mix.inputs[0]); links.new(clear.outputs[0], mix.inputs[1]); links.new(ink.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get("Material Output").inputs["Surface"])
    return mat


def material(name, *, emission=False, water=False, kind="surface", ground=None, volume=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    colors = nodes.new("ShaderNodeVertexColor")
    colors.layer_name = "Color"
    links.new(colors.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 0.82
    shader.inputs["Specular IOR Level"].default_value = 0.25
    if not water and not emission:
        coordinates=nodes.new("ShaderNodeTexCoord")
        vector=coordinates.outputs["Object"]
        if kind=="bark":
            stretch=nodes.new("ShaderNodeVectorMath")
            stretch.operation="MULTIPLY"
            stretch.inputs[1].default_value=(7,7,.55)
            links.new(vector,stretch.inputs[0])
            vector=stretch.outputs["Vector"]
        coarse=nodes.new("ShaderNodeTexNoise")
        coarse.inputs["Scale"].default_value={"terrain":.13,"stone":3.5,"bark":2.4,"metal":5,"cloth":95}.get(kind,4)
        coarse.inputs["Detail"].default_value=3
        coarse.inputs["Roughness"].default_value=.7
        links.new(vector,coarse.inputs["Vector"])
        variation=nodes.new("ShaderNodeValToRGB")
        variation.color_ramp.elements[0].color=(.56,.56,.56,1) if kind in ("terrain","stone","bark") else (.83,.83,.83,1)
        if STYLE is not None:                 # flat colour blocks: keep only a trace of mottling
            variation.color_ramp.elements[0].color=(.84,.84,.84,1) if kind in ("terrain","stone","bark") else (.93,.93,.93,1)
        variation.color_ramp.elements[1].color=(1.06,1.06,1.06,1)
        links.new(coarse.outputs["Fac"],variation.inputs["Fac"])
        multiply=nodes.new("ShaderNodeMixRGB")
        multiply.blend_type="MULTIPLY";multiply.inputs[0].default_value=1
        links.new(colors.outputs["Color"],multiply.inputs[1])
        links.new(variation.outputs["Color"],multiply.inputs[2])
        links.new(multiply.outputs["Color"],shader.inputs["Base Color"])
        fine=nodes.new("ShaderNodeTexNoise")
        fine.inputs["Scale"].default_value={"terrain":9,"stone":28,"bark":7,"metal":38,"cloth":180}.get(kind,35)
        fine.inputs["Detail"].default_value=2
        links.new(vector,fine.inputs["Vector"])
        bump=nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value={"terrain":.32,"stone":.35,"bark":.40,"cloth":.16,"foliage":.07,"skin":.06}.get(kind,.12)
        bump.inputs["Distance"].default_value={"terrain":.055,"stone":.035,"bark":.025,"cloth":.004,"foliage":.002,"skin":.002}.get(kind,.008)
        links.new(fine.outputs["Fac"],bump.inputs["Height"])
        links.new(bump.outputs["Normal"],shader.inputs["Normal"])
        shader.inputs["Roughness"].default_value={"stone":.88,"terrain":.95,"bark":.90,"foliage":.68,"metal":.48,"cloth":.92,"skin":.6}.get(kind,.82)
        if kind=="metal":
            shader.inputs["Metallic"].default_value=.72
        if kind=="skin":
            shader.inputs["Subsurface Weight"].default_value=.05
        if kind=="foliage":
            shader.inputs["Subsurface Weight"].default_value=.06
            translucent=nodes.new("ShaderNodeBsdfTranslucent")
            links.new(multiply.outputs["Color"],translucent.inputs["Color"])
            mix=nodes.new("ShaderNodeMixShader");mix.inputs[0].default_value=.14
            links.new(shader.outputs["BSDF"],mix.inputs[1])
            links.new(translucent.outputs[0],mix.inputs[2])
            links.new(mix.outputs[0],nodes.get("Material Output").inputs["Surface"])
        if kind=="terrain":
            # Earth patches are tied to world coordinates and vertex biome color.
            patches=nodes.new("ShaderNodeTexNoise")
            patches.inputs["Scale"].default_value=.045
            patches.inputs["Detail"].default_value=3
            links.new(coordinates.outputs["Object"],patches.inputs["Vector"])
            mask=nodes.new("ShaderNodeMapRange")
            mask.inputs["From Min"].default_value=.50
            mask.inputs["From Max"].default_value=.78
            mask.inputs["To Max"].default_value=.48
            links.new(patches.outputs["Fac"],mask.inputs["Value"])
            earth=nodes.new("ShaderNodeMixRGB")
            links.new(mask.outputs[0],earth.inputs[0])
            links.new(multiply.outputs[0],earth.inputs[1])
            earth.inputs[2].default_value=(*linear(np.array((.40,.35,.23))),1)
            links.new(earth.outputs[0],shader.inputs["Base Color"])
            if ground is not None:
                _ground_detail(nodes,links,earth.outputs[0],bump.outputs["Normal"],shader,ground)
    if emission:
        links.new(colors.outputs["Color"], shader.inputs["Emission Color"])
        shader.inputs["Emission Strength"].default_value = 2.0
    if water and volume:
        _clear_water(mat, shader, colors)
    elif water:
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
    if STYLE is not None:
        terrain = _anime_terrain(nodes, links, shader, STYLE) if kind == "terrain" and ground is None else (None, None)
        _stylize(mat, kind, STYLE, water=water, emission=emission, terrain_color=terrain[0], normal=terrain[1])
    if not water and not emission:
        add_wetness(mat)
    mat = add_haze(mat)
    if not emission:
        # Haze is distance fog seen by rays, not a light. With AUTO emission
        # sampling every hazed triangle (tens of millions of grass blades) became
        # a light-tree emitter. Rays still gather the haze, so the expected image
        # is unchanged; real glow materials keep light sampling.
        mat.cycles.emission_sampling = "NONE"
    return mat


def _clear_water(mat, shader, colors):
    """Refractive water with depth absorption, two scales of ripples, and
    shadow rays passing the surface so beds stay sunlit (closed water meshes)."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for link in list(shader.inputs["Base Color"].links):
        links.remove(link)
    shader.inputs["Base Color"].default_value = (0.93, 0.98, 0.97, 1)
    shader.inputs["Transmission Weight"].default_value = 1.0
    shader.inputs["Roughness"].default_value = 0.03
    shader.inputs["IOR"].default_value = 1.333
    shader.inputs["Specular IOR Level"].default_value = 0.5
    coordinates = nodes.new("ShaderNodeTexCoord")
    normal = None
    for scale, strength, distance in ((0.32, 0.22, 0.3), (3.2, 0.08, 0.04)):
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = scale
        noise.inputs["Detail"].default_value = 4
        noise.inputs["Roughness"].default_value = 0.55
        links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = strength
        bump.inputs["Distance"].default_value = distance
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        if normal is not None:
            links.new(normal, bump.inputs["Normal"])
        normal = bump.outputs["Normal"]
    links.new(normal, shader.inputs["Normal"])
    path = nodes.new("ShaderNodeLightPath")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.inputs["Color"].default_value = (0.9, 0.96, 0.95, 1)
    mix = nodes.new("ShaderNodeMixShader")
    links.new(path.outputs["Is Shadow Ray"], mix.inputs[0])
    links.new(shader.outputs["BSDF"], mix.inputs[1])
    links.new(transparent.outputs["BSDF"], mix.inputs[2])
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    links.new(mix.outputs[0], output.inputs["Surface"])
    absorb = nodes.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (0.36, 0.62, 0.55, 1)
    absorb.inputs["Density"].default_value = 0.25
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (0.55, 0.75, 0.7, 1)
    scatter.inputs["Density"].default_value = 0.012
    add = nodes.new("ShaderNodeAddShader")
    links.new(absorb.outputs[0], add.inputs[0])
    links.new(scatter.outputs[0], add.inputs[1])
    links.new(add.outputs[0], output.inputs["Volume"])


MATERIAL_KINDS = ("terrain", "stone", "bark", "foliage", "metal", "cloth", "skin")


def build_materials(metadata, data=None):
    """All shared materials; a snapshot's ground-detail texture goes on terrain."""
    ground = None
    spec = metadata.get("textures", {}).get("ground_detail")
    if spec is not None and data is not None:
        ground = ground_detail_image("Ground detail", data["texture_ground_detail"], spec)
    materials = dict(surface=material("Painted surfaces"),
                     emission=material("Elemental glow", emission=True),
                     water=material("Water", water=True, volume=bool(metadata.get("water_volume"))))
    for kind in MATERIAL_KINDS:
        materials[kind] = material(kind.title(), kind=kind, ground=ground if kind == "terrain" else None)
    if STYLE is not None:
        materials["outline"] = outline_material()
    return materials


def mesh_material(materials, spec):
    """The material for a mesh spec, creating faded occluder variants on demand."""
    if spec["water"]:
        return materials["water"]
    base = spec.get("material", "surface")
    if not spec.get("occluder"):
        return materials.get(base, materials["surface"])
    key = base + ":occluder"
    if key not in materials:
        materials[key] = occluder_variant(materials.get(base, materials["surface"]))
    if "emission:occluder" not in materials:
        materials["emission:occluder"] = occluder_variant(materials["emission"])
    return materials[key]


# Blender's bulk attribute transfer fails on very large arrays: the smooth
# Skylands campaign terrain (60M triangles) needs 722M per-corner colour floats
# and foreach_set raises "Array length mismatch". Exported colours are per
# triangle, so larger meshes store them once per face instead (a third of the
# floats) and read them through an Attribute node, as baked grass already does.
# The rendered colour is identical.
FACE_COLOR_TRIANGLES = 20_000_000


def face_color_material(base):
    """Copy of ``base`` reading the face-domain "Color" attribute."""
    name = base.name + " (face colors)"
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = base.copy()
    mat.name = name
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for old in [n for n in nodes if n.bl_idname == "ShaderNodeVertexColor"]:
        new = nodes.new("ShaderNodeAttribute")
        new.attribute_type = "GEOMETRY"
        new.attribute_name = old.layer_name
        for socket in ("Color", "Alpha"):
            for link in list(old.outputs[socket].links):
                links.new(new.outputs[socket], link.to_socket)
        nodes.remove(old)
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
    blend = bool(spec.get("blend"))
    per_face = not blend and len(triangles) > FACE_COLOR_TRIANGLES
    colors = linear(np.asarray(data[prefix + "colors"]))
    if blend:
        # Area-weighted average of the adjacent triangles' colours per welded
        # vertex, interpolated across faces.
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        weight = np.repeat(area, 3)
        total = np.bincount(inverse, weight, minlength=len(vertices))
        colors = np.stack([np.bincount(inverse, weight * np.repeat(colors[:, k], 3), minlength=len(vertices))
                           for k in range(3)], axis=1) / np.maximum(total, 1e-12)[:, None]
    elif not per_face:
        colors = np.repeat(colors, 3, axis=0)
    rgba = np.concatenate((colors, np.ones((len(colors), 1))), axis=1).astype(np.float32)
    if blend:
        attr = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
        attr.data.foreach_set("color", rgba.ravel())
        mesh.color_attributes.active_color = attr
    elif per_face:
        print(f"  per-face colors ({len(triangles):,} triangles)", flush=True)
        attr = mesh.attributes.new(name="Color", type="FLOAT_COLOR", domain="FACE")
        attr.data.foreach_set("color", rgba.ravel())
    else:
        attr = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
        attr.data.foreach_set("color", rgba.ravel())
        mesh.color_attributes.active_color = attr
    adapt = face_color_material if per_face else (lambda m: m)
    mesh.materials.append(adapt(mesh_material(materials, spec)))
    if not spec["water"]:
        mesh.materials.append(adapt(materials["emission:occluder" if spec.get("occluder") else "emission"]))
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


SUN_SCALE = 4.0
SKY_STRENGTH = 0.62
DEVICES = ("CPU", "CUDA", "OPTIX", "METAL")


def _socket(sockets, name):
    """The active socket called ``name``. Blender 4.x keeps one Min/Max/Value
    socket per data type (inactive ones disabled); 5.x shows only the active one."""
    return next(socket for socket in sockets if socket.name == name and socket.enabled)


def ground_cover(spec, objects, image=None):
    """Instance short grass on terrain near the camera, kept off paths and paving.

    Geometry nodes scatter world-anchored prototypes (a fixed seed, so clumps
    stay put as the camera moves); nothing is added to the exported snapshot.
    """
    target = objects[spec["target"]]
    collection = bpy.data.collections.new("Meadow grass prototypes")
    for name in spec["prototypes"]:
        obj = objects[name]
        for parent in list(obj.users_collection):
            parent.objects.unlink(obj)
        collection.objects.link(obj)
    group = bpy.data.node_groups.new("Meadow carpet", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    source, result = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    def vmath(op, a, b=None, value=None):
        node = nodes.new("ShaderNodeVectorMath"); node.operation = op
        links.new(a, node.inputs[0])
        if b is not None: links.new(b, node.inputs[1])
        if value is not None: node.inputs[1].default_value = value
        return node.outputs["Value" if op in ("DISTANCE", "LENGTH", "DOT_PRODUCT") else "Vector"]
    def math(op, a, b):
        node = nodes.new("ShaderNodeMath"); node.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)): node.inputs[i].default_value = v
            else: links.new(v, node.inputs[i])
        return node.outputs[0]
    camera = nodes.new("GeometryNodeObjectInfo"); camera.inputs["Object"].default_value = bpy.context.scene.camera
    position = nodes.new("GeometryNodeInputPosition").outputs["Position"]
    near = math("LESS_THAN", vmath("DISTANCE", position, camera.outputs["Location"]), spec["radius"] + 6)
    if "min_up" in spec:                          # level ground above the shore only
        facing = nodes.new("ShaderNodeSeparateXYZ"); links.new(nodes.new("GeometryNodeInputNormal").outputs[0], facing.inputs[0])
        height = nodes.new("ShaderNodeSeparateXYZ"); links.new(position, height.inputs[0])
        near = math("MULTIPLY", near, math("GREATER_THAN", facing.outputs["Z"], spec["min_up"]))
        near = math("MULTIPLY", near, math("GREATER_THAN", height.outputs["Z"], spec.get("min_height", 0.0)))
    scatter = nodes.new("GeometryNodeDistributePointsOnFaces")
    scatter.inputs["Density"].default_value = spec["density"]
    scatter.inputs["Seed"].default_value = spec.get("seed", 0) % 10000
    links.new(source.outputs[0], scatter.inputs["Mesh"]); links.new(near, scatter.inputs["Selection"])
    point = nodes.new("GeometryNodeInputPosition").outputs["Position"]
    distance = vmath("DISTANCE", point, camera.outputs["Location"])
    keep = math("LESS_THAN", distance, spec["radius"])
    clumps = nodes.new("ShaderNodeTexNoise"); clumps.inputs["Scale"].default_value = .22
    clumps.inputs["Detail"].default_value = 2
    links.new(point, clumps.inputs["Vector"])
    density = math("MULTIPLY", math("SUBTRACT", clumps.outputs["Fac"], spec.get("clump_floor", .45)), 2.6)
    if image is not None:
        mapping = json.loads(image["owb_mapping"])
        x0, z0 = mapping["origin"]; size = mapping["size"]
        uv = vmath("ADD", vmath("MULTIPLY", point, None, (1 / size, -1 / size, 0)), None, (-x0 / size, -z0 / size, 0))
        texture = nodes.new("GeometryNodeImageTexture"); texture.inputs["Image"].default_value = image
        texture.extension = "CLIP"
        links.new(uv, texture.inputs["Vector"])
        split = nodes.new("FunctionNodeSeparateColor"); links.new(texture.outputs["Color"], split.inputs["Color"])
        rock = nodes.new("GeometryNodeImageTexture"); rock.inputs["Image"].default_value = bpy.data.images[image.name + " rock"]
        rock.extension = "CLIP"; links.new(uv, rock.inputs["Vector"])
        rock_value = nodes.new("FunctionNodeSeparateColor"); links.new(rock.outputs["Color"], rock_value.inputs["Color"])
        bare = math("MAXIMUM", math("MAXIMUM", split.outputs[0], split.outputs[2]), math("MULTIPLY", rock_value.outputs[0], .8))
        density = math("MULTIPLY", density, math("SUBTRACT", 1.0, math("MULTIPLY", bare, 1.6)))
    chance = nodes.new("FunctionNodeRandomValue"); chance.data_type = "FLOAT"
    chance.inputs["Seed"].default_value = 11
    keep = math("MULTIPLY", keep, math("LESS_THAN", _socket(chance.outputs, "Value"), density))
    prototypes = nodes.new("GeometryNodeCollectionInfo"); prototypes.inputs["Collection"].default_value = collection
    prototypes.inputs["Separate Children"].default_value = True; prototypes.inputs["Reset Children"].default_value = True
    pick = nodes.new("FunctionNodeRandomValue"); pick.data_type = "INT"
    _socket(pick.inputs, "Min").default_value = 0; _socket(pick.inputs, "Max").default_value = len(spec["prototypes"]) - 1
    pick.inputs["Seed"].default_value = 12
    spin = nodes.new("FunctionNodeRandomValue"); spin.data_type = "FLOAT_VECTOR"
    _socket(spin.inputs, "Min").default_value = (0, 0, 0); _socket(spin.inputs, "Max").default_value = (0, 0, 6.283)
    spin.inputs["Seed"].default_value = 13
    euler = nodes.new("FunctionNodeEulerToRotation"); links.new(_socket(spin.outputs, "Value"), euler.inputs[0])
    size_ = nodes.new("FunctionNodeRandomValue"); size_.data_type = "FLOAT"
    _socket(size_.inputs, "Min").default_value = .75; _socket(size_.inputs, "Max").default_value = 1.35; size_.inputs["Seed"].default_value = 14
    instances = nodes.new("GeometryNodeInstanceOnPoints"); instances.inputs["Pick Instance"].default_value = True
    links.new(scatter.outputs["Points"], instances.inputs["Points"]); links.new(keep, instances.inputs["Selection"])
    links.new(prototypes.outputs[0], instances.inputs["Instance"]); links.new(_socket(pick.outputs, "Value"), instances.inputs["Instance Index"])
    links.new(euler.outputs[0], instances.inputs["Rotation"]); links.new(_socket(size_.outputs, "Value"), instances.inputs["Scale"])
    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(source.outputs[0], join.inputs[0]); links.new(instances.outputs[0], join.inputs[0])
    links.new(join.outputs[0], result.inputs[0])
    modifier = target.modifiers.new("Meadow carpet near camera", "NODES"); modifier.node_group = group
    print(f"Meadow carpet: radius {spec['radius']} m, density {spec['density']}/m2", flush=True)


def setup_scene(metadata, samples, threads, device="CPU"):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    if device != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = device
        preferences.get_devices()
        accelerators = [item for item in preferences.devices if item.type == device]
        if not accelerators:
            raise RuntimeError(f"No {device} Cycles device available; refusing CPU fallback")
        for item in preferences.devices:
            item.use = item.type == device
        scene.cycles.device = "GPU"
        print(f"Cycles {device}: " + ", ".join(item.name for item in accelerators), flush=True)
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
    # Neutral exposure: the earlier +0.5 washed out ground and lifted green bounce.
    scene.view_settings.exposure = 0.0
    if STYLE is not None:
        scene.view_settings.view_transform = STYLE["view"]
        scene.view_settings.look = STYLE["look"]
        scene.view_settings.exposure = STYLE["exposure"]
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
    grade = metadata.get("night_grade") or {}
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = (max(0, lighting["strength"]) * SUN_SCALE * (STYLE["sun_scale"] if STYLE else 1.0)
                  * grade.get("moon", 1.0))
    sun.color = STYLE["sun_color"] if STYLE else lighting["color"]
    sun.angle = np.radians(4)
    lamp = bpy.data.objects.new("Sun", sun)
    bpy.context.collection.objects.link(lamp)
    lamp.rotation_euler = Vector(-z_up(lighting["direction"])).to_track_quat("-Z", "Y").to_euler()

    world = bpy.data.worlds.new("Painted sky")
    world.use_nodes = True
    scene.world = world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = nodes.get("Background")
    background.inputs["Strength"].default_value = SKY_STRENGTH * grade.get("sky", 1.0)
    if grade:
        scene.view_settings.exposure += grade.get("exposure", 0.0)
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
    if STYLE is not None:
        background.inputs["Strength"].default_value = STYLE["sky_strength"]
        _anime_sky(nodes, links, background, STYLE, -z_up(np.asarray(lighting["direction"], float)))
    for index, spec in enumerate(metadata.get("point_lights", [])):
        light = bpy.data.lights.new(f"Task lamp {index}", "POINT")
        light.energy = spec["power"]
        light.color = spec["color"]
        light.shadow_soft_size = spec["radius"]
        lamp_obj = bpy.data.objects.new(f"Task lamp {index}", light)
        bpy.context.collection.objects.link(lamp_obj)
        lamp_obj.location = z_up(spec["pos"])
    if STYLE is not None:
        haze_group(dict(metadata.get("haze") or {}, **STYLE["haze"]), STYLE["haze_color"], STYLE["sky_strength"])
    else:
        haze_group(metadata.get("haze"), lighting["horizon"], SKY_STRENGTH)
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
    parser.add_argument("--device", choices=DEVICES, default="CPU")
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
        global STYLE
        STYLE = style_settings(metadata)
        scene = setup_scene(metadata, args.samples, args.threads, args.device)
        materials = build_materials(metadata, data)
        objects = {}
        for i, spec in enumerate(metadata["meshes"]):
            objects[spec["name"]] = add_mesh(data, i, spec, materials)
        if metadata.get("ground_cover"):
            ground_cover(metadata["ground_cover"], objects, bpy.data.images.get("Ground detail"))
        apply_weather(scene, metadata)
    scene.render.filepath = str(args.output.resolve())
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    if args.glb:
        bpy.ops.export_scene.gltf(filepath=str(args.glb.resolve()), export_format="GLB",
                                  export_cameras=True, export_lights=True, export_yup=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
