"""Perception-oracle observations: what a perfect object detector reports for the current frame.

An intermediate information condition between ``observation_only`` (native RGB +
text) and ``privileged_state`` (evaluator state, roles, rules). It isolates visual
parsing and localisation from discovery, procedure learning and memory:

* Included: entities whose anchor point projects inside the policy camera frustum,
  is within range, and has clear voxel line of sight from the camera. For each:
  its interaction ID (if any), a *visual* category, its painted symbol colour or
  visible size, world position, distance and bearing from the character.
* Excluded: functional roles (brake, release, inlet...), which milestone an entity
  belongs to, the current objective's location, rules or hidden constants,
  mechanism state, failure reasons, and anything out of frame or occluded.

Objects leave the list as soon as they leave the frame, so remembering where
things are is still the agent's job. Results under this condition are labelled
``information_condition: perception_oracle`` and reported separately.
"""
import math

import numpy as np

from ...render.camera import Camera, resolve_third_person

MAX_RANGE = 48.0
MAX_ENTRIES = 24


def camera(env):
    """The exact camera the policy renderer uses for this frame."""
    c = env.cfg.render_cfg; a = env.agent
    eye = a.pos + np.array((0, env.cfg.sim_cfg.eye_height, 0), np.float32)
    if c.third_person:
        return resolve_third_person(env.world, eye, a.yaw, a.pitch, c.cam_distance, c.cam_pitch_offset), c
    return Camera(eye, a.yaw, a.pitch, c.fov), c


class Frame:
    def __init__(self, env, max_range=MAX_RANGE):
        self.env = env
        self.cam, cfg = camera(env)
        self.cam.fov = cfg.fov
        self.f, self.r, self.u = self.cam.basis()
        self.tan_x = math.tan(math.radians(cfg.fov) / 2)
        self.tan_y = self.tan_x * cfg.height / cfg.width
        self.max_range = min(max_range, cfg.max_dist)

    def sees(self, point):
        d = np.asarray(point, float) - self.cam.pos
        depth = float(d @ self.f)
        dist = float(np.linalg.norm(d))
        if depth <= .1 or dist > self.max_range:
            return False
        if abs(float(d @ self.r)) / depth > self.tan_x or abs(float(d @ self.u)) / depth > self.tan_y:
            return False
        return not self.env.world.raycast(self.cam.pos, d, max(0., dist - .35)).hit

    def entry(self, point, **fields):
        a = self.env.agent; p = np.asarray(point, float); d = p - a.pos
        bearing = math.degrees(math.atan2(d[2], d[0]) - a.yaw)
        bearing = (bearing + 180) % 360 - 180          # + is to the character's right
        return dict(fields, pos=np.round(p, 1).tolist(), distance_m=round(float(np.linalg.norm(d)), 1),
                    bearing_deg=round(bearing))


def _finish(items, limit=MAX_ENTRIES):
    items.sort(key=lambda e: e['distance_m'])
    limit = len(items) if limit is None else limit
    return dict(in_frame=items[:limit], truncated=max(0, len(items) - limit))


def _creatures(frame, env):
    out = []
    for n in getattr(env.npcs, 'npcs', []):
        # Denned creatures are not drawn; ridden ones (the horse under you) are.
        if n.alive and not n.data.get('denned') and frame.sees(np.asarray(n.pos) + (0, n.spec.height / 2, 0)):
            out.append(frame.entry(n.pos, kind='creature', species=n.kind))
    return out


def wildlands(env, limit=MAX_ENTRIES, creatures=True):
    """Posts, plaques, movable blocks, staked plates and creatures visible in this frame.

    ``limit=None`` and ``creatures=False`` give the untruncated fixture list the
    evaluator-side ledger (``owb.bench.memory_ladder``) records."""
    from ...env.wildlands_reference import Y
    frame = Frame(env); items = []
    for tid, obj in sorted(env.targets.items()):
        role = obj['role']
        if role.startswith('weight'):
            m = env.by_id[obj['milestone']]
            body = env.bodies.get(m.state['bodies'][int(role[6:])])
            if body is not None and frame.sees(body.pos):
                items.append(frame.entry(body.pos, id=tid, kind='movable block',
                                         size_m=np.round(2 * np.asarray(body.half), 2).tolist()))
            continue
        plaque = role == 'inspect'
        h = .8 if plaque else 1.05
        centre = np.asarray(obj['pos'], float) - (0, 1 - h / 2, 0)   # the drawn post, as in _extra_entities
        if frame.sees(centre):
            items.append(frame.entry(obj['pos'], id=tid, kind='plaque' if plaque else 'post',
                                     colour=obj['symbol'] or 'grey'))
    for m in env.milestones:
        if m.kind != 'balance':
            continue
        x, z = m.site
        corners = [(cx, Y + .3, cz) for cx in (x + 2.4, x + 4.7) for cz in (z + .5, z + 4.7)]
        if any(frame.sees(c) for c in corners):
            items.append(frame.entry((x + 3.55, Y, z + 2.6), kind='orange-staked ground plate',
                                     corners_xz=[[c[0], c[2]] for c in corners]))
    if creatures:
        items += _creatures(frame, env)
    return _finish(items, limit)


def skylands(env, limit=MAX_ENTRIES, creatures=True):
    """Interaction targets (coarse visual category only) and creatures visible in this frame."""
    frame = Frame(env); items = []
    for tid, obj in sorted(env.targets.items()):
        role = obj['role']; pos = obj['pos']
        if role == 'part':
            part = env.parts[obj['part']]
            if part['mounted'] or env.held_part == obj['part']:
                continue
        if role == 'raft' and obj['owner'] in env.rafts:
            pos = env.rafts[obj['owner']]['pos'] + (0, 1, 0)
        if role == 'cargo' and obj['owner'] in env.cargo:
            pos = env.cargo[obj['owner']]['pos'] + (0, 1, 0)
        if frame.sees(pos):
            items.append(frame.entry(pos, id=tid, kind=role))
    if creatures:
        items += _creatures(frame, env)
    return _finish(items, limit)
