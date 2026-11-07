"""Third-person orbit camera with terrain-aware occlusion pull-in."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import math3d as mth
from .character import CharacterState, MoveState
from .scatter import PropField
from .terrain import Terrain


@dataclass
class CameraConfig:
    distance: float = 6.0
    min_distance: float = 2.8
    height_offset: float = 1.45      # pivot height above the character's feet
    fov_y: float = 58.0
    near: float = 0.08
    far: float = 900.0
    # Pitch is the elevation of the camera above its target: positive lifts
    # the camera and tilts the view down, which is the usual third-person feel.
    pitch_min_deg: float = -22.0     # camera below the shoulder, looking up
    pitch_max_deg: float = 70.0      # camera overhead, looking down
    yaw_rate: float = 2.6            # rad/s at full stick deflection
    pitch_rate: float = 1.7
    follow_smoothing: float = 0.0008  # fraction of gap left after one second
    distance_smoothing: float = 0.002
    collision_margin: float = 0.45
    collide_props: bool = False   # terrain only: foliage should not shove the camera
    auto_pitch_glide_deg: float = 22.0    # rise and look down while gliding


@dataclass
class CameraState:
    yaw: float = 0.0
    pitch: float = 0.18
    distance: float = 6.0
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    eye: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


class OrbitCamera:
    """Spring-follow orbit rig: yaw/pitch from the agent, distance from the world."""

    def __init__(self, terrain: Terrain, props: PropField | None = None, cfg: CameraConfig | None = None):
        self.terrain = terrain
        self.props = props
        self.architecture = None
        self.cfg = cfg or CameraConfig()

    def reset(self, char: CharacterState, yaw: float | None = None) -> CameraState:
        c = self.cfg
        st = CameraState(
            yaw=float(char.yaw if yaw is None else yaw),
            pitch=0.22,
            distance=c.distance,
        )
        st.pivot = char.pos + np.array([0.0, c.height_offset, 0.0], dtype=np.float32)
        st.eye = self._ideal_eye(st, st.distance)
        return st

    # ------------------------------------------------------------------ update
    def update(self, st: CameraState, char: CharacterState, look: np.ndarray, dt: float) -> CameraState:
        c = self.cfg
        look = np.asarray(look, dtype=np.float32).reshape(2)
        st.yaw = float((st.yaw + float(look[0]) * c.yaw_rate * dt + np.pi) % (2.0 * np.pi) - np.pi)
        pitch = st.pitch + float(look[1]) * c.pitch_rate * dt
        st.pitch = float(np.clip(pitch, np.radians(c.pitch_min_deg), np.radians(c.pitch_max_deg)))

        # While gliding, ease the camera down so the landing zone stays in frame.
        if char.state == MoveState.GLIDING:
            st.pitch = mth.damp(st.pitch, np.radians(c.auto_pitch_glide_deg), 0.25, dt)

        target_pivot = char.pos + np.array([0.0, c.height_offset, 0.0], dtype=np.float32)
        st.pivot = mth.damp(st.pivot, target_pivot, c.follow_smoothing, dt).astype(np.float32)

        wanted = self._free_distance(st, c.distance)
        # Snap in instantly when something intrudes, ease back out slowly.
        st.distance = wanted if wanted < st.distance else mth.damp(st.distance, wanted, c.distance_smoothing, dt)
        st.distance = float(np.clip(st.distance, c.min_distance, c.distance))
        st.eye = self._ideal_eye(st, st.distance)
        return st

    # -------------------------------------------------------------- internals
    def forward(self, st: CameraState) -> np.ndarray:
        """Unit view direction (from the eye toward the pivot)."""
        return mth.norm(st.pivot - st.eye)

    @staticmethod
    def boom_dir(yaw: float, pitch: float) -> np.ndarray:
        """Unit vector from the pivot toward the eye.

        Positive pitch puts the eye above the pivot, so the camera looks down.
        """
        cp = np.cos(pitch)
        return np.array([-np.sin(yaw) * cp, np.sin(pitch), np.cos(yaw) * cp], dtype=np.float32)

    def _ideal_eye(self, st: CameraState, distance: float) -> np.ndarray:
        back = self.boom_dir(st.yaw, st.pitch)
        eye = st.pivot + back * distance
        # Never let the eye sink under the ground or the sea surface.
        gh = float(self.terrain.height_at(eye[0], eye[2])) + self.cfg.collision_margin
        eye[1] = max(float(eye[1]), gh, self.terrain.cfg.sea_level - 1.0)
        return eye.astype(np.float32)

    def _free_distance(self, st: CameraState, desired: float) -> float:
        """Shortest unobstructed boom length, by sampling along the boom."""
        c = self.cfg
        back = self.boom_dir(st.yaw, st.pitch)
        samples = np.linspace(c.min_distance, desired, 10, dtype=np.float32)
        pts = st.pivot[None, :] + back[None, :] * samples[:, None]
        gh = self.terrain.height_at(pts[:, 0], pts[:, 2]) + c.collision_margin
        blocked = pts[:, 1] < gh
        if self.architecture is not None:
            blocked |= self.architecture.blocked(pts, c.collision_margin)
        if c.collide_props and self.props is not None and len(self.props) > 0:
            idx = self.props.query_radius(float(st.pivot[0]), float(st.pivot[2]), desired + 2.0)
            idx = idx[self.props.radius[idx] > 0.3] if len(idx) else idx
            for i in idx:
                p = self.props.pos[i]
                d = pts[:, [0, 2]] - p[[0, 2]][None, :]
                near = (d[:, 0] ** 2 + d[:, 1] ** 2) < (self.props.radius[i] + 0.35) ** 2
                blocked |= near & (pts[:, 1] < p[1] + self.props.height[i])
        if bool(blocked.any()):
            first = int(np.argmax(blocked))
            return float(samples[max(first - 1, 0)])
        return float(desired)

    def matrices(self, st: CameraState, aspect: float):
        """(view, projection) matrices for the renderer."""
        view = mth.look_at(st.eye, st.pivot)
        proj = mth.perspective(self.cfg.fov_y, aspect, self.cfg.near, self.cfg.far)
        return view, proj
