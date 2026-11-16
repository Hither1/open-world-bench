"""RoboCasa kitchen for the lemon-tart task.

Importing this module needs robocasa. `owb.tasks` does not import it, so the
milestone tests run without the kitchen assets. The spec is
`docs/mobile-manipulation-tasks.md`.
"""
from __future__ import annotations

import numpy as np
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T
from robocasa.environments.kitchen.kitchen import *

from owb.tasks import lemon_tart_lock
from owb.tasks import lemon_tart_prep as prep
from owb.tasks.lemon_tart import (
    COUNTER,
    DISHWASHER,
    FRIDGE,
    KNOBS,
    OBJECT_NAMES,
    RECIPE_INSTRUCTION,
    SHORT_INSTRUCTION,
    STOVE,
    TOASTER_OVEN,
    LemonTartTracker,
    assign_homes,
)

# Groups that the dishwasher placement filter would reject, so they are spawned
# on the counter and moved into the dishwasher after every reset.
OBJECT_GROUPS = {
    "butter": "butter_stick",
    "egg": "egg",
    "lemon": "lemon",
    "flour": "flour_bag",
    "sugar": "sugar_cube",
    "bowl": "bowl",
    "whisk": "whisk",
    "saucepan": "saucepan",
    "baking_sheet": "baking_sheet",
    "tart_pan": "plate",
}

OBJECT_SCALES = {
    "baking_sheet": 0.35,
    "tart_pan": 0.55,
    "flour": 0.50,
    "saucepan": 0.65,
    "bowl": 0.75,
}
# Drawer interiors in layout 001 are about 0.12 m tall. Reject larger samples.
DRAWER_MAX_SIZE = (0.32, 0.45, 0.10)
DRAWER_Z_RANGE = (0.0, 1.50)
# Style 001's ToasterOven003 has temperature and time only. 055 has the
# function knob the recipe scores as well.
TOASTER_OVEN_MODEL = "ToasterOven055"
# The PandaOmron's torso only goes up and its gripper bottoms out near
# z = 0.29 m. A drawer whose handle is lower than this cannot be opened by the
# robot, so it is never used as a home. In layout 001 this drops the bottom
# drawer of the four-drawer stack (handle at 0.155 m).
DRAWER_MIN_HANDLE_Z = 0.30
# rack1 is the upper dishwasher rack, the one that slides out past the door.
# rack0 sits at 0.22 m behind a door that only opens 35 degrees: out of reach.
DISHWASHER_RACK = "rack1"
# Dishes thinner than this count as flat and are propped on the rack's rim.
FLAT_DISH_HEIGHT = 0.03
FLAT_DISH_TILT = np.deg2rad(30.0)
FLAT_DISH_OVERHANG = 0.035
# Height of the rack's front wall above the rack region's floor.
RACK_WALL_HEIGHT = 0.048

COUNTER_SLOTS = ((-0.8, -0.6), (0.0, -0.6), (0.8, -0.6), (-0.4, 0.4), (0.4, 0.4))
REQUIRED_FIXTURES = ("toaster_oven", "stove", "dishwasher")


class LemonTart(Kitchen):
    """One kitchen, layout 001. Score containers and toaster-oven knobs.

    The tart pan stays on the counter. The bowl, whisk, saucepan, and baking
    sheet end in the dishwasher. Ingredient homes are not in the policy info.
    """

    def __init__(self, robots="PandaOmron", enable_fixtures=None, **kwargs):
        kwargs.setdefault("layout_ids", 1)
        kwargs.setdefault("style_ids", 1)
        kwargs.setdefault("horizon", 20_000)
        kwargs.setdefault("ignore_done", False)
        kwargs.setdefault("obj_registries", ("objaverse", "lightwheel", "aigen"))
        fixture_updates = dict(kwargs.pop("update_fxtr_cfg_dict", None) or {})
        oven_update = dict(fixture_updates.get("toaster_oven") or {})
        oven_update.setdefault("config_name", TOASTER_OVEN_MODEL)
        fixture_updates["toaster_oven"] = oven_update
        kwargs["update_fxtr_cfg_dict"] = fixture_updates
        enabled = list(enable_fixtures or [])
        for name in REQUIRED_FIXTURES:
            if name not in enabled:
                enabled.append(name)
        # Optional: ingredients that pour and cut, plus a knife in a drawer.
        # See owb/tasks/lemon_tart_prep.py. Off by default.
        self.usable = bool(kwargs.pop("usable_ingredients", False))
        cut_rule = kwargs.pop("cut_rule", prep.DEFAULT_CUT_RULE)
        self.prep = prep.PrepKit(self, cut_rule) if self.usable else None
        self.knife_home = None
        # Optional: the toaster oven door has to be unlocked with a button.
        # See owb/tasks/lemon_tart_lock.py. Off by default.
        lock_mode = kwargs.pop("oven_lock", None)
        self.oven_lock = lemon_tart_lock.OvenLock(self, lock_mode) if lock_mode else None
        # Optional: the oven's knobs turn up only in an order drawn at each reset.
        knob_mode = kwargs.pop("oven_knob_order", None)
        self._knob_mode = knob_mode
        self.knob_order = None
        self.expose_object_state = kwargs.pop("expose_object_state", False)
        self.expose_bindings = kwargs.pop("expose_bindings", False)
        super().__init__(robots=robots, enable_fixtures=enabled, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref(FRIDGE, dict(id=FixtureType.FRIDGE))
        self.dishwasher = self.register_fixture_ref(
            DISHWASHER, dict(id=FixtureType.DISHWASHER)
        )
        self.stove = self.register_fixture_ref(STOVE, dict(id=FixtureType.STOVE))
        self.toaster_oven = self.register_fixture_ref(
            TOASTER_OVEN, dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            COUNTER, dict(id=FixtureType.COUNTER, ref=self.fridge, size=(0.6, 0.4))
        )
        drawers = self.get_fixture(FixtureType.DRAWER, return_all=True) or []
        drawers = [f for f in drawers if float(f.pos[2]) >= DRAWER_MIN_HANDLE_Z]
        self.drawers = sorted(drawers, key=lambda fixture: fixture.name)
        self.homes = assign_homes([fixture.name for fixture in self.drawers], self.rng)
        self.storage_ids = (FRIDGE, *[fixture.name for fixture in self.drawers])
        self.tracker = LemonTartTracker(self.homes, self.storage_ids)
        self.init_robot_base_ref = self.fridge
        if self.usable:
            # a separate generator, so the default task's homes and placements
            # are the same with and without the option
            pick = np.random.default_rng([0 if self.seed is None else int(self.seed), 20261116])
            self.knife_home = self.drawers[int(pick.integers(len(self.drawers)))].name

    def _get_obj_cfgs(self):
        cfgs = []
        slot = 0
        sources = self.prep.sources() if self.usable else {}
        for name in OBJECT_NAMES:
            home = self.homes[name]
            group = OBJECT_GROUPS[name]
            scale = OBJECT_SCALES.get(name)
            if self.usable and name in sources:
                group, scale = sources[name], None
            cfg = dict(name=name, obj_groups=group)
            if scale is not None:
                cfg["object_scale"] = scale
            if home == FRIDGE:
                cfg["placement"] = dict(
                    fixture=self.fridge,
                    size=(0.28, 0.22),
                    pos=_fridge_pos(name),
                    sample_region_kwargs=dict(z_range=(1.0, 1.5)),
                )
            elif home == DISHWASHER:
                # Sample on the counter. _reset_internal moves these into the
                # dishwasher, so the first observation is already in there.
                x, y = COUNTER_SLOTS[slot % len(COUNTER_SLOTS)]
                slot += 1
                cfg["placement"] = dict(
                    fixture=self.counter,
                    size=(0.36, 0.28),
                    pos=(x, y),
                )
            else:
                cfg["max_size"] = DRAWER_MAX_SIZE
                cfg["placement"] = dict(
                    fixture=self._drawer(home),
                    size=(0.28, 0.36),
                    pos=(0.0, 0.0),
                    sample_region_kwargs=dict(z_range=DRAWER_Z_RANGE),
                )
            cfgs.append(cfg)
        if self.usable:
            # last, so that it does not disturb where the other objects land;
            # lying across the drawer, handle to the left, in the middle
            cfgs.append(
                dict(
                    name="knife",
                    obj_groups=prep.mjcf(prep.KNIFE_MJCF),
                    placement=dict(
                        fixture=self._drawer(self.knife_home),
                        size=(0.30, 0.16),
                        pos=(0.0, -0.4),
                        rotation=(-np.pi / 2 - 0.15, -np.pi / 2 + 0.15),
                        sample_region_kwargs=dict(z_range=DRAWER_Z_RANGE),
                    ),
                )
            )
        return cfgs

    def sample_object(self, groups, **kwargs):
        """The usable ingredients are not in RoboCasa's object registries: build them directly."""
        if prep.is_prep_asset(groups):
            name = next((n for n, path in self.prep.sources().items() if path == groups), None)
            if name is not None:
                # draw the default object too and discard it, so that every
                # other object comes out the same as in the default task
                default_kwargs = dict(kwargs, object_scale=OBJECT_SCALES.get(name))
                super().sample_object(OBJECT_GROUPS[name], **default_kwargs)
            info = dict(
                groups_containing_sampled_obj=[],
                groups=groups,
                cat=groups.rsplit("/", 1)[-1].split(".")[0],
                split=None,
                mjcf_path=groups,
            )
            return dict(mjcf_path=groups, scale=1.0), info
        return super().sample_object(groups, **kwargs)

    def _create_objects(self):
        super()._create_objects()
        if self.usable:
            self.prep.add_bodies()
        if self.oven_lock is not None:
            self.oven_lock.add_bodies()

    def grain_positions(self):
        """World positions of the loose grains, per ingredient. Empty unless usable."""
        return self.prep.grain_positions() if self.usable else {}

    def knob_state(self):
        """Knob order drawn at this reset, which knobs are free, blocked attempts. Empty without the option."""
        return self.knob_order.state() if self.knob_order is not None else {}

    def lock_state(self):
        """Oven lock: locked or not, presses, pulls on the locked door. Empty without the option."""
        return self.oven_lock.state() if self.oven_lock is not None else {}

    def prep_state(self):
        """Which ingredients have been cut, and how far apart their halves are."""
        return self.prep.state() if self.usable else {}

    def _drawer(self, name):
        for fixture in self.drawers:
            if fixture.name == name:
                return fixture
        raise KeyError(name)

    def _reset_internal(self):
        super()._reset_internal()
        self._park_dishwasher_objects()
        if self.usable:
            self.prep.reset()
        if self.oven_lock is not None:
            self.oven_lock.reset()
        if self._knob_mode:
            if self.knob_order is None:
                self.knob_order = lemon_tart_lock.KnobOrder(self, self._knob_mode)
            self.knob_order.reset()
        self._close_fixtures()
        self._zero_knobs()
        self.tracker.reset()

    def _park_dishwasher_objects(self):
        """Lay the dishwasher objects side by side on the rack, not overlapping.

        The rack is a set of low tines with no solid floor. A dish lying flat
        among them cannot be picked up again: the gripper cannot get a finger
        under its rim. Flat dishes are therefore propped on the rack's front
        rim, front edge up, the way dishes lean in a real rack.
        """
        names = [name for name in OBJECT_NAMES if self.homes[name] == DISHWASHER]
        sites = self.dishwasher.get_int_sites(relative=False)
        if not sites or not names:
            return
        key = DISHWASHER_RACK if DISHWASHER_RACK in sites else next(iter(sites))
        p0, px, py = [np.asarray(point, dtype=float) for point in sites[key][:3]]
        u, v = px - p0, py - p0
        width = float(np.linalg.norm(u))
        u_hat = u / width if width > 1e-6 else np.array([1.0, 0.0, 0.0])
        v_hat = v / max(float(np.linalg.norm(v)), 1e-6)
        placed = []
        for name in names:
            obj = self.objects[name]
            qpos = np.array(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)
            rot = T.quat2mat(T.convert_quat(qpos[3:7], to="xyzw"))
            size = np.asarray(obj.size, dtype=float)
            extent = np.abs(rot) @ size
            placed.append((obj, rot, extent, size[2] < FLAT_DISH_HEIGHT))
        # Deep items go to the left end, flat dishes to the right end. The
        # stretch of front wall left between them is where a hand can pull the
        # rack out.
        placed.sort(key=lambda item: item[3])
        left, right = 0.005, width - 0.005
        wall_top = p0[2] + RACK_WALL_HEIGHT
        for obj, rot, extent, flat in placed:
            along = float(abs(extent @ u_hat))
            if flat:
                pos = p0 + u_hat * (right - along / 2.0)
                right -= along + 0.01
            else:
                pos = p0 + u_hat * (left + along / 2.0)
                left += along + 0.01
            bottom = rot @ np.asarray(obj.bottom_offset, dtype=float)
            if flat:
                # front edge on the wall top, overhanging a little, rear on the tines
                depth = float(abs(extent @ v_hat))
                lever = depth / 2.0 - FLAT_DISH_OVERHANG
                tilt = T.quat2mat(T.axisangle2quat(np.array([-FLAT_DISH_TILT, 0.0, 0.0])))
                rot = tilt @ rot
                pos = pos + v_hat * (lever * np.cos(FLAT_DISH_TILT) - 0.004)
                pos[2] = wall_top - lever * np.sin(FLAT_DISH_TILT) + extent[2] / 2.0 + 0.006
                quat = T.convert_quat(T.mat2quat(rot), to="wxyz")
            else:
                pos = pos + v / 2.0 - bottom
                pos[2] = p0[2] - bottom[2] + 0.01
                quat = T.convert_quat(T.mat2quat(rot), to="wxyz")
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))
        self.sim.forward()

    def _close_fixtures(self):
        for fixture in (self.fridge, self.dishwasher, self.toaster_oven, *self.drawers):
            try:
                fixture.close_door(self)
            except Exception:
                continue

    def _zero_knobs(self):
        setters = {
            "function": self.toaster_oven.set_function,
            "temperature": self.toaster_oven.set_temperature,
            "time": self.toaster_oven.set_time,
        }
        present = set(self.sim.model.joint_names)
        for name, setter in setters.items():
            joint = self.toaster_oven._joint_names.get(name)
            if joint in present:
                setter(self, 0.0)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.expose_object_state:
            return observables
        prefixes = tuple(f"{name}_" for name in OBJECT_NAMES)
        for name, observable in observables.items():
            if name == "world_pose_in_gripper" or name.startswith(prefixes):
                observable.set_active(False)
                observable.set_enabled(False)
        return observables

    def instruction(self) -> str:
        if not hasattr(self, "tracker") or self.tracker.recipe_visible:
            # the hint is part of the recipe only: once the recipe is gone the
            # agent has to remember it
            hint = getattr(self, "oven_lock", None) is not None and self.oven_lock.mode == "recipe"
            text = RECIPE_INSTRUCTION + (lemon_tart_lock.HINT if hint else "")
            knobs = getattr(self, "knob_order", None)
            if knobs is not None and knobs.mode == "recipe":
                text += lemon_tart_lock.knob_hint(knobs.order)
            return text
        return SHORT_INSTRUCTION

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.instruction()
        if not self.expose_bindings:
            for cfg in ep_meta.get("object_cfgs", []):
                placement = cfg.get("placement")
                if isinstance(placement, dict):
                    placement["fixture"] = "hidden"
                cfg.pop("reset_region", None)
        return ep_meta

    def evaluator_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.instruction()
        ep_meta["lemon_tart_homes"] = dict(self.homes)
        return ep_meta

    def evaluator_state(self):
        return self.tracker.evaluator_state()

    def reward(self, action=None):
        self.tracker.update(self._sense())
        return float(self.tracker.consume_reward())

    def _check_success(self):
        self.tracker.update(self._sense())
        return self.tracker.succeeded

    def _post_action(self, action):
        self.update_state()
        if self.usable:
            self.prep.update()
        if self.oven_lock is not None:
            self.oven_lock.update()
        if self.knob_order is not None:
            self.knob_order.update()
        reward, done, info = super()._post_action(action)
        info["instruction"] = self.instruction()
        info["lemon_tart"] = self.tracker.public_state()
        if self.tracker.succeeded and not self.ignore_done:
            self.done = True
            done = True
        return reward, done, info

    def _sense(self):
        containers = {
            FRIDGE: self.fridge,
            DISHWASHER: self.dishwasher,
            TOASTER_OVEN: self.toaster_oven,
        }
        for drawer in self.drawers:
            containers[drawer.name] = drawer
        inside = {name: set() for name in OBJECT_NAMES}
        touching = {name: set() for name in OBJECT_NAMES}
        for name in OBJECT_NAMES:
            for fixture_id, fixture in containers.items():
                try:
                    if OU.obj_inside_of(self, name, fixture, partial_check=True):
                        inside[name].add(fixture_id)
                except Exception:
                    continue
            try:
                if OU.check_obj_fixture_contact(self, name, self.stove):
                    touching[name].add(STOVE)
            except Exception:
                pass
            try:
                if OU.check_obj_any_counter_contact(self, name):
                    touching[name].add(COUNTER)
            except Exception:
                pass
        try:
            if self.check_contact(self.objects["lemon"], self.objects["tart_pan"]):
                touching["lemon"].add("tart_pan")
        except Exception:
            pass
        open_ids = set()
        for fixture_id, fixture in containers.items():
            if fixture_id == TOASTER_OVEN:
                continue
            try:
                if not fixture.is_closed(self):
                    open_ids.add(fixture_id)
            except Exception:
                continue
        held = set()
        for name in OBJECT_NAMES:
            try:
                if not OU.gripper_obj_far(self, name):
                    held.add(name)
            except Exception:
                continue
        knob_state = self.toaster_oven.get_state()
        return {
            "inside": inside,
            "touching": touching,
            "open": open_ids,
            "held": held,
            "knobs": {name: float(knob_state.get(name) or 0.0) for name in KNOBS},
        }


def _fridge_pos(name: str):
    return {"butter": (-0.35, -0.8), "egg": (0.35, -0.8), "lemon": (0.0, -0.2)}[name]
