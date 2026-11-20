"""A toaster oven that has to be unlocked: `LemonTart(oven_lock="recipe" | "hidden")`.

The idea follows AdaManip (Wang et al., ICLR 2025): the door has a lock whose
state cannot be read off the door itself, so pulling the handle is not enough.

- The door is held shut until the button on the oven's front panel is pressed.
- Once unlocked, the door opens normally. When it has been opened and comes
  back to closed, it locks again; opening it a second time needs another press.
- `"recipe"`: the recipe text says so, and a lamp over the button shows the
  state (red locked, green unlocked).
- `"hidden"`: nothing is said and there is no lamp. The agent has to find out
  from a failed pull, and remember it for the second bake.

A second, independent option, `LemonTart(oven_knob_order="recipe" | "hidden")`,
makes the three knobs the recipe uses turn up only in a particular order:

- The order is a permutation of function, temperature and time, drawn afresh
  at every reset. It is not fixed in the code.
- A knob can be turned up only while every knob before it in the order is on,
  and turned back down only while every knob after it is off. So the knobs go
  up in the order and come back down in the reverse order; a knob out of turn
  is held where it is.
- The time knob runs down by itself. It does not while it is held.
- `"recipe"`: the recipe text gives the order. `"hidden"`: it does not, and the
  agent has to work it out and remember it for the second bake.

Both are off by default; the scored milestones are unchanged.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

MODES = ("recipe", "hidden")
DOOR_JOINT = "toaster_oven_main_group_door_joint"
OVEN_PREFIX = "toaster_oven_main_group"
BUTTON_JOINT = "oven_lock_button_joint"

# Offset of the button from the door hinge's anchor: on the front panel, to the
# left of the four knobs, at their height. (Not on the right: a hand turning
# the time knob would lean on it.)
BUTTON_OFFSET = np.array([-0.175, -0.036, 0.252])
BUTTON_HALF = (0.014, 0.008, 0.014)
BUTTON_TRAVEL = 0.010
PRESSED = 0.006  # pushed in this far counts as a press, metres
OPENED = 0.15  # the door counts as opened beyond this angle, rad
SHUT = 0.02  # ... and as shut again below this one
PULL_TORQUE = 1.0  # the lock holding back more than this counts as a pull, N m

HINT = (
    " The toaster oven door is locked. Press the button to the left of its knobs "
    "to unlock it before you pull the handle. The door locks again every time it closes."
)
RED, GREEN, GREY = (0.85, 0.10, 0.10, 1.0), (0.10, 0.80, 0.20, 1.0), (0.75, 0.75, 0.78, 1.0)


KNOB_JOINTS = {
    "function": f"{OVEN_PREFIX}_knob_function_joint",
    "temperature": f"{OVEN_PREFIX}_knob_temp_joint",
    "time": f"{OVEN_PREFIX}_knob_time_joint",
}
KNOB_BODIES = {name: joint[: -len("_joint")] for name, joint in KNOB_JOINTS.items()}
KNOB_SLACK = 0.02  # a held knob can still be wiggled this far, rad


def knob_hint(order):
    return (
        f" The toaster oven's knobs only turn up in this order: {order[0]}, then {order[1]}, "
        f"then {order[2]}. A knob stays stuck until the ones before it are on. They only turn back "
        f"down in the reverse order: {order[2]}, then {order[1]}, then {order[0]}."
    )


class KnobOrder:
    """Knobs that turn up only in an order drawn at every reset."""

    def __init__(self, env, mode):
        if mode not in MODES:
            raise ValueError(f"oven_knob_order must be one of {MODES}, got {mode!r}")
        from owb.tasks.lemon_tart import KNOB_OFF, KNOB_ON

        self.env = env
        self.mode = mode
        self.on_level = KNOB_ON
        self.off_level = KNOB_OFF
        # its own generator, so that the default task's sampling is untouched;
        # one draw per reset
        self.rng = np.random.default_rng(None if env.seed is None else [int(env.seed), 20261117])
        self.order = tuple(KNOB_JOINTS)
        self.full = {}
        self._clear()

    def _clear(self):
        self.caps = {}  # upper stops of knobs that may not go up
        self.floors = {}  # lower stops of knobs that may not come down
        self.blocked_turns = {name: 0 for name in KNOB_JOINTS}
        self._was_touching = {name: False for name in KNOB_JOINTS}
        self._quiet = {name: 0 for name in KNOB_JOINTS}

    def _ids(self, name):
        model = self.env.sim.model
        return model.joint_name2id(KNOB_JOINTS[name]), model.get_joint_qpos_addr(KNOB_JOINTS[name])

    def reset(self):
        names = list(KNOB_JOINTS)
        self.order = tuple(names[i] for i in self.rng.permutation(len(names)))
        model = self.env.sim.model._model
        self.full = {}
        for name in KNOB_JOINTS:
            jid, _ = self._ids(name)
            self.full[name] = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
        self._clear()
        self.update()

    def value(self, name):
        """Normalised knob position, on the knob's full range."""
        _, adr = self._ids(name)
        lo, hi = self.full[name]
        return (float(self.env.sim.data.qpos[adr]) - lo) / (hi - lo)

    def free(self, name):
        """May this knob be turned up right now?"""
        before = self.order[: self.order.index(name)]
        return all(self.value(other) >= self.on_level for other in before)

    def free_down(self, name):
        """May this knob be turned back down right now?"""
        after = self.order[self.order.index(name) + 1 :]
        return all(self.value(other) <= self.off_level for other in after)

    def _robot_on(self, name):
        sim = self.env.sim
        model, data = sim.model._model, sim.data._data
        for i in range(data.ncon):
            con = data.contact[i]
            names = [sim.model.body_id2name(model.geom_bodyid[g]) or "" for g in (con.geom1, con.geom2)]
            if KNOB_BODIES[name] in names and any(n.startswith(("gripper0", "robot0")) for n in names):
                return True
        return False

    def update(self):
        sim = self.env.sim
        model = sim.model._model
        for name in KNOB_JOINTS:
            jid, adr = self._ids(name)
            lo, hi = self.full[name]
            q = float(sim.data.qpos[adr])
            pushed = False
            if self.free(name):
                self.caps.pop(name, None)
                upper = hi
            else:
                # held: the stop follows it down, but it cannot go up
                cap = min(self.caps.get(name, q), q)
                self.caps[name] = cap
                upper = min(hi, max(cap, lo) + KNOB_SLACK)
                pushed = q > cap + 0.5 * KNOB_SLACK
            if self.free_down(name):
                self.floors.pop(name, None)
                lower = lo
            else:
                floor = max(self.floors.get(name, q), q)
                self.floors[name] = floor
                lower = max(lo, min(floor, hi) - KNOB_SLACK)
                pushed = pushed or q < floor - 0.5 * KNOB_SLACK
                if name == "time":
                    # a held timer does not run down: restart the oven's clock
                    self.env.toaster_oven._last_time_update = float(sim.data.time)
            model.jnt_range[jid][0] = lower
            model.jnt_range[jid][1] = max(upper, lower + 1e-3)
            # a blocked attempt: the robot is on the knob and has it against a stop
            # ... counted once per attempt: a new one starts after a second of quiet
            touching = pushed and self._robot_on(name)
            self._quiet[name] = 0 if touching else self._quiet[name] + 1
            if touching and not self._was_touching[name]:
                self.blocked_turns[name] += 1
                self._was_touching[name] = True
            elif self._quiet[name] > 20:
                self._was_touching[name] = False

    def state(self):
        return {
            "mode": self.mode,
            "order": list(self.order),
            "values": {name: self.value(name) for name in KNOB_JOINTS},
            "free": {name: self.free(name) for name in KNOB_JOINTS},
            "free_down": {name: self.free_down(name) for name in KNOB_JOINTS},
            "blocked_turns": dict(self.blocked_turns),
        }


class OvenLock:
    def __init__(self, env, mode):
        if mode not in MODES:
            raise ValueError(f"oven_lock must be one of {MODES}, got {mode!r}")
        self.env = env
        self.mode = mode
        self.locked = True
        self._clear()

    def _clear(self):
        self.opened = False
        self.presses = 0  # presses that unlocked the door
        self.idle_presses = 0  # presses while it was already unlocked
        self.locked_pulls = 0  # times the robot took hold of the locked door
        self.relocks = 0
        self._was_pressed = False
        self._was_touching = False
        self._quiet = 0

    # ---------- model ----------
    def add_bodies(self):
        model = self.env.model
        anchor = self._hinge_anchor()
        pos = anchor + BUTTON_OFFSET
        body = ET.SubElement(model.worldbody, "body", name="oven_lock_button", pos=" ".join(f"{v:.5f}" for v in pos))
        # pushed in along +y against a spring that returns it
        ET.SubElement(
            body, "joint", name=BUTTON_JOINT, type="slide", axis="0 1 0", range=f"0 {BUTTON_TRAVEL}",
            stiffness="120", damping="3", limited="true",
        )
        size = " ".join(str(v) for v in BUTTON_HALF)
        ET.SubElement(body, "geom", name="oven_lock_button_col", type="box", size=size, group="0", mass="0.03")
        ET.SubElement(
            body, "geom", name="oven_lock_button_vis", type="box", size=size, group="1", contype="0",
            conaffinity="0", rgba=" ".join(str(v) for v in GREY), mass="0",
        )
        if self.mode == "recipe":
            lamp = pos + np.array([0.0, 0.004, 0.034])
            ET.SubElement(
                model.worldbody, "geom", name="oven_lock_lamp", type="sphere", size="0.007", group="1", contype="0",
                conaffinity="0", pos=" ".join(f"{v:.5f}" for v in lamp), rgba=" ".join(str(v) for v in RED),
            )
        # the button is mounted on the panel: it must not collide with the oven
        for other in [b.get("name") for b in model.worldbody.iter("body") if (b.get("name") or "").startswith(OVEN_PREFIX)]:
            ET.SubElement(model.contact, "exclude", body1="oven_lock_button", body2=other)
        ET.SubElement(
            model.equality, "joint", name="oven_lock", joint1=DOOR_JOINT, polycoef="0 0 0 0 0", active="true",
            solref="0.004 1",
        )

    def _hinge_anchor(self):
        """World position of the door hinge, from the fixture's placement in the model tree."""
        from robosuite.utils.mjcf_utils import string_to_array

        def walk(elem, offset):
            pos = offset + (string_to_array(elem.get("pos")) if elem.get("pos") else 0.0)
            for joint in elem.findall("joint"):
                if joint.get("name") == DOOR_JOINT:
                    return pos + (string_to_array(joint.get("pos")) if joint.get("pos") else 0.0)
            for child in elem.findall("body"):
                found = walk(child, pos)
                if found is not None:
                    return found
            return None

        anchor = walk(self.env.model.worldbody, np.zeros(3))
        if anchor is None:
            raise RuntimeError("toaster oven door joint not found")
        return anchor

    # ---------- runtime ----------
    def _raw(self):
        sim = self.env.sim
        return sim.model._model, sim.data._data

    def _apply(self):
        import mujoco

        model, data = self._raw()
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "oven_lock")
        data.eq_active[eq] = 1 if self.locked else 0
        if self.mode == "recipe":
            lamp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "oven_lock_lamp")
            model.geom_rgba[lamp] = RED if self.locked else GREEN

    def reset(self):
        self.locked = True
        self._clear()
        self._apply()

    def button_travel(self):
        sim = self.env.sim
        return float(sim.data.qpos[sim.model.get_joint_qpos_addr(BUTTON_JOINT)])

    def door_angle(self):
        sim = self.env.sim
        return float(sim.data.qpos[sim.model.get_joint_qpos_addr(DOOR_JOINT)])

    def _lock_load(self):
        """Opening torque the lock is holding back on the door joint, N m.
        Negative when something pushes the door shut, as its own spring does."""
        import mujoco

        model, data = self._raw()
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "oven_lock")
        for i in range(data.nefc):
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and data.efc_id[i] == eq:
                return -float(data.efc_force[i])
        return 0.0

    def update(self):
        pressed = self.button_travel() > PRESSED
        if pressed and not self._was_pressed:
            if self.locked:
                self.locked = False
                self.opened = False
                self.presses += 1
                self._apply()
            else:
                self.idle_presses += 1
        self._was_pressed = pressed
        angle = self.door_angle()
        if not self.locked:
            if angle > OPENED:
                self.opened = True
            elif self.opened and angle < SHUT:
                self.locked = True
                self.relocks += 1
                self._apply()
        # a pull on the locked door: the lock is holding back a real torque,
        # not just the door's own closing spring or a hand brushing the handle
        pulling = self.locked and self._lock_load() > PULL_TORQUE
        self._quiet = 0 if pulling else self._quiet + 1
        if pulling and not self._was_touching:
            self.locked_pulls += 1
            self._was_touching = True
        elif self._quiet > 20:
            self._was_touching = False

    def state(self):
        return {
            "mode": self.mode,
            "locked": self.locked,
            "unlock_presses": self.presses,
            "presses_while_unlocked": self.idle_presses,
            "locked_pulls": self.locked_pulls,
            "relocks": self.relocks,
            "door_angle": self.door_angle(),
        }
