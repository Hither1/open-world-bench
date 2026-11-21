"""Horse riding: a learnable traversal skill built on native creatures.

A horse must first be calmed: approach without sprinting and offer it fruit
(``feed`` with fruit selected). A calmed horse accepts a rider (``ride``);
after the first ride it stays bonded. Sprinting near an unbonded horse spooks
it and it bolts. ``dismount`` leaves the horse standing where it is.

Riding proficiency grows with distance ridden and unlocks faster gaits:
walk, then trot, then canter. A proficient rider can gallop (hold sprint),
which draws on the horse's own stamina rather than the agent's. Mounted riders
cannot climb, glide, swim or operate fixtures. The rider keeps the agent's
collision volume; the horse is carried with it (documented simplification).

A bonded horse answers a ``whistle`` from up to ``WHISTLE_RANGE`` metres away
on the same level: it finds a dry walking route (hopping single-block steps)
and canters to where the rider whistled. It gives up if no route exists or it
is blocked for a few seconds.
"""
from __future__ import annotations

import heapq

import numpy as np

from ..core import blocks as B

COMMANDS = ("feed", "ride", "dismount", "whistle")
#: (minimum metres ridden, gait, speed multiplier over walking)
GAITS = ((0.0, "walk", 1.4), (80.0, "trot", 1.9), (250.0, "canter", 2.4))
GALLOP = 3.0
GALLOP_COST, STAMINA_REGEN, HORSE_STAMINA = 14.0, 6.0, 100.0
REACH, SPOOK_RADIUS = 2.6, 5.0
WHISTLE_RANGE, ROUTE_LIMIT = 150.0, 60000
#: Commands that remain possible from the saddle.
MOUNTED_OK = frozenset({"none", "eat", "equip", "stow", "attack", "dismount"})


def gait(distance):
    for threshold, name, mult in reversed(GAITS):
        if distance >= threshold:
            return name, mult
    return GAITS[0][1:]


class Riding:
    def __init__(self, env):
        self.env = env
        self.horse = None          # npc id while mounted
        self.distance = 0.0        # metres ridden (proficiency)
        self.stamina = HORSE_STAMINA
        self.galloping = False

    # -- state -----------------------------------------------------------
    @property
    def mounted(self):
        return self.horse is not None

    def _horse(self):
        return self.env.npcs.by_id(self.horse) if self.horse is not None else None

    def state(self):
        name, mult = gait(self.distance)
        return dict(mounted=self.mounted, horse=self.horse, distance=round(self.distance, 3),
                    gait=("gallop" if self.galloping else name) if self.mounted else name,
                    speed=GALLOP if self.galloping else mult, horse_stamina=round(self.stamina, 3))

    def _facing_horse(self):
        a = self.env.agent
        best = None
        for n in self.env.npcs.near(a.pos, REACH + 0.6):
            if n.kind != "horse":
                continue
            delta = n.pos - a.pos
            delta[1] = 0.0
            distance = float(np.linalg.norm(delta))
            if distance > REACH or (distance > 0.3 and np.dot(delta, a.forward_flat()) < 0.3 * distance):
                continue
            if best is None or distance < best[0]:
                best = (distance, n)
        return None if best is None else best[1]

    @staticmethod
    def _failed(reason):
        return [dict(type="action_failed", reason=reason)]

    # -- commands --------------------------------------------------------
    def dispatch(self, command, slot):
        if command == "feed":
            return self.feed(slot)
        if command == "ride":
            return self.ride()
        if command == "dismount":
            return self.dismount()
        if command == "whistle":
            return self.whistle()
        return []

    def feed(self, slot):
        if self.mounted:
            return self._failed("dismount first")
        horse = self._facing_horse()
        if horse is None:
            return self._failed("no horse within reach")
        stacks = self.env.agent.inventory.stacks
        stack = stacks[slot] if 0 <= slot < len(stacks) else None
        if stack is None or "fruit" not in stack.spec.tags:
            return self._failed("select fruit to offer")
        stack.count -= 1
        if not stack.count:
            self.env.agent.inventory._drop_stack(stack)
        horse.data["calm"] = True
        horse.state, horse.aggro_timer = "idle", 0.0
        return [dict(type="horse_calmed", npc=horse.id, item=stack.spec.id)]

    def ride(self):
        a = self.env.agent
        if self.mounted:
            return self._failed("already riding")
        if not a.on_ground or a.in_water or a.climbing or a.gliding:
            return self._failed("mount from firm ground")
        horse = self._facing_horse()
        if horse is None:
            return self._failed("no horse within reach")
        if not (horse.data.get("calm") or horse.data.get("bonded")):
            self._spook(horse)
            return self._failed("the horse shies away; calm it first") + [dict(type="horse_spooked", npc=horse.id)]
        horse.data.update(rider="agent", bonded=True)
        horse.vel[:] = 0.0
        self.horse = horse.id
        a.pos[0], a.pos[2] = horse.pos[0], horse.pos[2]
        a.yaw = float(horse.yaw)
        return [dict(type="mounted", npc=horse.id, gait=gait(self.distance)[0])]

    def dismount(self, reason="dismounted"):
        horse = self._horse()
        if horse is None:
            return self._failed("not riding")
        a = self.env.agent
        horse.pos = np.asarray(a.pos, float).copy()
        horse.yaw = float(a.yaw)
        horse.home = (float(horse.pos[0]), float(horse.pos[2]))
        horse.data.update(rider=None, roam=1.5, leash=3.0)
        horse.state, horse.patrol_target = "idle", None
        self.horse, self.galloping = None, False
        a.flags.pop("ride_mult", None)
        # Step off to the side when there is room; otherwise stay in place.
        from ..physics import collide as C
        side = np.array((-np.sin(a.yaw), 0.0, np.cos(a.yaw)))
        for offset in (side * 1.1, -side * 1.1):
            candidate = a.pos + offset
            half = np.array((0.3, 0.9, 0.3))
            if not C.box_collides(self.env.world, candidate + (0, 0.92, 0), half):
                a.pos[:] = candidate
                break
        return [dict(type=reason, npc=horse.id)]

    def whistle(self):
        a = self.env.agent
        if self.mounted:
            return self._failed("already riding")
        best = None
        for n in self.env.npcs.alive:
            if n.kind != "horse" or not n.data.get("bonded") or abs(n.pos[1] - a.pos[1]) > 2.0:
                continue
            distance = float(np.hypot(*(n.pos - a.pos)[[0, 2]]))
            if distance <= WHISTLE_RANGE and (best is None or distance < best[0]):
                best = (distance, n)
        if best is None:
            return self._failed("no bonded horse within earshot")
        distance, horse = best
        route = walking_route(self.env.world, horse.pos, a.pos)
        if route is None:
            return self._failed("the horse cannot find a way") + [dict(type="whistled", npc=horse.id, answered=False)]
        horse.data.update(route=route, stall=0.0)
        horse.state, horse.patrol_target, horse.aggro_timer = "idle", None, 0.0
        return [dict(type="whistled", npc=horse.id, answered=True, distance=round(distance, 2), route=len(route))]

    def _spook(self, horse):
        horse.data["calm"] = False
        horse.state, horse.aggro_timer = "flee", 3.0

    # -- per tick --------------------------------------------------------
    def before_tick(self, intent):
        a = self.env.agent
        if not self.mounted:
            a.flags.pop("ride_mult", None)
            return
        intent.climb = intent.glide = intent.crouch = False
        name, mult = gait(self.distance)
        self.galloping = bool(intent.sprint) and name == "canter" and self.stamina > 0.0
        intent.sprint = False
        a.flags["ride_mult"] = GALLOP if self.galloping else mult

    def after_tick(self, dt, previous):
        a, events = self.env.agent, []
        if not self.mounted:
            moving = float(np.hypot(*(a.pos - previous)[[0, 2]])) > 0.02
            for n in self.env.npcs.near(a.pos, SPOOK_RADIUS):
                if n.kind == "horse" and not n.data.get("bonded") and a.sprinting and moving:
                    if n.data.get("calm") or n.state != "flee":
                        self._spook(n)
                        events.append(dict(type="horse_spooked", npc=n.id))
            self.stamina = min(HORSE_STAMINA, self.stamina + STAMINA_REGEN * dt)
            return events
        horse = self._horse()
        if horse is None or not horse.alive or not a.alive:
            self.horse = None
            a.flags.pop("ride_mult", None)
            return events
        if a.in_water or self.env.world.get_f(a.pos[0], a.pos[1] + 0.3, a.pos[2]) == B.WATER:
            events.extend(self.dismount("thrown"))
            return events
        step = float(np.hypot(*(a.pos - previous)[[0, 2]]))
        if a.on_ground:
            self.distance += step
        before = gait(self.distance - step)[0]
        if gait(self.distance)[0] != before:
            events.append(dict(type="riding_improved", gait=gait(self.distance)[0], distance=round(self.distance, 1)))
        if self.galloping:
            self.stamina = max(0.0, self.stamina - GALLOP_COST * dt)
        else:
            self.stamina = min(HORSE_STAMINA, self.stamina + STAMINA_REGEN * dt)
        horse.pos = np.asarray(a.pos, float).copy()
        horse.yaw = float(a.yaw)
        horse.vel[:] = 0.0
        horse.vel[0], horse.vel[2] = (a.pos - previous)[[0, 2]] / max(dt, 1e-6)
        return events


def walking_route(world, start, goal, climb=1, drop=2, limit=ROUTE_LIMIT):
    """Dry four-connected walking route between two positions for a two-block
    tall animal that hops up ``climb`` blocks and steps down ``drop``.
    Returns a list of ``[x, z, y]`` cells after the start, or ``None``."""
    vox, solid = world.vox, B.SOLID
    sx, sy, sz = vox.shape

    def stand(x, z, y):
        return (1 <= x < sx - 1 and 1 <= z < sz - 1 and 1 <= y < sy - 2 and solid[vox[x, y - 1, z]]
                and vox[x, y - 1, z] != B.WATER and not solid[vox[x, y, z]] and not solid[vox[x, y + 1, z]]
                and vox[x, y, z] != B.WATER and vox[x, y + 1, z] != B.WATER)

    def settle(x, z, y):
        for h in (y, y + 1, y - 1):
            if stand(x, z, h):
                return h
        return None

    s = (int(np.floor(start[0])), int(np.floor(start[2])), int(np.floor(start[1] + 0.2)))
    g = (int(np.floor(goal[0])), int(np.floor(goal[2])))
    h0 = settle(*s)
    if h0 is None:
        return None
    s = (s[0], s[1], h0)
    queue, previous, cost, found, expanded = [(0, s)], {s: None}, {s: 0}, None, 0
    while queue and expanded < limit:
        _, q = heapq.heappop(queue)
        expanded += 1
        if (q[0], q[1]) == g or (abs(q[0] - g[0]) + abs(q[1] - g[1]) <= 1 and q != s):
            found = q
            break
        x, z, y = q
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            for ny in range(y + climb, y - drop - 1, -1):
                if stand(nx, nz, ny):
                    n = (nx, nz, ny)
                    c = cost[q] + 1 + 0.5 * abs(ny - y)
                    if c < cost.get(n, 1e18):
                        cost[n] = c
                        previous[n] = q
                        heapq.heappush(queue, (c + abs(nx - g[0]) + abs(nz - g[1]), n))
                    break
    if found is None:
        return None
    route = []
    while found != s:
        route.append([int(found[0]), int(found[1]), int(found[2])])
        found = previous[found]
    return route[::-1]
