"""Goal-first Teyvat campaigns: persistent facts, local mechanisms, no active step.

These are original benchmark adaptations of mountain/shrine/desert quest patterns,
not reproductions of Genshin quests. Only native interaction/combat hooks mutate
progress; the dependency graph is evaluator data, never a policy observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .elements import Element
from .events import Event
from .world import EntityKind

SUITE_VERSION = 'teyvat-campaign-v1'
GOALS = {
    'summit': 'Restore the Frostfall Spire at the mountain summit.',
    'cleansing': 'End the valley blight by defeating the sealed Blightroot Heart.',
    'desert': 'Recover the lost star atlas from the sealed Zenith Archive.',
}
DIFFICULTIES = dict(summit='easy', cleansing='medium', desert='hard')
ELEMENTS = (Element.EMBER, Element.TIDE, Element.GALE)


@dataclass(frozen=True)
class Milestone:
    id: str
    title: str
    requires: tuple[str, ...] = ()


@dataclass
class Site:
    id: str
    kind: str
    center: tuple[float, float]
    requirements: tuple[str, ...] = ()
    entities: dict[str, str] = field(default_factory=dict)
    sequence: tuple[int, ...] = ()
    progress: int = 0
    installed: bool = False
    solved: bool = False
    finished: bool = False
    hits: int = 0
    phase: int = 0
    phase_damage: float = 0.0
    vulnerable_until: float = 0.0


class Campaign:
    def __init__(self, world, task, seed, scale):
        if task not in GOALS:
            raise ValueError(f'unknown campaign {task!r}; choose from {tuple(GOALS)}')
        self.world, self.task, self.seed, self.scale = world, task, seed, scale
        self.rng = np.random.default_rng(seed+84019)
        self.sites: dict[str, Site] = {}
        self.roles: dict[str, tuple[str, str]] = {}
        self.milestones: dict[str, Milestone] = {}
        self.facts: set[str] = set()
        self.completed: list[dict] = []
        self.inventory: dict[str, int] = {}
        self.events: list[Event] = []
        self.discovered: set[str] = set()
        self.feedback = ''
        self.feedback_time = -100.0
        self.cold = 0.0
        self.charge_until = 0.0
        self.slate_energy = 0
        self.metrics = dict(wrong_applications=0, blocked_interactions=0, quartz_expirations=0,
                            cold_damage=0.0, distance_metres=0.0, recharge_visits=0,
                            archive_revisits=0, boss_windows=0)
        self._last_pos = None
        self._quartz_ready: dict[str, float] = {}
        self._archive_seen: set[str] = set()
        self.gates: list[dict] = []
        self.spawn_xz = self.point(0, 170)
        self.add_entity('camp', 'fire', self.spawn_xz, EntityKind.BEACON, 'Expedition camp hearth', 'campfire')
        self.add_entity('camp', 'guide', self.spawn_xz+(-5, 0), EntityKind.NPC, 'Expedition keeper')
        getattr(self, '_build_'+task)()
        self._validate_graph()

    def point(self, x, z):
        return np.array((x*self.scale, z*self.scale), float)

    def center(self, x, z):
        # Regions move between seeds; local mechanism distances stay readable.
        return tuple(self.point(x, z)+self.rng.uniform(-9, 9, 2)*self.scale)

    def milestone(self, key, title, requires=()):
        if key in self.milestones:
            raise ValueError(f'duplicate milestone {key}')
        self.milestones[key] = Milestone(key, title, tuple(requires))

    def _validate_graph(self):
        visiting, done = set(), set()
        def visit(key):
            if key in done:
                return
            if key in visiting or key not in self.milestones:
                raise ValueError('cyclic or missing campaign prerequisite')
            visiting.add(key)
            for prerequisite in self.milestones[key].requires:
                visit(prerequisite)
            visiting.remove(key); done.add(key)
        for key in self.milestones:
            visit(key)

    def emit(self, kind, **payload):
        self.events.append(Event(kind, payload))

    def say(self, message, *, blocked=False):
        self.feedback, self.feedback_time = message, self.world.elapsed
        if blocked:
            self.metrics['blocked_interactions'] += 1
        self.emit('local_feedback', text=message)

    def complete(self, key):
        if key in self.facts:
            return
        spec = self.milestones[key]
        if not set(spec.requires) <= self.facts:
            raise RuntimeError(f'invalid transition: {key} before {spec.requires}')
        self.facts.add(key)
        self.completed.append(dict(id=key, time=round(self.world.elapsed, 6)))
        self.emit('milestone_complete', milestone=key)
        if key == 'final':
            self.emit('activate', entity='campaign_final')

    def add_entity(self, site, role, xz, kind, label, visual=''):
        eid = f'{site}:{role}'
        self.world.spawn_at(eid, kind, *xz, radius=2.6, label=label,
                            visual_role='campaign_'+visual if visual else '')
        self.roles[eid] = (site, role)
        if site in self.sites:
            self.sites[site].entities[role] = eid
        return eid

    def add_site(self, sid, kind, center, requires=()):
        site = Site(sid, kind, tuple(center), tuple(requires),
                    sequence=tuple(int(x) for x in self.rng.permutation(ELEMENTS)))
        self.sites[sid] = site
        return site

    def add_ritual(self, sid, center, requires=(), *, kind='ritual'):
        site = self.add_site(sid, kind, center, requires)
        p = np.asarray(center)
        self.add_entity(sid, 'mural', p+(0, 8), EntityKind.PILLAR, f'{sid} inscribed tablet', 'mural')
        self.add_entity(sid, 'altar', p, EntityKind.ALTAR, f'{sid} ritual socket', 'seal')
        offsets = np.array(((-6, -3), (6, -3), (0, -8)), float)
        self.rng.shuffle(offsets)
        for element, offset in zip(ELEMENTS, offsets):
            role = element.name.lower()
            eid = self.add_entity(sid, role, p+offset, EntityKind.PILLAR,
                                  f'{sid} {role} receiver')
            self.world.get(eid).elemental.aura.element = element
            self.world.get(eid).elemental.aura.charge = 100000
        eid = self.add_entity(sid, 'guard', p+(0, -14), EntityKind.BOSS if kind=='boss' else EntityKind.ENEMY,
                              'Blightroot Heart' if kind=='boss' else f'{sid} sealed guardian')
        enemy = self.world.get(eid)
        enemy.hp = enemy.max_hp = 160.0 if kind=='boss' else 84.0
        enemy.damage = 3.0
        enemy.elemental.shield = 100.0
        return site

    def gate(self, name, xz, fact):
        position = self.world.ground_pos(*xz)
        self.gates.append(dict(id=name, position=position.tolist(), size=[7.2, 4.5, .55], opens=fact))

    def _build_summit(self):
        self.milestone('charged_strike', 'Demonstrate a quartz-charged strike')
        for i, (x, z) in enumerate(((-105, 40), (110, 25), (-20, 110))):
            sid = f'lower_{i}'
            site = self.add_site(sid, 'ice', self.center(x, z))
            self.add_entity(sid, 'ice', site.center, EntityKind.PILLAR, f'Frozen foothill shard {i+1}', 'ice')
            self.add_entity(sid, 'quartz', np.asarray(site.center)+(8, 3), EntityKind.COLLECTIBLE, 'Crimson quartz outcrop', 'quartz')
            self.add_entity(sid, 'fire', np.asarray(site.center)+(-8, 7), EntityKind.BEACON, 'Sheltered brazier', 'campfire')
            self.milestone(sid, f'Thaw foothill shard {i+1}', ('charged_strike',))
        self.milestone('ascent', 'Release the summit wind seal', tuple(f'lower_{i}' for i in range(3)))
        self.add_entity('ascent', 'gate', self.point(0, -45), EntityKind.BEACON, 'Summit wind-seal console', 'seal')
        self.gate('summit_ward', self.point(0, -50), 'ascent')
        for i, (x, z) in enumerate(((-55, -115), (60, -120), (0, -170))):
            sid = f'upper_{i}'
            site = self.add_site(sid, 'ice', self.center(x, z), ('ascent',))
            self.add_entity(sid, 'ice', site.center, EntityKind.PILLAR, f'Summit shard {i+1}', 'ice')
            self.add_entity(sid, 'quartz', np.asarray(site.center)+(8, 3), EntityKind.COLLECTIBLE, 'Crimson quartz outcrop', 'quartz')
            self.add_entity(sid, 'fire', np.asarray(site.center)+(-8, 7), EntityKind.BEACON, 'Summit brazier', 'campfire')
            self.milestone(sid, f'Thaw summit shard {i+1}', ('ascent',))
        self.milestone('final', 'Restore the Frostfall Spire', tuple(f'upper_{i}' for i in range(3)))
        self.add_entity('final', 'spire', self.point(0, -200), EntityKind.ALTAR, 'Frostfall Spire', 'spire')

    def _build_cleansing(self):
        self.milestone('lens', 'Recover the Echo Lens')
        self.add_entity('lens', 'tool', self.point(-30, 135), EntityKind.CHEST, 'Echo Lens reliquary', 'lens')
        for i, (x, z) in enumerate(((-130, 90), (125, 100), (-150, -50), (145, -55), (0, -160))):
            sid = f'shrine_{i}'
            site = self.add_ritual(sid, self.center(x, z), ('lens',))
            self.add_entity(sid, 'ward', np.asarray(site.center)+(11, 8), EntityKind.PILLAR, f'{sid} echo statue', 'lens')
            self.milestone(sid+'_ward', f'Recover shrine {i+1} ward', ('lens',))
            self.milestone(sid+'_puzzle', f'Restore shrine {i+1} pattern', (sid+'_ward',))
            self.milestone(sid, f'Cleanse shrine {i+1}', (sid+'_puzzle',))
        self.add_ritual('heart', self.point(0, -40), tuple(f'shrine_{i}' for i in range(5)), kind='boss')
        self.milestone('final', 'Defeat the Blightroot Heart', tuple(f'shrine_{i}' for i in range(5)))
        self.gate('heart_ward', self.point(0, -25), 'final')

    def _build_desert(self):
        self.milestone('slate', 'Recover the permission slate')
        self.add_entity('slate', 'tool', self.point(20, 145), EntityKind.CHEST, 'Unattuned permission slate', 'slate')
        self.add_entity('camp', 'recharge', self.spawn_xz+(6, 0), EntityKind.BEACON, 'Slate recharge obelisk', 'quartz')
        for i, (x, z) in enumerate(((-140, 40), (145, 25), (0, -145))):
            sid = f'temple_{i}'
            site = self.add_ritual(sid, self.center(x, z), ('slate',), kind='temple')
            self.milestone(sid+'_puzzle', f'Restore temple {i+1} circuit', ('slate',))
            self.milestone(sid, f'Attune temple {i+1} permission', (sid+'_puzzle',))
            self.add_entity(sid, 'archive', np.asarray(site.center)+(13, 9), EntityKind.CHEST,
                            f'{sid} rank-two archive seal', 'slate')
            self.add_entity(sid, 'recharge', np.asarray(site.center)+(-13, 10), EntityKind.BEACON,
                            'Field recharge obelisk', 'quartz')
            self.milestone(sid+'_record', f'Recover temple {i+1} star chart', ('rank_two',))
            self.gate(sid+'_door', np.asarray(site.center)+(13, 6), 'rank_two')
        self.milestone('rank_two', 'Upgrade slate clearance', tuple(f'temple_{i}' for i in range(3)))
        self.add_entity('calibration', 'console', self.point(0, 65), EntityKind.ALTAR, 'Central clearance calibrator', 'slate')
        self.milestone('atlas_key', 'Assemble the archive cipher', tuple(f'temple_{i}_record' for i in range(3)))
        self.add_entity('cipher', 'console', self.spawn_xz+(0, -9), EntityKind.ALTAR, 'Cartographer cipher desk', 'mural')
        self.add_ritual('zenith', self.point(0, -215), ('atlas_key',), kind='temple')
        self.milestone('zenith_puzzle', 'Power the Zenith lock', ('atlas_key',))
        self.milestone('zenith', 'Defeat the archive custodian', ('zenith_puzzle',))
        self.milestone('final', 'Recover the star atlas', ('zenith',))
        self.add_entity('final', 'atlas', self.point(0, -240), EntityKind.CHEST, 'Lost star atlas', 'slate')
        self.gate('zenith_door', self.point(0, -230), 'zenith')

    def near(self, char, eid, radius=2.6):
        e = self.world.get(eid)
        return np.linalg.norm((e.pos-char.pos)[[0, 2]]) <= radius and abs(float(e.pos[1]-char.pos[1])) <= 2.8

    def available(self, site):
        return set(site.requirements) <= self.facts

    def interact(self, char):
        choices = [e for e in self.world.entities.values()
                   if self.near(char, e.id, e.radius) and self.roles[e.id][1] != 'guard']
        if not choices:
            return
        e = min(choices, key=lambda e: (float(np.linalg.norm(e.pos-char.pos)), e.id))
        sid, role = self.roles[e.id]
        site = self.sites.get(sid)
        self.emit('interact', entity=e.id)
        if role == 'guide':
            messages = {
                'summit': 'Quartz coats one physical strike, then must be renewed. Warm braziers clear cold. Three foothill shards feed the wind seal; the summit repeats the ice mechanism.',
                'cleansing': 'The Echo Lens reads old statues and tablets. Each shrine has its own ward. Install it, follow the tablet glyph order with matching elements, then defeat its guardian. Five cleansed shrines seal the Heart.',
                'desert': 'The slate reads glyph tablets and opens permission sockets. Three temples grant separate marks. Bring all three to the central calibrator; revisit their rank-two archives. Recharge at obelisks. The cipher desk assembles recovered charts.',
            }
            self.say(messages[self.task]); return
        if role == 'fire':
            self.cold = 0.0
            self.say('Warmth restored.'); return
        if role == 'recharge':
            if 'slate' not in self.facts:
                self.say('A permission slate fits this obelisk.', blocked=True); return
            self.slate_energy = 6
            self.metrics['recharge_visits'] += 1
            self.say('Slate charge restored to 6.'); return
        if sid in ('lens', 'slate'):
            if sid not in self.facts:
                self.complete(sid); self.inventory[sid] = 1; e.used = True
                if sid == 'slate': self.slate_energy = 6
            self.say('Recovered '+e.label+'.'); return
        if role == 'quartz':
            if self.world.elapsed < self._quartz_ready.get(e.id, 0):
                self.say('The outcrop is regenerating.'); return
            self.charge_until = self.world.elapsed+45
            self._quartz_ready[e.id] = self.world.elapsed+2
            self.say('Crimson coating: one physical strike; expires after 45 seconds.'); return
        if sid == 'ascent':
            if not set(self.milestones['ascent'].requires) <= self.facts:
                self.say('Three unlit shard sockets keep the wind seal closed.', blocked=True); return
            self.complete('ascent'); e.used = True
            self.say('The summit wind seal dissolves.'); return
        if sid == 'calibration':
            if not set(self.milestones['rank_two'].requires) <= self.facts:
                self.say('Calibration needs all three temple permission marks.', blocked=True); return
            self.complete('rank_two'); self.inventory['clearance_rank'] = 2; e.used = True
            self.say('Rank-two clearance granted. Previously sealed archives now accept the slate.'); return
        if sid == 'cipher':
            if not set(self.milestones['atlas_key'].requires) <= self.facts:
                self.say('The cipher desk has three empty chart slots.', blocked=True); return
            self.complete('atlas_key'); self.inventory['atlas_key'] = 1; e.used = True
            self.say('Star charts assembled into the Zenith cipher.'); return
        if sid == 'final':
            if not set(self.milestones['final'].requires) <= self.facts:
                self.say('The final mechanism lacks its regional power or clearance.', blocked=True); return
            self.complete('final'); e.used = True; self.say(GOALS[self.task]+' Complete.'); return
        if site is None:
            return
        if role == 'archive':
            self._archive_seen.add(e.id)
            if 'rank_two' not in self.facts:
                self.say('Rank-two seal: return with upgraded clearance.', blocked=True); return
            key = sid+'_record'
            if key in self.facts:
                self.say('This star chart is already in your pack.'); return
            if self.slate_energy < 2:
                self.say('Opening this seal requires two slate charges.', blocked=True); return
            self.slate_energy -= 2
            self.inventory[key] = 1; self.complete(key); e.used = True
            self.metrics['archive_revisits'] += int(e.id in self._visited_locked_archives)
            self.say('Recovered a star chart; two slate charges spent.'); return
        if not self.available(site):
            self.say('This mechanism is dormant; its tool or regional seals are missing.', blocked=True); return
        if role == 'ward':
            key = sid+'_ward'
            if key not in self.facts:
                self.inventory[key] = 1; self.complete(key); e.used = True
            self.say('The Echo Lens reveals this shrine\'s ward.'); return
        if role == 'mural':
            if self.task == 'desert':
                if self.slate_energy < 1:
                    self.say('Reading glyphs needs one slate charge.', blocked=True); return
                self.slate_energy -= 1
            glyphs = ', '.join(Element(x).name.lower() for x in self.sequence(site))
            self.say(f'Tablet glyph order: {glyphs}. Match each receiver\'s element in that order.'); return
        if role == 'altar':
            if site.finished:
                self.say('This site remains restored.'); return
            if not site.installed:
                if self.task == 'cleansing' and site.kind != 'boss':
                    key = sid+'_ward'
                    if self.inventory.get(key, 0) < 1:
                        self.say('The socket needs this shrine\'s ward.', blocked=True); return
                    self.inventory[key] -= 1
                if self.task == 'desert':
                    if self.slate_energy < 1:
                        self.say('Attuning the socket needs one slate charge.', blocked=True); return
                    self.slate_energy -= 1
                site.installed = True
                self.say('Socket attuned. Receivers are listening.'); return
            guardian = self.world.get(site.entities['guard'])
            if not guardian.alive and site.solved:
                site.finished = True; e.used = True
                self.complete(sid)
                if self.task == 'desert': self.inventory[sid+'_permission'] = 1
                self.say('Site restored; its regional seal remains active.'); return
            self.say('The guardian or receiver circuit is still active.', blocked=True)
        elif role == 'ice':
            self.say('Ordinary ice damage fails here. Crimson quartz leaves a short-lived coating.')
        else:
            self.say('Elemental receiver: apply its matching element in the tablet order.')

    @property
    def _visited_locked_archives(self):
        return getattr(self, '_locked_visits', set())

    def sequence(self, site):
        # The second boss phase rotates the pattern; brute damage cannot skip it.
        return site.sequence[site.phase:]+site.sequence[:site.phase]

    def apply_element(self, char, element):
        receivers = []
        for site in self.sites.values():
            if site.kind == 'ice' or not self.available(site) or not site.installed or site.finished:
                continue
            for role in ('ember', 'tide', 'gale'):
                eid = site.entities[role]
                if self.near(char, eid): receivers.append((site, role, eid))
        if not receivers:
            return
        site, role, eid = min(receivers, key=lambda row: np.linalg.norm(self.world.get(row[2]).pos-char.pos))
        if site.solved:
            self.say('The circuit is already complete.'); return
        sequence = self.sequence(site)
        if int(element) == sequence[site.progress] and role == element.name.lower():
            site.progress += 1; self.world.get(eid).used = True
            self.emit('receiver_accepted', entity=eid)
            if site.progress == 3:
                site.solved = True
                self.world.get(site.entities['guard']).elemental.shield = 0
                if site.kind == 'boss':
                    site.vulnerable_until = self.world.elapsed+20
                    self.metrics['boss_windows'] += 1
                else:
                    self.complete(site.id+'_puzzle')
                self.say('Pattern complete. The guardian is exposed.')
            else:
                self.say(f'Receiver accepted; {site.progress} lamps lit.')
        else:
            site.progress = 0
            for name in ('ember', 'tide', 'gale'): self.world.get(site.entities[name]).used = False
            self.metrics['wrong_applications'] += 1
            self.say('Pattern rejected; lamps reset. Inspect the tablet and try again.')

    def strike_ice(self, char):
        candidates = [s for s in self.sites.values() if s.kind=='ice' and not s.finished
                      and self.near(char, s.entities['ice'], 2.8)]
        if not candidates:
            return
        site = min(candidates, key=lambda s: np.linalg.norm(self.world.get(s.entities['ice']).pos-char.pos))
        if not self.available(site):
            self.say('The upper shard is isolated by the wind seal.', blocked=True); return
        delta = self.world.get(site.entities['ice']).pos-char.pos
        flat = delta[[0, 2]]
        forward = np.array((np.sin(char.yaw), -np.cos(char.yaw)))
        if np.linalg.norm(flat)>1e-6 and flat@forward/np.linalg.norm(flat)<.64:
            return
        if self.charge_until <= self.world.elapsed:
            self.say('The strike glances off strange ice.'); return
        self.charge_until = 0
        site.hits += 1
        self.complete('charged_strike')
        self.emit('charged_ice_hit', entity=site.entities['ice'], hits=site.hits)
        self.say(f'Ice cracked ({site.hits}/2). The quartz coating is spent.')
        if site.hits >= 2:
            site.finished = True
            self.world.get(site.entities['ice']).used = True
            self.complete(site.id)

    def damage_guard(self, enemy, damage, element):
        sid, role = self.roles[enemy.id]
        site = self.sites[sid]
        if not self.available(site) or not site.solved:
            self.say('The guardian\'s regional seal blocks damage.'); return []
        actual = min(float(damage), enemy.hp)
        if site.kind == 'boss':
            actual = min(actual, 80-site.phase_damage)
            site.phase_damage += actual
        enemy.hp = max(0., enemy.hp-actual)
        events = [Event('enemy_hit', dict(entity=enemy.id, damage=actual, hp=enemy.hp,
                                         element=element.name.lower()))]
        if enemy.hp == 0:
            enemy.alive = False
            events.append(Event('defeat', dict(entity=enemy.id)))
            if site.kind == 'boss':
                site.finished = True; self.complete('final')
        elif site.kind == 'boss' and site.phase_damage >= 80:
            site.phase += 1; site.phase_damage = 0
            self.reset_pattern(site)
            self.say('The Heart reseals. Its tablet glyphs have rotated.')
        return events

    def reset_pattern(self, site):
        site.progress = 0; site.solved = False; site.vulnerable_until = 0
        self.world.get(site.entities['guard']).elemental.shield = 100
        for role in ('ember', 'tide', 'gale'): self.world.get(site.entities[role]).used = False

    def tick(self, char, dt):
        if self._last_pos is not None:
            self.metrics['distance_metres'] += float(np.linalg.norm(char.pos-self._last_pos))
        self._last_pos = char.pos.copy()
        if self.charge_until and self.world.elapsed >= self.charge_until:
            self.charge_until = 0; self.metrics['quartz_expirations'] += 1
        if self.task == 'summit':
            near_heat = any(role=='fire' and self.near(char,eid,7) for eid,(_,role) in self.roles.items())
            self.cold = float(np.clip(self.cold+dt*(-18 if near_heat else .30), 0, 100))
            if self.cold >= 100:
                damage = min(char.hp, dt*2)
                char.hp -= damage; self.metrics['cold_damage'] += damage
                self.emit('player_hit', entity='cold', damage=damage)
        for site in self.sites.values():
            if site.kind == 'boss' and site.solved and not site.finished and self.world.elapsed >= site.vulnerable_until:
                self.reset_pattern(site); self.say('The exposure window closed; the pattern can be restored again.')
        # Merely seeing a locked archive can be remembered and revisited later.
        for eid, (_,role) in self.roles.items():
            if role=='archive' and 'rank_two' not in self.facts and self.near(char,eid,4):
                if not hasattr(self,'_locked_visits'): self._locked_visits=set()
                self._locked_visits.add(eid)

    def state(self):
        from dataclasses import asdict
        return dict(version=SUITE_VERSION, task=self.task, seed=self.seed,
                    facts=sorted(self.facts), completed=self.completed, inventory=self.inventory,
                    sites={key:asdict(s) for key,s in self.sites.items()}, cold=self.cold,
                    charge_until=self.charge_until, slate_energy=self.slate_energy,
                    quartz_ready=self._quartz_ready, locked_visits=sorted(self._visited_locked_archives),
                    metrics=self.metrics, feedback=self.feedback, feedback_time=self.feedback_time,
                    discovered=sorted(self.discovered))

    def verify(self):
        failures = []
        for key in self.facts:
            if not set(self.milestones[key].requires) <= self.facts:
                failures.append('missing prerequisite: '+key)
        for sid, site in self.sites.items():
            if sid in self.facts and not site.finished:
                failures.append('unfinished site: '+sid)
            if site.finished and site.kind!='ice' and self.world.get(site.entities['guard']).alive:
                failures.append('living guardian: '+sid)
            if site.finished and site.kind=='ice' and site.hits<2:
                failures.append('unthawed ice: '+sid)
        if 'final' not in self.facts: failures.append('final goal incomplete')
        if self.task=='cleansing' and not self.sites['heart'].finished: failures.append('Heart still active')
        return dict(success=not failures, failures=failures, completed=len(self.facts),
                    required=len(self.milestones), metrics=dict(self.metrics),
                    scope='Procedural campaign; native-action feasibility, not a learned-agent score')
