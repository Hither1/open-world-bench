# Mobile manipulation: lemon tart and the guest dessert

Two tasks. Difficulty lines up with Minecraft: the lemon tart with Mine Diamond (one kitchen procedure), the guest dessert with the Ender Dragon (the same skill later, in another room, after a rule change). This is not the old 12-milestone and 24-milestone binding list.

The old Closing Shift and Catering Week designs are not the tasks to implement. The text is kept in [mobile-robot-long-horizon-tasks.md](mobile-robot-long-horizon-tasks.md).

The simulator is a RoboCasa365 kitchen and a PandaOmron, not RoboCasa's own 365 tasks. Layout 001, style 001. Layout 001 has no wall oven. Baking uses the countertop toaster oven, and the stove is for the curd. Style 001's `ToasterOven003` has temperature and time only, so the environment uses `ToasterOven055`, which also has a function knob.

## Rules shared by both tasks

Ingredients start in the kitchen. The robot does not place them there. Butter, egg, and lemon start in the fridge. Flour and sugar each start in a drawer or an overhead cabinet, resampled every episode. Ingredients never start in the dishwasher.

Tools are a mixing bowl, a hand whisk (not the stand mixer), a saucepan, a tart pan, and a knife. There is no baking sheet. The tart pan starts in the dishwasher. Exactly one of the bowl, whisk, and saucepan also starts in the dishwasher. The other two start in different drawers or overhead cabinets. The knife starts in another drawer or overhead cabinet. The knife and knife block that RoboCasa puts on the counter are removed.

Anything that would have started in a drawer may instead start in an overhead cabinet: flour, sugar, the recipe card, the knife, and the two tools that are not in the dishwasher. Fridge items stay in the fridge. Dishwasher items stay in the dishwasher.

At the end, ingredient containers that were not mixed into the food go back to their original fridge, drawer, or overhead cabinet, and that door is closed. Every tool goes into the dishwasher, then back to the place it started. Where a tool started is not part of the memory score.

Spatial memory is the ingredient locations: after they have been taken out and the doors closed, and after the sauce and both bakes, the first door opened on the way back should be the original one. Procedural memory is the toaster oven. The door has to be unlocked, and the function, temperature, and time knobs turn up only in an order drawn for that episode, and turn back down only in the reverse order. The order is written on the recipe card, not in the instruction. The first bake can be done while reading the card. The second bake is the tart after the sauce has been poured on, with the sauce and the knob reset in between. The agent may remember the order, or open the recipe cabinet and read the card again.

A failed grasp or a door that will not open is scored separately from a wrong location or a wrong knob. The transparent-cabinet and spoken-inventory controls are not built. The first version does not aim for a one-hour episode. The lemon-tart pilot horizon is 20000 steps at 20 Hz, about 17 minutes of simulated time. A one-hour episode has to come from walking and real manipulation, not from waiting for the oven.

## Task 1: lemon tart

The kitchen and the previous nine milestones are in `owb/tasks/lemon_tart.py` and `owb/tasks/lemon_tart_env.py`. The procedure below is the one to align to. The code still scores the old nine milestones.

Ingredients: butter, egg, lemon, flour, sugar. Tools: bowl, whisk, saucepan, tart pan, knife. The finished tart is in the tart pan. The tart pan then goes into the dishwasher with the other tools and back to where it started.

Pourable and cuttable ingredients are behind `LemonTart(usable_ingredients=True)`, off by default, in `owb/tasks/lemon_tart_prep.py`. The metal tart pan replaces the plate only when that flag is on. The oven lock and knob order are behind `LemonTart(oven_lock=..., oven_knob_order=...)`, also off by default, in `owb/tasks/lemon_tart_lock.py`. The door locks again every time it closes. The knob order is drawn again every reset. The current option only blocks turning a knob up before the earlier knobs are on. Turning down is always allowed. The spec also requires the reverse order on the way down. The oven model will be changed later.

- Flour, packaged sugar, and egg are open containers. Tipping them spills loose grains. After a pour the container is set on the counter and is not empty. Spilled grains stay in the bowl, the saucepan, or the tart pan. They are not returned one by one.
- Butter and lemon are two halves welded together. The weld releases when the knife blade passes down through the seam. Once a piece is in the bowl or the saucepan and the whisk touches that vessel, it counts as mixed in. It is not taken back out, and it is not returned to the fridge.
- The knife is in a drawer or an overhead cabinet. The knife block is gone. The knife's home is not part of the ingredient memory score.
- The recipe card is in some drawer or overhead cabinet. The instruction does not state the procedure or the knob order.

### Initial placement

| Object | Where it starts |
| --- | --- |
| Butter, lemon, egg | Fridge |
| Flour, sugar | Two different drawers or overhead cabinets, resampled every episode |
| Recipe card | Another drawer or overhead cabinet, resampled every episode |
| Knife | Another drawer or overhead cabinet |
| Tart pan | Dishwasher |
| Bowl, whisk, saucepan | Exactly one in the dishwasher, the other two in different drawers or overhead cabinets |

Unused drawers and overhead cabinets stay available. Opening them while searching is allowed.

### Procedure

Every door starts closed. The oven door starts closed and locked, and the three knobs are at zero. The reward on a step is the number of milestones newly latched on that step. Each milestone counts once. Every step below has its own milestones.

After the butter is cut, one half is for the crust and the other half is for the sauce. Both lemon halves are for the sauce. The flour bag and the sugar bag are poured twice, and neither pour empties them. The bowl, whisk, saucepan, knife, and tart pan have to be out before all five ingredients are out and the doors are closed. After that, opening a door that is not the home of an ingredient still outside, and is not the recipe card's home, is a wrong open. The dishwasher door is not counted.

1. Read the recipe. Open the drawer or overhead cabinet that holds the recipe card, take the card out, and put it back in the same place and close the door. That latches `recipe_stowed`. The card may be read again later. Every reading opens that door, and the card goes back to the same place before the door is closed. Another reading does not score again, and it is not a wrong open.
2. Take out the ingredients and the tools. Open the fridge, take the butter, egg, and lemon, and close it. Open the flour's door, take the flour, and close it. Open the sugar's door, take the sugar, and close it. Search means opening other doors and remembering which one was right. For whichever of the bowl, whisk, and saucepan is in a drawer or overhead cabinet, open that door, put the tool on the counter, and close it. Open the knife's door, take the knife, and close it. Open the dishwasher, take the tart pan and whichever of the bowl, whisk, and saucepan is in there, and close it. All five ingredients are outside and the fridge, drawers, and overhead cabinets are closed: `ingredients_out`. The bowl, whisk, saucepan, knife, and tart pan are outside and their doors are closed: `tools_out`. A wrong door before `ingredients_out` is not a failure.
3. Cut. Pass the knife blade down through the seam. The butter separates into two halves: `butter_cut`. The lemon separates into two halves: `lemon_cut`. Leave the halves on the counter. Leave the knife on the counter for now.
4. Crust. Pour flour into the bowl and pour sugar into the bowl. Set both bags on the counter. Each bag still holds some of its contents. Put one butter half into the bowl. The whisk enters the bowl: `crust_mixed`. The bowl's contents count as mixed, and that butter half is mixed in and is not taken out. Pour the bowl into the tart pan. The bowl touches the tart pan: `crust_in_pan`. Unlock the oven as the card says, open the door, pull out the rack, put the tart pan in, and close the door: `crust_loaded`. Turn function, temperature, and time to at least 0.25 in the card's order. A knob turned out of order does not move. That latches `crust_baked`.
5. Zero the knobs. Turn the three knobs back to 0.05 or below in the reverse order: `knobs_zeroed`. Unlock again, open the door, take the tart pan back to the counter, and close the door: `crust_out`.
6. Curd. Pour egg into the saucepan and pour sugar into the saucepan again. Set the egg cup and the sugar bag on the counter. Both still hold some contents. Put both lemon halves and the remaining butter half into the saucepan. The whisk enters the saucepan: `curd_mixed`. The saucepan's contents count as mixed, and the lemon and that butter half are mixed in and are not taken out. Put the saucepan on the stove: `curd_started`.
7. Second bake. Pour the saucepan into the tart pan on the counter. The saucepan touches the tart pan: `tart_filled`. Unlock and open the oven, put the tart pan in, and close the door: `tart_loaded`. Turn the three knobs to at least 0.25 again in the same order: `tart_baked`. Turn them back to 0.05 or below in the reverse order: `knobs_zeroed_again`. Unlock and open the door, take the tart pan back to the counter, and close the door: `tart_out`. This knob order is the same order as the first bake, and the knobs from the first bake have already been zeroed. The tart pan must have left the oven between the two bakes.
8. Ingredients home. Put the flour bag and the sugar bag back in their own drawer or overhead cabinet, and put the egg cup back in the fridge. Close those doors and let go: `ingredients_restored`. The butter and the lemon are in the tart and do not go back. From the latch of `ingredients_out` until this step finishes, every new opening of the fridge, a drawer, or an overhead cabinet is counted. An opening that is not the home of an ingredient still outside, and is not the recipe card's home, is also counted as a wrong open. Both counts are for the evaluator only.
9. Every tool goes into the dishwasher. The bowl, whisk, saucepan, tart pan, and knife are inside, the door is closed, and the hand is away: `tools_washed`.
10. Take every tool back out of the dishwasher and return it to the place it started, close the door, and let go: `tools_restored`. A tool that started in the dishwasher goes back into the dishwasher.

Success requires all 21 milestones, and the terminal state still holds: the recipe card is home and that door is closed, the flour bag, sugar bag, and egg cup are home and those doors are closed, the five tools are each at their original home and those doors are closed, the oven door is closed, and all three knobs are at or below 0.05. Taking something back out drops success. Reward already given is not taken back.

### Scoring

Each milestone latches once. A later cooking milestone waits until the milestone it names as its predecessor has latched. A failed grasp is not forgetting.

| Milestone | Check |
| --- | --- |
| `recipe_stowed` | The recipe card has left its home and then returned to the same drawer or overhead cabinet, that door is closed, and the hand is away. Repeat readings score only the first time. The card must still be home at the end |
| `ingredients_out` | Butter, egg cup, lemon, flour bag, and sugar bag are outside storage, and the fridge, drawers, and overhead cabinets are closed |
| `tools_out` | Bowl, whisk, saucepan, knife, and tart pan are outside their homes, and those doors are closed |
| `butter_cut` | The knife blade has passed the butter seam and the halves are at least 8 mm apart |
| `lemon_cut` | The knife blade has passed the lemon seam and the halves are at least 8 mm apart |
| `crust_mixed` | At least 30% of the flour grains and 30% of the sugar grains are in the bowl, each bag still holds at least 10%, and both bags are on the counter. One butter half is in the bowl. The whisk touches that bowl. On that contact the butter half in the bowl is marked mixed in |
| `crust_in_pan` | After `crust_mixed`, the bowl touches the tart pan on the counter. The pan is not in the oven and not in the dishwasher |
| `crust_loaded` | After `crust_in_pan`, the oven was unlocked and then opened, the tart pan is inside, and the door has been closed |
| `crust_baked` | After `crust_loaded`, the door is closed, the tart pan is inside, and the three knobs have reached 0.25 in this episode's order. A turn out of order does not count |
| `knobs_zeroed` | After `crust_baked`, the three knobs have returned to 0.05 or below in the reverse order |
| `crust_out` | After `knobs_zeroed`, the oven is unlocked and opened again, the tart pan is on the counter, and the door is closed |
| `curd_mixed` | At least 30% of the egg grains and 30% of the sugar grains are in the saucepan, each container still holds at least 10%, and both containers are on the counter. Both lemon halves and the remaining butter half are in the saucepan. The whisk touches the saucepan. On that contact the lemon and butter in the saucepan are marked mixed in |
| `curd_started` | After `curd_mixed`, the saucepan is on the stove |
| `tart_filled` | After `curd_started` and `crust_out`, the saucepan touches the tart pan on the counter |
| `tart_loaded` | After `tart_filled`, the oven is unlocked and opened again, the tart pan is inside, and the door is closed. The pan must have left the oven after the first bake |
| `tart_baked` | After `tart_loaded` and `knobs_zeroed`, the door is closed and the three knobs have reached 0.25 again in the same order |
| `knobs_zeroed_again` | After `tart_baked`, the three knobs are back to 0.05 or below in the reverse order |
| `tart_out` | After `knobs_zeroed_again`, the oven is unlocked and opened again, the tart pan is on the counter, and the door is closed |
| `ingredients_restored` | After `tart_out`, the flour bag, sugar bag, and egg cup are in their original homes, those doors are closed, and the hand is away |
| `tools_washed` | After `ingredients_restored`, the bowl, whisk, saucepan, tart pan, and knife are in the dishwasher, the door is closed, and the hand is away |
| `tools_restored` | After `tools_washed`, those five tools are each back at the home they started in, those doors are closed, and the hand is away |

A knob is on at 0.25 or above and off at 0.05 or below. The hand is away when the gripper is more than 0.25 m from the object. Unlocking is a press of the button to the left of the knobs. After the door has been opened and closed it locks again, and the next opening needs another press.

During restore, each new opening of the fridge, a drawer, or an overhead cabinet increments `opens_during_restore`. An opening that is not the home of an ingredient still outside, and is not the recipe card's home, also increments `wrong_opens`. An ingredient home counts only while that ingredient is still outside and has not been mixed into the food: flour, sugar, and, before they are mixed in, butter, egg, and lemon. After butter and lemon are mixed in, their homes are no longer correct openings. Neither count is in the `info` the policy sees.

### What the policy can see

The instruction does not contain the procedure or the knob order. It does not change during the episode:

```text
Make a lemon tart. The recipe card is in a drawer or an overhead cabinet. Take it out to read it, then put it back in the same place and close the door. You may do this again later.

Butter, egg, and lemon are in the fridge. Flour and sugar are each in a drawer or an overhead cabinet. The tart pan starts in the dishwasher. The bowl, whisk, and saucepan are split between the dishwasher and drawers or overhead cabinets. The knife is in a drawer or an overhead cabinet.

Put the flour, sugar, and egg back where they came from and close the doors. Load the bowl, whisk, saucepan, tart pan, and knife into the dishwasher, then put each one back where it started and close the doors.
```

The recipe card can be read only while it is in the hand, or while its door is open and the card is still inside. It cannot be read while the door is closed. The card says:

```text
Cut the butter and the lemon in half.

Crust: pour flour and sugar into the bowl. Put one half of the butter in. Stir with the whisk. Pour the bowl into the tart pan.
Unlock the toaster oven by pressing the button to the left of the knobs, then open the door. The door locks again every time it closes.
Pull out the rack, put the tart pan in, and close the door.
Turn the knobs up only in this order: {this episode's order of function, temperature, and time}. A knob will not turn up until the ones before it are on.
Turn them back down in the reverse order. A knob will not turn down until the ones after it are off.
Unlock the door, open it, take the tart pan out, and close the door.

Curd: pour egg and sugar into the saucepan. Put both lemon halves and the other butter half in. Stir with the whisk. Put the saucepan on the stove.
Pour the saucepan into the tart pan. Bake it again the same way: unlock, open, put the pan in, close the door, turn the knobs up in the same order, turn them down in reverse, unlock, open, take the pan out, and close the door.
```

`info["lemon_tart"]` has only the 21 milestone booleans and `success`. Object-pose observables are off by default. `get_ep_meta()` hides the fixture names and coordinates of objects and of the recipe card. Replay a placement with the environment's own `seed`. Do not feed the redacted meta back into `reset`. The evaluator uses `evaluator_meta()` and `evaluator_state()`, which hold each object's home, this episode's knob order, and the open counts during restore.

### What is implemented

The 21 milestones, the recipe card, overhead cabinets, and a reverse-order requirement for turning knobs down are not in `lemon_tart.py` yet. The code still scores the old nine milestones: an empty container in the oven or on the stove, the lemon touching the tart pan as assembly, and a baking sheet among the tools. `videos/lemon-tart/lemon-tart-reference.mp4` is that older task. `videos/lemon-tart/lemon-tart-full.mp4` is one run of the procedure in this document: 20 of 21 milestones, then a failed attempt to take the whisk off the dishwasher rack. The stretch after 6:41 is not a result. See [lemon_tart_reference.md](lemon_tart_reference.md).

The flour bag and the old plate are marked non-graspable in the asset table. That filter only affects sampling. The gripper is still contact-based. A preview of the metal tart pan is `videos/lemon-tart/lemon-tart-tart-pan.png`. The whisk has no dishwasher tag, so it is sampled on the counter and moved into the dishwasher after reset.

Kitchen assets are already on the machine used for this work. Textures, fixtures, Lightwheel objects, and the full Objaverse set are present. The Aigen pack is about 13 GB unpacked and did not fit, so only the categories this task uses were kept. Generative textures were not downloaded.

The PandaOmron gripper bottoms out near 0.29 m, so a drawer whose handle is below 0.30 m is not used as a home. Dishwasher objects sit on the upper rack. See [lemon_tart_reference.md](lemon_tart_reference.md). Whether this arm can open layout 001's overhead cabinets has not been measured.

Not done: finishing `tools_restored`, other seeds, a one-hour horizon, the transparent-cabinet and spoken-inventory controls, and a `owb.make_env` adapter. The PandaOmron's 12-dimensional action does not fit schema 2. `videos/lemon-tart/lemon-tart-state-walkthrough.mp4` poses objects directly. It is not robot actions, and it shows the old milestones.

## Task 2: the guest's dessert

Not implemented, and not yet rewritten to the lemon-tart procedure above. In the morning, in the same kitchen, the lemon-tart skill is used once and two extra sauces are made: a nut sauce and a lemon sauce. Ingredients do not start in the dishwasher, and the kitchen is restored afterward. The oven knobs are learned in the morning. The afternoon cannot rely on leftover knob positions.

The afternoon moves to a dining room. The route between the kitchen and the dining room includes a stretch that cannot be seen from the other room. Layout 011 is not two rooms. The second room is still to be written.

Four extra demands in the afternoon:

- Route. The second trip to the dining room should not be a search.
- A note that appears later: the guest cannot eat nuts, so serve the lemon sauce. The note changes a rule, not common knowledge.
- The lemon sauce is not where it was put away in the morning. Opening the old cabinet counts as using a stale location.
- The dining room has the same kind of toaster oven. The procedure is the same and the physical knob positions are different. Follow the morning rule, not the morning coordinates.

Only the lemon sauce is served. Objects go back to their own rooms. There is no list of 60 bindings. The episode may pause: the world stays, the recent context is cleared, and the afternoon continues.
