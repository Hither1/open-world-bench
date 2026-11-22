"""Run bounded observation-only model pilots on the Skylands campaign courses.

Same protocol and outputs as ``owb.scripts.wildlands_model``; see
docs/skylands-model-pilots.md for the Skylands action/observation adapter.
"""
from owb.env.skylands_campaign import SkylandsCampaignEnv
from owb.bench.baselines.skylands_model import SkylandsModelPolicy,local_view
from owb.scripts.skylands_campaign import SOURCES
from owb.scripts.skylands_reference import checked_step
from owb.scripts.wildlands_model import Course,main

MISSIONS={
 'a':'Recover the navigation core from the depths vault and restore the observatory navigation service.',
 'b':'Restore the depths, surface and sky station outputs and deliver the relief shipment to the designated refuge.',
}

SKYLANDS=Course('skylands_campaign',lambda task,seed: SkylandsCampaignEnv(task,seed,render_enabled=False),
                SkylandsModelPolicy,MISSIONS,checked_step,
                SOURCES+('owb/bench/baselines/skylands_model.py','owb/scripts/skylands_model.py',
                         'owb/bench/baselines/wildlands_model.py','owb/scripts/wildlands_model.py',
                         'owb/bench/baselines/codex_policy.py','owb/sim/construction.py'),
                observe=lambda env,obs: dict(obs,local_targets=local_view(env)))

if __name__=='__main__': main(SKYLANDS,__doc__)
