"""Lossless face merging and whole-trace time coverage."""
import numpy as np
from owb.core import blocks as B
from owb.render.skylands_campaign import exact_surfaces
from owb.scripts.skylands_campaign_video import sampling


def test_merged_faces_keep_exact_surface_area_and_bounds():
    vox=np.zeros((8,7,9),np.uint8);vox[1:7,1:4,2:8]=B.STONE
    natural,blocks,architecture,_,water,area=exact_surfaces(vox)
    expected=2*(6*3+6*6+3*6)
    assert area==expected and len(natural)==12 and not len(architecture) and not len(water)
    tri_area=np.linalg.norm(np.cross(natural[:,1]-natural[:,0],natural[:,2]-natural[:,0]),axis=1).sum()/2
    assert tri_area==expected
    np.testing.assert_array_equal(natural.min(axis=(0,1)),[1,1,2]);np.testing.assert_array_equal(natural.max(axis=(0,1)),[7,4,8])
    assert (blocks==B.STONE).all()


def test_sampling_covers_complete_timeline_and_preserves_key_actions():
    rows=[dict(step=i,explanation='Walk the physical route',events=[dict(type='part_mounted')] if i==57 else []) for i in range(1,201)]
    selected=sampling(rows);assert 57 in selected and 200 in selected
    cursor=0
    for step,item in selected.items():
        assert item['source_start_step']==cursor and item['source_end_step']==step
        cursor=step
    assert cursor==200
