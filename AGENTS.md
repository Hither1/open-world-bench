# Rendering preference

Use Blender Cycles for user-facing Teyvat and Wildlands screenshots, demos,
previews, and recordings by default. The user explicitly prefers the full
renderer. Do not silently substitute CPU renders, low-resolution upscaling,
disabled visual features, or simplified scenery to save time. A lightweight
render must be explicitly requested and labeled.

Keep presentation separate from policy observations and simulation. Preserve
the actual task scene and action trace; changing terrain or collisions requires
new ground-truth validation. A compact prototype is still a compact prototype
when rendered in Cycles. Record the renderer and settings with artifacts.

Tests and headless logic checks may explicitly use inexpensive observation
rendering. For environments without a Cycles exporter, document the limitation
instead of implying their native output is Cycles.

# Asset fidelity and diversity

The user wants more realistic, varied environments, not repeated placeholder
cones and spheres. Prefer botanical branching, irregular rock shapes, distinct
functional structures, and differentiated natural materials. Preserve coherent
role cues across variants. Key appearance to stable world/entity identity so
objects do not change appearance as the camera moves, and never consume the
simulation RNG for visual detail.
