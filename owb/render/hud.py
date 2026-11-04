"""Small scalable hearts and a stamina ring for human-facing Wildlands views."""
import numpy as np

_HEART = np.array([
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
], bool)


def draw_hud(rgb, agent):
    height, width = rgb.shape[:2]
    scale = max(1, min(width // 200, height // 100))
    heart = np.repeat(np.repeat(_HEART, scale, axis=0), scale, axis=1)
    count = max(1, min(12, int(np.ceil(agent.max_health / 4))))
    for i in range(count):
        x, y = 9 * scale + i * (heart.shape[1] + 3 * scale), 9 * scale
        if x + heart.shape[1] > width or y + heart.shape[0] > height:
            break
        region = rgb[y:y + heart.shape[0], x:x + heart.shape[1]]
        region[heart] = (67, 72, 62)
        fraction = np.clip(agent.health / (agent.max_health / count) - i, 0, 1)
        fill = heart & (np.arange(heart.shape[1])[None, :] < heart.shape[1] * fraction)
        region[fill] = (226, 90, 67)
    radius = 7 * scale
    cx, cy = 17 * scale, 33 * scale
    y, x = np.ogrid[:height, :width]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (d <= radius) & (d >= radius - 2 * scale)
    angle = (np.arctan2(y - cy, x - cx) + np.pi / 2) % (2 * np.pi)
    rgb[ring] = (55, 67, 54)
    rgb[ring & (angle <= 2 * np.pi * agent.stamina / max(agent.stamina_cap, 1))] = (154, 208, 79)
