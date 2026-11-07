"""A dependency-free HUD: 3x5 bitmap font, quest banner, stamina and HP bars.

Only used for videos and human-facing previews; agent observations are drawn
without it so the HUD can never leak information the policy should not have.
"""
from __future__ import annotations

import numpy as np

_GLYPHS: dict[str, tuple[str, str, str, str, str]] = {
    "A": ("###", "# #", "###", "# #", "# #"),
    "B": ("## ", "# #", "## ", "# #", "## "),
    "C": (" ##", "#  ", "#  ", "#  ", " ##"),
    "D": ("## ", "# #", "# #", "# #", "## "),
    "E": ("###", "#  ", "## ", "#  ", "###"),
    "F": ("###", "#  ", "## ", "#  ", "#  "),
    "G": (" ##", "#  ", "# #", "# #", " ##"),
    "H": ("# #", "# #", "###", "# #", "# #"),
    "I": ("###", " # ", " # ", " # ", "###"),
    "J": ("  #", "  #", "  #", "# #", " # "),
    "K": ("# #", "# #", "## ", "# #", "# #"),
    "L": ("#  ", "#  ", "#  ", "#  ", "###"),
    "M": ("# #", "###", "###", "# #", "# #"),
    "N": ("# #", "## ", "###", " ##", "# #"),
    "O": (" # ", "# #", "# #", "# #", " # "),
    "P": ("## ", "# #", "## ", "#  ", "#  "),
    "Q": (" # ", "# #", "# #", " # ", "  #"),
    "R": ("## ", "# #", "## ", "# #", "# #"),
    "S": (" ##", "#  ", " # ", "  #", "## "),
    "T": ("###", " # ", " # ", " # ", " # "),
    "U": ("# #", "# #", "# #", "# #", "###"),
    "V": ("# #", "# #", "# #", "# #", " # "),
    "W": ("# #", "# #", "###", "###", "# #"),
    "X": ("# #", "# #", " # ", "# #", "# #"),
    "Y": ("# #", "# #", " # ", " # ", " # "),
    "Z": ("###", "  #", " # ", "#  ", "###"),
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("## ", "  #", " # ", "#  ", "###"),
    "3": ("###", "  #", " ##", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "## ", "  #", "## "),
    "6": (" ##", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", " # ", " # ", " # "),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "## "),
    ".": ("   ", "   ", "   ", "   ", " # "),
    ",": ("   ", "   ", "   ", " # ", "#  "),
    ":": ("   ", " # ", "   ", " # ", "   "),
    "-": ("   ", "   ", "###", "   ", "   "),
    "/": ("  #", "  #", " # ", "#  ", "#  "),
    "!": (" # ", " # ", " # ", "   ", " # "),
    "?": ("## ", "  #", " # ", "   ", " # "),
    "(": (" ##", "#  ", "#  ", "#  ", " ##"),
    ")": ("## ", "  #", "  #", "  #", "## "),
    "[": (" ##", " # ", " # ", " # ", " ##"),
    "]": ("## ", " # ", " # ", " # ", "## "),
    "'": (" # ", " # ", "   ", "   ", "   "),
    "+": ("   ", " # ", "###", " # ", "   "),
    "%": ("# #", "  #", " # ", "#  ", "# #"),
    " ": ("   ", "   ", "   ", "   ", "   "),
}

GLYPH_W, GLYPH_H = 3, 5


def text_width(text: str, scale: int = 1, spacing: int = 1) -> int:
    return len(text) * (GLYPH_W + spacing) * scale


def draw_text(img: np.ndarray, x: int, y: int, text: str, color=(255, 255, 255),
              scale: int = 1, spacing: int = 1, shadow: bool = True) -> np.ndarray:
    """Blit `text` into a uint8 image at (x, y). Returns the same array."""
    h, w = img.shape[:2]
    col = np.asarray(color, dtype=np.uint8)
    cx = x
    for ch in text.upper():
        glyph = _GLYPHS.get(ch)
        if glyph is None:
            cx += (GLYPH_W + spacing) * scale
            continue
        for row, line in enumerate(glyph):
            for colx, c in enumerate(line):
                if c != "#":
                    continue
                px, py = cx + colx * scale, y + row * scale
                if shadow:
                    sx0, sy0 = px + scale, py + scale
                    if 0 <= sy0 < h and 0 <= sx0 < w:
                        img[sy0:min(sy0 + scale, h), sx0:min(sx0 + scale, w)] = (
                            img[sy0:min(sy0 + scale, h), sx0:min(sx0 + scale, w)] // 3)
                if 0 <= py < h and 0 <= px < w:
                    img[py:min(py + scale, h), px:min(px + scale, w)] = col
        cx += (GLYPH_W + spacing) * scale
    return img


def wrap(text: str, max_chars: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > max_chars and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines


def panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.45) -> None:
    h_img, w_img = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (region * (1.0 - alpha)).astype(np.uint8)


def bar(img: np.ndarray, x: int, y: int, w: int, h: int, frac: float, color, back=(20, 22, 28)) -> None:
    h_img, w_img = img.shape[:2]
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x or y1 <= y:
        return
    img[y:y1, x:x1] = np.asarray(back, dtype=np.uint8)
    fill = int(round((x1 - x - 2) * float(np.clip(frac, 0.0, 1.0))))
    if fill > 0:
        img[y + 1:y1 - 1, x + 1:x + 1 + fill] = np.asarray(color, dtype=np.uint8)


def draw_hud(img: np.ndarray, env) -> np.ndarray:
    """Overlay quest text, bars and the interact prompt onto a rendered frame."""
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    scale = max(1, w // 320)
    pad = 6 * scale

    # Quest banner, top-left.
    lines = wrap(env.quests.instruction(), max_chars=max(18, w // (4 * scale)))[:3]
    banner_h = pad + len(lines) * 7 * scale
    panel(img, 0, 0, w, banner_h + pad // 2, alpha=0.5)
    for i, line in enumerate(lines):
        draw_text(img, pad, pad // 2 + i * 7 * scale, line, (248, 238, 206), scale)

    # Time of day, top-right.
    tl = f"{env.world.time_label()} {int(env.world.time_of_day * 24):02d}:00"
    draw_text(img, w - text_width(tl, scale) - pad, pad // 2, tl, (200, 220, 245), scale)

    # HP and stamina bars, bottom-left.
    bw, bh = int(w * 0.28), 5 * scale
    by = h - pad - bh * 2 - 4 * scale
    ch_cfg = env.controller.cfg
    bar(img, pad, by, bw, bh, env.char.hp / max(ch_cfg.max_hp, 1e-6), (226, 92, 92))
    bar(img, pad, by + bh + 3 * scale, bw, bh,
        env.char.stamina / max(ch_cfg.stamina_max, 1e-6), (118, 212, 164))
    draw_text(img, pad, by - 7 * scale, f"{env.char.state.name}", (210, 225, 240), scale)

    if env.party is not None:
        from ..elements import ELEMENT_COLORS
        from ..party import MAX_ENERGY

        x = max(pad, w - 98 * scale)
        for i, member in enumerate(env.party.members):
            y = banner_h + pad + i * 23 * scale
            color = tuple(int(v * 255) for v in ELEMENT_COLORS[member.spec.element])
            marker = ">" if i == env.party.active_index else " "
            label = f"{marker}{i + 1} {member.spec.name}"
            draw_text(img, x, y, label, color, scale)
            bar(img, x, y + 7 * scale, 85 * scale, 4 * scale,
                member.hp / env.party.member_max_hp(member), (226, 92, 92))
            bar(img, x, y + 12 * scale, 85 * scale, 3 * scale,
                member.energy / MAX_ENERGY, color)
        active = env.party.active
        skill = "READY" if env.party.ready_skill() else f"{active.skill_cd:.1f}S"
        burst = "READY" if env.party.ready_burst() else f"{active.energy:.0f}/{MAX_ENERGY:.0f}"
        draw_text(img, pad, by - 15 * scale,
                  f"SKILL {skill}  BURST {burst}", (245, 230, 180), scale)

    # Progress, bottom-right.
    snap = env.quests.snapshot()
    prog = f"{snap['completed_objectives']}/{env.graph.total_objectives} OBJ  {snap['completed_quests']}/{len(env.graph.quests)} QUESTS"
    draw_text(img, w - text_width(prog, scale) - pad, h - pad - 5 * scale, prog, (226, 226, 236), scale)

    # Interact prompt, centred.
    hint = env.world.last_interact_hint
    if hint:
        tw = text_width(hint, scale)
        draw_text(img, (w - tw) // 2, int(h * 0.62), hint, (255, 245, 180), scale)
    return img
