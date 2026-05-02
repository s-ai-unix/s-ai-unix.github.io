#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
ILLUSTRATIONS = ROOT / "static/images/illustrations"
COVERS = ROOT / "static/images/covers"
SIZE = 1024
SCALE = 3
W = SIZE * SCALE
H = SIZE * SCALE

INK = (31, 35, 45, 255)
CREAM = (250, 244, 230, 255)
PAPER = (255, 249, 238, 255)
RED = (236, 74, 58, 255)
ORANGE = (255, 155, 51, 255)
YELLOW = (255, 213, 74, 255)
GREEN = (67, 196, 120, 255)
CYAN = (43, 190, 217, 255)
BLUE = (70, 122, 255, 255)
PURPLE = (148, 92, 224, 255)
PINK = (238, 92, 163, 255)


def sc(v: float) -> int:
    return round(v * SCALE)


def box(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    return sc(x0), sc(y0), sc(x1), sc(y1)


def pt(x: float, y: float) -> tuple[int, int]:
    return sc(x), sc(y)


def make_canvas(seed: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    random.seed(seed)
    img = Image.new("RGBA", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    for _ in range(900):
        x = random.randrange(W)
        y = random.randrange(H)
        tone = random.randint(-10, 12)
        alpha = random.randint(10, 24)
        color = (
            max(0, min(255, PAPER[0] + tone)),
            max(0, min(255, PAPER[1] + tone)),
            max(0, min(255, PAPER[2] + tone)),
            alpha,
        )
        draw.point((x, y), fill=color)
    return img, draw


def finish(img: Image.Image) -> Image.Image:
    img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return img.convert("RGB")


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    finish(img).save(path, "PNG", optimize=True)


def line(draw: ImageDraw.ImageDraw, points, fill=INK, width=5, joint="curve") -> None:
    draw.line([pt(x, y) for x, y in points], fill=fill, width=sc(width), joint=joint)


def ellipse(draw: ImageDraw.ImageDraw, xy, fill, outline=INK, width=5) -> None:
    draw.ellipse(box(*xy), fill=fill, outline=outline, width=sc(width))


def rect(draw: ImageDraw.ImageDraw, xy, fill, outline=INK, width=5, radius=0) -> None:
    if radius:
        draw.rounded_rectangle(box(*xy), radius=sc(radius), fill=fill, outline=outline, width=sc(width))
    else:
        draw.rectangle(box(*xy), fill=fill, outline=outline, width=sc(width))


def polygon(draw: ImageDraw.ImageDraw, points, fill, outline=INK, width=5) -> None:
    draw.polygon([pt(x, y) for x, y in points], fill=fill)
    draw.line([pt(x, y) for x, y in points + [points[0]]], fill=outline, width=sc(width), joint="curve")


def shadow(img: Image.Image, mask: Image.Image, dx=12, dy=16, blur=18, alpha=70) -> None:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    blurred = mask.filter(ImageFilter.GaussianBlur(sc(blur))).point(lambda p: int(p * alpha / 255))
    layer.putalpha(blurred)
    img.alpha_composite(layer, (sc(dx), sc(dy)))


def draw_blob(draw: ImageDraw.ImageDraw, cx, cy, r, color, n=11, wobble=0.18) -> None:
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1 + math.sin(i * 1.7) * wobble)
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    polygon(draw, pts, color, width=4)


def draw_gear(draw: ImageDraw.ImageDraw, cx, cy, r1, r2, teeth, fill):
    pts = []
    for i in range(teeth * 2):
        a = 2 * math.pi * i / (teeth * 2)
        r = r2 if i % 2 == 0 else r1
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    polygon(draw, pts, fill, width=4)
    ellipse(draw, (cx - r1 * 0.35, cy - r1 * 0.35, cx + r1 * 0.35, cy + r1 * 0.35), PAPER, width=4)


def draw_chip(draw: ImageDraw.ImageDraw, x, y, w, h, fill=CYAN):
    rect(draw, (x, y, x + w, y + h), fill, radius=20, width=5)
    for i in range(7):
        yy = y + 24 + i * (h - 48) / 6
        line(draw, [(x - 36, yy), (x, yy)], width=4)
        line(draw, [(x + w, yy), (x + w + 36, yy)], width=4)
    for i in range(5):
        xx = x + 34 + i * (w - 68) / 4
        line(draw, [(xx, y - 34), (xx, y)], width=4)
        line(draw, [(xx, y + h), (xx, y + h + 34)], width=4)


def cellular_sky(draw: ImageDraw.ImageDraw, x0, y0, cols, rows, cell):
    for y in range(rows):
        for x in range(cols):
            if (x * 7 + y * 11 + (x ^ y)) % 5 in (0, 2):
                c = [BLUE, PURPLE, PINK, CYAN, YELLOW][(x + y) % 5]
                rect(draw, (x0 + x * cell, y0 + y * cell, x0 + (x + 1) * cell, y0 + (y + 1) * cell), c, width=2)


def theory_01(path: Path) -> None:
    img, draw = make_canvas(301)
    ellipse(draw, (180, 250, 820, 820), (237, 239, 245, 255), width=6)
    rect(draw, (236, 348, 788, 702), (250, 250, 252, 255), radius=40, width=5)
    for i, color in enumerate([CYAN, YELLOW, PINK, GREEN, ORANGE]):
        x = 300 + i * 88
        ellipse(draw, (x, 438, x + 64, 502), color, width=4)
        line(draw, [(x + 32, 502), (x + 32, 604)], width=4)
    for x in range(274, 745, 74):
        line(draw, [(x, 604), (x + 40, 604)], width=4)
    draw_gear(draw, 210, 230, 54, 72, 12, ORANGE)
    draw_gear(draw, 798, 236, 46, 62, 11, PURPLE)
    line(draw, [(512, 124), (512, 310), (588, 348)], width=6)
    ellipse(draw, (478, 94, 546, 162), BLUE, width=5)
    save(img, path)


def theory_02(path: Path) -> None:
    img, draw = make_canvas(302)
    rect(draw, (106, 480, 918, 626), (245, 235, 203, 255), radius=22, width=5)
    for i in range(11):
        x = 126 + i * 72
        rect(draw, (x, 500, x + 54, 606), [YELLOW, CYAN, PINK, GREEN][i % 4], width=3, radius=8)
        if i % 3 == 0:
            line(draw, [(x + 14, 530), (x + 40, 530), (x + 28, 578)], width=3)
    rect(draw, (392, 244, 632, 448), (232, 238, 250, 255), radius=34, width=6)
    ellipse(draw, (444, 292, 492, 340), RED, width=4)
    ellipse(draw, (534, 292, 582, 340), BLUE, width=4)
    line(draw, [(512, 448), (512, 514)], width=7)
    for x in [260, 760]:
        draw_blob(draw, x, 246, 74, (239, 241, 248, 150), n=9)
        line(draw, [(x - 34, 246), (x + 34, 246)], width=4)
        line(draw, [(x, 212), (x, 280)], width=4)
    line(draw, [(168, 700), (308, 676), (486, 724), (680, 676), (858, 710)], width=5)
    save(img, path)


def theory_03(path: Path) -> None:
    img, draw = make_canvas(303)
    rect(draw, (86, 102, 938, 764), (20, 32, 62, 255), radius=54, width=6)
    cellular_sky(draw, 124, 140, 20, 13, 38)
    for x, y, r, c in [(240, 246, 10, YELLOW), (694, 194, 9, CYAN), (780, 440, 12, PINK), (410, 554, 8, GREEN)]:
        ellipse(draw, (x - r, y - r, x + r, y + r), c, width=2)
    for y in [810, 848, 886]:
        line(draw, [(110, y), (914, y)], fill=(84, 78, 68, 255), width=3)
    polygon(draw, [(276, 774), (512, 632), (748, 774)], (255, 246, 190, 255), width=5)
    save(img, path)


def theory_04(path: Path) -> None:
    img, draw = make_canvas(304)
    for i in range(8):
        offset = i * 46
        line(draw, [(142 + offset, 166), (142 + offset, 798), (208 + offset, 798)], width=4)
        line(draw, [(142, 166 + offset), (802, 166 + offset)], width=4)
    rect(draw, (410, 330, 640, 660), (235, 232, 214, 255), radius=26, width=6)
    ellipse(draw, (466, 430, 586, 550), (221, 225, 235, 255), width=5)
    line(draw, [(526, 330), (526, 660)], width=5)
    ellipse(draw, (318, 706, 370, 758), RED, width=4)
    line(draw, [(344, 706), (388, 648), (430, 628)], width=5)
    draw_blob(draw, 746, 262, 70, PURPLE, n=10)
    save(img, path)


def theory_05(path: Path) -> None:
    img, draw = make_canvas(305)
    ellipse(draw, (128, 234, 536, 660), (255, 205, 184, 255), width=5)
    for i in range(9):
        x0 = 212 + i * 24
        line(draw, [(x0, 318), (x0 + 80, 356), (x0 + 28, 420), (x0 + 96, 486)], width=4)
    draw_chip(draw, 532, 286, 306, 306, CYAN)
    for i in range(7):
        line(draw, [(410 + i * 20, 484), (542, 440 + i * 18)], fill=[RED, BLUE, GREEN, PURPLE][i % 4], width=4)
    ellipse(draw, (438, 438, 584, 584), (255, 249, 238, 220), width=4)
    line(draw, [(176, 740), (850, 740)], width=5)
    save(img, path)


def theory_06(path: Path) -> None:
    img, draw = make_canvas(306)
    polygon(draw, [(126, 760), (420, 480), (522, 760)], (206, 217, 223, 255), width=5)
    polygon(draw, [(522, 760), (642, 432), (908, 760)], (189, 196, 207, 255), width=5)
    rect(draw, (208, 310, 438, 438), (247, 244, 226, 255), radius=18, width=5)
    ellipse(draw, (270, 184, 378, 292), YELLOW, width=5)
    line(draw, [(324, 292), (324, 310)], width=6)
    line(draw, [(438, 374), (682, 652), (704, 920)], fill=RED, width=5)
    for i in range(8):
        line(draw, [(650 + i * 24, 720 + i * 20), (690 + i * 14, 750 + i * 34)], fill=(70, 73, 84, 255), width=3)
    save(img, path)


def prediction_01(path: Path) -> None:
    img, draw = make_canvas(401)
    for i, color in enumerate([CYAN, YELLOW, PINK, GREEN, PURPLE]):
        rect(draw, (116 + i * 80, 188 + i * 22, 484 + i * 80, 336 + i * 22), (255, 255, 255, 210), radius=12, width=4)
        for k in range(4):
            line(draw, [(150 + i * 80, 224 + i * 22 + k * 26), (430 + i * 80, 224 + i * 22 + k * 26)], fill=color, width=3)
    polygon(draw, [(500, 510), (586, 404), (716, 430), (794, 534), (704, 666), (570, 640)], (125, 226, 255, 255), width=6)
    for p in [(586, 404), (716, 430), (794, 534), (704, 666), (570, 640), (500, 510)]:
        line(draw, [(640, 536), p], fill=(255, 255, 255, 190), width=3)
    line(draw, [(210, 668), (832, 720)], fill=INK, width=4)
    save(img, path)


def prediction_02(path: Path) -> None:
    img, draw = make_canvas(402)
    ellipse(draw, (188, 284, 398, 514), (255, 220, 174, 255), width=5)
    line(draw, [(292, 514), (292, 660)], width=5)
    rect(draw, (214, 650, 374, 826), BLUE, radius=30, width=5)
    ellipse(draw, (230, 350, 270, 390), (255, 255, 255, 255), width=3)
    ellipse(draw, (318, 350, 358, 390), (255, 255, 255, 255), width=3)
    line(draw, [(222, 326), (270, 314)], width=4)
    line(draw, [(320, 314), (368, 326)], width=4)
    for i, c in enumerate([RED, ORANGE, YELLOW, GREEN, CYAN, PURPLE, PINK]):
        y = 260 + i * 52
        line(draw, [(450, y), (840, 512)], fill=c, width=8)
        ellipse(draw, (420, y - 16, 452, y + 16), c, width=3)
    ellipse(draw, (806, 478, 884, 556), YELLOW, width=5)
    save(img, path)


def prediction_03(path: Path) -> None:
    img, draw = make_canvas(403)
    rect(draw, (150, 234, 874, 360), (220, 224, 232, 255), radius=18, width=6)
    rect(draw, (196, 360, 828, 540), (238, 242, 248, 255), radius=22, width=5)
    for i, c in enumerate([RED, BLUE, GREEN, PURPLE, ORANGE, CYAN]):
        draw_blob(draw, 170 + i * 120, 638 + (i % 2) * 36, 42, c, n=8)
        line(draw, [(170 + i * 120, 536), (170 + i * 120, 604)], fill=c, width=5)
    polygon(draw, [(452, 690), (512, 618), (596, 640), (626, 720), (544, 792), (464, 760)], (124, 224, 255, 255), width=6)
    for p in [(452, 690), (512, 618), (596, 640), (626, 720), (544, 792), (464, 760)]:
        line(draw, [(536, 710), p], fill=(255, 255, 255, 180), width=3)
    save(img, path)


def prediction_04(path: Path) -> None:
    img, draw = make_canvas(404)
    rect(draw, (96, 156, 928, 742), (28, 38, 66, 255), radius=40, width=6)
    for y in range(198, 690, 42):
        for x in range(142, 884, 42):
            if (x + y) % 5:
                ellipse(draw, (x - 5, y - 5, x + 5, y + 5), [CYAN, PURPLE, GREEN, PINK][(x + y) % 4], outline=None, width=0)
    polygon(draw, [(318, 664), (510, 442), (710, 664)], (83, 191, 132, 255), width=5)
    ellipse(draw, (474, 298, 596, 420), YELLOW, width=5)
    rect(draw, (606, 526, 746, 666), ORANGE, radius=10, width=4)
    line(draw, [(260, 706), (790, 706)], fill=(255, 255, 255, 220), width=4)
    save(img, path)


def prediction_05(path: Path) -> None:
    img, draw = make_canvas(405)
    colors = [CYAN, BLUE, PURPLE, PINK, ORANGE, YELLOW]
    for i in range(7):
        x = 170 + i * 90
        y = 760 - i * 76
        rect(draw, (x, y, x + 118, y + 58), colors[i % len(colors)], radius=10, width=4)
        draw_gear(draw, x + 58, y - 64, 22, 32, 10, colors[(i + 2) % len(colors)])
    for i in range(6):
        line(draw, [(230 + i * 90, 718 - i * 76), (320 + i * 90, 642 - i * 76)], width=5)
    draw_blob(draw, 778, 192, 102, (230, 238, 255, 255), n=12)
    ellipse(draw, (742, 154, 814, 226), YELLOW, width=5)
    save(img, path)


def prediction_06(path: Path) -> None:
    img, draw = make_canvas(406)
    rect(draw, (154, 430, 526, 670), (225, 218, 202, 255), radius=24, width=6)
    rect(draw, (214, 318, 466, 444), (236, 236, 236, 255), radius=16, width=5)
    for i in range(5):
        rect(draw, (210 + i * 58, 524, 252 + i * 58, 566), [RED, ORANGE, YELLOW, GREEN, CYAN][i], radius=6, width=3)
    line(draw, [(466, 382), (660, 302), (856, 188)], width=5)
    for i in range(14):
        x = 548 + i * 26
        y = 346 - i * 11 + math.sin(i) * 22
        ellipse(draw, (x - 7, y - 7, x + 7, y + 7), [YELLOW, CYAN, PINK, GREEN][i % 4], width=2)
    rect(draw, (522, 604, 898, 690), (255, 255, 255, 210), radius=12, width=4)
    line(draw, [(550, 646), (864, 646)], fill=PURPLE, width=4)
    save(img, path)


def cover_theory(path: Path) -> None:
    img, draw = make_canvas(501)
    ellipse(draw, (124, 160, 900, 850), (235, 238, 246, 255), width=7)
    cellular_sky(draw, 206, 220, 14, 10, 44)
    rect(draw, (320, 358, 704, 604), (255, 249, 238, 230), radius=42, width=6)
    draw_gear(draw, 420, 480, 50, 68, 12, ORANGE)
    draw_chip(draw, 520, 414, 126, 126, CYAN)
    line(draw, [(210, 770), (814, 770)], width=5)
    save(img, path)


def cover_prediction(path: Path) -> None:
    img, draw = make_canvas(502)
    rect(draw, (126, 190, 898, 740), (248, 241, 226, 255), radius=52, width=7)
    for i, c in enumerate([CYAN, BLUE, PURPLE, PINK, ORANGE, YELLOW, GREEN]):
        line(draw, [(180, 280 + i * 56), (470, 496), (824, 312 + i * 38)], fill=c, width=8)
    polygon(draw, [(448, 494), (516, 398), (628, 424), (694, 514), (620, 626), (500, 604)], (130, 229, 255, 255), width=7)
    ellipse(draw, (764, 250, 844, 330), YELLOW, width=5)
    for i in range(5):
        rect(draw, (202 + i * 72, 642, 260 + i * 72, 700), [RED, ORANGE, GREEN, CYAN, PURPLE][i], radius=8, width=3)
    save(img, path)


def main() -> None:
    tasks = [
        ("theory-01.png", theory_01),
        ("theory-02.png", theory_02),
        ("theory-03.png", theory_03),
        ("theory-04.png", theory_04),
        ("theory-05.png", theory_05),
        ("theory-06.png", theory_06),
        ("prediction-01.png", prediction_01),
        ("prediction-02.png", prediction_02),
        ("prediction-03.png", prediction_03),
        ("prediction-04.png", prediction_04),
        ("prediction-05.png", prediction_05),
        ("prediction-06.png", prediction_06),
    ]
    for filename, renderer in tasks:
        renderer(ILLUSTRATIONS / filename)
        print(f"wrote static/images/illustrations/{filename}")
    cover_theory(COVERS / "ai-first-principles-3-cover.png")
    print("wrote static/images/covers/ai-first-principles-3-cover.png")
    cover_prediction(COVERS / "ai-first-principles-4-cover.png")
    print("wrote static/images/covers/ai-first-principles-4-cover.png")


if __name__ == "__main__":
    main()
