#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""Build the social sharing images from the Bloom kit, at each platform's own pixel spec.

WHY A GENERATOR AND NOT A FOLDER OF PNGs

The Bloom identity arrived as a kit from the operator, and the one previous replacement of it
(`b02383a`) records what goes wrong with hand-managed brand files: an asset survives under a name
the new kit reuses, and months later one icon does not match the rest. A script that rebuilds every
size from one source removes that failure by construction, and it also makes "does the card still
say the right thing" a diff rather than an archaeology exercise.

WHAT IS REUSED RATHER THAN REDRAWN, AND WHY IT MATTERS

The mark is `assets/brand/mark.svg`, the operator's vector, rasterised at each size.

The wordmark is LIFTED FROM `docs/public/social-card.png` rather than re-set in a font. This
machine has DejaVu and JetBrains Mono and neither is the geometric sans the kit uses, so setting
"SENBONZAKURA" here would produce a second, subtly different wordmark: the exact "one icon that does
not match the rest" failure the kit replacement was written about. Lifting the rendered pixels keeps
one wordmark in the project. It is only ever scaled DOWN from its native 1280 width, so nothing is
upscaled.

The supporting line is set in DejaVu, deliberately and visibly: it is secondary text, the mark and
the wordmark carry the identity, and a dependency on a font nobody else has would make this script
unreproducible for the next person.

THE SIZES, AND WHY THERE ARE THIS MANY

GitHub renders a repository's social preview at 2:1 and crops anything else. The four social
platforms all want roughly 1.91:1 and each states a slightly different pixel size; handing a
platform its exact size is what stops it re-encoding, and a re-encode on a gradient is where banding
comes from. The square is for a feed post rather than a link preview, which is a different surface
with a different crop.

Every layout keeps its content inside a centred safe box, because a link card is cropped differently
on almost every surface that shows it and the one thing that must survive is the mark and the name.
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import subprocess
import sys

#: REDESIGNED 2026-10-13 by the operator. The cards were navy with the mark and the name stacked
#: and centred. They are now white, with the mark on the left sitting in a soft gradient bleed, and
#: the name and tagline set to its right. What that buys, and it is the point of the change: a
#: reader scrolling past learns what the project IS, where before they learned only that it exists.
#:
#: The petals keep their own colour. An earlier pass recoloured them to the docs' bronze and the
#: operator rejected it: `mark.svg` is the source for the favicons, the icons and the site nav, so
#: the mark has to look the same in a preview as it does on the page the preview leads to.
WHITE = (255, 255, 255)

#: Near-black for the type. Not pure black: against a white card at thumbnail size, #000 reads as a
#: hole rather than as ink.
INK = (28, 28, 30)

#: The bleed behind the mark, lifted from the documentation site's hero rather than invented here:
#:
#:     --vp-home-hero-image-background-image:
#:         linear-gradient(-45deg, rgba(61,77,107,0.55) 45%, rgba(199,123,90,0.45) 55%);
#:     --vp-home-hero-image-filter: blur(48px);
#:
#: Two departures from those values, both deliberate. The alphas are raised, because the docs' were
#: tuned against a dark page and wash out to nothing on white. And the axis is turned so the WARM
#: stop falls at the bottom left: at the docs' angle the cool stop lands behind the petals' dark
#: purple bases and the two muddy each other.
HALO_WARM = (199, 123, 90, 205)
HALO_COOL = (61, 77, 107, 190)

#: The kit's deepest stop. No longer a background, kept because `glow()` and the measuring helper
#: still refer to the old card's navy when keying the lifted wordmark out of it.
NAVY = (20, 26, 66)

#: The kit's pink, used for the supporting line at low weight so it sits under the wordmark.
PETAL_PINK = (249, 141, 176)

#: The glow behind the mark, the kit's mid purple. Subtle by design: it gives the flat navy some
#: depth on a phone screen without becoming a second design element.
GLOW_PURPLE = (139, 71, 145)

#: The line under the wordmark. This is the project's tagline, already shipped on five surfaces, so
#: it is quoted rather than reinvented: `pyproject.toml`'s description opens with it.
TAGLINE = "The world's top open-weight AI model workshop."

ROOT = pathlib.Path(__file__).resolve().parents[2]
MARK = ROOT / "assets" / "brand" / "mark.svg"
WORDMARK_SOURCE = ROOT / "docs" / "public" / "social-card.png"
OUT_DIR = ROOT / "assets" / "brand" / "social"

#: Where the wordmark sits in `social-card.png`, measured from that file rather than assumed:
#: the content bands are y 84 to 423 (the mark) and y 553 to 605 (the wordmark), x 311 to 970.
#: Re-measure with the `--measure` flag if the source card is ever replaced.
WORDMARK_BOX = (311, 553, 971, 606)

#: (filename, width, height, why this exact size). One entry per surface that will actually be fed
#: one of these, with the platform's own stated spec rather than a shared approximation.
SIZES = [
    ("github-social-preview.png", 1280, 640,
     "GitHub renders a repository's social preview at 2:1 and crops anything else"),
    ("linkedin.png", 1200, 627, "LinkedIn's stated link share size"),
    ("x.png", 1200, 628, "X's summary_large_image card, 1.91:1"),
    ("facebook.png", 1200, 630, "Facebook's stated og:image size"),
    ("threads.png", 1200, 630, "Threads reads og:image, same spec as Facebook"),
    ("square.png", 1080, 1080, "a feed post rather than a link preview, cropped square"),
]


def _need(module):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"build_brand_assets needs {module}. Install it, or run this on a machine that "
                 f"has it; this script is a developer convenience and is not part of any build.")


def measure(path):
    """Print the content bands of a card, for re-deriving WORDMARK_BOX after a kit replacement."""
    Image = _need("PIL.Image") and __import__("PIL.Image", fromlist=["Image"])
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    bg = px[5, 5]

    def differs(x, y):
        return sum(abs(a - b) for a, b in zip(px[x, y], bg, strict=True)) > 40

    rows = [y for y in range(h) if any(differs(x, y) for x in range(0, w, 3))]
    if not rows:
        print(f"{path}: no content found against background {bg}")
        return
    bands, start = [], rows[0]
    for a, b in itertools.pairwise(rows):
        if b - a > 6:
            bands.append((start, a))
            start = b
    bands.append((start, rows[-1]))
    print(f"{path}: {w}x{h}, background {bg}")
    for y0, y1 in bands:
        cols = [x for x in range(w) if any(differs(x, y) for y in range(y0, y1 + 1, 2))]
        print(f"  band y {y0}-{y1}  x {cols[0]}-{cols[-1]}")


def render_mark(target_height):
    """The mark, cropped to its own ink and scaled so the FLOWER is `target_height` tall.

    CROPPED, AND THAT IS THE WHOLE POINT. `mark.svg` is a 512 square in which the flower occupies
    80% of the width and only 48% of the height, so sizing by the square silently renders a flower
    at half the height asked for. The first build of this kit did exactly that and produced cards
    whose mark was half the size of the one on the operator's own card. Composition should be stated
    in terms of the thing a reader sees, which is the ink, not the viewBox around it.
    """
    import io

    Image = __import__("PIL.Image", fromlist=["Image"])
    # Rendered large and scaled down, so the crop and the resize both happen on plenty of pixels.
    out = subprocess.run(
        ["rsvg-convert", "-w", "2048", "-h", "2048", str(MARK)],
        capture_output=True, check=True, timeout=120).stdout
    full = Image.open(io.BytesIO(out)).convert("RGBA")
    ink = full.crop(full.getchannel("A").getbbox())
    width = max(1, round(ink.width * target_height / ink.height))
    return ink.resize((width, target_height), Image.LANCZOS)


def wordmark():
    """The rendered wordmark, lifted from the kit's own card. See the module docstring."""
    Image = __import__("PIL.Image", fromlist=["Image"])
    card = Image.open(WORDMARK_SOURCE).convert("RGB")
    crop = card.crop(WORDMARK_BOX)
    # The lift carries the card's navy with it, so the background is keyed out into alpha. The
    # wordmark is white on navy, so luminance IS the mask, which keeps the letter edges' antialiasing
    # instead of hard-cutting them into a jagged stencil.
    Image_ = Image
    grey = crop.convert("L")
    out = Image_.new("RGBA", crop.size, (255, 255, 255, 0))
    out.putalpha(grey.point(lambda v: 0 if v < 28 else min(255, int((v - 28) * 255 / (235 - 28)))))
    white = Image_.new("RGBA", crop.size, (255, 255, 255, 255))
    white.putalpha(out.getchannel("A"))
    return white


def glow(size, radius_frac=0.62, strength=0.30):
    """A soft radial wash behind the mark, so a flat navy card has some depth on a phone."""
    Image = __import__("PIL.Image", fromlist=["Image"])
    ImageFilter = __import__("PIL.ImageFilter", fromlist=["ImageFilter"])
    ImageDraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    r = int(size * radius_frac / 2)
    c = size // 2
    draw.ellipse((c - r, c - r, c + r, c + r), fill=(*GLOW_PURPLE, int(255 * strength)))
    return layer.filter(ImageFilter.GaussianBlur(size * 0.13))


#: The brand's own face, vendored beside the mark with its licence. Hanken Grotesk is what the SITE
#: loads (`next/font/google` in the landing page's layout, now in its own repository); the documentation site uses Inter. Hanken is
#: the right one for a card because a reader who clicks a card lands on the site, so the preview
#: should look like the place it leads.
#:
#: Two static cuts rather than the variable original, instantiated from its own `wght` axis at 500
#: and 700. Instantiated rather than synthesised: Pillow cannot set a variable axis and has no real
#: bold, so asking it for weight would have got a smeared regular.
#:
#: Vendored rather than fetched, so a clean clone can rebuild the cards with no network and no
#: dependency on a web build having run. SIL Open Font License, text alongside in `OFL.txt`.
BRAND_FONTS = ROOT / "assets" / "brand" / "fonts"


def brand_face(weight, px):
    """Hanken Grotesk at `weight` and `px`, or None when the vendored file is missing."""
    ImageFont = __import__("PIL.ImageFont", fromlist=["ImageFont"])
    path = BRAND_FONTS / f"HankenGrotesk-{weight}.ttf"
    if not path.is_file():
        return None
    return ImageFont.truetype(str(path), px)


def halo(size):
    """The documentation site's hero bleed: a -45deg two-stop ramp, ellipse-masked, blurred hard.

    Drawn on a canvas larger than the blob so the blur falls off into transparency instead of being
    clipped into a hard edge at the tile boundary, which is what made the first attempt look like a
    pasted circle rather than light.
    """
    Image = __import__("PIL.Image", fromlist=["Image"])
    ImageDraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
    ImageFilter = __import__("PIL.ImageFilter", fromlist=["ImageFilter"])

    blur = max(8, round(size * 0.14))
    pad = blur * 3
    w = h = size + pad * 2
    grad = Image.new("RGBA", (w, h))
    px = grad.load()
    for y in range(h):
        for x in range(w):
            t = ((x / w) + (1 - y / h)) / 2
            if t <= 0.42:
                c = HALO_WARM
            elif t >= 0.58:
                c = HALO_COOL
            else:
                u = (t - 0.42) / 0.16
                c = tuple(round(HALO_WARM[i] + (HALO_COOL[i] - HALO_WARM[i]) * u) for i in range(4))
            px[x, y] = c
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((pad, pad, pad + size, pad + size), fill=255)
    grad.putalpha(Image.composite(grad.getchannel("A"), Image.new("L", (w, h), 0), mask))
    return grad.filter(ImageFilter.GaussianBlur(blur))


def _font(px):
    """DejaVu Sans at `px`, or None when it is not on this machine."""
    ImageFont = __import__("PIL.ImageFont", fromlist=["ImageFont"])
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/TTF/DejaVuSans.ttf",
                      "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if pathlib.Path(candidate).is_file():
            return ImageFont.truetype(candidate, px)
    return None


def _tracked_text(draw, text, font, tracking, fill, centre_x, y):
    """Draw `text` with extra space between letters, and return its width.

    The wordmark is widely tracked, so a supporting line set solid underneath it reads as belonging
    to a different design. Pillow has no letter-spacing, so the characters are placed individually.
    """
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = centre_x - total / 2
    for ch, w in zip(text, widths, strict=True):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + tracking
    return total


#: The tagline, broken where it should break rather than wherever the column runs out. "AI model
#: workshop." is the half a reader needs to keep, so it gets its own line instead of being orphaned
#: mid-phrase by a width calculation.
TAGLINE_LINES = ["The world's top open-weight", "AI model workshop."]


def _tracked(draw, text, font, tracking):
    return sum(draw.textlength(c, font=font) for c in text) + tracking * (len(text) - 1)


def _draw_tracked(draw, text, font, tracking, fill, x, y):
    for c in text:
        draw.text((x, y), c, font=font, fill=fill)
        x += draw.textlength(c, font=font) + tracking


def compose(width, height):
    """One card. Every proportion is stated against the canvas so each size composes the same.

    TWO ARRANGEMENTS, because one does not survive both shapes. The wide cards put the mark on the
    left and the type beside it, which is what the operator asked for and what uses a 1.91:1 canvas
    properly. The square has no width to give away, so it stacks: mark above, type below, same
    bleed, same faces. Resizing the wide layout into a square would have shrunk the type to
    illegibility, which is the failure the per-platform sizes exist to avoid.
    """
    Image = __import__("PIL.Image", fromlist=["Image"])
    ImageDraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])

    card = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(card)
    square = width == height

    title_face = brand_face(700, 10)
    if title_face is None:
        print(f"{BRAND_FONTS}: the vendored brand font is missing, so the cards cannot be set.",
              file=sys.stderr)
        raise SystemExit(2)

    if square:
        return _compose_square(card, draw, width, height)
    return _compose_wide(card, draw, width, height)


def _compose_wide(card, draw, width, height):
    """Mark left, name and tagline right. The arrangement for every link-preview size."""
    pad = round(width * 0.06)
    mark_h = round(height * 0.30)
    mark = render_mark(mark_h)
    blob = round(mark_h * 2.05)
    cx, cy = pad + blob // 2, height // 2

    h_img = halo(blob)
    card.paste(h_img, (cx - h_img.width // 2, cy - h_img.height // 2), h_img)
    card.paste(mark, (cx - mark.width // 2, cy - mark.height // 2), mark)

    col_x = pad + blob + round(width * 0.045)
    avail = width - col_x - pad

    # The name fills the column on ONE line. Tracked, because the wordmark it replaces was.
    t_px = round(height * 0.115)
    track = round(t_px * 0.14)
    f_title = brand_face(700, t_px)
    while _tracked(draw, "SENBONZAKURA", f_title, track) > avail and t_px > 20:
        t_px -= 1
        track = round(t_px * 0.14)
        f_title = brand_face(700, t_px)
    t_w = _tracked(draw, "SENBONZAKURA", f_title, track)

    s_px = round(height * 0.055)
    f_sub = brand_face(500, s_px)
    while max(draw.textlength(ln, font=f_sub) for ln in TAGLINE_LINES) > avail and s_px > 12:
        s_px -= 1
        f_sub = brand_face(500, s_px)

    leading = round(s_px * 1.30)
    gap = round(height * 0.052)
    block = t_px + gap + leading * len(TAGLINE_LINES)
    top = (height - block) // 2

    _draw_tracked(draw, "SENBONZAKURA", f_title, track, INK, col_x, top)
    # The tagline is centred on the NAME rather than ranged left with it: two short lines hanging
    # off a long tracked word read as an afterthought, centred they read as a unit.
    centre = col_x + t_w / 2
    y = top + t_px + gap
    for line in TAGLINE_LINES:
        w = draw.textlength(line, font=f_sub)
        draw.text((centre - w / 2, y), line, font=f_sub, fill=INK)
        y += leading
    return card


def _compose_square(card, draw, width, height):
    """Mark above, name and tagline below, centred. The feed-post arrangement."""
    pad = round(width * 0.08)
    mark_h = round(height * 0.26)
    mark = render_mark(mark_h)
    blob = round(mark_h * 2.05)
    cx = width // 2
    cy = round(height * 0.36)

    h_img = halo(blob)
    card.paste(h_img, (cx - h_img.width // 2, cy - h_img.height // 2), h_img)
    card.paste(mark, (cx - mark.width // 2, cy - mark.height // 2), mark)

    avail = width - pad * 2
    t_px = round(height * 0.082)
    track = round(t_px * 0.14)
    f_title = brand_face(700, t_px)
    while _tracked(draw, "SENBONZAKURA", f_title, track) > avail and t_px > 18:
        t_px -= 1
        track = round(t_px * 0.14)
        f_title = brand_face(700, t_px)
    t_w = _tracked(draw, "SENBONZAKURA", f_title, track)

    s_px = round(height * 0.042)
    f_sub = brand_face(500, s_px)
    while max(draw.textlength(ln, font=f_sub) for ln in TAGLINE_LINES) > avail and s_px > 12:
        s_px -= 1
        f_sub = brand_face(500, s_px)

    leading = round(s_px * 1.32)
    gap = round(height * 0.040)
    top = cy + blob // 2 - round(height * 0.03)

    _draw_tracked(draw, "SENBONZAKURA", f_title, track, INK, cx - t_w / 2, top)
    y = top + t_px + gap
    for line in TAGLINE_LINES:
        w = draw.textlength(line, font=f_sub)
        draw.text((cx - w / 2, y), line, font=f_sub, fill=INK)
        y += leading
    return card


def _compose_retired(width, height):
    """The navy stacked card, kept only so its measurements are not lost. Not called."""
    Image = __import__("PIL.Image", fromlist=["Image"])
    ImageDraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])

    card = Image.new("RGB", (width, height), NAVY)
    square = width == height

    # MEASURED AGAINST THE KIT'S OWN CARD, not chosen by eye: on `social-card.png` the flower is 340
    # of 640 tall, so 53%, and the wordmark is 659 of 1280 wide, so 51%. The square gets a smaller
    # mark because its canvas is taller and the same fraction would crowd the name.
    mark_h = int(height * (0.39 if square else 0.53))
    word_width = int(width * (0.52 if square else 0.51))
    gap = int(height * (0.035 if square else 0.055))

    mark = render_mark(mark_h)
    word = wordmark()
    word = word.resize((word_width, max(1, round(word.height * word_width / word.width))),
                       Image.LANCZOS)

    tag_px = max(13, int(height * (0.026 if square else 0.030)))
    font = _font(tag_px)
    tag_gap = int(height * 0.030)
    tag_block = (tag_gap + tag_px) if font else 0

    total = mark_h + gap + word.height + tag_block
    # OPTICALLY CENTRED ON THE SQUARE, mathematically centred on the wide cards. A tall canvas with
    # a bottom-weighted block reads as sinking, and the fix is to sit it slightly above true centre;
    # on a 1.91:1 card there is not enough vertical room for the effect to be worth the asymmetry.
    top = (height - total) // 2 - (int(height * 0.025) if square else 0)

    halo = glow(int(mark.width * 1.9))
    card.paste(Image.alpha_composite(Image.new("RGBA", halo.size, (*NAVY, 255)), halo).convert("RGB"),
               ((width - halo.width) // 2, top + mark_h // 2 - halo.height // 2))

    card.paste(mark, ((width - mark.width) // 2, top), mark)
    card.paste(word, ((width - word.width) // 2, top + mark_h + gap), word)

    if font:
        draw = ImageDraw.Draw(card)
        # White rather than the kit pink, at a weight that reads as a caption. Pink DejaVu under a
        # white geometric wordmark looked like a second brand rather than a supporting line.
        _tracked_text(draw, TAGLINE, font, tracking=max(1.0, tag_px * 0.055),
                      fill=(214, 206, 230), centre_x=width / 2,
                      y=top + mark_h + gap + word.height + tag_gap)
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(
        allow_abbrev=False, prog="build_brand_assets", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR), help=f"where to write (default: {OUT_DIR})")
    ap.add_argument("--measure", metavar="PNG",
                    help="print a card's content bands, to re-derive WORDMARK_BOX after a kit swap")
    a = ap.parse_args(argv)

    _need("PIL")
    if a.measure:
        measure(a.measure)
        return 0

    for path in (MARK, WORDMARK_SOURCE):
        if not path.is_file():
            sys.exit(f"missing kit asset: {path}")

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, width, height, why in SIZES:
        card = compose(width, height)
        target = out_dir / name
        card.save(target, "PNG", optimize=True)
        kb = target.stat().st_size / 1024
        print(f"  {name:28} {width}x{height}  {kb:6.0f} KB   {why}")
        if kb > 1024:
            print("    WARNING: over 1 MB, which GitHub refuses for a social preview")
    print(f"\n{len(SIZES)} asset(s) written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
