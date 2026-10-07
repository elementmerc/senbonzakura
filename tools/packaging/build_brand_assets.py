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

#: The kit's deepest stop, and the background of every card. Read from `mark.svg` rather than typed
#: in twice: the gradient's last stop is the identity's background colour.
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


def compose(width, height):
    """One card. Every proportion is stated against the canvas so each size composes the same."""
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
