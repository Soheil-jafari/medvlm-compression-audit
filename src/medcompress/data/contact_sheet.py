"""Contact sheets of image pairs, for checking the near-duplicate grouping by eye (RC-D1).

A sheet is a private output: it shows dataset images, so it is written outside the repository
and never committed. Each pair is drawn side by side with the crop of the finer measure as a
yellow box, and a caption (the crop correlation, the dHash distance, the splits). No image
name is drawn.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

PairCell = tuple[str | Path, str | Path, str]
"""(first image, second image, caption)."""

THUMB = (200, 160)
PAD = 8
CAPTION = 16
TITLE = 22


def _thumb(path: str | Path, crop_fraction: float | None, size: tuple[int, int] = THUMB):  # type: ignore[no-untyped-def]
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(path) as image:
        small = ImageOps.contain(image.convert("RGB"), size)
    if crop_fraction is not None:
        draw = ImageDraw.Draw(small)
        w, h = small.size
        left, top = w * (1 - crop_fraction) / 2, h * (1 - crop_fraction) / 2
        draw.rectangle((left, top, w - left, h - top), outline=(255, 220, 0), width=2)
    return small


def render_pairs(
    bands: Sequence[tuple[str, Sequence[PairCell]]],
    out: Path,
    *,
    crop_fraction: float | None,
    columns: int = 2,
    numbered: bool = True,
    thumb: tuple[int, int] = THUMB,
) -> Path:
    """Draw ``bands`` (a title and its pairs, ``columns`` pairs per row) into one PNG at ``out``.

    ``crop_fraction``: draw that central box on each image (``None``: no box, as on the
    labelling sheets). ``numbered``: put the pair's place in its band before its caption.
    ``thumb``: the largest size of each image.
    """
    from PIL import Image, ImageDraw

    from medcompress.config import repo_root

    out = Path(out).resolve()
    if out.is_relative_to(repo_root().resolve()):
        raise ValueError(f"contact sheets show dataset images; write them outside the repo: {out}")
    cell_w = 2 * thumb[0] + 3 * PAD
    cell_h = thumb[1] + CAPTION + 2 * PAD
    height = PAD
    for _, cells in bands:
        height += TITLE + max(1, -(-len(cells) // columns)) * cell_h
    sheet = Image.new("RGB", (columns * cell_w + PAD, height), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    y = PAD
    for title, cells in bands:
        draw.text((PAD, y + 4), title, fill=(255, 255, 255))
        y += TITLE
        if not cells:
            draw.text((PAD, y + PAD), "(no pairs)", fill=(200, 200, 200))
        for k, (first, second, caption) in enumerate(cells):
            x0 = (k % columns) * cell_w + PAD
            y0 = y + (k // columns) * cell_h
            for j, path in enumerate((first, second)):
                picture = _thumb(path, crop_fraction, thumb)
                sheet.paste(picture, (x0 + j * (thumb[0] + PAD), y0))
            text = f"{k + 1}. {caption}" if numbered else caption
            draw.text((x0, y0 + thumb[1] + 2), text, fill=(255, 255, 255))
            frame = (x0 - 4, y0 - 4, x0 + 2 * thumb[0] + PAD + 4, y0 + thumb[1] + CAPTION + 2)
            draw.rectangle(frame, outline=(110, 110, 110), width=1)
        y += max(1, -(-len(cells) // columns)) * cell_h
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
