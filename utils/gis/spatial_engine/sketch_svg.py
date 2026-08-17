"""
Renders a schematic SVG sketch of a parcel's boundary — answers "can we
visualize the plot and its shape" using the real AGKK boundary geometry
we already fetch, rather than a placeholder.

Layout/style adapted from the user's own chsi-app project
(github.com/Kamend1/chsi-app, sketch.py), which draws the same kind of
schematic (grid background, translucent fill, centered identifier + area
labels, "not an official document" disclaimer) from sample boundary
points as a prototype pending the real official document. Since we have
AGKK's actual boundary coordinates rather than sample data, this renders
the genuine parcel shape — still schematic (no true north arrow, no
scale bar), not a substitute for AGKK's own official скица.

Unlike chsi-app's sketch.py, this does not attempt to label neighbours
along specific edges — we only know *which* parcels are neighbours (from
the Intersects spatial query), not which edge of the subject polygon each
one borders, and guessing that mapping wrong would be worse than omitting
it. Neighbours are listed as plain text elsewhere in the panel instead.
"""
from __future__ import annotations

import html

_VIEW_W = 420
_VIEW_H = 320
_PAD = 40


def _bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def render_parcel_svg(geometry_geojson: dict, cadastral_id: str, area_sqm: float | None = None) -> str:
    if geometry_geojson.get("type") != "Polygon":
        return _empty_svg("Скицата поддържа само единичен полигон")

    ring = geometry_geojson["coordinates"][0]
    points = [(pt[0], pt[1]) for pt in ring]
    if len(points) < 3:
        return _empty_svg("Няма гранични данни за скица")

    ident = html.escape(cadastral_id)
    minx, miny, maxx, maxy = _bounds(points)
    w = max(maxx - minx, 1e-9)
    h = max(maxy - miny, 1e-9)
    scale = min((_VIEW_W - 2 * _PAD) / w, (_VIEW_H - 2 * _PAD) / h)

    def tx(x: float) -> float:
        return _PAD + (x - minx) * scale

    # Flip Y: geographic north-up (larger latitude higher on screen) vs SVG's
    # top-down coordinate system.
    def ty(y: float) -> float:
        return _PAD + (maxy - y) * scale

    poly = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in points)
    cx = sum(tx(x) for x, _y in points) / len(points)
    cy = sum(ty(y) for _x, y in points) / len(points)
    area_txt = f"{area_sqm:.0f} кв.м" if area_sqm is not None else ""

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_VIEW_W} {_VIEW_H}" role="img" aria-label="Скица на имот {ident}">
  <rect x="0" y="0" width="{_VIEW_W}" height="{_VIEW_H}" fill="#fbfaf6"/>
  <defs>
    <pattern id="gis-sketch-grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e8e4d8" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect x="0" y="0" width="{_VIEW_W}" height="{_VIEW_H}" fill="url(#gis-sketch-grid)"/>
  <polygon points="{poly}" fill="#1f4d3a14" stroke="#1f4d3a" stroke-width="2"/>
  <text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" fill="#1f4d3a"
        font-family="monospace" font-size="12" font-weight="600">{ident}</text>
  <text x="{cx:.1f}" y="{cy + 16:.1f}" text-anchor="middle" fill="#4a4a4a"
        font-family="sans-serif" font-size="10">{area_txt}</text>
  <text x="{_VIEW_W / 2:.0f}" y="{_VIEW_H - 8}" text-anchor="middle" fill="#9a9a9a"
        font-family="sans-serif" font-size="9">Схематична скица по данни на АГКК — не заменя официална скица</text>
</svg>'''


def _empty_svg(message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_VIEW_W} {_VIEW_H}">'
        f'<text x="{_VIEW_W / 2:.0f}" y="{_VIEW_H / 2:.0f}" text-anchor="middle" fill="#6b6b6b" '
        f'font-family="sans-serif" font-size="14">{html.escape(message)}</text></svg>'
    )
