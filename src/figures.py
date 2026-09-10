"""Charts for the README, written straight to SVG from the results CSVs.

No plotting dependency. Three charts do not justify one, and hand-writing the
SVG means every mark on them is traceable to a number in a committed file
rather than to a notebook that no longer exists.

Colours are explicit hex, NOT CSS variables: GitHub sanitises SVG and strips
external context, so a variable-driven palette renders black-on-black. Each
colour is chosen to stay legible on both the light and dark GitHub themes,
which is why the axis furniture is mid-grey rather than near-black.

    py src/figures.py
"""
import pandas as pd

from paths import PROCESSED_DIR, ROOT

FIG_DIR = ROOT / "docs" / "figures"

INK = "#8B949E"      # axis text and grid, legible on both GitHub themes
BLUE = "#3B6FD4"     # the model / the arm under test
GREY = "#8A9199"     # the baseline it is being compared against
PALE = "#B9BFC7"     # the do-nothing baseline
OCHRE = "#C08A2E"    # the market
RED = "#D6543F"      # today / warning


def _svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{title}">'
            f'<title>{title}</title>{body}</svg>')


def _txt(x, y, s, size=11, fill=INK, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,'
            f'Arial,sans-serif">{s}</text>')


def checkpoint_chart(out=None):
    """Position RPS for the three arms, across the season."""
    R = pd.read_csv(PROCESSED_DIR / "checkpoint_backtest.csv")
    piv = R.pivot_table(index="target", columns="arm", values="rps")
    xs = list(piv.index)

    # A legend, not labels at the line ends: "August ratings + real table" is
    # ~150px of text and would run past the viewBox, which SVG will not tell you
    # about -- it simply draws outside the canvas and GitHub clips it.
    w, h = 720, 350
    pl, pr, pt, pb = 62, 34, 60, 46
    iw, ih = w - pl - pr, h - pt - pb
    ymax = 0.12

    def X(v):
        return pl + iw * (v - xs[0]) / (xs[-1] - xs[0])

    def Y(v):
        return pt + ih * (1 - v / ymax)

    b = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="none"/>']
    b.append(_txt(pl, 20, "Position RPS through the season (lower is better)",
                  size=13, fill=INK, weight="600"))
    for g in range(5):
        v = ymax * g / 4
        b.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{pl+iw}" y2="{Y(v):.1f}" '
                 f'stroke="{INK}" stroke-width="1" opacity="0.22"/>')
        b.append(_txt(pl - 9, Y(v) + 4, f"{v:.2f}", anchor="end"))
    for v in xs:
        b.append(_txt(X(v), h - 22, str(v), anchor="middle"))
    b.append(_txt(pl + iw / 2, h - 6, "Premier League matches played",
                  anchor="middle", size=11))

    series = [("frozen", PALE, "Frozen August forecast"),
              ("banked", GREY, "August ratings + real table"),
              ("refit", BLUE, "Full weekly re-fit")]
    lx = pl
    for _, colour, label in series:
        b.append(f'<rect x="{lx}" y="32" width="20" height="9" rx="1.5" '
                 f'fill="{colour}"/>')
        b.append(_txt(lx + 26, 40, label, size=11, fill=INK))
        lx += 34 + len(label) * 5.9
    for col, colour, label in series:
        pts = " ".join(f"{X(x):.1f},{Y(piv.loc[x, col]):.1f}" for x in xs)
        wid = 2.6 if col == "refit" else 1.8
        b.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                 f'stroke-width="{wid}" stroke-linejoin="round"/>')
        for x in xs:
            b.append(f'<circle cx="{X(x):.1f}" cy="{Y(piv.loc[x, col]):.1f}" '
                     f'r="3" fill="{colour}"/>')

    return _write(out or FIG_DIR / "checkpoint_rps.svg", _svg(
        w, h, "".join(b), "Position RPS by checkpoint for three forecast arms"))


def weight_chart(out=None):
    """Share of fitting weight carried by the current season."""
    pairs = [(1, .008), (4, .033), (8, .067), (15, .125), (22, .180),
             (30, .239), (38, .298)]
    w, h = 720, 300
    pl, pr, pt, pb = 62, 34, 34, 46
    iw, ih = w - pl - pr, h - pt - pb
    ymax = 0.35

    def X(g):
        return pl + iw * (g - 1) / 37

    def Y(v):
        return pt + ih * (1 - v / ymax)

    b = [_txt(pl, 20, "How much of the model's fitting weight is THIS season?",
              size=13, weight="600")]
    for g in range(6):
        v = ymax * g / 5
        b.append(f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{pl+iw}" y2="{Y(v):.1f}" '
                 f'stroke="{INK}" stroke-width="1" opacity="0.22"/>')
        b.append(_txt(pl - 9, Y(v) + 4, f"{v*100:.0f}%", anchor="end"))
    for g in (1, 10, 20, 30, 38):
        b.append(_txt(X(g), h - 22, f"GW{g}", anchor="middle"))

    pts = " ".join(f"{X(g):.1f},{Y(v):.1f}" for g, v in pairs)
    b.append(f'<polygon points="{X(1):.1f},{Y(0):.1f} {pts} '
             f'{X(38):.1f},{Y(0):.1f}" fill="{BLUE}" opacity="0.13"/>')
    b.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" '
             f'stroke-width="2.6" stroke-linejoin="round"/>')
    b.append(f'<line x1="{X(4):.1f}" y1="{pt}" x2="{X(4):.1f}" y2="{Y(0):.1f}" '
             f'stroke="{RED}" stroke-width="1.4" stroke-dasharray="4 3"/>')
    b.append(f'<circle cx="{X(4):.1f}" cy="{Y(0.033):.1f}" r="3.6" fill="{RED}"/>')
    b.append(_txt(X(4) + 9, pt + 14, "GW4: 3.3%", size=11, fill=RED, weight="600"))
    b.append(_txt(pl + iw / 2, h - 6, "Five seasons of history versus the season "
                  "in progress, at a one-year half-life", anchor="middle"))
    return _write(out or FIG_DIR / "weight_share.svg", _svg(
        w, h, "".join(b), "Share of fitting weight from the current season"))


def ladder_chart(out=None):
    """Held-out match log loss, each layer against the market."""
    rows = [("Uniform (1/3 each)", 1.0986, PALE),
            ("Base rates", 1.0682, PALE),
            ("Model, goals only", 0.9868, GREY),
            ("+ expected goals", 0.9858, GREY),
            ("+ market prior", 0.9790, GREY),
            ("+ availability", 0.9760, BLUE),
            ("Bookmaker closing line", 0.9640, OCHRE)]
    lo, hi = 0.955, 1.105
    w = 720
    rh, pt = 30, 40
    h = pt + rh * len(rows) + 34
    pl, pr = 178, 74
    iw = w - pl - pr

    b = [_txt(18, 20, "Held-out match log loss, 3,420 matches over 9 seasons "
              "(lower is better)", size=13, weight="600")]
    for i, (label, v, colour) in enumerate(rows):
        y = pt + i * rh
        bw = iw * (hi - v) / (hi - lo)
        b.append(_txt(pl - 12, y + 15, label, anchor="end", size=11.5,
                      fill=INK, weight="600" if colour in (BLUE, OCHRE) else "400"))
        b.append(f'<rect x="{pl}" y="{y+4}" width="{bw:.1f}" height="16" '
                 f'rx="2" fill="{colour}"/>')
        b.append(_txt(pl + bw + 8, y + 16, f"{v:.4f}", size=11, fill=colour,
                      weight="600"))
    b.append(_txt(18, h - 10, "The market is the benchmark, not a competitor: "
                  "this model has never beaten the closing line.", size=11))
    return _write(out or FIG_DIR / "logloss_ladder.svg", _svg(
        w, h, "".join(b), "Held-out match log loss by model layer"))


def _write(path, svg):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}  ({len(svg)/1024:.1f} KB)")
    return path


def build_all():
    print("figures:")
    return [checkpoint_chart(), weight_chart(), ladder_chart()]


if __name__ == "__main__":
    build_all()
