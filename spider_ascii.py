"""
Hanging spider - 3D point cloud rendered as ASCII.

Pure Python (no numpy). The scene is a z-buffered point cloud that is
projected onto a character grid; each frame is drawn as a handful of
monospace text layers, one per brightness level, so it stays smooth.
"""
import math as m
import random as r

# ---------- settings ----------
TICK = 33              # ms per frame (~30 fps)
FONT_SIZE = 8
BG = "#05070d"
SCALE = 140.0          # pixels per world unit
THREAD_LEN = 2.6       # silk length (world units)
REVEAL_START = 12      # frame the thread starts to descend
REVEAL_FRAMES = 150    # frames until the spider is fully formed
BAND = 7               # rows of "decoding" glyphs behind the reveal front
SWING_AMP = 0.22       # radians
SPIN_SPEED = 0.012     # radians per frame

RAMP = " .:-=+*#%@"
SCRAMBLE = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZabdefghjkmnqrty#%&$"
NB = 6                 # brightness layers

# cool, restrained palette: deep steel blue -> ice white
DEEP = (0.17, 0.32, 0.66)
ICE = (0.93, 0.97, 1.00)
ACCENT = (0.35, 0.82, 0.86)
HUD = (0.30, 0.40, 0.58)


def mix(a, b, k):
    return tuple(p + (q - p) * k for p, q in zip(a, b))


def hx(c):
    return "#%02x%02x%02x" % tuple(int(max(0, min(1, v)) * 255) for v in c)


def smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


# ---------- model ----------
def _fib_sphere(n):
    ga = m.pi * (3 - m.sqrt(5))
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        rad = m.sqrt(max(0.0, 1 - y * y))
        a = ga * i
        yield rad * m.cos(a), y, rad * m.sin(a)


def _ellipsoid(c, rad, n, tex=None):
    out = []
    for ux, uy, uz in _fib_sphere(n):
        nx, ny, nz = ux / rad[0], uy / rad[1], uz / rad[2]
        ln = m.sqrt(nx * nx + ny * ny + nz * nz)
        t = tex(ux, uy, uz) if tex else 1.0
        out.append((c[0] + rad[0] * ux, c[1] + rad[1] * uy,
                    c[2] + rad[2] * uz, nx / ln, ny / ln, nz / ln, t))
    return out


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(v):
    ln = m.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def _tube(A, B, r0, r1, step=0.05, nc=5):
    a = (B[0] - A[0], B[1] - A[1], B[2] - A[2])
    ln = m.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    u = _unit(a)
    ref = (1, 0, 0) if abs(u[0]) < 0.9 else (0, 1, 0)
    p = _unit(_cross(u, ref))
    q = _cross(u, p)
    nl = max(2, int(ln / step))
    out = []
    for i in range(nl):
        s = (i + 0.5) / nl
        rr = r0 + (r1 - r0) * s
        for j in range(nc):
            ph = 2 * m.pi * (j + 0.5 * (i % 2)) / nc
            cp, sp = m.cos(ph), m.sin(ph)
            n = (p[0] * cp + q[0] * sp, p[1] * cp + q[1] * sp,
                 p[2] * cp + q[2] * sp)
            out.append((A[0] + a[0] * s + n[0] * rr,
                        A[1] + a[1] * s + n[1] * rr,
                        A[2] + a[2] * s + n[2] * rr,
                        n[0], n[1], n[2], 1.0))
    return out


def build_model():
    pts = []

    # abdomen: banded, with a lighter dorsal stripe
    def abd_tex(ux, uy, uz):
        t = 0.90 + 0.10 * m.cos(8 * uy)
        if abs(ux) < 0.20 and uz < 0:
            t += 0.30
        return t

    pts += _ellipsoid((0, 0.55, 0), (0.52, 0.68, 0.52), 1100, abd_tex)
    # cephalothorax
    pts += _ellipsoid((0, -0.45, 0), (0.33, 0.30, 0.33), 230)
    # eyes (small, bright)
    for sx in (-1, 1):
        pts += _ellipsoid((sx * 0.12, -0.36, 0.28), (0.055, 0.055, 0.055),
                          16, lambda a, b, c: 2.2)
    # fangs
    for sx in (-1, 1):
        pts += _tube((sx * 0.09, -0.68, 0.16), (sx * 0.06, -0.92, 0.20),
                     0.035, 0.015, step=0.04, nc=4)

    # eight legs: three segments each, knees raised, tips hanging down
    base_angles = (28, 62, 106, 148)
    scales = (0.95, 0.88, 0.92, 1.05)
    for side in (-1, 1):
        for a_deg, k in zip(base_angles, scales):
            a = m.radians(a_deg)
            d = (side * m.sin(a), 0.0, m.cos(a))
            b0 = (d[0] * 0.26, -0.45, d[2] * 0.26)
            j1 = (b0[0] + d[0] * 0.72 * k, b0[1] + 0.52 * k,
                  b0[2] + d[2] * 0.72 * k)
            j2 = (j1[0] + d[0] * 0.58 * k, j1[1] - 0.30 * k,
                  j1[2] + d[2] * 0.58 * k)
            j3 = (j2[0] + d[0] * 0.22 * k, j2[1] - 0.78 * k,
                  j2[2] + d[2] * 0.22 * k)
            pts += _tube(b0, j1, 0.100, 0.085, nc=6)
            pts += _tube(j1, j2, 0.085, 0.068, nc=6)
            pts += _tube(j2, j3, 0.068, 0.040, nc=6)
    return pts


MODEL = build_model()

# light & view directions in view space (x right, y up, z toward viewer)
LIGHT = _unit((-0.45, 0.55, 0.70))
VIEW = (0.0, 0.0, 1.0)
DEPTH = 14.0

# ---------- grid configuration (set by main or by tests) ----------
COLS = ROWS = 0
CW = CH = 1.0
_rng = r.Random(9)


def configure(cols, rows, cw, ch):
    global COLS, ROWS, CW, CH
    COLS, ROWS, CW, CH = cols, rows, cw, ch


def _to_model(v, ct, st, cp, sp):
    """Rotate a view-space vector back into model space."""
    x1 = v[0] * ct + v[1] * st
    y1 = -v[0] * st + v[1] * ct
    z1 = v[2]
    return (x1 * cp - z1 * sp, y1, x1 * sp + z1 * cp)


def render(f):
    """Return (layers, scramble, telemetry) for frame f.

    layers   : NB lists of characters (rows*cols), one per brightness level
    scramble : list of characters for the decoding band
    """
    n = COLS * ROWS
    reveal = (f - REVEAL_START) / REVEAL_FRAMES
    front = smooth(reveal) * (ROWS + BAND + 2) - 1
    done = reveal >= 1.0

    amp = SWING_AMP * smooth((f - REVEAL_START - 40) / 140)
    theta = amp * m.sin(0.045 * f)
    phi = SPIN_SPEED * f
    ct, st = m.cos(theta), m.sin(theta)
    cp, sp = m.cos(phi), m.sin(phi)

    cx = COLS * CW / 2.0
    py = -14.0            # pivot sits just above the top of the window
    L = THREAD_LEN

    zb = [-1e9] * n
    pid = [-1] * n
    k0 = SCALE * DEPTH
    for idx, (x, y, z, _a, _b, _c, _d) in enumerate(MODEL):
        x1 = x * cp + z * sp
        Z = z * cp - x * sp
        y1 = y - L
        X = x1 * ct - y1 * st
        Y = x1 * st + y1 * ct
        k = k0 / (DEPTH - Z)
        col = int((cx + X * k) / CW)
        row = int((py - Y * k) / CH)
        if 0 <= col < COLS and 0 <= row < ROWS:
            i = row * COLS + col
            if Z > zb[i]:
                zb[i] = Z
                pid[i] = idx

    lm = _to_model(LIGHT, ct, st, cp, sp)
    vm = _to_model(VIEW, ct, st, cp, sp)

    layers = [[" "] * n for _ in range(NB)]
    scr = [" "] * n
    sweep = -99.0
    if done:
        sweep = ((f - REVEAL_START - REVEAL_FRAMES) % 300) / 300.0 \
            * (ROWS + 24) - 8
    rng = _rng
    ramp_n = len(RAMP) - 1

    def put(i, row, ch_, b, is_thread=False):
        if row > front:
            return
        if row > front - BAND:
            scr[i] = rng.choice(SCRAMBLE)
            return
        if abs(row - sweep) < 1.6:
            b = min(1.0, b + 0.45)
        g = b ** 1.35
        li = min(NB - 1, int(g * NB))
        layers[li][i] = ch_ if is_thread else RAMP[
            max(1, min(ramp_n, int(g * (ramp_n + 1))))]

    for i in range(n):
        j = pid[i]
        if j < 0:
            continue
        _x, _y, _z, nx, ny, nz, tx = MODEL[j]
        diff = max(0.0, nx * lm[0] + ny * lm[1] + nz * lm[2])
        facing = max(0.0, nx * vm[0] + ny * vm[1] + nz * vm[2])
        rim = (1.0 - facing) ** 2
        b = tx * (0.10 + 0.62 * diff) + 0.32 * rim
        b = max(0.08, min(1.0, b))
        put(i, i // COLS, None, b)

    # silk thread: from the pivot to the top of the abdomen, swinging with it
    top = -L + 1.23
    steps = int((-top) * SCALE / (CH * 0.8)) + 1
    for s in range(steps + 1):
        yy = -top * s / steps
        X = yy * st
        Y = -yy * ct
        col = int((cx + X * SCALE) / CW)
        row = int((py - Y * SCALE) / CH)
        if 0 <= col < COLS and 0 <= row < ROWS:
            i = row * COLS + col
            if pid[i] < 0 and layers[0][i] == " " and scr[i] == " " \
                    and all(layers[q][i] == " " for q in range(1, NB)):
                put(i, row, "|" if abs(theta) < 0.22 else (
                    "/" if theta < 0 else "\\"), 0.30, True)

    tele = "SWING %+6.1f deg   SPIN %5.1f deg" % (
        m.degrees(theta), m.degrees(phi) % 360)
    return layers, scr, tele


# ---------- display ----------
def main():
    import turtle as t
    import tkinter.font as tkf

    s = t.Screen()
    s.setup(800, 800)
    s.bgcolor(BG)
    s.title("Spider")
    s.tracer(0)
    cv = s.getcanvas()

    avail = set(tkf.families(root=cv))
    fam = next((f for f in ("Consolas", "Menlo", "DejaVu Sans Mono",
                            "Cascadia Mono", "Courier New")
                if f in avail), "Courier")
    font = tkf.Font(root=cv, family=fam, size=FONT_SIZE)
    cw = font.measure("M")
    ch = font.metrics("linespace")
    cols = int(780 // cw)
    rows = int(780 // ch)
    configure(cols, rows, cw, ch)

    x0 = -cols * cw / 2
    y0 = -rows * ch / 2
    bg = tuple(int(BG[i:i + 2], 16) / 255 for i in (1, 3, 5))

    items = []
    for k in range(NB):
        c = mix(DEEP, ICE, (k / (NB - 1)) ** 1.15)
        items.append(cv.create_text(x0, y0, text="", anchor="nw",
                                    font=font, fill=hx(c)))
    scr_item = cv.create_text(x0, y0, text="", anchor="nw", font=font,
                              fill=hx(ACCENT))
    hud_font = tkf.Font(root=cv, family=fam, size=FONT_SIZE - 1)
    hud = cv.create_text(x0 + 6, -y0 - 6, text="", anchor="sw",
                         font=hud_font, fill=hx(bg))
    tag = cv.create_text(-x0 - 6, -y0 - 6, text="3D POINT CLOUD  /  ASCII",
                         anchor="se", font=hud_font, fill=hx(bg))

    state = {"f": 0}

    def join(grid):
        return "\n".join("".join(grid[q * cols:(q + 1) * cols])
                         for q in range(rows))

    def step():
        try:
            f = state["f"]
            layers, scr, tele = render(f)
            for item, grid in zip(items, layers):
                cv.itemconfig(item, text=join(grid))
            cv.itemconfig(scr_item, text=join(scr))
            fade = smooth((f - REVEAL_START - REVEAL_FRAMES) / 60)
            col = hx(mix(bg, HUD, fade))
            cv.itemconfig(hud, text=tele, fill=col)
            cv.itemconfig(tag, fill=col)
            state["f"] = f + 1
            cv.after(TICK, step)
        except Exception:
            return

    try:
        cv.after(200, step)
        t.done()
    except (t.Terminator, Exception):
        pass


if __name__ == "__main__":
    main()
