"""Conceptual mechanism figure: why orientation gain = wake pool x rose narrowness.

All curves are COMPUTED, not drawn: L(theta) from an analytic Jensen model on
reference square grids (dense 3.3D vs sparse 7D), roses are von Mises
distributions, and panel (d) is their literal circular convolution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nc_style import PAL, apply_style, panel_label, savefig
apply_style()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrow

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "FigC_mechanism_schematic.png")

# ----------------------------------------------------------------------
# Analytic Jensen wake model on an N x N square grid
# ----------------------------------------------------------------------
CT, K = 0.8, 0.05
A0 = 1.0 - np.sqrt(1.0 - CT)


def farm_loss(theta_deg, spacing_D, n=6):
    """Fractional power loss of an n x n grid for wind from direction theta."""
    idx = np.arange(n) - (n - 1) / 2.0
    X, Y = np.meshgrid(idx * spacing_D, idx * spacing_D)
    xs, ys = X.ravel(), Y.ravel()
    t = np.radians(theta_deg)
    xr = xs * np.cos(t) + ys * np.sin(t)          # downwind coordinate
    yr = -xs * np.sin(t) + ys * np.cos(t)         # crosswind coordinate
    p_tot = 0.0
    for i in range(len(xs)):
        dx = xr[i] - xr                            # >0 when j is upstream of i
        dy = np.abs(yr[i] - yr)
        up = dx > 1e-6
        rw = 0.5 + K * dx[up]
        inwake = dy[up] < rw
        d = A0 / (1.0 + 2.0 * K * dx[up][inwake]) ** 2
        deficit = np.sqrt((d ** 2).sum()) if d.size else 0.0
        p_tot += max(0.0, 1.0 - deficit) ** 3
    return 1.0 - p_tot / len(xs)


TH = np.arange(0, 360, 2.0)
L_dense = np.array([farm_loss(t, 3.3) for t in TH])
L_sparse = np.array([farm_loss(t, 7.0) for t in TH])


def vonmises(th_deg, mu_deg, kappa):
    th = np.radians(th_deg)
    mu = np.radians(mu_deg)
    p = np.exp(kappa * np.cos(th - mu))
    return p / p.sum()


rose_narrow = vonmises(TH, 45.0, 12.0)                       # monsoon spike
rose_wide = 0.6 * vonmises(TH, 250.0, 1.1) + 0.4 * vonmises(TH, 60.0, 0.9)
rose_wide /= rose_wide.sum()


def annual_loss(L, rose):
    """Circular convolution: mean loss as a function of array heading phi."""
    out = np.empty_like(L)
    n = len(L)
    for k in range(n):
        out[k] = (rose * np.roll(L, -k)).sum()
    return out


combos = {
    "dense_narrow": annual_loss(L_dense, rose_narrow),
    "dense_wide": annual_loss(L_dense, rose_wide),
    "sparse_narrow": annual_loss(L_sparse, rose_narrow),
    "sparse_wide": annual_loss(L_sparse, rose_wide),
}
for k, v in combos.items():
    print("%-14s loss %5.1f-%5.1f%%  recoverable %4.1f pp"
          % (k, v.min() * 100, v.max() * 100, (v.max() - v.min()) * 100))

# ----------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 5.9), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC1 = fig.add_subplot(gs[1, 0], projection="polar")
axD = fig.add_subplot(gs[1, 1])

# ======================================================================
# (a) lattice anisotropy: shadowing axis vs escape axis (schematic)
# ======================================================================
axA.set_xlim(-0.6, 12.2)
axA.set_ylim(-1.3, 5.6)
axA.set_aspect("equal")
axA.axis("off")


def draw_grid(ax, x0, wind_deg, tag):
    idx = np.arange(4)
    gx, gy = np.meshgrid(idx * 1.15, idx * 1.15)
    gx = gx + x0
    t = np.radians(wind_deg)
    ux, uy = np.cos(t), np.sin(t)
    for x, y in zip(gx.ravel(), gy.ravel()):
        wake = Polygon([(x, y - 0.10), (x, y + 0.10),
                        (x + 2.4 * ux - 0.42 * uy, y + 2.4 * uy + 0.42 * ux),
                        (x + 2.4 * ux + 0.42 * uy, y + 2.4 * uy - 0.42 * ux)],
                       closed=True, facecolor=PAL["highlight"], alpha=0.16,
                       edgecolor="none", zorder=2)
        ax.add_patch(wake)
    ax.scatter(gx.ravel(), gy.ravel(), s=16, color=PAL["ink"], zorder=4)
    ax.add_patch(FancyArrow(x0 - 0.05 - 1.3 * ux, 1.7 - 1.3 * uy,
                            1.0 * ux, 1.0 * uy, width=0.06, head_width=0.24,
                            color=PAL["baseline"], zorder=5))
    ax.text(x0 + 1.7, -1.05, tag, ha="center", fontsize=6.6, color=PAL["ink"])


draw_grid(axA, 1.3, 0.0, "wind along rows,\ndeep shadowing")
draw_grid(axA, 8.1, 30.0, "wind off-axis,\nwakes escape")
axA.text(0.56, 1.035, "a lattice is anisotropic:\nshadowing axes and escape axes",
         transform=axA.transAxes, ha="center", va="bottom", fontsize=6.8,
         color=PAL["ink"], linespacing=1.25)
panel_label(axA, "a", dy=1.16)

# ======================================================================
# (b) computed L(theta): density widens the angular contrast
# ======================================================================
axB.plot(TH, L_dense * 100, color=PAL["highlight"], lw=1.5,
         label="dense grid, 3.3 D")
axB.plot(TH, L_sparse * 100, color=PAL["baseline"], lw=1.5,
         label="sparse grid, 7 D")
axB.set_xlim(0, 180)
axB.set_ylim(0, 80)
axB.set_xticks([0, 45, 90, 135, 180])
axB.set_xlabel("Wind direction relative to rows (deg)")
axB.set_ylabel("Farm wake loss (%)")
axB.legend(loc="upper center", ncols=1, fontsize=6.0, handlelength=1.3,
           borderaxespad=0.2, bbox_to_anchor=(0.32, 1.0))
axB.annotate("aligned shadowing\n(rows, columns, diagonals)",
             xy=(91, 61), xytext=(103, 70), fontsize=6.0, color=PAL["ink"],
             linespacing=1.2,
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6))
axB.annotate("crosswind escape", xy=(15, 9), xytext=(30, 2.5), fontsize=6.0,
             color=PAL["ink"],
             arrowprops=dict(arrowstyle="-", color=PAL["neutral"], lw=0.6))
panel_label(axB, "b")

# ======================================================================
# (c) the two roses: narrow spike vs wide multimodal (polar)
# ======================================================================
th_r = np.radians(TH)
axC1.plot(th_r, rose_narrow / rose_narrow.max(), color=PAL["accent"], lw=1.5)
axC1.fill(th_r, rose_narrow / rose_narrow.max(), color=PAL["accent"], alpha=0.25)
axC1.plot(th_r, rose_wide / rose_wide.max(), color=PAL["neutral"], lw=1.5)
axC1.fill(th_r, rose_wide / rose_wide.max(), color=PAL["neutral"], alpha=0.18)
axC1.set_theta_zero_location("N")
axC1.set_theta_direction(-1)
axC1.set_yticklabels([])
axC1.tick_params(labelsize=6)
axC1.set_title("energy rose acts as a smoothing kernel:\nnarrow spike (amber) keeps the contrast,\nwide rose (grey) averages it away",
               fontsize=6.6, pad=8, linespacing=1.25)
panel_label(axC1, "c", dx=-0.12, dy=1.22)

# ======================================================================
# (d) convolution: annual loss vs heading — the product condition appears
# ======================================================================
PHI = TH
styles = {
    "dense_narrow": (PAL["highlight"], "-", "dense lattice, narrow rose"),
    "dense_wide": (PAL["highlight"], (0, (4, 2)), "dense lattice, wide rose"),
    "sparse_narrow": (PAL["baseline"], "-", "sparse lattice, narrow rose"),
    "sparse_wide": (PAL["baseline"], (0, (4, 2)), "sparse lattice, wide rose"),
}
for k, (c, ls, lab) in styles.items():
    axD.plot(PHI, combos[k] * 100, color=c, ls=ls, lw=1.4, label=lab)

v = combos["dense_narrow"] * 100
axD.annotate("", xy=(100, v.min()), xytext=(100, v.max()),
             arrowprops=dict(arrowstyle="<->", color=PAL["highlight"], lw=1.2))
axD.text(104, v.max() + 1.6,
         "recoverable by choosing the heading: %.0f pp\n(only this combination; flat dashed lines = 0)"
         % (v.max() - v.min()),
         fontsize=6.2, color=PAL["highlight"], va="bottom", linespacing=1.25,
         ha="left")

axD.set_xlim(0, 180)
axD.set_ylim(10, 48)
axD.set_xticks([0, 45, 90, 135, 180])
axD.set_xlabel("Array heading (deg)")
axD.set_ylabel("Annual mean wake loss (%)")
axD.legend(loc="upper left", fontsize=5.8, handlelength=1.6,
           labelspacing=0.3, borderaxespad=0.3)
axD.text(0.5, 0.025, "annual loss(heading) is the rose convolved with the lattice response",
         transform=axD.transAxes, ha="center", va="bottom", fontsize=6.2,
         color=PAL["neutral"])
panel_label(axD, "d")

savefig(fig, OUT)
print("exists:", os.path.exists(OUT), "bytes:", os.path.getsize(OUT))
