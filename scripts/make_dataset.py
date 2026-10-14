#!/usr/bin/env python3
"""Generates a synthetic wide-angle dataset with exact ground truth.

Each scene is an analytic function of continuous image coordinates (rotated
checkerboard, straight grid lines, discs, colour gradients), so it can be
rendered both

  * as the ideal pinhole image  -> data/ground_truth/frame_XXXX.ppm
  * through a Brown-Conrady lens -> data/distorted/frame_XXXX.ppm

without any resampling error. The lens is written to data/camera.txt in the
key=value format read by bin/lens_undistort --camera.

Only numpy is required. Output is binary PPM (P6).
"""

import argparse
import os
import sys
import time

import numpy as np

# Strong barrel distortion, typical of a ~120 degree action-camera lens.
DEFAULT_LENS = {
    "focal_scale": 0.6,
    "k1": -0.28,
    "k2": 0.07,
    "k3": 0.0,
    "p1": 0.0015,
    "p2": -0.0010,
}

SUPERSAMPLE = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))


def undistort_normalized(xd, yd, lens, iterations=40):
    """Inverts the lens model: distorted normalized coords -> ideal coords."""
    k1, k2, k3, p1, p2 = (lens[k] for k in ("k1", "k2", "k3", "p1", "p2"))
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    # Residual check of the fixed-point inversion.
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    rx = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd
    ry = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd
    return x, y, float(np.max(np.hypot(rx, ry)))


class Scene:
    """Random analytic scene; evaluate at arbitrary pixel coordinates."""

    def __init__(self, rng, width, height):
        self.w, self.h = width, height
        self.cell = rng.uniform(0.045, 0.09) * width
        self.angle = rng.uniform(-0.35, 0.35)
        self.offset = rng.uniform(0, self.cell, size=2)
        self.color_a = rng.uniform(0.55, 1.0, size=3)
        self.color_b = rng.uniform(0.0, 0.35, size=3)
        self.line_every = int(rng.integers(3, 6))
        self.line_color = rng.uniform(0.0, 1.0, size=3)
        n_discs = int(rng.integers(4, 9))
        self.discs = [
            (rng.uniform(0.1, 0.9) * width, rng.uniform(0.1, 0.9) * height,
             rng.uniform(0.03, 0.09) * width, rng.uniform(0.0, 1.0, size=3))
            for _ in range(n_discs)
        ]
        self.grad = rng.uniform(-0.25, 0.25, size=(2, 3))

    def render(self, px, py):
        """px, py: arrays of pixel coordinates -> float RGB in [0, 1]."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        xr = c * px + s * py + self.offset[0]
        yr = -s * px + c * py + self.offset[1]
        ix = np.floor(xr / self.cell)
        iy = np.floor(yr / self.cell)
        checker = ((ix + iy) % 2 == 0)[..., None]
        img = np.where(checker, self.color_a.astype(np.float32),
                       self.color_b.astype(np.float32))
        # Smooth colour gradient so the image is not piecewise constant.
        img = img + (px[..., None] / self.w) * self.grad[0] \
                  + (py[..., None] / self.h) * self.grad[1]
        # Thick straight lines: they become curves under barrel distortion,
        # which makes the correction easy to judge by eye.
        period = self.cell * self.line_every
        thick = 0.12 * self.cell
        on_line = ((np.mod(xr, period) < thick) |
                   (np.mod(yr, period) < thick))[..., None]
        img = np.where(on_line, self.line_color, img)
        for cx, cy, radius, color in self.discs:
            inside = ((px - cx) ** 2 + (py - cy) ** 2 < radius ** 2)[..., None]
            img = np.where(inside, color, img)
        return np.clip(img, 0.0, 1.0)


def sample_grid(coord_fn, width, height):
    """Scene coordinates of the 4 sub-pixel samples of every pixel.

    The lens is the same for every frame, so this is computed once and
    reused, just like the undistortion map on the GPU side.
    """
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    grid, worst = [], 0.0
    for du, dv in SUPERSAMPLE:
        sx, sy, residual = coord_fn(u + du, v + dv)
        worst = max(worst, residual)
        grid.append((sx.astype(np.float32), sy.astype(np.float32)))
    return grid, worst


def render_supersampled(scene, grid):
    """Averages the sub-pixel samples of |grid| into an RGB8 image."""
    acc = None
    for sx, sy in grid:
        sample = scene.render(sx, sy)
        acc = sample if acc is None else acc + sample
    img = np.clip(acc / len(grid) * 255.0 + 0.5, 0, 255)
    return img.astype(np.uint8)


def write_ppm(path, img):
    h, w, _ = img.shape
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(np.ascontiguousarray(img).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data", help="output root")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prefix", default="frame")
    parser.add_argument("--no_ground_truth", action="store_true",
                        help="skip writing ground-truth images")
    args = parser.parse_args()

    lens = dict(DEFAULT_LENS)
    w, h = args.width, args.height
    fx = fy = lens["focal_scale"] * w
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)

    dist_dir = os.path.join(args.out, "distorted")
    gt_dir = os.path.join(args.out, "ground_truth")
    os.makedirs(dist_dir, exist_ok=True)
    if not args.no_ground_truth:
        os.makedirs(gt_dir, exist_ok=True)
    with open(os.path.join(args.out, "camera.txt"), "w") as f:
        f.write("# Brown-Conrady lens used by scripts/make_dataset.py\n")
        f.write("# fx = fy = focal_scale * width, principal point at centre\n")
        for key, value in lens.items():
            f.write(f"{key}={value}\n")

    def ideal(u, v):
        return u, v, 0.0

    def through_lens(u, v):
        x, y, residual = undistort_normalized((u - cx) / fx, (v - cy) / fy,
                                              lens)
        return fx * x + cx, fy * y + cy, residual

    start = time.time()
    lens_grid, worst = sample_grid(through_lens, w, h)
    ideal_grid, _ = sample_grid(ideal, w, h)
    rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        scene = Scene(rng, w, h)
        name = f"{args.prefix}_{i:04d}.ppm"
        write_ppm(os.path.join(dist_dir, name),
                  render_supersampled(scene, lens_grid))
        if not args.no_ground_truth:
            truth = render_supersampled(scene, ideal_grid)
            write_ppm(os.path.join(gt_dir, name), truth)
        if (i + 1) % 25 == 0 or i + 1 == args.count:
            print(f"  generated {i + 1}/{args.count} ({w}x{h})", flush=True)
    print(f"done in {time.time() - start:.1f}s -> {args.out} "
          f"(max lens-inversion residual {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
