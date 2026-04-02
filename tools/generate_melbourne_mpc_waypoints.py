import math
from pathlib import Path

SRC = Path("/root/F1-Tenth-Duke-local/Code/sim_ws/src/f1tenth_gym_ros/maps/Melbourne_map.csv")
DST = Path("/root/F1-Tenth-Duke-local/Code/sim_ws/src/f1tenth_mpc-main/mpc/waypoints/Melbourne_map_mpc.csv")

V_MIN = 1.5
V_MAX = 5.0
K_SOFT = 0.22
K_HARD = 0.55


def wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def load_points(path: Path):
    pts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x, y = [float(v) for v in line.split(",")[:2]]
            pts.append((x, y))
    return pts


def smooth_points(pts, window=2):
    n = len(pts)
    smoothed = []
    for i in range(n):
        xs = 0.0
        ys = 0.0
        count = 0
        for k in range(-window, window + 1):
            px, py = pts[(i + k) % n]
            xs += px
            ys += py
            count += 1
        smoothed.append((xs / count, ys / count))
    return smoothed


def compute_yaw(pts):
    n = len(pts)
    yaw = []
    for i in range(n):
        px, py = pts[(i - 2) % n]
        nx, ny = pts[(i + 2) % n]
        yaw.append(wrap(math.atan2(ny - py, nx - px)))
    return yaw


def compute_curvature(pts, yaw):
    n = len(pts)
    curvatures = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]

        ds1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        ds2 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        ds = max((ds1 + ds2) * 0.5, 1e-6)

        dyaw = wrap(yaw[(i + 1) % n] - yaw[(i - 1) % n]) * 0.5
        curvature = abs(dyaw) / ds
        curvatures.append(curvature)
    return curvatures


def curvature_to_speed(kappa):
    if kappa <= K_SOFT:
        return V_MAX
    if kappa >= K_HARD:
        return V_MIN
    ratio = (kappa - K_SOFT) / (K_HARD - K_SOFT)
    return V_MAX - ratio * (V_MAX - V_MIN)


def smooth_scalar(values, window=2, wrap_angle_mode=False):
    n = len(values)
    out = []
    for i in range(n):
        vals = [values[(i + k) % n] for k in range(-window, window + 1)]
        if wrap_angle_mode:
            base = vals[0]
            adjusted = [base]
            for val in vals[1:]:
                diff = val - base
                while diff > math.pi:
                    val -= 2.0 * math.pi
                    diff = val - base
                while diff < -math.pi:
                    val += 2.0 * math.pi
                    diff = val - base
                adjusted.append(val)
            avg = sum(adjusted) / len(adjusted)
            out.append(wrap(avg))
        else:
            out.append(sum(vals) / len(vals))
    return out


def main():
    pts = load_points(SRC)
    pts = smooth_points(pts, window=2)
    yaw = compute_yaw(pts)
    yaw = smooth_scalar(yaw, window=3, wrap_angle_mode=True)
    curvature = compute_curvature(pts, yaw)

    speeds = [curvature_to_speed(k) for k in curvature]
    speeds = smooth_scalar(speeds, window=4)

    lines = []
    for (x, y), psi, v in zip(pts, yaw, speeds):
        lines.append(f"{x:.6f},{y:.6f},{psi:.6f},{v:.6f}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("\n".join(lines))

    print(f"Saved to: {DST}")
    print(f"Speed range: min={min(speeds):.3f}, max={max(speeds):.3f}")
    print("First 10 lines:")
    for line in lines[:10]:
        print(line)


if __name__ == "__main__":
    main()
