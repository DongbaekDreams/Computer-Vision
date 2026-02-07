"""Console output formatting."""

import os
import sys

ANSI_OK = sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")
RESET = "\x1b[0m"


def ansi_fg_gray(level_0_1):
    if not ANSI_OK:
        return ""
    level_0_1 = max(0.0, min(1.0, float(level_0_1)))
    code = 232 + int(round(level_0_1 * 23))
    return f"\x1b[38;5;{code}m"


def cgray(text, level):
    if not ANSI_OK:
        return text
    return f"{ansi_fg_gray(level)}{text}{RESET}"


def fmt_deg(x):
    return "--" if x is None else f"{int(x):3d}°"


def console_line(elapsed_s, fps, infer_ms, Lh, Rh, Lk, Rk, La, Ra, Ls, Rs, Le, Re):
    t_inf = (infer_ms - 6.0) / 20.0
    t_inf = max(0.0, min(1.0, t_inf))
    t_fps = (fps - 15.0) / 45.0
    t_fps = max(0.0, min(1.0, t_fps))

    s_fps = cgray(f"{fps:5.1f}", 0.35 + 0.55 * t_fps)
    s_inf = cgray(f"{infer_ms:5.1f}ms", 0.30 + 0.60 * t_inf)

    return (
        f"t={elapsed_s:6.1f}s | FPS {s_fps} | infer {s_inf} | "
        f"Hip L {fmt_deg(Lh)} R {fmt_deg(Rh)} | "
        f"Knee L {fmt_deg(Lk)} R {fmt_deg(Rk)} | "
        f"Ank L {fmt_deg(La)} R {fmt_deg(Ra)} | "
        f"Sh L {fmt_deg(Ls)} R {fmt_deg(Rs)} | "
        f"El L {fmt_deg(Le)} R {fmt_deg(Re)}"
    )
