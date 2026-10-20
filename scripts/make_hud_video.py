"""Overlay the survival crab's inner state on Jumper-sim clips and join them into one video.

    python scripts/make_hud_video.py --out crab.mp4 \\
        threat.mp4:inst_threat.csv:threat  hunt.mp4:inst_hunt.csv:hunt:2:18  ...

Each clip is  VIDEO:CSV:SCENARIO[:START_S:END_S].  The CSV is the per-tick log written by play_with_brain.py --instinct.
Video time equals simulation time (the sim videos are rendered at real simulated speed), so frame i is at t = i / fps.
Needs opencv-python and ffmpeg. No music (the contest makes entrants license their own).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 720
INFO = {
    "hunt": ("HUNT", "jumping-spider stalk, freeze, pounce",
             "The ball is a stand-in for prey. A frozen, randomly wired spider-style network (not a real connectome) steers; the pounce is a forward-biased jump."),
    "threat": ("ESCAPE", "fly-style looming reflex",
               "The ball is a stand-in for a threat charging at the crab. The reflex is a hand-written looming score on its angular size, not a fly connectome."),
    "tun": ("SHUTDOWN", "tardigrade-style cryptobiosis",
            "A scripted dry spell raises stress. The crab crouches and drops its metabolism, then revives when it passes. A simple stand-in model."),
    "court": ("COURTSHIP", "gesture display",
              "The ball is a stand-in for a mate. The display uses Jumper's built-in gestures; acceptance is scripted."),
    "ledge": ("LEDGE JUMP", "pounce up onto a 4 cm ledge",
              "The ball (stand-in prey) sits on a raised slab 4 cm high. The crab stalks, freezes, then uses the forward-biased jump to leap up onto the ledge."),
    "court2": ("COURTSHIP", "display to a second crab",
               "The partner is a second Jumper crab model, posed (not driven): it faces the suitor and sways. Acceptance is scripted."),
    "soft": ("FLOOR SENSE", "soft-ground adaptation",
             "Softer, sinking ground gives a middling grip reading, so the crab is cautious but still moves. Compare ice (very low) and grass (full)."),
    "ice": ("FLOOR SENSE", "slippery-floor adaptation",
            "The crab estimates grip from how far it really moves. On ice it crouches and slows, and does not pounce."),
}
STATE_COL = {"wander": (200, 200, 200), "hunt": (80, 200, 255), "flee": (60, 60, 255), "flee_jump": (60, 60, 255), "flee_run": (60, 60, 255),
             "tun": (255, 160, 60), "revive": (120, 255, 160), "court": (200, 120, 255)}
FOOT = "Simulation only. Jumper by KingKong Robotics (Apache-2.0). Behaviour logic: github.com/adhritpradeepashoka/spiderbrain-moveset (AGPL-3.0)"


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t"] = float(r["t_s"])
    return rows


def at(rows, t):
    lo, hi = 0, len(rows) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if rows[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid - 1
    return rows[lo]


def text(img, s, org, scale=0.7, col=(255, 255, 255), th=2, shadow=True):
    if shadow:      # same thickness as the text (a thicker shadow gets different glyph spacing and smears), offset instead
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (2, 2)):
            cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def bar(img, x, y, w, label, v, col, mark=None):
    cv2.rectangle(img, (x, y), (x + w, y + 14), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, v))), y + 14), col, -1)
    cv2.rectangle(img, (x, y), (x + w, y + 14), (230, 230, 230), 1)
    if mark is not None:
        mx = x + int(w * max(0.0, min(1.0, mark)))
        cv2.line(img, (mx, y - 3), (mx, y + 17), (0, 255, 255), 2)
    text(img, label, (x + w + 8, y + 13), 0.45, th=1)


def hud(frame, r, scenario, surface=''):
    out = cv2.resize(frame, (W, H), interpolation=cv2.INTER_CUBIC)
    state = r["state"]
    if state == "tun":
        tint = np.full_like(out, (90, 50, 20))
        out = cv2.addWeighted(out, 0.72, tint, 0.28, 0)
    col = STATE_COL.get(state, (255, 255, 255))
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (W, 78), (0, 0, 0), -1)
    cv2.rectangle(ov, (0, H - 162), (W, H), (0, 0, 0), -1)
    out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
    text(out, state.upper().replace("_", " "), (16, 38), 1.0, col, 2)
    text(out, r["why"][:70], (16, 66), 0.55, (230, 230, 230), 1)
    if state == "tun":
        text(out, "metabolism 2%", (W - 230, 38), 0.8, (255, 190, 120), 2)
    title, bio, cap = INFO[scenario]
    text(out, f"{title}: {bio}", (16, H - 144), 0.6, (255, 255, 255), 1)
    f = lambda k: float(r[k]) if r.get(k) not in (None, "") else 0.0
    bar(out, 16, H - 124, 200, "energy", f("energy"), (90, 200, 90))
    bar(out, 16, H - 104, 200, "fear", f("fear"), (60, 60, 240))
    bar(out, 16, H - 84, 200, "mate urge", f("mate"), (200, 120, 255))
    bar(out, 330, H - 124, 200, "viability" + (" (| = no shutdown)" if scenario == "tun" else ""), f("viability"), (240, 160, 60), mark=f("shadow") if scenario == "tun" else None)
    bar(out, 330, H - 104, 200, "grip (slip sense)", f("grip"), (200, 200, 90))
    # bottom label: what the crab is doing, and what it is standing on
    ov = out.copy()
    cv2.rectangle(ov, (0, H - 40), (W, H), (0, 0, 0), -1)
    out = cv2.addWeighted(ov, 0.75, out, 0.25, 0)
    act = {"hunt": "HUNTING", "threat": "ESCAPING A THREAT", "tun": "SHUTDOWN (tardigrade)", "court": "COURTSHIP", "court2": "COURTSHIP", "ledge": "LEDGE JUMP", "ice": "ADAPTING TO THE FLOOR", "soft": "ADAPTING TO THE FLOOR"}.get(scenario, scenario.upper())
    text(out, f"{act}   |   SURFACE: {surface or 'grass pitch (normal grip)'}", (16, H - 12), 0.7, (255, 255, 255), 2)
    for k, line in enumerate(wrap(cap, 118)[:2]):
        text(out, line, (16, H - 53 + 11 * k), 0.45, (210, 210, 210), 1, shadow=False)
    return out


def card(lines, secs, fps, vw):
    img = np.zeros((H, W, 3), np.uint8)
    y = 250 - 28 * (len(lines) - 3)
    for k, (s, sc, col) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        cv2.putText(img, s, (max(16, (W - tw) // 2), y + 56 * k), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
    for _ in range(int(secs * fps)):
        vw.write(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", default="crab.mp4")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-cards", action="store_true")
    a = ap.parse_args()
    raw = Path(a.out).with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    if not a.no_cards:
        card([("SURVIVAL INSTINCT CRAB", 1.3, (255, 255, 255)), ("one Jumper, five instincts", 0.8, (180, 220, 255)),
              ("spider + fly + tardigrade behaviour logic", 0.7, (200, 200, 200)), ("all in simulation", 0.7, (200, 200, 200))], 3.0, a.fps, vw)
    for spec in a.clips:
        p = spec.split(":")
        vid, cs, scen = p[0], p[1], p[2]
        t0 = float(p[3]) if len(p) > 3 else 0.0
        t1 = float(p[4]) if len(p) > 4 else 1e9
        surf = p[5] if len(p) > 5 else ''
        rows = read_csv(cs)
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not a.no_cards:
            card([(INFO[scen][0], 1.5, STATE_COL.get({"hunt": "hunt", "threat": "flee", "tun": "tun", "court": "court", "court2": "court"}.get(scen, "wander"), (255, 255, 255))),
                  (INFO[scen][1], 0.9, (220, 220, 220))], 1.6, a.fps, vw)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t < t0:
                continue
            if t > t1:
                break
            vw.write(hud(fr, at(rows, t), scen, surf))
        cap.release()
    if not a.no_cards:
        card([("Bio-inspired behaviour logic", 0.9, (255, 255, 255)), ("not real connectomes. Only one crab is controlled:", 0.7, (200, 200, 200)),
              ("prey and threat are stand-in objects; the mate crab is posed.", 0.7, (200, 200, 200)), ("Open source: spiderbrain-moveset on GitHub", 0.7, (180, 220, 255)),
              ("Jumper by KingKong Robotics", 0.7, (200, 200, 200))], 4.0, a.fps, vw)
    vw.release()
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", a.out],
                       capture_output=True, text=True)
    if r.returncode:
        print("ffmpeg h264 re-encode failed; keeping the mp4v file:", r.stderr[-300:])
        raw.replace(a.out)
    else:
        raw.unlink()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
