#!/usr/bin/env python3
"""
Hand-gesture controller for the Google Earth flight simulator (web version).

Setup:    pip install mediapipe opencv-python pyautogui
          (use opencv-python, NOT opencv-python-headless, or no window opens)

Test:     python gesture_earth.py --test-keys     (keys only, no camera)
Fly:      python gesture_earth.py
          Click the Earth tab within 5 seconds, then hold both CLOSED hands up
          in front of the camera, palms toward it, as if gripping a yoke. Keep
          them still for one second. That pose becomes level flight.

Two-hand yoke (default)
  CLOSED hands steer:
      tilt the yoke   right hand lower = bank right, left hand lower = bank left
                      (equivalently: left hand up = bank right)
      raise / lower   both up = climb, both down = dive
      level pose      wings level, nose on horizon
  OPEN both hands:
      release stick   plane levels off; hold 1 s to set a new level pose
  SINGLE OPEN hand (hold ~0.25 s):
      right hand open, left closed   thrust up
      left hand open, right closed   thrust down
  Partial / unknown pose is treated as release.

Attitude mode: hands ask for a TARGET bank/pitch angle. The plane banks to that
angle and stops; return your hands to level and the plane unwinds back to level.

Profiles:
  --profile gentle   30 deg max bank, 50% duty cap
  --profile normal   40 deg max bank, 65% duty cap   (default)
  --profile sharp    55 deg max bank, 85% duty cap

Preview window keys (click the preview first):
  1 2 3 profile   - = gain down/up   c recenter   g guide   q quit
Quit from the terminal with Ctrl+C.

One hand (--style palm): open palm steers, fist = thrust up.
"""
import argparse
import math
import os
import sys
import time
import urllib.request

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]
ASPECT = 4 / 3
FIST_LEVEL = 0.70

# roll_deg    hand tilt that counts as full deflection
# pitch_full  hand-height change (frame fraction) that counts as full deflection
# bank_deg / pitch_deg   attitude target at full deflection
# att_cap     max duty cycle on the arrow keys (main safety limit)
# dz / expo   dead zone and fine-control curve
# min_cutoff / beta   one-euro filter for the raw pose
# roll_rate / pitch_rate   estimated game rotation rate, deg/s (only scales angles)
PROFILES = {
    "gentle": dict(roll_deg=30.0, pitch_full=0.16, bank_deg=30.0, pitch_deg=12.0,
                   att_cap=0.50, dz=0.16, expo=1.30, min_cutoff=1.0, beta=0.40,
                   roll_rate=60.0, pitch_rate=30.0),
    "normal": dict(roll_deg=28.0, pitch_full=0.14, bank_deg=40.0, pitch_deg=18.0,
                   att_cap=0.65, dz=0.13, expo=1.15, min_cutoff=1.1, beta=0.55,
                   roll_rate=60.0, pitch_rate=30.0),
    "sharp":  dict(roll_deg=24.0, pitch_full=0.12, bank_deg=55.0, pitch_deg=25.0,
                   att_cap=0.85, dz=0.10, expo=1.00, min_cutoff=1.4, beta=0.90,
                   roll_rate=75.0, pitch_rate=40.0),
}
PROFILE_ORDER = ["gentle", "normal", "sharp"]


# ---------------------------------------------------------------- hand maths
def dist(a, b):
    return math.hypot((a.x - b.x) * ASPECT, a.y - b.y)


def clamp(v, a, b):
    return max(a, min(b, v))


def deadzone(v, dz):
    return math.copysign(max(0.0, abs(v) - dz) / (1 - dz), v)


def expo(v, power):
    return math.copysign(abs(v) ** power, v)


def analyse(lm, fist_level=FIST_LEVEL):
    pts = [lm[i] for i in (0, 5, 9, 13, 17)]
    x = 1.0 - sum(p.x for p in pts) / 5.0
    y = sum(p.y for p in pts) / 5.0
    size = dist(lm[0], lm[9]) or 0.1

    class _C:
        pass
    c = _C()
    c.x, c.y = sum(p.x for p in pts) / 5.0, y
    curl = sum(dist(lm[t], c) for t in TIPS) / 4.0 / size
    extended = [dist(lm[t], lm[0]) > dist(lm[p], lm[0]) * 1.15 for t, p in zip(TIPS, PIPS)]

    gesture = "partial"
    if curl < fist_level:
        gesture = "fist"
    elif sum(extended) >= 3 and curl > 1.1:
        gesture = "open"
    return {"x": x, "y": y, "gesture": gesture, "curl": curl}


class OneEuro:
    def __init__(self, min_cutoff=1.5, beta=1.0, dcut=1.0):
        self.min_cutoff, self.beta, self.dcut = min_cutoff, beta, dcut
        self.reset()

    def configure(self, min_cutoff=None, beta=None):
        """Update filter parameters without resetting state."""
        if min_cutoff is not None:
            self.min_cutoff = min_cutoff
        if beta is not None:
            self.beta = beta

    def reset(self):
        self.t = self.x = None
        self.dx = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        self.dx += self._alpha(dt, self.dcut) * ((x - self.x) / dt - self.dx)
        self.x += self._alpha(dt, self.min_cutoff + self.beta * abs(self.dx)) * (x - self.x)
        return self.x


class Gesture:
    def __init__(self, confirm=6):
        self.confirm, self.raw, self.streak, self.value = confirm, None, 0, "none"

    def __call__(self, g):
        g = g or "none"
        self.streak = self.streak + 1 if g == self.raw else 1
        self.raw = g
        if self.streak >= self.confirm:
            self.value = g
        return self.value


# ---------------------------------------------------------------- attitude hold
class AttitudeHold:
    DEADBAND_S, E_FULL, PANIC_CAP = 0.05, 0.15, 0.9

    def __init__(self, rate_deg_s, max_deg, cap):
        self.rate = max(1.0, rate_deg_s)
        self.est = 0.0
        self.last_t = None
        self.configure(max_deg, cap)

    def configure(self, max_deg, cap, rate=None):
        """Update limits without resetting the estimate (safe mid-flight)."""
        self.max_deg = max(1.0, max_deg)
        self.cap = clamp(cap, 0.0, 1.0)
        if rate is not None:
            self.rate = max(1.0, rate)

    def reset(self):
        self.est = 0.0
        self.last_t = None

    @property
    def max_s(self):
        return self.max_deg / self.rate

    @property
    def deg(self):
        return self.est * self.rate

    def command(self, deflection, panic=False):
        err = clamp(deflection, -1.0, 1.0) * self.max_s - self.est
        if abs(err) < self.DEADBAND_S:
            return 0.0
        cap = max(self.cap, self.PANIC_CAP) if panic else self.cap
        return clamp(err / self.E_FULL, -1.0, 1.0) * cap

    def integrate(self, direction, now):
        if self.last_t is None:
            self.last_t = now
            return
        dt = clamp(now - self.last_t, 0.001, 0.1)
        self.last_t = now
        self.est = clamp(self.est + direction * dt, -3 * self.max_s, 3 * self.max_s)


# ---------------------------------------------------------------- the pilots
def _blank():
    return {"roll": 0.0, "pitch": 0.0, "roll_in": 0.0, "pitch_in": 0.0,
            "roll_target_deg": 0.0, "roll_est_deg": 0.0,
            "pitch_target_deg": 0.0, "pitch_est_deg": 0.0,
            "thrust": 0, "state": "searching", "progress": 0.0, "release_progress": 0.0,
            "notice": "", "gestures": {}, "pose": None, "panic": False}


class YokePilot:
    HOLD_S, STILL_DEG, STILL_Y, LOST_S, HOLD_LAST_S, RECENTER_S = 1.0, 3.0, 0.02, 1.5, 0.25, 1.0
    PANIC_SEP, PANIC_S = 0.22, 0.3
    THRUST_DELAY = 0.25

    def __init__(self, sign_roll=1, sign_pitch=1, sensitivity=1.0, profile="normal"):
        self.sr, self.sp, self.sens = sign_roll, sign_pitch, sensitivity
        self.g = {"L": Gesture(), "R": Gesture()}
        self.neutral = self.anchor = None
        self.anchor_t = self.notice_until = 0.0
        self.last_two = -1e9
        self.open_t, self.recentered, self.close_t, self.mixed_t = None, False, None, None
        self.last = _blank()
        self.f_angle = OneEuro()
        self.f_ymid = OneEuro()
        self.att_roll = AttitudeHold(60.0, 40.0, 0.65)
        self.att_pitch = AttitudeHold(30.0, 18.0, 0.65)
        self.set_profile(profile)

    def set_profile(self, name):
        self.profile_name = name
        self.p = dict(PROFILES[name])
        self.f_angle.configure(self.p["min_cutoff"], self.p["beta"])
        self.f_ymid.configure(self.p["min_cutoff"], self.p["beta"])
        self.att_roll.configure(self.p["bank_deg"], self.p["att_cap"], self.p["roll_rate"])
        self.att_pitch.configure(self.p["pitch_deg"], self.p["att_cap"], self.p["pitch_rate"])

    @staticmethod
    def pose(L, R):
        dx, dy = (R["x"] - L["x"]) * ASPECT, R["y"] - L["y"]
        return math.degrees(math.atan2(dy, dx)), (L["y"] + R["y"]) / 2.0

    def _rebase(self, pose):
        self.neutral = pose
        self.att_roll.reset()
        self.att_pitch.reset()

    def recalibrate(self):
        if self.last["pose"]:
            self._rebase(self.last["pose"])

    def integrate(self, roll_dir, pitch_dir, now):
        self.att_roll.integrate(roll_dir, now)
        self.att_pitch.integrate(pitch_dir, now)

    def update(self, hands, now):
        if len(hands) < 2:
            self.g["L"](None)
            self.g["R"](None)
            if now - self.last_two <= self.HOLD_LAST_S:
                return self.last
            if now - self.last_two > self.LOST_S:
                self.neutral = self.anchor = None
                self.att_roll.reset()
                self.att_pitch.reset()
            self.open_t, self.recentered, self.mixed_t = None, False, None
            out = _blank()
            out["state"] = "searching" if not hands else "need both hands"
            self.last = out
            return out

        L, R = hands[0], hands[-1]
        self.last_two = now
        raw_angle, raw_ymid = self.pose(L, R)
        angle = self.f_angle(raw_angle, now)
        ymid = self.f_ymid(raw_ymid, now)

        out = _blank()
        out["pose"] = (angle, ymid)
        out["gestures"] = {"L": self.g["L"](L["gesture"]), "R": self.g["R"](R["gesture"])}
        if math.hypot((R["x"] - L["x"]) * ASPECT, R["y"] - L["y"]) < self.PANIC_SEP:
            self.close_t = self.close_t if self.close_t is not None else now
        else:
            self.close_t = None
        out["panic"] = self.close_t is not None and now - self.close_t >= self.PANIC_S

        if self.neutral is None:
            if (self.anchor is None or abs(angle - self.anchor[0]) > self.STILL_DEG
                    or abs(ymid - self.anchor[1]) > self.STILL_Y):
                self.anchor, self.anchor_t = (angle, ymid), now
            elif now - self.anchor_t >= self.HOLD_S:
                self._rebase(self.anchor)
            out["state"] = "calibrating"
            out["progress"] = clamp((now - self.anchor_t) / self.HOLD_S, 0, 1)
            if self.neutral is None:
                self.last = out
                return out

        na, ny = self.neutral
        roll_norm = expo(deadzone(clamp((angle - na) / (self.p["roll_deg"] / self.sens), -1, 1),
                                  self.p["dz"]), self.p["expo"])
        pitch_norm = expo(deadzone(clamp(-(ymid - ny) / (self.p["pitch_full"] / self.sens), -1, 1),
                                   self.p["dz"]), self.p["expo"])
        gl, gr = out["gestures"]["L"], out["gestures"]["R"]
        panic = out["panic"]

        if panic:
            roll_norm = pitch_norm = 0.0

        if gl == "open" and gr == "open":
            self.open_t = self.open_t if self.open_t is not None else now
            self.mixed_t = None
            roll_norm = pitch_norm = 0.0
            held = now - self.open_t
            out["release_progress"] = 0.0 if self.recentered else clamp(held / self.RECENTER_S, 0, 1)
            if held >= self.RECENTER_S and not self.recentered:
                self.recentered, self.notice_until = True, now + 1.5
                self._rebase((angle, ymid))

        elif gl == "fist" and gr == "fist":
            self.open_t, self.recentered, self.mixed_t = None, False, None

        elif gl == "fist" and gr == "open":
            self.open_t, self.recentered = None, False
            roll_norm = pitch_norm = 0.0
            self.mixed_t = self.mixed_t if self.mixed_t is not None else now
            if not panic and now - self.mixed_t >= self.THRUST_DELAY:
                out["thrust"] = 1

        elif gl == "open" and gr == "fist":
            self.open_t, self.recentered = None, False
            roll_norm = pitch_norm = 0.0
            self.mixed_t = self.mixed_t if self.mixed_t is not None else now
            if not panic and now - self.mixed_t >= self.THRUST_DELAY:
                out["thrust"] = -1

        else:
            self.open_t, self.recentered, self.mixed_t = None, False, None
            roll_norm = pitch_norm = 0.0

        out["roll_in"] = roll_norm
        out["pitch_in"] = pitch_norm
        out["roll"] = self.att_roll.command(self.sr * roll_norm, panic)
        out["pitch"] = self.att_pitch.command(self.sp * pitch_norm, panic)
        out["roll_target_deg"] = roll_norm * self.p["bank_deg"]
        out["roll_est_deg"] = self.att_roll.deg
        out["pitch_target_deg"] = pitch_norm * self.p["pitch_deg"]
        out["pitch_est_deg"] = self.att_pitch.deg
        out["notice"] = "RECENTERED" if now < self.notice_until else ""
        out["state"] = "flying"
        self.last = out
        return out


class PalmPilot:
    HOLD_S, STILL, LOST_S = 1.0, 0.05, 1.5

    def __init__(self, sign_roll=1, sign_pitch=1, sensitivity=1.0, profile="normal", still=None):
        self.sr, self.sp, self.sens = sign_roll, sign_pitch, sensitivity
        self.still = self.STILL if still is None else still
        self.g = Gesture(4)
        self.neutral = self.anchor = self.pos = None
        self.last_pos = None
        self.anchor_t, self.last_seen, self.last = 0.0, -1e9, _blank()
        self.fx, self.fy = OneEuro(), OneEuro()
        self.att_roll = AttitudeHold(60.0, 40.0, 0.65)
        self.att_pitch = AttitudeHold(30.0, 18.0, 0.65)
        self.set_profile(profile)

    def set_profile(self, name):
        self.profile_name = name
        self.p = dict(PROFILES[name])
        self.fx.configure(self.p["min_cutoff"], self.p["beta"])
        self.fy.configure(self.p["min_cutoff"], self.p["beta"])
        self.att_roll.configure(self.p["bank_deg"], self.p["att_cap"], self.p["roll_rate"])
        self.att_pitch.configure(self.p["pitch_deg"], self.p["att_cap"], self.p["pitch_rate"])

    def _choose(self, hands):
        if self.last_pos is None:
            return hands[-1]
        return min(hands, key=lambda h: math.hypot(h["x"] - self.last_pos[0],
                                                   h["y"] - self.last_pos[1]))

    def _rebase(self, pos):
        self.neutral = pos
        self.att_roll.reset()
        self.att_pitch.reset()

    def recalibrate(self):
        if self.pos:
            self._rebase(self.pos)

    def integrate(self, roll_dir, pitch_dir, now):
        self.att_roll.integrate(roll_dir, now)
        self.att_pitch.integrate(pitch_dir, now)

    def update(self, hands, now):
        out = _blank()
        if not hands:
            self.g(None)
            if now - self.last_seen > self.LOST_S:
                self.neutral = self.anchor = self.pos = self.last_pos = None
                self.att_roll.reset()
                self.att_pitch.reset()
                self.fx.reset()
                self.fy.reset()
            self.last = out
            return out
        h = self._choose(hands)
        self.last_pos = (h["x"], h["y"])
        self.last_seen = now
        self.pos = (self.fx(h["x"], now), self.fy(h["y"], now))
        out["gestures"] = {"R": self.g(h["gesture"])}
        out["pose"] = self.pos
        if self.neutral is None:
            p = (h["x"], h["y"])
            if self.anchor is None or math.hypot(p[0] - self.anchor[0], p[1] - self.anchor[1]) > self.still:
                self.anchor, self.anchor_t = p, now
            elif now - self.anchor_t >= self.HOLD_S:
                self._rebase(self.anchor)
            out["state"] = "calibrating"
            out["progress"] = clamp((now - self.anchor_t) / self.HOLD_S, 0, 1)
            if self.neutral is None:
                self.last = out
                return out
        nx, ny = self.neutral
        roll_norm = expo(deadzone(clamp((self.pos[0] - nx) * self.sens / 0.22, -1, 1),
                                  self.p["dz"]), self.p["expo"])
        pitch_norm = expo(deadzone(clamp(-(self.pos[1] - ny) * self.sens / 0.20, -1, 1),
                                   self.p["dz"]), self.p["expo"])
        out["roll_in"] = roll_norm
        out["pitch_in"] = pitch_norm
        out["roll"] = self.att_roll.command(self.sr * roll_norm)
        out["pitch"] = self.att_pitch.command(self.sp * pitch_norm)
        out["roll_target_deg"] = roll_norm * self.p["bank_deg"]
        out["roll_est_deg"] = self.att_roll.deg
        out["pitch_target_deg"] = pitch_norm * self.p["pitch_deg"]
        out["pitch_est_deg"] = self.att_pitch.deg
        if out["gestures"]["R"] == "fist":
            out["thrust"] = 1
        out["state"] = "flying"
        self.last = out
        return out


# ---------------------------------------------------------------- key output
class KeyDriver:
    MIN_HOLD = 0.04

    def __init__(self, kb):
        self.kb = kb
        self.acc = {"roll": 0.0, "pitch": 0.0}
        self.down = {}

    def _apply(self, wanted, now):
        for k in list(self.down):
            if k not in wanted and now - self.down[k] >= self.MIN_HOLD:
                self.kb.keyUp(k)
                del self.down[k]
        for k in wanted:
            self.kb.keyDown(k)
            self.down.setdefault(k, now)

    def update(self, roll, pitch, now):
        wanted = set()
        for axis, d, neg, pos in (("roll", roll, "left", "right"), ("pitch", pitch, "down", "up")):
            self.acc[axis] = clamp(self.acc[axis] + d, -1.5, 1.5)
            if self.acc[axis] >= 1:
                self.acc[axis] -= 1
                wanted.add(pos)
            elif self.acc[axis] <= -1:
                self.acc[axis] += 1
                wanted.add(neg)
        self._apply(wanted, now)
        return (("right" in self.down) - ("left" in self.down),
                ("up" in self.down) - ("down" in self.down))

    def release(self):
        self.acc = {"roll": 0.0, "pitch": 0.0}
        for k in list(self.down):
            self.kb.keyUp(k)
        self.down.clear()


class _NullKB:
    def keyDown(self, k): pass
    def keyUp(self, k): pass
    def press(self, k): pass


def _make_input_struct():
    import ctypes as C

    class MOUSEINPUT(C.Structure):
        _fields_ = [("dx", C.c_int32), ("dy", C.c_int32), ("mouseData", C.c_uint32),
                    ("dwFlags", C.c_uint32), ("time", C.c_uint32), ("dwExtraInfo", C.c_size_t)]

    class KEYBDINPUT(C.Structure):
        _fields_ = [("wVk", C.c_uint16), ("wScan", C.c_uint16), ("dwFlags", C.c_uint32),
                    ("time", C.c_uint32), ("dwExtraInfo", C.c_size_t)]

    class HARDWAREINPUT(C.Structure):
        _fields_ = [("uMsg", C.c_uint32), ("wParamL", C.c_uint16), ("wParamH", C.c_uint16)]

    class _U(C.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(C.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", C.c_uint32), ("u", _U)]

    return INPUT


class WinKeys:
    SCAN = {"left": 0x4B, "right": 0x4D, "up": 0x48, "down": 0x50, "pageup": 0x49, "pagedown": 0x51}
    KEYUP, EXTENDED, SCANCODE = 0x0002, 0x0001, 0x0008

    def __init__(self, send_input=None):
        import ctypes
        self.INPUT = _make_input_struct()
        self._send = send_input or ctypes.windll.user32.SendInput
        self._warned = False

    def _event(self, key, up):
        import ctypes
        inp = self.INPUT()
        inp.type = 1
        inp.ki.wVk = 0
        inp.ki.wScan = self.SCAN[key]
        inp.ki.dwFlags = self.SCANCODE | self.EXTENDED | (self.KEYUP if up else 0)
        if self._send(1, ctypes.byref(inp), ctypes.sizeof(self.INPUT)) != 1 and not self._warned:
            self._warned = True
            print("WARNING: Windows refused the key press. If Edge is running as administrator, "
                  "run this script as administrator too.")

    def keyDown(self, key): self._event(key, False)
    def keyUp(self, key):   self._event(key, True)

    def press(self, key):
        self.keyDown(key)
        time.sleep(0.04)
        self.keyUp(key)


def foreground_title():
    if os.name != "nt":
        return None
    import ctypes
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def make_keyboard(args, pyautogui):
    if os.name == "nt" and args.backend == "auto":
        return WinKeys()
    return pyautogui


# ---------------------------------------------------------------- preview HUD
AMBER, GREEN, RED, WHITE, GREY = (77, 184, 255), (164, 227, 107), (102, 122, 255), (240, 240, 240), (150, 150, 150)

GUIDE_YOKE = [
    ("CLOSED HANDS = STEER", None, "h"),
    ("Tilt yoke", "right lower = bank right", GREEN),
    ("Raise / lower both", "up = climb, down = dive", GREEN),
    ("OPEN BOTH = RELEASE", None, "h"),
    ("Level out", "open both hands", RED),
    ("Hold open 1 s", "set new level pose", AMBER),
    ("ONE HAND OPEN = THRUST", None, "h"),
    ("Right hand open", "thrust up", GREEN),
    ("Left hand open", "thrust down", RED),
    ("KEYS", None, "h"),
    ("1 / 2 / 3", "gentle / normal / sharp", WHITE),
    ("- / =", "gain down / up", WHITE),
    ("c    g    q", "recenter / guide / quit", WHITE),
]


def _text(cv2, frame, s, org, scale=0.5, col=WHITE, thick=1):
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def draw_guide(frame, style="yoke"):
    import cv2
    rows = GUIDE_YOKE
    h, w = frame.shape[:2]
    pw = min(280, w // 2)
    x0, y0, bh = w - pw - 6, 6, h - 12
    shade = frame.copy()
    cv2.rectangle(shade, (x0, y0), (w - 6, y0 + bh), (10, 10, 10), -1)
    cv2.addWeighted(shade, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x0, y0), (w - 6, y0 + bh), AMBER, 2)
    _text(cv2, frame, "CONTROLS", (x0 + 10, y0 + 22), 0.62, AMBER, 2)
    y = y0 + 50
    for head, body, kind in rows:
        if kind == "h":
            _text(cv2, frame, head, (x0 + 10, y), 0.52, AMBER, 2)
            y += 26
        else:
            _text(cv2, frame, head, (x0 + 10, y), 0.46, WHITE, 1)
            _text(cv2, frame, body, (x0 + 10, y + 15), 0.40, kind, 1)
            y += 34


def draw_hud(frame, out, msg, marks, style, dry_run=False, info="", guide=False):
    import cv2
    h, w = frame.shape[:2]
    state = out["state"]
    flying = state == "flying"
    col = GREEN if flying else AMBER

    if guide:
        draw_guide(frame, style)
    if style == "yoke" and len(marks) >= 2:
        a, b = marks[0], marks[-1]
        cv2.line(frame, (a["px"], a["py"]), (b["px"], b["py"]), col, 3, cv2.LINE_AA)
    for m in marks:
        fist = m["gesture"] == "fist"
        cv2.circle(frame, (m["px"], m["py"]), 14 if fist else 9,
                   RED if fist else col, -1 if fist else 2, cv2.LINE_AA)
        _text(cv2, frame, f'{m["side"]} {m["gesture"]}',
              (m["px"] - 30, m["py"] - 20), 0.5, RED if fist else WHITE)

    _text(cv2, frame, msg, (10, 24), 0.6, AMBER if not flying else GREEN, 2)
    if state == "calibrating":
        cv2.rectangle(frame, (10, 34), (10 + int(180 * out["progress"]), 42), AMBER, -1)
        cv2.rectangle(frame, (10, 34), (190, 42), GREY, 1)
    if out.get("release_progress", 0) > 0:
        cv2.rectangle(frame, (10, 34), (10 + int(180 * out["release_progress"]), 42), RED, -1)
        cv2.rectangle(frame, (10, 34), (190, 42), GREY, 1)
        _text(cv2, frame, "hold OPEN hands to recenter", (200, 44), 0.45, RED)
    if out.get("panic"):
        _text(cv2, frame, "PANIC: hands together - levelling", (10, 82), 0.5, RED)
    if out.get("notice"):
        _text(cv2, frame, out["notice"], (w // 2 - 80, h // 2 - 90), 0.9, GREEN, 2)

    cx, by = w // 2, h - 46
    cv2.line(frame, (cx - 100, by), (cx + 100, by), GREY, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, by - 6), (cx, by + 6), GREY, 2)
    cv2.circle(frame, (cx + int(out["roll_in"] * 100), by), 7, col, -1, cv2.LINE_AA)
    _text(cv2, frame, f'roll {abs(out["roll"]) * 100:.0f}% of full rate',
          (cx - 62, by + 22), 0.4, GREY)
    py0 = h // 2
    cv2.line(frame, (w - 26, py0 - 70), (w - 26, py0 + 70), GREY, 2, cv2.LINE_AA)
    cv2.line(frame, (w - 32, py0), (w - 20, py0), GREY, 2)
    cv2.circle(frame, (w - 26, py0 - int(out["pitch_in"] * 70)), 7, col, -1, cv2.LINE_AA)
    _text(cv2, frame, "up", (w - 40, py0 - 80), 0.4, GREY)

    if flying:
        _text(cv2, frame,
              f"bank {out['roll_target_deg']:+.0f}->{out['roll_est_deg']:+.0f}   "
              f"pitch {out['pitch_target_deg']:+.0f}->{out['pitch_est_deg']:+.0f}",
              (10, h - 28), 0.4, GREY)
    if out["thrust"]:
        _text(cv2, frame, "THRUST +" if out["thrust"] > 0 else "THRUST -",
              (cx - 50, by - 20), 0.7, GREEN if out["thrust"] > 0 else RED, 2)
    if info:
        _text(cv2, frame, info, (10, h - 10), 0.45, GREY)
    if dry_run:
        _text(cv2, frame, "DRY RUN: no input sent", (10, 62), 0.5, RED)


# ---------------------------------------------------------------- helpers
def run_key_test(kb):
    print("KEY TEST: switch to the Google Earth flight simulator now.")
    for i in range(5, 0, -1):
        t = foreground_title()
        print(f"  starting in {i}...   focused window: {t if t is not None else '(not checked on this OS)'}")
        time.sleep(1)
    t = foreground_title()
    if t is not None and "google earth" not in t.lower():
        print(f"  WARNING: keys will go to '{t}', not Google Earth. Click the Earth page, then run again.")
    for k in ("right", "left", "up", "down"):
        print(f"  holding {k} for 1.2s")
        end = time.monotonic() + 1.2
        while time.monotonic() < end:
            kb.keyDown(k)
            time.sleep(0.03)
        kb.keyUp(k)
        time.sleep(0.8)
    for k in ("pageup", "pagedown"):
        print(f"  tapping {k} x3")
        for _ in range(3):
            kb.press(k)
            time.sleep(0.35)
    print("Done. Did the plane react? If not, click once inside the simulation "
          "(toggles mouse/keyboard steering) and run --test-keys again.")


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand model (about 8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        sys.exit(f"Could not download the hand model ({e}).\nDownload it in a browser from:\n  {MODEL_URL}\n"
                 f"and save it as:\n  {MODEL_PATH}")


def parse_args():
    p = argparse.ArgumentParser(description="Gesture control for the Google Earth flight simulator")
    p.add_argument("--style", choices=["yoke", "palm"], default="yoke")
    p.add_argument("--profile", choices=PROFILE_ORDER, default="normal",
                   help="steering profile: gentle / normal / sharp")
    p.add_argument("--sensitivity", type=float, default=1.0)
    p.add_argument("--fist-level", type=float, default=FIST_LEVEL)
    p.add_argument("--test-keys", action="store_true")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--delay", type=float, default=5.0)
    p.add_argument("--invert-pitch", action="store_true")
    p.add_argument("--invert-roll", action="store_true")
    p.add_argument("--no-thrust", action="store_true")
    p.add_argument("--no-guide", action="store_true")
    p.add_argument("--backend", choices=["auto", "pyautogui"], default="auto")
    p.add_argument("--any-window", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    global ASPECT
    args = parse_args()
    try:
        import pyautogui
    except ImportError:
        sys.exit("Missing pyautogui. Run: pip install pyautogui")
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    if args.test_keys:
        run_key_test(make_keyboard(args, pyautogui))
        return

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as e:
        sys.exit(f"Missing package ({e}). Run: pip install mediapipe opencv-python pyautogui")
    ensure_model()

    detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1. On Windows, check Settings > Privacy > "
                 "Camera, and on macOS allow your terminal under Privacy & Security > Camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win = "Hand tracking (q = quit)"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)
        cv2.moveWindow(win, 0, 0)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
    except cv2.error:
        sys.exit("OpenCV has no window support. Run: "
                 "pip uninstall opencv-python-headless && pip install opencv-python")

    sr, sp = (-1 if args.invert_roll else 1), (-1 if args.invert_pitch else 1)
    Pilot = YokePilot if args.style == "yoke" else PalmPilot
    pilot = Pilot(sr, sp, args.sensitivity, args.profile)
    live = not args.dry_run
    kb = make_keyboard(args, pyautogui)
    keys = KeyDriver(kb if live else _NullKB())
    focus_ok, focus_checked, last_thrust = True, 0.0, 0.0
    show_guide = not args.no_guide
    aspect_set = False
    start = time.monotonic()
    bad_reads = 0
    print(f"Running ({args.style}, {args.profile}). Click the Google Earth tab now. "
          f"Control starts in {args.delay:.0f}s. Ctrl+C to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                bad_reads += 1
                if bad_reads > 30:
                    sys.exit("The camera stopped returning frames. Close other apps using it and try again.")
                time.sleep(0.03)
                continue
            bad_reads = 0
            if not aspect_set:
                ASPECT = frame.shape[1] / frame.shape[0]
                aspect_set = True
            now = time.monotonic()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect_for_video(image, int((now - start) * 1000))
            hands = sorted((analyse(lm, args.fist_level) for lm in result.hand_landmarks),
                           key=lambda h: h["x"])
            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            marks = [{"px": int(h["x"] * fw), "py": int(h["y"] * fh), "gesture": h["gesture"],
                      "side": "L" if (len(hands) == 2 and i == 0) else "R" if len(hands) == 2 else "hand"}
                     for i, h in enumerate(hands)]

            if now - focus_checked > 0.25:
                focus_checked = now
                title = foreground_title()
                focus_ok = args.any_window or title is None or "google earth" in title.lower()
            armed = now - start >= args.delay
            o = pilot.update(hands, now)
            go = armed and focus_ok
            roll, pitch = (o["roll"], o["pitch"]) if go else (0.0, 0.0)
            roll_dir, pitch_dir = keys.update(roll, pitch, now)
            pilot.integrate(roll_dir, pitch_dir, now)
            thrust = o["thrust"] if (go and o["state"] == "flying") else 0
            if thrust and live and not args.no_thrust and now - last_thrust > 0.25:
                last_thrust = now
                kb.press("pageup" if thrust > 0 else "pagedown")

            if not armed:
                msg = f"Click the Earth tab: starting in {int(args.delay - (now - start)) + 1}"
            elif not focus_ok:
                msg = "Paused: click the Google Earth window"
            elif o["state"] == "searching":
                msg = "Show both hands like a yoke" if args.style == "yoke" else "Show your open hand"
            elif o["state"] == "need both hands":
                msg = "Level: bring the other hand back"
            elif o["state"] == "calibrating":
                msg = "Hold still to set level..."
            else:
                msg = "FLYING"
            shown = dict(o, roll=roll, pitch=pitch, thrust=thrust)
            draw_hud(frame, shown, msg, marks, args.style, args.dry_run,
                     f"{pilot.profile_name}  gain x{pilot.sens:.2f}", show_guide)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                pilot.recalibrate()
            elif k == ord("g"):
                show_guide = not show_guide
            elif k in (ord("1"), ord("2"), ord("3")):
                pilot.set_profile(PROFILE_ORDER[k - ord("1")])
            elif k == ord("-"):
                pilot.sens = max(0.3, pilot.sens * 0.85)
            elif k in (ord("="), ord("+")):
                pilot.sens = min(3.0, pilot.sens * 1.18)
    except KeyboardInterrupt:
        pass
    finally:
        keys.release()
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()