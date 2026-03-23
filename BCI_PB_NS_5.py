import pygame
import random
import sys
import psutil
import os
import json
import threading
import gc
import queue
import math
import array
import time
import socket
import csv
import asyncio
import websockets
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet

""" changes made from last version:
1. sequential ssvep and related setting removed
2. udp trigger added in order to embed tags into eeg file automatically
3. creates a csv file with time stamp and event labels in case udp didn't work """

# =============================================================================
# SYNCHRONIZATION CONFIGURATION
# Edit these values if your setup or analysis requirements change.
# =============================================================================

# EPOCH_INTERVAL_SECONDS — time in seconds between each TICK epoch boundary marker.
# This controls how often a TICK trigger is sent to Unicorn Recorder (via UDP)
# and written to the CSV backup. Each tick marks the start of a new 5-second
# epoch used in power spectral analysis for SSVEP classifier training.
# Change this value if your supervisor requires a different epoch length.
EPOCH_INTERVAL_SECONDS = 5

# UDP trigger numbers sent to Unicorn Recorder on 127.0.0.1:1000.
# Unicorn Recorder only accepts single ASCII characters as trigger values.
# These numbers are embedded into the EEG recording file at the exact sample
# where each event occurs, allowing offline analysis to locate game events.
UDP_TRIGGER_GAME_START = b"1"   # sent when Unity signals the game screen has started
UDP_TRIGGER_TICK       = b"2"   # sent every EPOCH_INTERVAL_SECONDS during gameplay
UDP_TRIGGER_GAME_END   = b"3"   # sent when Unity signals the game screen has ended

# UDP destination — Unicorn Recorder listens on this address and port by default.
UDP_IP   = "127.0.0.1"
UDP_PORT = 1000

# WebSocket server settings — Unity connects to this to send GAME_START / GAME_END.
WS_HOST = "localhost"
WS_PORT = 8765


# --- 1. SYSTEM OPTIMIZATION ---
def allocate_resources():
    try:
        p = psutil.Process()
        if sys.platform == 'win32': p.nice(psutil.HIGH_PRIORITY_CLASS)
        else: p.nice(-10)
    except: pass


# --- 2. AUDIO SYNTHESIS ---
def create_beep(frequency, duration_ms):
    sample_rate = 44100
    n_samples = int((duration_ms / 1000.0) * sample_rate)
    buf = array.array('h')
    amplitude = 16000
    for i in range(n_samples):
        t = float(i) / sample_rate
        buf.append(int(amplitude * math.sin(2.0 * math.pi * frequency * t)))
    return pygame.mixer.Sound(buffer=buf)


# --- 3. SAVE SYSTEM & CONSTANTS ---
SAVE_FILE = "focus_suite_save.json"

DEFAULT_COLORS = {
    "BG": (10, 12, 18), "MENU_BG": (15, 15, 25), "BOARD_BG": (12, 14, 22),
    "GRID_LINE": (35, 40, 55), "FLOOR": (40, 45, 60), "BUFFER_LINE": (255, 165, 0),
    "WARMUP": (100, 100, 120), "PIECE": (0, 255, 255), "TARGET_GHOST": (100, 100, 60),
    "FLICKER_ON": (255, 255, 255), "FLICKER_OFF": (30, 30, 40),
    "CROSS_ON": (0, 255, 100), "CROSS_OFF": (0, 150, 50),
    "BTN_NORM": (40, 50, 80), "BTN_HOVER": (70, 90, 140), "BTN_DISABLED": (25, 25, 35),
    "TEXT": (240, 240, 240), "TEXT_DIM": (100, 100, 100), "TITLE": (0, 255, 255),
    "POPUP_BG": (20, 20, 30, 230), "INPUT_TXT": (50, 255, 50)
}

def load_game():
    default_data = {
        "pinball_high_score": 0, "stats": {}, "colors": DEFAULT_COLORS.copy(),
        "global_config": {
            "photodiode_sync": False, "flicker_mode": "SYNC_RR", "RR": 60,
            "target_hz": 15, "target_hz2": 12,   # two independent SSVEP frequencies
            "custom_hz": 15, "custom_hz2": 12,
            "handedness": "RIGHT", "wall_bounce": False,
            "override_key": "SPACE", "ram_limit": 2,
            "pad_size": 90, "pad_spacing": 160, "pad_layout": "DIAMOND"
        }
    }
    if not os.path.exists(SAVE_FILE): return default_data
    try:
        with open(SAVE_FILE, 'r') as f:
            data = json.load(f)
            for k, v in default_data.items():
                if k not in data: data[k] = v
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if sub_k not in data[k]: data[k][sub_k] = sub_v
            return data
    except: return default_data

def save_game(data):
    try:
        with open(SAVE_FILE, 'w') as f: json.dump(data, f)
    except: pass

def rgb_to_hex(rgb): return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
def hex_to_rgb(hex_str):
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) == 6:
        try: return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        except: return None
    return None
def dim_color(rgb, factor=0.2): return tuple(max(0, int(c * factor)) for c in rgb)

def closest_point_on_segment(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0: return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    return x1 + t * dx, y1 + t * dy

BASE_W, BASE_H, HEADER_H = 800, 600, 60
VALID_FREQUENCIES = {
    60: [10,12,15,20,30], 75: [15,25], 120: [10,12,15,20,24,30],
    144: [12,16,18,24], 165: [11,15,33], 240: [10,12,15,20,24,30],
    360: [10,12,15,18,20,24,30], 540: [10,12,15,18,20,27,30]
}
RAM_OPTIONS = [1, 2, 3, 4]


# ==========================================
# BASE STATE ARCHITECTURE
# ==========================================
class GameState:
    def __init__(self, engine):
        self.engine = engine
        self.font_title = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_btn   = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_ui    = pygame.font.SysFont("Arial", 20, bold=True)
        self.cfg    = self.engine.game_data["global_config"]
        self.colors = self.engine.game_data["colors"]
        self._is_background = False   # set True by engine when drawing as a backdrop

    def handle_events(self, events): pass
    def update(self, dt): pass
    def draw(self, surface): pass

    def get_logical_mouse_pos(self):
        mx, my = pygame.mouse.get_pos()
        win_w, win_h = self.engine.display.get_size()
        scale = min(win_w / BASE_W, win_h / BASE_H)
        off_x, off_y = (win_w - (BASE_W * scale)) // 2, (win_h - (BASE_H * scale)) // 2
        return (mx - off_x) / scale, (my - off_y) / scale

    def draw_button(self, surface, text, y, action, w=480, h=40,
                    color_override=None, disabled=False, draw_only=False):
        mx, my = self.get_logical_mouse_pos()
        rect = pygame.Rect((BASE_W - w) // 2, y, w, h)
        inactive = draw_only or self._is_background
        hover = rect.collidepoint((mx, my)) and not disabled and not inactive

        bg_color   = self.colors["BTN_DISABLED"] if disabled else (self.colors["BTN_HOVER"] if hover else self.colors["BTN_NORM"])
        text_color = self.colors["TEXT_DIM"] if disabled else (color_override if color_override else self.colors["TEXT"])

        pygame.draw.rect(surface, bg_color,   rect, border_radius=8)
        pygame.draw.rect(surface, text_color, rect, 2, border_radius=8)

        txt = self.font_btn.render(text, True, text_color)
        surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        if hover and self.engine.mouse_clicked and self.engine.click_cooldown == 0:
            self.engine.mouse_clicked = False
            self.engine.click_cooldown = 8
            action()

    def draw_ssvep_pad(self, surface, cx, cy, pad_sz, is_on, label, cross_color_on, cross_color_off):
        """
        Draw an SSVEP stimulus pad with a central crosshair focal point.
        The crosshair gives the user a stable fixation target, reducing
        the attentional smearing caused by visual crowding.
        """
        r = pygame.Rect(0, 0, pad_sz, pad_sz)
        r.center = (cx, cy)
        bg = self.colors["FLICKER_ON"] if is_on else self.colors["FLICKER_OFF"]
        pygame.draw.rect(surface, bg, r, border_radius=10)
        pygame.draw.rect(surface, self.colors["TEXT_DIM"], r, 2, border_radius=10)

        font_small = pygame.font.SysFont("Arial", 10, bold=True)
        lbl = font_small.render(label, True, (0, 0, 0) if is_on else self.colors["TEXT"])
        surface.blit(lbl, (r.centerx - lbl.get_width() // 2, r.bottom + 8))

        cross_col = cross_color_on if is_on else cross_color_off
        arm = pad_sz // 5
        thickness = 3
        pygame.draw.line(surface, cross_col, (cx - arm, cy), (cx + arm, cy), thickness)
        pygame.draw.line(surface, cross_col, (cx, cy - arm), (cx, cy + arm), thickness)
        ball_r = max(3, pad_sz // 20)
        pygame.draw.circle(surface, bg, (cx, cy), ball_r + 1)
        pygame.draw.circle(surface, cross_col, (cx, cy), ball_r)


# ==========================================
# SETTINGS & MENUS
# ==========================================
class SettingsMenuState(GameState):
    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render("GLOBAL SETTINGS", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 60))

        self.draw_button(surface, " THEME & COLORS",      180, lambda: self.engine.change_state(ThemeSettingsState(self.engine)))
        self.draw_button(surface, " HARDWARE & DISPLAY",  250, lambda: self.engine.change_state(HardwareSettingsState(self.engine)))
        self.draw_button(surface, " GAMEPLAY & CONTROLS", 320, lambda: self.engine.change_state(GameplaySettingsState(self.engine)))
        self.draw_button(surface, "RESTORE ALL DEFAULTS", 420, self.engine.restore_defaults, color_override=(255, 100, 100))
        self.draw_button(surface, "RETURN",               490, self.engine.pop_state)


class HardwareSettingsState(GameState):
    def toggle_flicker_mode(self):
        self.cfg["flicker_mode"] = "CUSTOM_ASYNC" if self.cfg["flicker_mode"] == "SYNC_RR" else "SYNC_RR"
        if self.cfg["flicker_mode"] == "SYNC_RR":
            vl = VALID_FREQUENCIES[self.cfg["RR"]]
            if self.cfg["target_hz"]  not in vl: self.cfg["target_hz"]  = vl[0]
            if self.cfg["target_hz2"] not in vl: self.cfg["target_hz2"] = vl[1] if len(vl) > 1 else vl[0]

    def cycle_rr(self):
        rrs = list(VALID_FREQUENCIES.keys())
        self.cfg["RR"] = rrs[(rrs.index(self.cfg["RR"]) + 1) % len(rrs)]
        vl = VALID_FREQUENCIES[self.cfg["RR"]]
        if self.cfg["target_hz"]  not in vl: self.cfg["target_hz"]  = vl[0]
        if self.cfg["target_hz2"] not in vl: self.cfg["target_hz2"] = vl[1] if len(vl) > 1 else vl[0]

    def cycle_hz(self, key):
        vl  = VALID_FREQUENCIES[self.cfg["RR"]]
        cur = self.cfg[key]
        self.cfg[key] = vl[(vl.index(cur) + 1) % len(vl) if cur in vl else 0]

    def cycle_ram(self):
        self.cfg["ram_limit"] = RAM_OPTIONS[(RAM_OPTIONS.index(self.cfg["ram_limit"]) + 1) % len(RAM_OPTIONS)]
        self.engine.apply_ram_allocation()

    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render("HARDWARE & DISPLAY", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 30))

        pd_txt = "ON" if self.cfg["photodiode_sync"] else "OFF"
        self.draw_button(surface, f"PHOTODIODE FLASHER: {pd_txt}", 110,
                         lambda: self.cfg.update({"photodiode_sync": not self.cfg["photodiode_sync"]}))
        self.draw_button(surface, f"FLICKER MODE: {self.cfg['flicker_mode']}", 160, self.toggle_flicker_mode)

        is_custom = self.cfg["flicker_mode"] == "CUSTOM_ASYNC"
        self.draw_button(surface, f"SCREEN REFRESH: {self.cfg['RR']} Hz", 210, self.cycle_rr, disabled=is_custom)

        if is_custom:
            self.draw_button(surface, f"PAD 1 (ANGLE) CUSTOM HZ: {self.cfg['custom_hz']} (CLICK)", 260,
                             lambda: self.engine.push_state(TextInputState(self.engine, "custom_hz", "ENTER PAD 1 HZ:", True)))
            self.draw_button(surface, f"PAD 2 (SHOOT) CUSTOM HZ: {self.cfg['custom_hz2']} (CLICK)", 310,
                             lambda: self.engine.push_state(TextInputState(self.engine, "custom_hz2", "ENTER PAD 2 HZ:", True)))
        else:
            self.draw_button(surface, f"PAD 1 (ANGLE) FLASH HZ: {self.cfg['target_hz']}", 260,
                             lambda: self.cycle_hz("target_hz"))
            self.draw_button(surface, f"PAD 2 (SHOOT) FLASH HZ: {self.cfg['target_hz2']}", 310,
                             lambda: self.cycle_hz("target_hz2"))

        hint = pygame.font.SysFont("Arial", 15).render(
            "  PAD 1 and PAD 2 should have DIFFERENT frequencies for BCI classification accuracy.",
            True, (180, 180, 100))
        surface.blit(hint, (BASE_W // 2 - hint.get_width() // 2, 355))

        self.draw_button(surface, f"RAM ALLOCATION: {self.cfg['ram_limit']} GB", 390, self.cycle_ram)
        self.draw_button(surface, "BACK TO SETTINGS", 460, lambda: self.engine.change_state(SettingsMenuState(self.engine)))


class GameplaySettingsState(GameState):
    def cycle_list(self, key, options):
        self.cfg[key] = options[(options.index(self.cfg[key]) + 1) % len(options)]

    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render("GAMEPLAY & CONTROLS", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 30))

        # Sequential SSVEP and its settings have been removed.
        # Both SSVEP pads now always flicker simultaneously (STATIC / simultaneous mode).
        self.draw_button(surface, f"HANDEDNESS: {self.cfg['handedness']}", 90,
                         lambda: self.cycle_list("handedness", ["RIGHT", "LEFT", "SPLIT"]))
        self.draw_button(surface, f"KEYBOARD OVERRIDE: {self.cfg['override_key']} (CLICK)", 150,
                         lambda: self.engine.push_state(TextInputState(
                             self.engine, "override_key", "ENTER OVERRIDE KEY (e.g. SPACE, WASD):", False)))
        self.draw_button(surface, f"PAD SIZE: {self.cfg['pad_size']} PX", 210,
                         lambda: self.engine.push_state(TextInputState(
                             self.engine, "pad_size", "ENTER PAD SIZE (PX):", True)))
        self.draw_button(surface, f"PAD SPACING: {self.cfg['pad_spacing']} PX", 270,
                         lambda: self.engine.push_state(TextInputState(
                             self.engine, "pad_spacing", "ENTER SPREAD (PX):", True)))
        self.draw_button(surface, "BACK TO SETTINGS", 350,
                         lambda: self.engine.change_state(SettingsMenuState(self.engine)))


class ThemeSettingsState(GameState):
    def draw(self, surface):
        opts = [("BACKGROUND", "BG"), ("TEXT", "TEXT"), ("FALLING ENTITY", "PIECE"),
                ("TARGET AREA", "TARGET_GHOST"), ("CROSSHAIR", "CROSS_ON")]
        for i, (label, key) in enumerate(opts):
            self.draw_button(surface, f"{label}: {rgb_to_hex(self.colors[key])}", 110 + i * 60,
                             lambda k=key: self.engine.push_state(
                                 TextInputState(self.engine, k, "ENTER HEX CODE (e.g. #FF00FF):", False, is_color=True)))
        self.draw_button(surface, "RESTORE DEFAULT COLORS", 430, self.engine.restore_colors, color_override=(255, 100, 100))
        self.draw_button(surface, "BACK TO SETTINGS",       500, lambda: self.engine.change_state(SettingsMenuState(self.engine)))


class TextInputState(GameState):
    def __init__(self, engine, target_key, prompt, numbers_only, is_color=False, on_confirm=None):
        super().__init__(engine)
        self.target_key  = target_key
        self.prompt      = prompt
        self.numbers_only = numbers_only
        self.is_color    = is_color
        self.on_confirm  = on_confirm
        if self.is_color: self.input_text = rgb_to_hex(self.colors[self.target_key])
        else: self.input_text = str(self.cfg[self.target_key])
        self.overlay = pygame.Surface((BASE_W, BASE_H), pygame.SRCALPHA)
        self.overlay.fill(self.colors["POPUP_BG"])

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if self.is_color:
                        new_rgb = hex_to_rgb(self.input_text)
                        if new_rgb:
                            self.colors[self.target_key] = new_rgb
                            if self.target_key == "CROSS_ON": self.colors["CROSS_OFF"] = dim_color(new_rgb, 0.5)
                    elif self.numbers_only and self.input_text.isdigit():
                        self.cfg[self.target_key] = int(self.input_text)
                    elif not self.numbers_only:
                        self.cfg[self.target_key] = self.input_text.upper()
                    if self.on_confirm: self.on_confirm()
                    self.engine.pop_state()
                elif event.key == pygame.K_BACKSPACE: self.input_text = self.input_text[:-1]
                elif event.key == pygame.K_ESCAPE: self.engine.pop_state()
                else:
                    char = event.unicode
                    if self.is_color and char.upper() in "0123456789ABCDEF#":
                        if len(self.input_text) < 7: self.input_text += char.upper()
                    elif self.numbers_only and char.isdigit():
                        self.input_text += char
                    elif not self.numbers_only and not self.is_color:
                        if len(self.input_text) < 10: self.input_text += char.upper()

    def draw(self, surface):
        surface.blit(self.overlay, (0, 0))
        p_txt = self.font_btn.render(self.prompt, True, self.colors["TEXT"])
        surface.blit(p_txt, (BASE_W // 2 - p_txt.get_width() // 2, 220))
        box = pygame.Rect(BASE_W // 2 - 150, 270, 300, 50)
        pygame.draw.rect(surface, self.colors["BTN_NORM"], box)
        pygame.draw.rect(surface, self.colors["INPUT_TXT"], box, 2)
        i_txt = self.font_title.render(self.input_text + "_", True, self.colors["INPUT_TXT"])
        surface.blit(i_txt, (box.centerx - i_txt.get_width() // 2, box.centery - i_txt.get_height() // 2))


# ==========================================
# SCOREBOARD & STATS
# ==========================================
class ScoreBoardState(GameState):
    def __init__(self, engine):
        super().__init__(engine)
        self.game_mode = "PINBALL"
        self.load_scores()

    def load_scores(self):
        self.high_scores = []
        stats = self.engine.game_data.get("stats", {})
        for uid, records in stats.items():
            scores = [r["score"] for r in records if r.get("game") == self.game_mode]
            if scores: self.high_scores.append((uid, max(scores)))
        self.high_scores.sort(key=lambda x: x[1], reverse=True)
        self.high_scores = self.high_scores[:5]

    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render("TOP 5 SCORES", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 50))
        y_offset = 150
        for i, (uid, score) in enumerate(self.high_scores):
            txt = self.font_title.render(f"{i+1}. {uid}   -   {score} PTS", True, self.colors["TEXT"])
            surface.blit(txt, (BASE_W // 2 - txt.get_width() // 2, y_offset))
            y_offset += 55
        if not self.high_scores:
            txt = self.font_ui.render("NO SCORES RECORDED YET.", True, self.colors["TEXT_DIM"])
            surface.blit(txt, (BASE_W // 2 - txt.get_width() // 2, 300))
        self.draw_button(surface, "BACK TO MAIN MENU", 500, lambda: self.engine.change_state(MainMenuState(self.engine)))


class StatsMenuState(GameState):
    def __init__(self, engine):
        super().__init__(engine)
        self.users = list(self.engine.game_data.get("stats", {}).keys())

    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render("SELECT USER FOR STATS", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 40))
        y_offset = 130
        for uid in self.users[:5]:
            self.draw_button(surface, f"VIEW PROGRESS: {uid}", y_offset,
                             lambda u=uid: self.engine.change_state(GraphState(self.engine, u)))
            y_offset += 60
        if not self.users:
            txt = self.font_ui.render("NO USERS FOUND.", True, self.colors["TEXT_DIM"])
            surface.blit(txt, (BASE_W // 2 - txt.get_width() // 2, 250))
        self.draw_button(surface, "BACK TO MAIN MENU", 500, lambda: self.engine.change_state(MainMenuState(self.engine)))


class GraphState(GameState):
    def __init__(self, engine, user_id):
        super().__init__(engine)
        self.user_id = user_id
        self.scores  = [r["score"] for r in self.engine.game_data.get("stats", {}).get(self.user_id, [])
                        if r.get("game") == "PINBALL"]

    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render(f"PROGRESS: {self.user_id}", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 30))
        g_rect = pygame.Rect(100, 160, 600, 280)
        pygame.draw.rect(surface, self.colors["BOARD_BG"], g_rect)
        pygame.draw.rect(surface, self.colors["GRID_LINE"], g_rect, 2)
        if len(self.scores) > 1:
            max_score = max(self.scores) or 1
            points = []
            for i, score in enumerate(self.scores):
                x = g_rect.left + i * (g_rect.width / (len(self.scores) - 1))
                y = g_rect.bottom - (score / max_score) * g_rect.height
                points.append((x, y))
                pygame.draw.circle(surface, self.colors["PIECE"], (int(x), int(y)), 6)
            pygame.draw.lines(surface, self.colors["PIECE"], False, points, 3)
            min_lbl = self.font_ui.render("0",          True, self.colors["TEXT_DIM"])
            max_lbl = self.font_ui.render(str(max_score), True, self.colors["TEXT_DIM"])
            surface.blit(min_lbl, (g_rect.left - 25, g_rect.bottom - 10))
            surface.blit(max_lbl, (g_rect.left - 45, g_rect.top - 5))
        elif len(self.scores) == 1:
            txt = self.font_ui.render(f"1 SESSION: {self.scores[0]} PTS.", True, self.colors["TEXT"])
            surface.blit(txt, (BASE_W // 2 - txt.get_width() // 2, g_rect.centery))
        else:
            txt = self.font_ui.render("NO DATA FOR THIS GAME.", True, self.colors["TEXT"])
            surface.blit(txt, (BASE_W // 2 - txt.get_width() // 2, g_rect.centery))
        self.draw_button(surface, "BACK TO USERS", 500, lambda: self.engine.change_state(StatsMenuState(self.engine)))


# ==========================================
# MAIN MENU & LOADING
# ==========================================
class MainMenuState(GameState):
    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render("BCI PINBALL STATION", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 50))
        self.draw_button(surface, " PLAY PINBALL",  150, lambda: self.engine.change_state(LoadingState(self.engine)))
        self.draw_button(surface, " SETTINGS",      220, lambda: self.engine.push_state(SettingsMenuState(self.engine)))
        self.draw_button(surface, " SCORE BOARD",   290, lambda: self.engine.change_state(ScoreBoardState(self.engine)))
        self.draw_button(surface, " STATS",         360, lambda: self.engine.change_state(StatsMenuState(self.engine)))
        self.draw_button(surface, " EXIT GAME",     430, self.engine.quit)


class LoadingState(GameState):
    def __init__(self, engine):
        super().__init__(engine)
        self.control_mode    = "BCI SIGNAL"
        self.buffer_active   = False
        self.buffer_start_time = 0

    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        if not self.buffer_active:
            title = self.font_title.render("LOADING: PINBALL", True, self.colors["TITLE"])
            surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 60))
            info  = self.font_ui.render(f"Override Keys: [{self.cfg['override_key']}] and [ENTER/ARROWS]", True, self.colors["TEXT"])
            info2 = self.font_ui.render("Left pad = Angle (15 deg). Right pad = Shoot.", True, self.colors["TEXT_DIM"])
            surface.blit(info,  (BASE_W // 2 - info.get_width()  // 2, 140))
            surface.blit(info2, (BASE_W // 2 - info2.get_width() // 2, 170))
            self.draw_button(surface, f"INPUT MODE: {self.control_mode}", 250,
                             lambda: setattr(self, 'control_mode', "KEYBOARD" if self.control_mode == "BCI SIGNAL" else "BCI SIGNAL"),
                             color_override=(0, 255, 100))
            self.draw_button(surface, "START SEQUENCE", 320,
                             lambda: [setattr(self, 'buffer_active', True), setattr(self, 'buffer_start_time', time.time())])
            self.draw_button(surface, "CANCEL", 460, lambda: self.engine.change_state(MainMenuState(self.engine)))
        else:
            elapsed   = time.time() - self.buffer_start_time
            remaining = max(0, 3 - int(elapsed))
            txt   = self.font_title.render("GET READY...", True, (255, 165, 0))
            count = pygame.font.SysFont("Arial", 80, bold=True).render(str(remaining), True, self.colors["TEXT"])
            surface.blit(txt,   (BASE_W // 2 - txt.get_width()   // 2, 200))
            surface.blit(count, (BASE_W // 2 - count.get_width() // 2, 300))
            if elapsed >= 3:
                self.engine.game_data["current_input_mode"] = self.control_mode
                self.engine.change_state(PinballGameState(self.engine))


class SaveScoreState(GameState):
    def __init__(self, engine, score):
        super().__init__(engine)
        self.score    = score
        self.user_id  = ""
        self.error_msg = ""

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE: self.user_id = self.user_id[:-1]
                elif event.key == pygame.K_RETURN:
                    if len(self.user_id) == 5:
                        if self.user_id not in self.engine.game_data["stats"]:
                            self.engine.game_data["stats"][self.user_id] = []
                        self.engine.game_data["stats"][self.user_id].append({"game": "PINBALL", "score": self.score})
                        self.engine.change_state(MainMenuState(self.engine))
                    else: self.error_msg = "MUST BE 2 LETTERS AND 2 NUMBERS (e.g. AB-12)"
                elif len(self.user_id) < 5:
                    char = event.unicode.upper()
                    if len(self.user_id) < 2 and char.isalpha(): self.user_id += char
                    elif len(self.user_id) == 2:
                        self.user_id += "-"
                        if char.isdigit(): self.user_id += char
                    elif len(self.user_id) >= 3 and char.isdigit(): self.user_id += char

    def draw(self, surface):
        surface.fill(self.colors["MENU_BG"])
        title = self.font_title.render("SESSION COMPLETE", True, self.colors["TITLE"])
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 80))
        s_txt = self.font_ui.render(f"FINAL SCORE: {self.score}", True, (50, 255, 50))
        surface.blit(s_txt, (BASE_W // 2 - s_txt.get_width() // 2, 160))
        inst = self.font_ui.render("ENTER ID (2 Letters, 2 Numbers) TO SAVE:", True, self.colors["TEXT"])
        surface.blit(inst, (BASE_W // 2 - inst.get_width() // 2, 230))
        box = pygame.Rect(BASE_W // 2 - 100, 280, 200, 50)
        pygame.draw.rect(surface, self.colors["BTN_NORM"], box)
        pygame.draw.rect(surface, self.colors["TEXT"], box, 2)
        id_txt = self.font_title.render(self.user_id, True, self.colors["TITLE"])
        surface.blit(id_txt, (box.centerx - id_txt.get_width() // 2, box.centery - id_txt.get_height() // 2))
        if self.error_msg:
            err = self.font_ui.render(self.error_msg, True, (255, 50, 50))
            surface.blit(err, (BASE_W // 2 - err.get_width() // 2, 350))


class PinballPauseState(GameState):
    def __init__(self, engine, pinball_state):
        super().__init__(engine)
        self.pinball_state = pinball_state
        self.overlay = pygame.Surface((BASE_W, BASE_H), pygame.SRCALPHA)
        self.overlay.fill(self.colors["POPUP_BG"])

    def draw(self, surface):
        surface.blit(self.overlay, (0, 0))
        title = self.font_title.render("PINBALL PAUSED", True, (255, 165, 0))
        surface.blit(title, (BASE_W // 2 - title.get_width() // 2, 100))
        self.draw_button(surface, "RESUME GAME",              250, self.engine.pop_state)
        self.draw_button(surface, "GLOBAL SETTINGS",          320, lambda: self.engine.push_state(SettingsMenuState(self.engine)))
        self.draw_button(surface, "END SESSION & SAVE SCORE", 390,
                         lambda: self.engine.change_state(SaveScoreState(self.engine, self.pinball_state.score)),
                         color_override=(255, 100, 100))


# ==========================================
# PINBALL GAME ENGINE
# ==========================================
class PinballGameState(GameState):
    def __init__(self, engine):
        super().__init__(engine)
        self.score  = 0
        self.level  = 1
        self.points_this_level = 0
        self.balls_left = 10

        self.state          = "AIMING"
        self.angle          = 90
        self.idle_timer     = 0
        self.victory_timer  = 0
        self.current_multiplier = 1

        self.update_bounds()
        self.ball_x, self.ball_y = self.play_cx, HEADER_H + 30
        self.vx, self.vy = 0, 0
        self.ball_r = 8
        self.generate_board()

        self.feedback_frames = 0
        self.feedback_color  = None

        # Sequential SSVEP, dwell counting, and action gating have been removed.
        # Both pads flicker simultaneously at their own frequencies (STATIC mode).
        # BCI commands from the classifier are passed through directly to game actions
        # without any confirmation window or gating logic.

        self.victory_balls = []

    def get_slot_scores(self):
        tier = min((self.level - 1) // 30, 3)
        mult = 2 ** tier
        base = [0, 10, 20, 40, 50, 40, 20, 10, 0]
        return [b * mult for b in base]

    def get_level_target(self):
        sub_tier = ((self.level - 1) % 30) // 10
        return 100 + (sub_tier * 100)

    def update_bounds(self):
        if self.cfg["handedness"] == "SPLIT":
            self.play_area_left  = BASE_W // 6
            self.play_area_right = BASE_W - (BASE_W // 6)
        elif self.cfg["handedness"] == "LEFT":
            self.play_area_left  = 160
            self.play_area_right = BASE_W - 20
        else:
            self.play_area_left  = 20
            self.play_area_right = BASE_W - 160
        self.play_cx = self.play_area_left + (self.play_area_right - self.play_area_left) / 2

    def spawn_orbs(self):
        self.orbs = []
        for _ in range(random.randint(2, 4)):
            x = random.randint(int(self.play_area_left + 60), int(self.play_area_right - 60))
            y = random.randint(HEADER_H + 200, BASE_H - 200)
            self.orbs.append([x, y, 14, random.choice(["MULTI", "BALL", "XP"])])

    def generate_board(self):
        self.pegs    = []
        self.walls   = []
        self.pushers = []

        self.pegs.append([int(self.play_cx),      HEADER_H + 120, 14])
        self.pegs.append([int(self.play_cx) - 40, HEADER_H + 180, 10])
        self.pegs.append([int(self.play_cx) + 40, HEADER_H + 180, 10])

        for _ in range(max(5, 15 - (self.level // 10))):
            x = random.randint(int(self.play_area_left + 20), int(self.play_area_right - 20))
            y = random.randint(HEADER_H + 150, BASE_H - 150)
            self.pegs.append([x, y, random.randint(8, 14)])

        if self.level >= 10:
            w_h = HEADER_H + random.randint(200, 300)
            self.walls.append([(self.play_area_left,  w_h), (self.play_area_left  + 100, w_h + 100)])
            self.walls.append([(self.play_area_right, w_h), (self.play_area_right - 100, w_h + 100)])
            if self.level >= 40:
                self.walls.append([(self.play_cx - 60, BASE_H - 200), (self.play_cx + 60, BASE_H - 200)])

        if self.level >= 20:
            for _ in range(min(5, self.level // 10)):
                x = random.randint(int(self.play_area_left + 80), int(self.play_area_right - 80))
                y = random.randint(HEADER_H + 200, BASE_H - 200)
                self.pushers.append([x, y, 16])

        self.spawn_orbs()

    def execute_angle(self):
        if self.state == "AIMING":
            self.angle += 15
            if self.angle > 150: self.angle = 30
            self.engine.send_marker("ANGLE_CHANGED")

    def execute_shoot(self):
        if self.state == "AIMING" and self.balls_left > 0:
            self.state      = "SHOOTING"
            self.balls_left -= 1
            self.idle_timer = 0
            self.current_multiplier = 1
            speed    = 400
            self.vx  = speed * math.cos(math.radians(self.angle))
            self.vy  = speed * math.sin(math.radians(self.angle))
            self.engine.send_marker("BALL_SHOT")

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.engine.push_state(PinballPauseState(self.engine, self))
                elif self.feedback_frames == 0 and self.engine.game_data.get("current_input_mode") == "KEYBOARD":
                    if event.key == pygame.K_SPACE:  self.execute_angle()
                    elif event.key == pygame.K_RETURN: self.execute_shoot()

    def apply_points(self, pts):
        final_pts = pts * self.current_multiplier
        self.score             += final_pts
        self.points_this_level += final_pts
        target = self.get_level_target()
        if self.points_this_level >= target:
            self.level             += 1
            self.points_this_level -= target
            self.balls_left        += 10 + ((self.level // 10) * 5)
            self.generate_board()
            self.engine.snd_perfect.play()
        self.engine.game_data["pinball_high_score"] = max(self.score, self.engine.game_data["pinball_high_score"])

    def handle_scoring(self, idx, forced_miss=False):
        if forced_miss:
            self.feedback_color = (255, 50, 50)
            self.engine.snd_miss.play()
        else:
            slots = self.get_slot_scores()
            pts   = slots[idx]
            if pts > 0:
                self.apply_points(pts)
                self.feedback_color = (50, 255, 50) if pts == max(slots) else (255, 165, 0)
                self.engine.snd_partial.play()
            else:
                self.feedback_color = (255, 50, 50)
                self.engine.snd_miss.play()
        hz_to_use = self.cfg["RR"] if self.cfg["flicker_mode"] == "SYNC_RR" else 60
        self.feedback_frames = int(hz_to_use * 1.0)

    def update(self, dt):
        self.update_bounds()

        if self.state == "VICTORY":
            self.victory_timer += dt
            if self.victory_timer > 10000:
                self.engine.change_state(SaveScoreState(self.engine, self.score))
                return
            for _ in range(3):
                self.victory_balls.append([
                    self.play_cx, HEADER_H,
                    random.uniform(-300, 300), random.uniform(-100, 200)
                ])
            for b in self.victory_balls:
                b[0] += b[2] * (dt / 1000.0)
                b[1] += b[3] * (dt / 1000.0)
                b[3] += 500  * (dt / 1000.0)
            return

        if self.feedback_frames > 0:
            self.feedback_frames -= 1
            if self.feedback_frames == 0:
                if self.level >= 100:
                    self.state = "VICTORY"
                    self.engine.snd_perfect.play()
                    return
                if self.balls_left <= 0:
                    self.engine.change_state(SaveScoreState(self.engine, self.score))
                self.state = "AIMING"
                self.ball_x, self.ball_y = self.play_cx, HEADER_H + 30
            return

        # --- BCI command handling (direct passthrough, no gating) ---
        # In STATIC simultaneous SSVEP both pads flicker at all times.
        # The classifier identifies which frequency is dominant and sends
        # either "ANGLE" or "SHOOT" directly. No dwell count or confirmation
        # window is needed — each command fires immediately on arrival.
        try:
            while True:
                cmd = self.engine.bci_queue.get_nowait()
                if self.engine.game_data.get("current_input_mode") == "BCI SIGNAL":
                    if cmd == "ANGLE": self.execute_angle()
                    elif cmd == "SHOOT": self.execute_shoot()
        except queue.Empty:
            pass

        if self.state == "SHOOTING":
            dt_sec = dt / 1000.0

            if abs(self.vx) < 5 and abs(self.vy) < 5: self.idle_timer += dt
            else: self.idle_timer = 0
            if self.idle_timer > 10000:
                self.handle_scoring(0, forced_miss=True)
                return

            for o in self.orbs[:]:
                dist = math.hypot(self.ball_x - o[0], self.ball_y - o[1])
                if dist < self.ball_r + o[2]:
                    if o in self.orbs: self.orbs.remove(o)
                    self.engine.snd_perfect.play()
                    if   o[3] == "MULTI": self.current_multiplier *= 2
                    elif o[3] == "BALL":  self.balls_left += 1
                    elif o[3] == "XP":    self.apply_points(100)

            self.vy     += 600 * dt_sec
            self.ball_x += self.vx * dt_sec
            self.ball_y += self.vy * dt_sec

            if self.ball_x - self.ball_r < self.play_area_left:
                self.ball_x = self.play_area_left  + self.ball_r; self.vx *= -0.8
            elif self.ball_x + self.ball_r > self.play_area_right:
                self.ball_x = self.play_area_right - self.ball_r; self.vx *= -0.8

            for p in self.pegs:
                dx, dy = self.ball_x - p[0], self.ball_y - p[1]
                dist   = math.hypot(dx, dy)
                if dist < self.ball_r + p[2]:
                    if dist == 0: dist = 1; nx, ny = 0, -1
                    else: nx, ny = dx / dist, dy / dist
                    dot = self.vx * nx + self.vy * ny
                    self.vx = (self.vx - 2 * dot * nx) * 0.7
                    self.vy = (self.vy - 2 * dot * ny) * 0.7
                    self.ball_x += nx * ((self.ball_r + p[2]) - dist)
                    self.ball_y += ny * ((self.ball_r + p[2]) - dist)

            for w in self.walls:
                x1, y1, x2, y2 = w[0][0], w[0][1], w[1][0], w[1][1]
                px, py = closest_point_on_segment(self.ball_x, self.ball_y, x1, y1, x2, y2)
                dist   = math.hypot(self.ball_x - px, self.ball_y - py)
                if dist < self.ball_r:
                    if dist == 0: dist = 1; nx, ny = 0, -1
                    else: nx, ny = (self.ball_x - px) / dist, (self.ball_y - py) / dist
                    self.ball_x += nx * (self.ball_r - dist)
                    self.ball_y += ny * (self.ball_r - dist)
                    dot = self.vx * nx + self.vy * ny
                    self.vx = (self.vx - 2 * dot * nx) * 0.7
                    self.vy = (self.vy - 2 * dot * ny) * 0.7

            for p in self.pushers:
                dx, dy = self.ball_x - p[0], self.ball_y - p[1]
                dist   = math.hypot(dx, dy)
                if dist < self.ball_r + p[2]:
                    if dist == 0: dist = 1; nx, ny = 0, -1
                    else: nx, ny = dx / dist, dy / dist
                    self.ball_x += nx * ((self.ball_r + p[2]) - dist)
                    self.ball_y += ny * ((self.ball_r + p[2]) - dist)
                    self.vx = nx * 500
                    self.vy = ny * 500
                    self.engine.snd_partial.play()

            if self.ball_y > BASE_H - 40:
                slot_w = (self.play_area_right - self.play_area_left) / 9
                idx    = int((self.ball_x - self.play_area_left) / slot_w)
                idx    = max(0, min(8, idx))
                self.handle_scoring(idx)

    def get_single_flicker_state(self, hz):
        """Compute on/off state for a given frequency, respecting flicker mode."""
        if self.cfg["flicker_mode"] == "SYNC_RR":
            frms = max(2, round(self.cfg["RR"] / hz))
            return (self.engine.frame_count % frms) < (frms // 2)
        else:
            p_ms = 1000.0 / hz
            return (pygame.time.get_ticks() % p_ms) < (p_ms / 2)

    def _resolve_hz(self, pad_index):
        """
        Return the correct flash frequency for pad 0 (ANGLE) or pad 1 (SHOOT).
        CUSTOM_ASYNC mode uses custom_hz / custom_hz2.
        SYNC_RR mode uses target_hz / target_hz2.
        """
        if self.cfg["flicker_mode"] == "CUSTOM_ASYNC":
            return self.cfg["custom_hz"] if pad_index == 0 else self.cfg["custom_hz2"]
        else:
            return self.cfg["target_hz"] if pad_index == 0 else self.cfg["target_hz2"]

    def get_flicker_states(self):
        """
        Returns (on1, on2) for ANGLE pad and SHOOT pad respectively.
        Both pads always flicker simultaneously at their own independent frequencies.
        This is the correct simultaneous dual-frequency SSVEP paradigm.
        Sequential mode has been removed.
        """
        hz1 = max(1, self._resolve_hz(0))
        hz2 = max(1, self._resolve_hz(1))
        return self.get_single_flicker_state(hz1), self.get_single_flicker_state(hz2)

    def draw(self, surface):
        surface.fill(self.colors["BG"])

        score_txt  = self.font_ui.render(f"SCORE: {self.score} | Lvl {self.level}/100 | BALLS: {self.balls_left}", True, self.colors["TEXT"])
        target_txt = self.font_ui.render(f"NEXT LVL: {self.points_this_level}/{self.get_level_target()}", True, self.colors["TEXT_DIM"])
        surface.blit(score_txt,  (20, 10))
        surface.blit(target_txt, (20, 35))

        if self.current_multiplier > 1:
            m_txt = self.font_title.render(f"x{self.current_multiplier}", True, (50, 150, 255))
            surface.blit(m_txt, (BASE_W // 2 - m_txt.get_width() // 2, 10))

        pygame.draw.line(surface, self.colors["GRID_LINE"], (self.play_area_left,  HEADER_H), (self.play_area_left,  BASE_H - 40), 3)
        pygame.draw.line(surface, self.colors["GRID_LINE"], (self.play_area_right, HEADER_H), (self.play_area_right, BASE_H - 40), 3)

        slot_w = (self.play_area_right - self.play_area_left) / 9
        slots  = self.get_slot_scores()
        for i, pts in enumerate(slots):
            x    = self.play_area_left + i * slot_w
            rect = pygame.Rect(x, BASE_H - 40, slot_w, 40)
            pygame.draw.rect(surface, self.colors["BOARD_BG"], rect)
            pygame.draw.rect(surface, self.colors["GRID_LINE"], rect, 2)
            c   = (50,255,50) if pts == max(slots) else ((255,165,0) if pts > 0 else (255,50,50))
            txt = pygame.font.SysFont("Arial", 14, bold=True).render(str(pts), True, c)
            surface.blit(txt, (x + (slot_w - txt.get_width()) // 2, BASE_H - 30))

        for p in self.pegs:
            pygame.draw.circle(surface, self.colors["TARGET_GHOST"], (p[0], p[1]), p[2])
            pygame.draw.circle(surface, self.colors["TEXT_DIM"],     (p[0], p[1]), p[2], 2)
        for w in self.walls:
            pygame.draw.line(surface, (200, 200, 255), w[0], w[1], 4)
        for p in self.pushers:
            pygame.draw.circle(surface, (200, 50, 50),   (p[0], p[1]), p[2])
            pygame.draw.circle(surface, (255, 255, 255), (p[0], p[1]), p[2], 2)
        for o in self.orbs:
            c = (50,150,255) if o[3]=="MULTI" else ((50,255,50) if o[3]=="BALL" else (255,215,0))
            pygame.draw.circle(surface, c,             (o[0], o[1]), o[2])
            pygame.draw.circle(surface, (255,255,255), (o[0], o[1]), o[2], 2)

        cx, cy = self.play_cx, HEADER_H + 30
        if self.state == "AIMING":
            end_x = cx + 50 * math.cos(math.radians(self.angle))
            end_y = cy + 50 * math.sin(math.radians(self.angle))
            pygame.draw.line(surface, self.colors["PIECE"], (cx, cy), (end_x, end_y), 4)

        b_color = self.feedback_color if self.feedback_frames > 0 else self.colors["PIECE"]
        pygame.draw.circle(surface, b_color, (int(self.ball_x), int(self.ball_y)), self.ball_r)

        if self.state == "VICTORY":
            v_txt = self.font_title.render("MAX LEVEL REACHED!", True, (255, 215, 0))
            surface.blit(v_txt, (BASE_W // 2 - v_txt.get_width() // 2, BASE_H // 2))
            for b in self.victory_balls:
                pygame.draw.circle(surface,
                    (random.randint(100,255), random.randint(100,255), 255),
                    (int(b[0]), int(b[1])), 8)

        # ---- SSVEP PADS ----
        pad_sz = self.cfg.get("pad_size",    90)
        off    = self.cfg.get("pad_spacing", 160)
        on1, on2   = self.get_flicker_states()
        cross_on   = self.colors["CROSS_ON"]
        cross_off  = self.colors["CROSS_OFF"]

        if self.cfg["handedness"] == "SPLIT":
            pad_y  = BASE_H // 2
            pad1_x = BASE_W // 12
            pad2_x = BASE_W - (BASE_W // 12)
            self.draw_ssvep_pad(surface, pad1_x, pad_y, pad_sz, on1, "ANGLE", cross_on, cross_off)
            self.draw_ssvep_pad(surface, pad2_x, pad_y, pad_sz, on2, "SHOOT", cross_on, cross_off)
        else:
            inset = 10
            if self.cfg["handedness"] == "RIGHT":
                sidebar_left  = self.play_area_right + 10
                sidebar_right = BASE_W - inset
            else:
                sidebar_left  = inset
                sidebar_right = self.play_area_left - 10
            pad_x = (sidebar_left + sidebar_right) // 2
            self.draw_ssvep_pad(surface, pad_x, BASE_H // 2 - off, pad_sz, on1, "ANGLE", cross_on, cross_off)
            self.draw_ssvep_pad(surface, pad_x, BASE_H // 2 + off, pad_sz, on2, "SHOOT", cross_on, cross_off)

        # Photodiode sync square (tracks PAD 1 state)
        if self.cfg["photodiode_sync"]:
            pygame.draw.rect(surface, (255,255,255) if on1 else (0,0,0),
                             pygame.Rect(BASE_W - 40, BASE_H - 40, 40, 40))


# ==========================================
# MASTER ENGINE
# ==========================================
class GameEngine:
    def __init__(self):
        allocate_resources()
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.init()

        pygame.display.set_caption("BCI PINBALL STATION")
        try:
            icon_surf = pygame.image.load("icon.png")
            pygame.display.set_icon(icon_surf)
        except: pass

        self.snd_perfect = create_beep(880, 200)
        self.snd_partial = create_beep(440, 200)
        self.snd_miss    = create_beep(150, 400)

        self.game_data = load_game()
        self.colors    = self.game_data["colors"]
        self.apply_ram_allocation()

        self.flags       = pygame.RESIZABLE | pygame.DOUBLEBUF | pygame.HWSURFACE
        self.display     = pygame.display.set_mode((BASE_W, BASE_H), self.flags, vsync=1)
        self.game_surface = pygame.Surface((BASE_W, BASE_H))
        self.clock       = pygame.time.Clock()

        self.bci_queue   = queue.Queue()

        # Synchronization state — managed by WebSocket handler and tick loop
        self._session_start_time = None   # set when GAME_START is received from Unity
        self._csv_file           = None   # open file handle for the trigger CSV backup
        self._csv_writer         = None   # csv.writer instance writing to _csv_file
        self._tick_active        = False  # True while TICK timer is running
        self._tick_thread        = None   # reference to the running tick thread

        self.init_lsl()
        self.init_udp()
        self.init_websocket()

        self.running       = True
        self.mouse_clicked = False
        self.click_cooldown = 0
        self.frame_count   = 0
        self.state_stack   = [MainMenuState(self)]

    # ------------------------------------------------------------------
    # RAM
    # ------------------------------------------------------------------
    def apply_ram_allocation(self):
        limit = self.game_data["global_config"]["ram_limit"]
        if   limit == 1: gc.set_threshold(700,   10,  10)
        elif limit == 2: gc.set_threshold(5000,  50,  50)
        else:            gc.set_threshold(50000, 500, 500)

    def restore_defaults(self):
        self.game_data["global_config"] = {
            "photodiode_sync": False, "flicker_mode": "SYNC_RR", "RR": 60,
            "target_hz": 15, "target_hz2": 12,
            "custom_hz": 15, "custom_hz2": 12,
            "handedness": "RIGHT", "wall_bounce": False,
            "override_key": "SPACE", "ram_limit": 2,
            "pad_size": 90, "pad_spacing": 160, "pad_layout": "DIAMOND"
        }
        self.restore_colors()

    def restore_colors(self):
        self.colors = DEFAULT_COLORS.copy()
        self.game_data["colors"] = self.colors

    # ------------------------------------------------------------------
    # LSL — sends game event markers to any LSL-capable recording tool.
    # Also starts the BCI command listener thread.
    # LSL stream name: "Focus_Markers"  (type: Markers)
    # BCI command stream listened for: "BCI_Commands"
    # ------------------------------------------------------------------
    def init_lsl(self):
        info = StreamInfo(
            name='Focus_Markers', type='Markers',
            channel_count=1, nominal_srate=0,
            channel_format='string', source_id='focus_game_outlet'
        )
        self.lsl_outlet = StreamOutlet(info)
        threading.Thread(target=self.bci_listener_thread, daemon=True).start()

    def send_marker(self, marker_string):
        """Broadcast a string marker on the LSL Focus_Markers stream."""
        self.lsl_outlet.push_sample([marker_string])

    def bci_listener_thread(self):
        """
        Background thread: listens for classifier commands on the LSL
        'BCI_Commands' stream and places them on bci_queue for the game loop.
        """
        try:
            streams = resolve_byprop('name', 'BCI_Commands')
            inlet   = StreamInlet(streams[0])
            while True:
                sample, _ = inlet.pull_sample()
                if sample:
                    cmd = sample[0]
                    if   cmd in ["MOVE_BLOCK", "MOVE", "ANGLE"]: self.bci_queue.put("ANGLE")
                    elif cmd == "SHOOT":                          self.bci_queue.put("SHOOT")
                    else:                                         self.bci_queue.put(cmd)
        except: pass

    # ------------------------------------------------------------------
    # UDP — sends numeric trigger codes to Unicorn Recorder.
    # Unicorn Recorder embeds these into the EEG file at the exact sample.
    # Destination: 127.0.0.1 : 1000  (Unicorn Recorder default)
    # Trigger map:
    #   b"1" = GAME_START
    #   b"2" = TICK (epoch boundary, fires every EPOCH_INTERVAL_SECONDS)
    #   b"3" = GAME_END
    # ------------------------------------------------------------------
    def init_udp(self):
        """Create the UDP socket used to send triggers to Unicorn Recorder."""
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_udp_trigger(self, trigger_byte):
        """
        Send a single-byte ASCII trigger to Unicorn Recorder via UDP.
        trigger_byte should be one of: UDP_TRIGGER_GAME_START,
        UDP_TRIGGER_TICK, or UDP_TRIGGER_GAME_END defined at the top of
        this file.
        """
        try:
            self._udp_sock.sendto(trigger_byte, (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"[UDP] Failed to send trigger {trigger_byte}: {e}")

    # ------------------------------------------------------------------
    # CSV BACKUP — written simultaneously with every UDP trigger.
    # Provides a human-readable fallback if UDP delivery fails.
    # Format: two columns — timestamp_seconds, event_label
    # File is named with the session start time for uniqueness.
    # ------------------------------------------------------------------
    def open_csv_log(self):
        """
        Create a new trigger CSV file named with the current timestamp.
        Called when GAME_START is received from Unity.
        """
        filename = f"triggers_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._csv_file   = open(filename, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestamp_seconds", "event_label"])  # header row
        print(f"[CSV] Trigger log opened: {filename}")

    def write_csv_row(self, event_label):
        """
        Write one row to the trigger CSV.
        timestamp_seconds is elapsed time since GAME_START.
        """
        if self._csv_writer is None: return
        elapsed = round(time.time() - self._session_start_time, 4)
        self._csv_writer.writerow([elapsed, event_label])

    def close_csv_log(self):
        """Flush and close the trigger CSV file. Called on GAME_END."""
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None
            print("[CSV] Trigger log closed.")

    # ------------------------------------------------------------------
    # TICK LOOP — fires every EPOCH_INTERVAL_SECONDS during gameplay.
    # Runs in its own daemon thread started at GAME_START.
    # Each tick sends a UDP trigger and writes a CSV row.
    # ------------------------------------------------------------------
    def _tick_loop(self):
        """
        Background thread: sends TICK triggers at regular intervals.
        Interval is controlled by EPOCH_INTERVAL_SECONDS at the top of
        this file — change that value to adjust epoch length for analysis.
        Stops automatically when _tick_active is set to False at GAME_END.
        """
        while self._tick_active:
            time.sleep(EPOCH_INTERVAL_SECONDS)
            if not self._tick_active: break  # check again after sleep in case GAME_END arrived
            self.send_udp_trigger(UDP_TRIGGER_TICK)    # b"2" = TICK epoch boundary
            self.send_marker("TICK")                   # also broadcast on LSL
            self.write_csv_row("TICK")
            print(f"[TICK] Epoch boundary at {round(time.time() - self._session_start_time, 2)}s")

    # ------------------------------------------------------------------
    # WEBSOCKET SERVER — receives GAME_START and GAME_END from Unity.
    # Unity connects to ws://localhost:8765 and sends plain text messages.
    # ------------------------------------------------------------------
    def init_websocket(self):
        """Start the WebSocket server in a background daemon thread."""
        threading.Thread(target=self._run_ws_server, daemon=True).start()
        print(f"[WS] WebSocket server starting on ws://{WS_HOST}:{WS_PORT}")

    def _run_ws_server(self):
        """Entry point for the WebSocket server thread."""
        asyncio.run(self._ws_server_main())

    async def _ws_server_main(self):
        """Async WebSocket server — handles Unity connection and messages."""
        async with websockets.serve(self._ws_handler, WS_HOST, WS_PORT):
            await asyncio.Future()  # run forever until the process exits

    async def _ws_handler(self, websocket):
        """
        Handle incoming WebSocket messages from Unity.
        Expected messages:
            "GAME_START" — game screen has become active (win or loss screen not yet shown)
            "GAME_END"   — game screen has ended (fired on both win and loss)
        """
        async for message in websocket:
            msg = message.strip()
            print(f"[WS] Received from Unity: {msg}")

            if msg == "GAME_START":
                self._session_start_time = time.time()

                # Open CSV backup file
                self.open_csv_log()

                # Send GAME_START trigger via UDP to Unicorn Recorder (b"1")
                self.send_udp_trigger(UDP_TRIGGER_GAME_START)

                # Also broadcast on LSL for any LSL-capable recording tools
                self.send_marker("GAME_START")

                # Write first CSV row (timestamp will be 0.0 as this is t=0)
                self.write_csv_row("GAME_START")

                # Start the TICK epoch boundary timer
                self._tick_active = True
                self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
                self._tick_thread.start()

                print("[SYNC] GAME_START — UDP trigger sent, CSV opened, TICK timer started.")

            elif msg == "GAME_END":
                # Stop the TICK timer first so no stray tick fires after GAME_END
                self._tick_active = False

                # Send GAME_END trigger via UDP to Unicorn Recorder (b"3")
                self.send_udp_trigger(UDP_TRIGGER_GAME_END)

                # Also broadcast on LSL
                self.send_marker("GAME_END")

                # Write final CSV row then close the file
                self.write_csv_row("GAME_END")
                self.close_csv_log()

                print("[SYNC] GAME_END — UDP trigger sent, TICK timer stopped, CSV closed.")

    # ------------------------------------------------------------------
    # STATE MANAGEMENT
    # ------------------------------------------------------------------
    def change_state(self, new_state):
        self.state_stack[-1] = new_state
        self.mouse_clicked   = False
        self.click_cooldown  = 8

    def push_state(self, new_state):
        if type(self.state_stack[-1]) == type(new_state): return
        self.state_stack.append(new_state)
        self.mouse_clicked  = False
        self.click_cooldown = 8

    def pop_state(self):
        if len(self.state_stack) > 1:
            self.state_stack.pop()
            self.mouse_clicked  = False
            self.click_cooldown = 8

    def quit(self): self.running = False

    # ------------------------------------------------------------------
    # MAIN GAME LOOP
    # ------------------------------------------------------------------
    def run(self):
        while self.running:
            self.frame_count += 1
            dt = self.clock.tick(
                self.game_data["global_config"]["RR"]
                if self.game_data["global_config"]["flicker_mode"] == "SYNC_RR"
                else 60
            )

            if self.click_cooldown > 0: self.click_cooldown -= 1

            self.mouse_clicked = False
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT: self.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.click_cooldown == 0: self.mouse_clicked = True

            current_state = self.state_stack[-1]
            current_state.handle_events(events)
            current_state.update(dt)

            self.game_surface.fill(self.colors["BG"])
            if len(self.state_stack) > 1 and isinstance(current_state, (
                PinballPauseState, TextInputState, SettingsMenuState,
                ThemeSettingsState, HardwareSettingsState, GameplaySettingsState
            )):
                bg_state = self.state_stack[-2]
                bg_state._is_background = True
                bg_state.draw(self.game_surface)
                bg_state._is_background = False
            current_state._is_background = False
            current_state.draw(self.game_surface)

            win_w, win_h = self.display.get_size()
            scale        = min(win_w / BASE_W, win_h / BASE_H)
            new_w, new_h = int(BASE_W * scale), int(BASE_H * scale)
            scaled_surf  = pygame.transform.scale(self.game_surface, (new_w, new_h))
            self.display.fill(self.colors["BG"])
            self.display.blit(scaled_surf, ((win_w - new_w) // 2, (win_h - new_h) // 2))
            pygame.display.flip()

        # Ensure CSV is closed cleanly on unexpected exit
        if self._csv_file: self.close_csv_log()
        save_game(self.game_data)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    GameEngine().run()
