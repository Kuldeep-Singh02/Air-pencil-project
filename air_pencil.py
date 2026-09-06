"""
Air Pencil - Webcam-based Virtual Drawing System (Smart Edition)
------------------------------------------------------------------
Draw in the air using your index fingertip. The webcam tracks your hand
using MediaPipe, and your fingertip movement is rendered as strokes on
a virtual canvas overlaid on the live video.

Smart features:
    - Straight-line snapping: if a single stroke (between pen-down and
      pen-up) is roughly straight, it is automatically redrawn as a
      perfectly straight line at the same angle/position you drew it.
    - Auto handwriting recognition: whenever you pause drawing for a
      short moment, whatever is currently on the canvas is passed
      through OCR (Tesseract) and the recognized text is shown in a
      side panel next to the video.

Gestures:
    - Only INDEX finger up            -> Drawing mode (pen down)
    - INDEX + MIDDLE fingers up       -> Selection mode (move over the
                                          color palette at the top to
                                          pick a color / eraser, no drawing)
    - Close hand into a fist          -> Pen up (pause between strokes/letters)
    - All 5 fingers up (open palm)    -> Clear the canvas + recognized text
    - Press 'r'                       -> Force text recognition right now
    - Press 's'                       -> Save the current canvas as an image
    - Press 'q' or ESC                -> Quit the application

Run:
    python air_pencil.py

Optional (for handwriting recognition to work):
    1. pip install pytesseract pillow
    2. Install the Tesseract-OCR engine (separate program, not a pip
       package): https://github.com/UB-Mannheim/tesseract/wiki
       (Windows installer). Default install path is auto-detected below.
    If Tesseract is not installed, drawing still works fine - only the
    text-recognition side panel will stay empty.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math
import textwrap

# ---------------------------------------------------------------------------
# Optional OCR support
# ---------------------------------------------------------------------------
OCR_AVAILABLE = False
try:
    import pytesseract

    # Common default install path on Windows - adjust if you installed elsewhere
    default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and os.path.exists(default_win_path):
        pytesseract.pytesseract.tesseract_cmd = default_win_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAM_WIDTH, CAM_HEIGHT = 1280, 720
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 90
SAVE_DIR = "saved_drawings"
SMOOTHING = 0.5          # 0 = no smoothing, closer to 1 = smoother but more lag
MISS_TOLERANCE = 6       # frames the hand can briefly go undetected without breaking the line
STRAIGHTEN_TOLERANCE = 0.14  # how "wobbly" a stroke can be and still snap straight (fraction of stroke length)
IDLE_RECOGNIZE_SEC = 1.3     # pause (seconds) after which auto text-recognition runs
SIDE_PANEL_W = 380
OCR_WHITELIST = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Color palette (name, BGR color) shown as swatches at the top of the screen
PALETTE = [
    ("Eraser", (0, 0, 0)),
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),
]
# One extra slot at the end of the header is reserved for the dedicated
# "Clear" button, so an accidental open-palm flicker can never wipe the
# canvas anymore - clearing only happens via this button (hover + hold)
# or the 'c' key.
TOTAL_SLOTS = len(PALETTE) + 1
SWATCH_W = CAM_WIDTH // TOTAL_SLOTS
CLEAR_SLOT_INDEX = len(PALETTE)
HEADER_H = 90
CLEAR_HOLD_FRAMES = 22   # ~1-1.5 sec hover over the Clear button to confirm


class HandTracker:
    """Thin wrapper around MediaPipe Hands for fingertip / gesture detection."""

    TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips

    def __init__(self, max_hands=1, detection_conf=0.6, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.landmarks = []  # list of (id, x, y) for the current frame

    def process(self, frame_bgr):
        """Detect hand landmarks in the frame. Returns True if a hand is found."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        self.landmarks = []

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            h, w, _ = frame_bgr.shape
            for idx, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks.append((idx, cx, cy))
            self.mp_draw.draw_landmarks(
                frame_bgr, hand, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(160, 160, 160), thickness=1),
            )
            return True
        return False

    def fingers_up(self):
        """Return a list of 5 booleans [thumb, index, middle, ring, pinky]."""
        if not self.landmarks:
            return [False] * 5

        fingers = []
        lm_dict = {i: (x, y) for i, x, y in self.landmarks}

        fingers.append(lm_dict[self.TIP_IDS[0]][0] > lm_dict[self.TIP_IDS[0] - 1][0])
        for tip_id in self.TIP_IDS[1:]:
            fingers.append(lm_dict[tip_id][1] < lm_dict[tip_id - 2][1])

        return fingers

    def get_landmark(self, idx):
        for i, x, y in self.landmarks:
            if i == idx:
                return x, y
        return None


def draw_header(frame, current_color, clear_progress=0.0):
    """Draw the color-palette header bar, plus the dedicated Clear button
    at the end. clear_progress (0-1) fills the Clear button as the user
    hovers over it, so they get clear visual feedback before it triggers."""
    for i, (name, color) in enumerate(PALETTE):
        x1, x2 = i * SWATCH_W, (i + 1) * SWATCH_W
        swatch_color = (40, 40, 40) if name == "Eraser" else color
        cv2.rectangle(frame, (x1, 0), (x2, HEADER_H), swatch_color, -1)
        cv2.putText(frame, name, (x1 + 15, HEADER_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if color == current_color and name != "Eraser":
            cv2.rectangle(frame, (x1, 0), (x2, HEADER_H), (255, 255, 255), 3)

    # Clear button (last slot, spans to the right edge of the frame)
    cx1, cx2 = CLEAR_SLOT_INDEX * SWATCH_W, CAM_WIDTH
    cv2.rectangle(frame, (cx1, 0), (cx2, HEADER_H), (30, 30, 130), -1)
    if clear_progress > 0:
        fill_w = int((cx2 - cx1) * clear_progress)
        cv2.rectangle(frame, (cx1, 0), (cx1 + fill_w, HEADER_H), (0, 0, 255), -1)
    cv2.putText(frame, "Clear", (cx1 + 15, HEADER_H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, 0), (CAM_WIDTH, HEADER_H), (255, 255, 255), 2)


def is_stroke_straight(points):
    """Check whether a list of (x, y) points is roughly a straight line."""
    if len(points) < 3:
        return True

    x0, y0 = points[0]
    x1, y1 = points[-1]
    line_len = math.hypot(x1 - x0, y1 - y0)
    if line_len < 12:
        return True

    dx, dy = x1 - x0, y1 - y0
    max_dev = 0
    for (px, py) in points:
        dev = abs(dy * (px - x0) - dx * (py - y0)) / line_len
        max_dev = max(max_dev, dev)

    return max_dev <= max(10, STRAIGHTEN_TOLERANCE * line_len)


def commit_stroke(canvas, points, color, thickness):
    """Finalize a finished stroke onto the permanent canvas - straightened
    automatically if it was roughly a straight line."""
    if len(points) < 2:
        if len(points) == 1:
            cv2.circle(canvas, points[0], thickness // 2, color, -1)
        return

    is_eraser = color == (0, 0, 0)
    if not is_eraser and is_stroke_straight(points):
        cv2.line(canvas, points[0], points[-1], color, thickness, lineType=cv2.LINE_AA)
    else:
        for i in range(1, len(points)):
            cv2.line(canvas, points[i - 1], points[i], color, thickness, lineType=cv2.LINE_AA)


def group_boxes_into_characters(boxes):
    """Group raw stroke bounding boxes into per-character groups. Strokes
    whose x-ranges are touching/overlapping (e.g. the dot and stem of
    'i'/'j', or a crossbar drawn separately over a vertical stroke like
    't'/'A') are merged into one character. Strokes separated by a normal
    horizontal writing gap are kept as distinct characters, so a whole
    word/name segments into separate letters correctly."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    avg_h = float(np.median([b[3] for b in boxes]))
    gap_threshold = max(14, avg_h * 0.16)  # small: only true multi-stroke parts merge

    groups = [[boxes[0]]]
    for b in boxes[1:]:
        group = groups[-1]
        group_right = max(g[0] + g[2] for g in group)
        gap = b[0] - group_right
        if gap < gap_threshold:
            group.append(b)
        else:
            groups.append([b])

    merged = []
    for g in groups:
        x0 = min(gg[0] for gg in g)
        y0 = min(gg[1] for gg in g)
        x1 = max(gg[0] + gg[2] for gg in g)
        y1 = max(gg[1] + gg[3] for gg in g)
        merged.append((x0, y0, x1 - x0, y1 - y0))
    return merged


# Multiple OCR attempts per character (different engine/page-segmentation
# combos react very differently to shaky, hand/air-drawn strokes). We run
# all of them and keep whichever result Tesseract itself is most
# confident about, instead of trusting a single fixed mode.
OCR_CONFIGS = [
    "--oem 3 --psm 10",   # LSTM engine, single character
    "--oem 3 --psm 8",    # LSTM engine, single word
    "--oem 0 --psm 10",   # legacy engine, single character
    "--oem 3 --psm 7",    # LSTM engine, single text line
]


def clean_stroke_for_ocr(crop_mask):
    """Smooth out webcam-tracking jitter/wobble in a cropped stroke mask so
    it looks closer to a clean printed glyph before handing it to OCR:
    slightly bolden thin strokes, then blur+re-threshold to soften jagged
    pixel edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bold = cv2.dilate(crop_mask, kernel, iterations=1)
    blurred = cv2.GaussianBlur(bold, (5, 5), 0)
    _, smooth = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    return smooth


def recognize_single_character(square_img):
    """Try several OCR configs on one character image and keep the result
    Tesseract reports the highest confidence for. Returns '?' if nothing
    usable (within the allowed whitelist) came back from any attempt."""
    best_char, best_conf = "", -1.0
    for cfg in OCR_CONFIGS:
        full_cfg = f"{cfg} -c tessedit_char_whitelist={OCR_WHITELIST}"
        try:
            data = pytesseract.image_to_data(
                square_img, config=full_cfg, output_type=pytesseract.Output.DICT
            )
        except Exception:
            continue
        for i, txt in enumerate(data.get("text", [])):
            txt = txt.strip()
            if not txt:
                continue
            ch = txt[0]
            if ch not in OCR_WHITELIST:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, KeyError):
                conf = -1.0
            if conf > best_conf:
                best_conf, best_char = conf, ch
    return best_char if best_char else "?"


def recognize_canvas_text(canvas):
    """Run OCR on the current canvas, one character at a time, and return
    the recognized text (or ''). Only letters (a-z, A-Z) and digits (0-9)
    are ever returned - nothing else is allowed through."""
    if not OCR_AVAILABLE:
        return ""

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    if mask.sum() == 0:
        return ""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 60]
    if not raw_boxes:
        return ""

    boxes = group_boxes_into_characters(raw_boxes)

    chars = []
    for (x, y, w, h) in boxes:
        pad = 22
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
        crop = mask[y0:y1, x0:x1]
        crop = clean_stroke_for_ocr(crop)

        # Paste onto a square, padded canvas (helps Tesseract a lot with
        # single glyphs) then invert to black-on-white and upscale.
        side = max(crop.shape) + 50
        square = np.zeros((side, side), dtype=np.uint8)
        oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        square[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        square = cv2.bitwise_not(square)
        square = cv2.resize(square, (220, 220), interpolation=cv2.INTER_CUBIC)

        chars.append(recognize_single_character(square))

    return "".join(chars)


def wrap_text_lines(text, width_chars):
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            continue
        lines.extend(textwrap.wrap(paragraph, width=width_chars) or [""])
    return lines


def build_side_panel(recognized_text, ocr_status_msg):
    panel = np.full((CAM_HEIGHT, SIDE_PANEL_W, 3), 250, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (SIDE_PANEL_W - 1, HEADER_H), (230, 230, 230), -1)
    cv2.putText(panel, "Recognized Text", (18, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.line(panel, (0, HEADER_H), (SIDE_PANEL_W, HEADER_H), (200, 200, 200), 2)

    y = HEADER_H + 45
    for line in wrap_text_lines(recognized_text, width_chars=24):
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        y += 38
        if y > CAM_HEIGHT - 40:
            break

    cv2.putText(panel, ocr_status_msg, (14, CAM_HEIGHT - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
    return panel


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not access the webcam. Check camera permissions / index.")
        return

    tracker = HandTracker(max_hands=1)
    canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    stroke_preview = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

    draw_color = (0, 0, 255)   # default: red
    xp, yp = 0, 0
    smooth_x, smooth_y = 0, 0
    miss_count = 0
    prev_time = 0

    current_stroke = []        # points of the stroke currently being drawn
    was_drawing = False        # drawing state in the previous frame
    clear_hover_frames = 0     # how many consecutive frames the finger has hovered the Clear button

    recognized_text = ""
    ocr_status_msg = "OCR ready" if OCR_AVAILABLE else "OCR not installed (pytesseract missing)"
    new_content_pending = False
    last_activity_time = time.time()

    print("Air Pencil (Smart Edition) started.")
    print("Keys: 'c' clear canvas | 'r' recognize now | 's' save | 'q'/Esc quit")
    if not OCR_AVAILABLE:
        print("NOTE: pytesseract not installed - text recognition panel will stay empty.")
        print("      Run: pip install pytesseract pillow  (and install Tesseract-OCR engine)")

    def do_clear():
        nonlocal canvas, stroke_preview, current_stroke, recognized_text, new_content_pending, xp, yp
        canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        stroke_preview = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        current_stroke = []
        recognized_text = ""
        new_content_pending = False
        xp, yp = 0, 0

    while True:
        success, frame = cap.read()
        if not success:
            print("ERROR: Failed to read frame from webcam.")
            break

        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        frame = cv2.flip(frame, 1)

        hand_found = tracker.process(frame)
        drawing_now = False
        hovering_clear = False

        if hand_found:
            miss_count = 0
            fingers = tracker.fingers_up()
            ix, iy = tracker.get_landmark(8)
            mx, my = tracker.get_landmark(12)

            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = ix, iy
            else:
                smooth_x = int(smooth_x + (ix - smooth_x) * SMOOTHING)
                smooth_y = int(smooth_y + (iy - smooth_y) * SMOOTHING)
            ix, iy = smooth_x, smooth_y

            index_up = fingers[1]
            middle_up = fingers[2]

            if index_up and middle_up:
                xp, yp = 0, 0
                cv2.rectangle(frame, (ix - 10, iy - 10), (ix + 10, iy + 10), draw_color, cv2.FILLED)
                if iy < HEADER_H:
                    idx_slot = min(ix // SWATCH_W, TOTAL_SLOTS - 1)
                    if idx_slot == CLEAR_SLOT_INDEX:
                        hovering_clear = True
                    else:
                        name, color = PALETTE[idx_slot]
                        draw_color = (0, 0, 0) if name == "Eraser" else color

            elif index_up:
                drawing_now = True
                if iy > HEADER_H:
                    if not current_stroke:
                        current_stroke = [(ix, iy)]
                    else:
                        current_stroke.append((ix, iy))
                        cv2.line(stroke_preview, current_stroke[-2], current_stroke[-1],
                                  draw_color, BRUSH_THICKNESS if draw_color != (0, 0, 0) else ERASER_THICKNESS,
                                  lineType=cv2.LINE_AA)
                xp, yp = ix, iy

            else:
                xp, yp = 0, 0
        else:
            miss_count += 1
            if miss_count > MISS_TOLERANCE:
                xp, yp = 0, 0
                smooth_x, smooth_y = 0, 0
            else:
                drawing_now = was_drawing  # tolerate a brief flicker mid-stroke

        # --- Stroke start/end handling ---
        if was_drawing and not drawing_now:
            thickness = ERASER_THICKNESS if draw_color == (0, 0, 0) else BRUSH_THICKNESS
            commit_stroke(canvas, current_stroke, draw_color, thickness)
            current_stroke = []
            stroke_preview = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            if draw_color != (0, 0, 0):
                new_content_pending = True
            last_activity_time = time.time()
        was_drawing = drawing_now

        # --- Clear button: hover-and-hold to confirm (no accidental clears) ---
        if hovering_clear:
            clear_hover_frames += 1
            if clear_hover_frames >= CLEAR_HOLD_FRAMES:
                do_clear()
                clear_hover_frames = 0
                cv2.putText(frame, "Canvas Cleared", (CAM_WIDTH // 2 - 150, CAM_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            clear_hover_frames = 0
        clear_progress = min(1.0, clear_hover_frames / CLEAR_HOLD_FRAMES)

        # --- Auto text recognition after a short pause ---
        if OCR_AVAILABLE and new_content_pending and not drawing_now:
            if time.time() - last_activity_time > IDLE_RECOGNIZE_SEC:
                text = recognize_canvas_text(canvas)
                if text:
                    recognized_text = text
                new_content_pending = False

        # --- Merge canvas + live stroke preview onto the frame ---
        combined = cv2.bitwise_or(canvas, stroke_preview)
        gray = cv2.cvtColor(combined, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, combined)

        draw_header(frame, draw_color, clear_progress)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (CAM_WIDTH - 150, HEADER_H + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Fist = pen up | 'c' clear | 'r' recognize", (20, CAM_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        side_panel = build_side_panel(recognized_text, ocr_status_msg)
        display = np.hstack([frame, side_panel])

        cv2.imshow("Air Pencil - Webcam Virtual Drawing", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"drawing_{int(time.time())}.png")
            cv2.imwrite(filename, canvas)
            print(f"Saved drawing to {filename}")
        elif key == ord('r'):
            text = recognize_canvas_text(canvas)
            if text:
                recognized_text = text
            new_content_pending = False
        elif key == ord('c'):
            do_clear()
            clear_hover_frames = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
