# Air-pencil-project
# Air Pencil — Webcam-based Virtual Drawing System (Smart Edition)

Draw in the air using just your index finger — no stylus, tablet, or extra
hardware needed. A standard webcam + MediaPipe hand tracking is enough.

## How it works

1. OpenCV captures live video from your webcam.
2. MediaPipe Hands detects 21 landmark points on your hand every frame.
3. The position of your index fingertip (landmark 8) is tracked frame to
   frame. Points are buffered for the current stroke and previewed live;
   when you lift the pen, the finished stroke is committed to the canvas.
4. **Straight-line snapping**: when a finished stroke is only slightly
   wobbly (its points stay close to the straight line connecting its
   start and end), it is automatically redrawn as a perfectly straight
   line at the exact angle you drew it — great for letters like I, K, A,
   or straight underlines. Genuinely curved strokes (O, S, C, etc.) are
   left as freehand.
5. **Handwriting-to-text panel**: whenever you pause after drawing
   something new (or press `r`), the canvas is run through OCR
   (Tesseract) and the recognized text appears in a side panel next to
   the video.

## Setup

```bash
pip install -r requirements.txt
python air_pencil.py
```

Requires a working webcam. Tested with Python 3.11 (MediaPipe does not yet
support the newest Python releases — see the setup PDF for details).

### Optional: enabling the text-recognition panel

The straight-line snapping works out of the box with no extra setup. The
handwriting-to-text panel needs one extra one-time install:

1. `pytesseract` and `pillow` are already in `requirements.txt`, so
   `pip install -r requirements.txt` covers the Python side.
2. Separately install the **Tesseract-OCR** engine itself (this is not a
   Python package, it's a small standalone program):
   Windows installer — https://github.com/UB-Mannheim/tesseract/wiki
   Use the default install location suggested by the installer
   (`C:\Program Files\Tesseract-OCR\`) — the script auto-detects it there.

If Tesseract isn't installed, the app still runs fine — drawing and
straight-line snapping work normally, the side panel just stays empty and
shows "OCR not installed".

## Gestures / Controls

| Gesture                                   | Action                                   |
|--------------------------------------------|-------------------------------------------|
| Only index finger up                        | Drawing mode — draws a stroke              |
| Index + middle finger up                    | Selection mode — hover over the top bar to pick a color / eraser (no drawing) |
| Close hand into a fist                      | Pen up — pause between letters/strokes without drawing |
| All 5 fingers up (open palm)                | Clears the canvas and the recognized text  |
| Keyboard `r`                                | Force text recognition right now           |
| Keyboard `s`                                | Saves the current drawing as a PNG (in `saved_drawings/`) |
| Keyboard `q` or `Esc`                       | Quits the application                      |

## Project structure

```
air_pencil/
├── air_pencil.py       # main application
├── requirements.txt    # dependencies
└── saved_drawings/     # auto-created; saved PNG drawings go here
```

## Notes for your report / viva

- Fingertip detection uses MediaPipe's pre-trained hand landmark model —
  no custom model training is required for the base version.
- Gesture logic compares the y-coordinate of each fingertip landmark
  against its corresponding lower joint (PIP) to decide if that finger
  is "up" or "down".
- Fingertip position is smoothed frame-to-frame (exponential smoothing)
  to reduce jitter, and a short "miss tolerance" window prevents the
  line from breaking if the hand is briefly lost for a frame or two.
- Straight-line detection works by measuring each point's perpendicular
  distance from the straight line connecting the stroke's first and last
  point; if the maximum deviation is small relative to the stroke's
  length, the whole stroke is replaced with one clean straight line.
- The header bar acts as an on-screen color palette — its x-position maps
  directly to a color in the `PALETTE` list in the code.
- FPS is displayed on screen — useful to report real-time performance in
  your synopsis / evaluation.

## Possible extensions (for future scope section)

- Multi-hand support for two-handed / collaborative drawing.
- Replace Tesseract with a custom-trained CNN for handwriting recognition,
  for better accuracy on cursive/stylized air-writing.
- Deploy as a browser-based tool using TensorFlow.js.
