"""
Single FlexiTac sensor visualizer.
Reads from /dev/ttyUSB0 (12x32 grid) and displays a live OpenCV colormap window.

Usage:
    python flexitac_single_viz.py [port]        (default: /dev/ttyUSB0)
"""

import sys
import time
import threading
import signal

import numpy as np
import serial
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
PORT            = '/dev/ttyUSB1'
BAUDRATE        = 1_000_000
CALIBRATION_N   = 30        # frames averaged for baseline
THRESHOLD       = 12        # subtract from raw - median
NOISE_SCALE     = 60        # divisor when no real contact
TEMPORAL_ALPHA  = 0.2       # EMA weight for new frame  (0=frozen, 1=raw)
WINDOW_SCALE    = 30        # pixels per grid cell
ROWS, COLS      = 12, 32
# ──────────────────────────────────────────────────────────────────────────────


def calibrate(ser: serial.Serial) -> np.ndarray:
    """Collect CALIBRATION_N complete frames and return their median."""
    print(f"Calibrating on {ser.port}  ({CALIBRATION_N} frames)…")
    frames = []
    current = None

    while len(frames) < CALIBRATION_N:
        if ser.in_waiting == 0:
            time.sleep(0.001)
            continue
        try:
            line = ser.readline().decode('utf-8').strip()
        except Exception:
            line = ""

        if len(line) < 10:                      # delimiter / empty line
            if current is not None and len(current) == ROWS:
                frames.append(np.array(current, dtype=np.float32))
            current = []
            continue

        if current is not None:
            try:
                vals = [int(v) for v in line.split()]
                if len(vals) == COLS:
                    current.append(vals)
            except ValueError:
                pass

    median = np.median(frames, axis=0)          # shape (12, 32)
    print("Calibration done.")
    return median


def read_thread(ser: serial.Serial, median: np.ndarray,
                buf: list, running: list) -> None:
    """Background thread: parse serial stream → write normalised frame to buf[0]."""
    current = None
    backup  = None

    while running[0]:
        if ser.in_waiting == 0:
            time.sleep(0.001)
            continue
        try:
            line = ser.readline().decode('utf-8').strip()
        except Exception:
            line = ""

        if len(line) < 10:
            if current is not None and len(current) == ROWS:
                row_lens = [len(r) for r in current]
                if all(l == COLS for l in row_lens):
                    backup = np.array(current, dtype=np.float32)
            current = []

            if backup is not None:
                raw = backup - median - THRESHOLD
                raw = np.clip(raw, 0, 100)
                if np.max(raw) < THRESHOLD:
                    norm = raw / NOISE_SCALE
                else:
                    norm = raw / np.max(raw)
                buf[0] = norm.copy()
            continue

        if current is not None:
            try:
                vals = [int(v) for v in line.split()]
                if len(vals) == COLS:
                    current.append(vals)
            except ValueError:
                pass


def main(port: str = PORT) -> None:
    running = [True]

    def handle_signal(sig, frame):
        print("\nStopping…")
        running[0] = False

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Opening {port} at {BAUDRATE} baud…")
    ser = serial.Serial(port, BAUDRATE, timeout=1)
    time.sleep(1)
    ser.flush()

    median = calibrate(ser)

    buf = [np.zeros((ROWS, COLS), dtype=np.float32)]

    t = threading.Thread(target=read_thread, args=(ser, median, buf, running),
                         daemon=True)
    t.start()

    win = "FlexiTac"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, COLS * WINDOW_SCALE, ROWS * WINDOW_SCALE)

    prev = np.zeros((ROWS, COLS), dtype=np.float32)
    print("Visualising — press  q  or  Ctrl+C  to quit.")

    while running[0]:
        frame = buf[0]
        smoothed = TEMPORAL_ALPHA * frame + (1 - TEMPORAL_ALPHA) * prev
        prev = smoothed

        disp = (smoothed * 255).astype(np.uint8)
        color = cv2.applyColorMap(disp, cv2.COLORMAP_VIRIDIS)
        cv2.imshow(win, color)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            running[0] = False

    running[0] = False
    t.join(timeout=2.0)
    ser.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else PORT
    main(port)
