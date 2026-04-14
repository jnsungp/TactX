"""
EGaIn sensor implementation (matplotlib visualization).
Reads from a single serial port that provides data for both sensors (S0 and S1).
"""

import serial
import threading
import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button

from .base_sensor import BaseTactileSensor
from sensor_configs import EGAIN_CONFIG


class EGaInSensor(BaseTactileSensor):
    def __init__(self):
        self.port = EGAIN_CONFIG["port"]
        self.baudrate = EGAIN_CONFIG["baudrate"]
        self.serial_conn = None

        self.is_running = False
        self._data_lock = threading.Lock()
        self._current_data = [0] * 12

        self.min_val = 0.0
        self.max_val = 100.0

        # Raw log (keep last N lines)
        self._log_lock = threading.Lock()
        self._log_lines = []
        self._log_max_lines = 30

        # Matplotlib objects
        self._fig = None
        self._im0 = None
        self._im1 = None
        self._txt0 = []
        self._txt1 = []
        self._log_text_artist = None
        self._anim = None

    def initialize(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)

            # Arduino helps: reset via DTR
            self.serial_conn.dtr = False
            time.sleep(0.5)
            self.serial_conn.dtr = True

            print("Waiting for Arduino Uno to boot...")
            time.sleep(3.0)
            self.serial_conn.reset_input_buffer()

            self.is_running = True
            self._thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
            self._thread.start()
            print(f"Connected to {self.port} and started background reader.")

        except Exception as e:
            print(f"Failed to initialize sensor: {e}")
            raise

    def _serial_reader_loop(self):
        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                data_str = line.split()
                if len(data_str) == 12:
                    data = [int(x) for x in data_str]
                    with self._data_lock:
                        self._current_data = data

                    with self._log_lock:
                        self._log_lines.append(line)
                        if len(self._log_lines) > self._log_max_lines:
                            self._log_lines = self._log_lines[-self._log_max_lines :]

                else:
                    # Useful debug if Arduino sends unexpected lines
                    print(f"Data length mismatch: expected 12, got {len(data_str)} | '{line}'")

            except Exception as e:
                print(f"Serial Error: {e}")

    # ---------- Visualization (Matplotlib) ----------

    def visualize(self):
        """
        Run matplotlib GUI (blocking). Call this on MAIN thread.
        """
        # Use a backend that works on Ubuntu desktop (Qt5Agg is common).
        # Usually no need to set explicitly; but if you have issues:
        # import matplotlib
        # matplotlib.use("Qt5Agg")

        self._fig = plt.figure(figsize=(10, 7))
        self._fig.canvas.manager.set_window_title("High-Speed Pressure Map (EGaIn)")

        # Layout: top = controls, middle = 2 heatmaps, bottom = log text
        gs = self._fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 8.0, 3.0])

        ax_ctrl = self._fig.add_subplot(gs[0, :])
        ax0 = self._fig.add_subplot(gs[1, 0])
        ax1 = self._fig.add_subplot(gs[1, 1])
        ax_log = self._fig.add_subplot(gs[2, :])

        # Controls area: we'll hide the axis and place widgets manually
        ax_ctrl.axis("off")

        # Heatmaps: 2x3 grids
        init0 = np.zeros((2, 3), dtype=float)
        init1 = np.zeros((2, 3), dtype=float)

        ax0.set_title("Sensor 0")
        ax1.set_title("Sensor 1")

        # grayscale: low -> white, high -> black (match your wx style)
        self._im0 = ax0.imshow(init0, vmin=self.min_val, vmax=self.max_val, cmap="gray_r", interpolation="nearest")
        self._im1 = ax1.imshow(init1, vmin=self.min_val, vmax=self.max_val, cmap="gray_r", interpolation="nearest")

        for ax in (ax0, ax1):
            ax.set_xticks(range(3))
            ax.set_yticks(range(2))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(1.5, -0.5)  # keep row0 on top
            ax.grid(False)

        # Put numbers on cells
        self._txt0 = []
        self._txt1 = []
        for r in range(2):
            for c in range(3):
                self._txt0.append(ax0.text(c, r, "0", ha="center", va="center", fontsize=14))
                self._txt1.append(ax1.text(c, r, "0", ha="center", va="center", fontsize=14))

        # Log view
        ax_log.set_title("Raw Serial Log (latest)")
        ax_log.axis("off")
        self._log_text_artist = ax_log.text(
            0.01, 0.98, "", ha="left", va="top", family="monospace", fontsize=9, transform=ax_log.transAxes
        )

        # Controls: TextBox + Button
        # Create small axes for widgets (figure coordinates)
        ax_min = self._fig.add_axes([0.15, 0.92, 0.12, 0.05])  # [left, bottom, width, height]
        ax_max = self._fig.add_axes([0.35, 0.92, 0.12, 0.05])
        ax_btn = self._fig.add_axes([0.52, 0.92, 0.10, 0.05])

        tb_min = TextBox(ax_min, "Min", initial=str(self.min_val))
        tb_max = TextBox(ax_max, "Max", initial=str(self.max_val))
        btn = Button(ax_btn, "Apply")

        def apply_range(_event=None):
            try:
                new_min = float(tb_min.text)
                new_max = float(tb_max.text)
                # avoid divide-by-zero / nonsense
                if new_max <= new_min:
                    print("Max must be > Min")
                    return
                self.min_val = new_min
                self.max_val = new_max
                self._im0.set_clim(self.min_val, self.max_val)
                self._im1.set_clim(self.min_val, self.max_val)
                self._fig.canvas.draw_idle()
            except ValueError:
                print("Invalid min/max")

        btn.on_clicked(apply_range)

        # Close handling
        self._fig.canvas.mpl_connect("close_event", lambda _e: self.cleanup())

        # Animation update
        def update(_frame_idx):
            with self._data_lock:
                d = list(self._current_data)

            s0 = np.array(d[0:6], dtype=float).reshape(2, 3)
            s1 = np.array(d[6:12], dtype=float).reshape(2, 3)

            self._im0.set_data(s0)
            self._im1.set_data(s1)

            # Update text values + choose readable text color (black/white) by intensity
            # (simple threshold based on midpoint)
            mid = 0.5 * (self.min_val + self.max_val) if self.max_val > self.min_val else self.min_val

            idx = 0
            for r in range(2):
                for c in range(3):
                    v = int(s0[r, c])
                    t = self._txt0[idx]
                    t.set_text(str(v))
                    t.set_color("white" if v > mid else "black")
                    idx += 1

            idx = 0
            for r in range(2):
                for c in range(3):
                    v = int(s1[r, c])
                    t = self._txt1[idx]
                    t.set_text(str(v))
                    t.set_color("white" if v > mid else "black")
                    idx += 1

            with self._log_lock:
                log_txt = "\n".join(self._log_lines[-self._log_max_lines :])
            self._log_text_artist.set_text(log_txt)

            return [self._im0, self._im1, *self._txt0, *self._txt1, self._log_text_artist]

        # roughly 60 FPS like your wx timer (16ms). You can relax to 30 FPS if CPU heavy.
        self._anim = FuncAnimation(self._fig, update, interval=16, blit=False)

        print("Launching Matplotlib GUI.")
        plt.show(block=True)

    # ---------- Shared memory update API ----------

    def read_and_update(self, shm_array0, shm_array1):
        with self._data_lock:
            data_clipped = np.asarray(self._current_data, dtype=np.float32)
        np.maximum(data_clipped, self.min_val, out=data_clipped)
        shm_array0[:] = data_clipped[0:6]
        shm_array1[:] = data_clipped[6:12]

    def cleanup(self):
        self.is_running = False
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
        except Exception:
            pass

    def _on_close(self, *_args, **_kwargs):
        self.cleanup()

    @property
    def element_count(self):
        return int(np.prod([6, 2]))  # 12
