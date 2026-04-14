"""Simple status display module for coordinated terminal output."""

import sys
import threading

_lock = threading.Lock()


def initialize():
    """Initialize status display by printing 4 blank lines."""
    print("\n\n\n")


def update_line(line_num, text):
    """Update a specific status line (1-4) without affecting others."""
    with _lock:
        sys.stdout.write(f"\033[{5-line_num}A")
        sys.stdout.write("\033[K")
        sys.stdout.write(text)
        sys.stdout.write(f"\033[{5-line_num}B")
        sys.stdout.write("\r")
        sys.stdout.flush()


def print_message(text):
    """Print a message on the 4th status line, persisting until overridden."""
    update_line(4, text)











