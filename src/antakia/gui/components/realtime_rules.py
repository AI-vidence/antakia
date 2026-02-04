"""
Debouncer for real-time rule computation.

Delays rule computation to avoid excessive CPU usage during interactive selection.
"""

import threading
from typing import Callable, Optional


class RealtimeRulesDebouncer:
    """
    Debouncer that delays callback execution.

    Useful for real-time rule computation during selection changes.
    If trigger() is called multiple times within delay_seconds,
    only the last call will actually fire the callback.
    """

    def __init__(self, callback: Callable, delay_seconds: float = 0.5):
        """
        Initialize the debouncer.

        Parameters
        ----------
        callback : Callable
            Function to call after the delay
        delay_seconds : float
            Delay before firing (default 0.5s)
        """
        self.callback = callback
        self.delay_seconds = delay_seconds
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def trigger(self):
        """
        Trigger the debouncer.

        Cancels any pending timer and starts a new one.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay_seconds, self._fire)
            self._timer.start()

    def _fire(self):
        """Execute the callback."""
        with self._lock:
            self._timer = None
        try:
            self.callback()
        except Exception:
            pass  # Silently ignore errors in callback

    def cancel(self):
        """Cancel any pending timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
