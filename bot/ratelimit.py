from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    """In-memory sliding window rate limiter, per user."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: dict[int, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        """Check if user_id is within rate limit, recording the request if so."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        # Evict expired entries
        self._timestamps[user_id] = [t for t in self._timestamps[user_id] if t > cutoff]
        if len(self._timestamps[user_id]) >= self.max_requests:
            return False
        self._timestamps[user_id].append(now)
        return True
