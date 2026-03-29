from __future__ import annotations

import time

from bot.ratelimit import RateLimiter


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed(user_id=123) is True

    def test_blocks_requests_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            assert limiter.is_allowed(user_id=123) is True
        assert limiter.is_allowed(user_id=123) is False

    def test_independent_per_user(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed(user_id=111) is True
        assert limiter.is_allowed(user_id=111) is True
        assert limiter.is_allowed(user_id=111) is False
        # Different user should still be allowed
        assert limiter.is_allowed(user_id=222) is True

    def test_allows_after_window_expires(self):
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.is_allowed(user_id=123) is True
        assert limiter.is_allowed(user_id=123) is False
        time.sleep(1.1)
        assert limiter.is_allowed(user_id=123) is True

    def test_sliding_window_evicts_old_entries(self):
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        assert limiter.is_allowed(user_id=123) is True
        time.sleep(0.6)
        assert limiter.is_allowed(user_id=123) is True
        # First request should be almost expired
        time.sleep(0.5)
        # First request expired, second still within window
        assert limiter.is_allowed(user_id=123) is True

    def test_default_values(self):
        limiter = RateLimiter()
        assert limiter.max_requests == 20
        assert limiter.window_seconds == 60
