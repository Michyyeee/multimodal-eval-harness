"""Thread-safe Token Bucket Rate Limiter for QPS management (Zero dependencies)."""

import threading
import time


class TokenBucketRateLimiter:
    """Thread-safe token bucket algorithm to enforce QPS limits across multi-threaded model calls."""

    def __init__(self, rate: float = 5.0, capacity: float = 5.0):
        """
        Args:
            rate: Token replenishment rate in tokens per second (target QPS).
            capacity: Maximum burst capacity of the token bucket.
        """
        self.rate = max(0.1, float(rate))
        self.capacity = max(1.0, float(capacity))
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """Blocks until the requested number of tokens can be consumed."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now

                # Replenish tokens based on elapsed time
                self.tokens = min(self.capacity, self.tokens + (elapsed * self.rate))

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Calculate wait time needed for next token
                needed = tokens - self.tokens
                wait_seconds = needed / self.rate

            # Sleep outside the lock so other threads can proceed
            time.sleep(min(wait_seconds, 0.5))
