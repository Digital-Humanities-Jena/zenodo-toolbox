from collections import deque
from contextlib import contextmanager
import logging
import sqlite3
import time
import threading
from typing import Generator, Tuple

logger = logging.getLogger("zenodo-toolbox")


class RateLimiter:
    """
    A class to implement rate limiting for API requests.

    Args:
        rate_per_min (int): Maximum number of requests allowed per minute.
        rate_per_hour (int): Maximum number of requests allowed per hour.
    """

    def __init__(self, rate_per_min, rate_per_hour):
        self.rate_per_min = rate_per_min
        self.rate_per_hour = rate_per_hour
        self.requests_per_minute = deque(maxlen=rate_per_min)
        self.requests_per_hour = deque(maxlen=rate_per_hour)
        self.lock = threading.Lock()

    def wait_for_rate_limit(self) -> None:
        """
        Checks and enforces rate limits, waiting if necessary.

        This method checks both per-minute and per-hour rate limits and
        waits if either limit has been reached.
        """
        with self.lock:
            now = time.time()

            # Check if rate limit per minute has been reached
            if len(self.requests_per_minute) == self.requests_per_minute.maxlen:
                if now - self.requests_per_minute[0] < 60:
                    wait_time = 60 - (now - self.requests_per_minute[0])
                    logger.info(f"Per-Minute Rate Limit reached. Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

            # Check if rate limit per hour has been reached
            if len(self.requests_per_hour) == self.requests_per_hour.maxlen:
                if now - self.requests_per_hour[0] < 3600:
                    wait_time = 3600 - (now - self.requests_per_hour[0])
                    logger.info(f"Per-Hour Rate Limit reached. Waiting for {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

    def record_request(self) -> None:
        """
        Records a new request in both per-minute and per-hour tracking deques.
        """
        with self.lock:
            now = time.time()
            self.requests_per_minute.append(now)
            self.requests_per_hour.append(now)


class RateLimiterParallel:
    """A rate limiter that enforces per-minute and per-hour request limits using SQLite for parallel processing."""

    def __init__(self, rate_per_min: int, rate_per_hour: int, db_path: str = "rate_limiter.db") -> None:
        """
        Initialize the RateLimiterParallel.

        Args:
            rate_per_min: Maximum number of requests allowed per minute.
            rate_per_hour: Maximum number of requests allowed per hour.
            db_path: Path to the SQLite database file.
        """
        self.rate_per_min = rate_per_min
        self.rate_per_hour = rate_per_hour
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database and create necessary tables and indexes."""
        with self._get_db_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    timestamp REAL
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)")

    @contextmanager
    def _get_db_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Create and yield a database connection.

        Returns:
            A context manager yielding a SQLite connection.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    def wait_for_rate_limit(self) -> None:
        while True:
            minute_count, hour_count, earliest_minute, earliest_hour = self._get_counts_and_earliest_timestamp()
            now = time.time()

            if minute_count >= self.rate_per_min:
                wait_time = earliest_minute + 60 - now
            elif hour_count >= self.rate_per_hour:
                wait_time = earliest_hour + 3600 - now
            else:
                break

            if wait_time > 0:
                time.sleep(wait_time)
            else:
                # If wait_time is negative, it means the window has passed
                # but we need to recheck the counts
                continue

    def record_request(self) -> None:
        """Record a new request in the database."""
        with self._get_db_connection() as conn:
            now = time.time()
            conn.execute("INSERT INTO requests (timestamp) VALUES (?)", (now,))
            conn.commit()

    def clean_old_records(self) -> None:
        """Remove records older than one hour from the database."""
        with self._get_db_connection() as conn:
            hour_ago = time.time() - 3600
            conn.execute("DELETE FROM requests WHERE timestamp <= ?", (hour_ago,))
            conn.commit()

    def _get_counts_and_earliest_timestamp(self) -> Tuple[int, int, float, float]:
        """
        Get current request counts and earliest timestamp for both minute and hour windows.
        """
        with self._get_db_connection() as conn:
            now = time.time()
            minute_ago = now - 60
            hour_ago = now - 3600

            minute_data = conn.execute(
                "SELECT COUNT(*), MIN(timestamp) FROM requests WHERE timestamp > ?", (minute_ago,)
            ).fetchone()
            hour_data = conn.execute(
                "SELECT COUNT(*), MIN(timestamp) FROM requests WHERE timestamp > ?", (hour_ago,)
            ).fetchone()

            return (minute_data[0], hour_data[0], minute_data[1] or now, hour_data[1] or now)
