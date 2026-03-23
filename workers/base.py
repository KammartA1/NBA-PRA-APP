"""
workers/base.py
===============
Abstract base class for all background workers.

Provides:
  - Retry logic with exponential backoff
  - Automatic WorkerStatus tracking in the database
  - Structured logging
  - One-shot and scheduled execution modes
"""

from __future__ import annotations

import abc
import json
import logging
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from database.connection import session_scope, get_session, init_db
from database.models import WorkerStatus

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BaseWorker(abc.ABC):
    """Base class every worker inherits from.

    Subclasses MUST implement :meth:`execute`.

    Parameters
    ----------
    name : str
        Unique worker name (used as key in ``worker_status`` table).
    interval_seconds : int
        Default scheduling interval.
    max_retries : int
        How many times to retry a failed ``execute()`` before giving up.
    retry_delay : float
        Base delay in seconds between retries (doubles each attempt).
    """

    def __init__(
        self,
        name: str,
        interval_seconds: int = 300,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:
        self.name = name
        self.interval_seconds = interval_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(f"workers.{name}")
        self._last_run_ok: bool = False

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Run the worker's core logic.

        Returns a dict with at least ``{"ok": bool, ...}`` plus any extra
        diagnostic keys the subclass wants to persist in worker_status
        metadata.
        """
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the worker with retry/error handling and status logging.

        This is the entry point for both one-shot and scheduled modes.
        Returns the result dict from :meth:`execute` (or an error dict).
        """
        self.logger.info("[%s] Starting run", self.name)
        self._update_status("running")

        last_error: Optional[str] = None
        result: Dict[str, Any] = {"ok": False}

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.execute()
                if result.get("ok"):
                    self.logger.info(
                        "[%s] Completed successfully on attempt %d",
                        self.name,
                        attempt,
                    )
                    self._last_run_ok = True
                    self._update_status(
                        "idle",
                        last_success=_utcnow(),
                        metadata=result,
                    )
                    return result
                # execute returned ok=False without raising
                last_error = result.get("error", "execute() returned ok=False")
                self.logger.warning(
                    "[%s] Attempt %d returned ok=False: %s",
                    self.name,
                    attempt,
                    last_error,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                self.logger.error(
                    "[%s] Attempt %d raised %s: %s",
                    self.name,
                    attempt,
                    type(exc).__name__,
                    exc,
                )

            # Retry backoff
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** (attempt - 1))
                self.logger.info(
                    "[%s] Retrying in %.1fs (attempt %d/%d)",
                    self.name,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)

        # All retries exhausted
        self._last_run_ok = False
        self._update_status(
            "error",
            last_error=last_error,
            metadata={"error": last_error, "retries_exhausted": True},
        )
        self.logger.error(
            "[%s] All %d retries exhausted. Last error: %s",
            self.name,
            self.max_retries,
            (last_error or "unknown")[:500],
        )
        result["ok"] = False
        result["error"] = last_error
        return result

    def run_once(self) -> Dict[str, Any]:
        """Alias for :meth:`run` -- explicit one-shot entry point."""
        return self.run()

    def run_forever(self, stop_event=None) -> None:
        """Blocking loop: run, sleep for ``interval_seconds``, repeat.

        Parameters
        ----------
        stop_event : threading.Event | None
            If provided, the loop exits when the event is set.
        """
        self.logger.info(
            "[%s] Entering run_forever loop (interval=%ds)",
            self.name,
            self.interval_seconds,
        )
        while True:
            self.run()
            if stop_event and stop_event.is_set():
                self.logger.info("[%s] Stop event received, exiting loop", self.name)
                break
            # Interruptible sleep
            waited = 0
            while waited < self.interval_seconds:
                if stop_event and stop_event.is_set():
                    break
                chunk = min(5, self.interval_seconds - waited)
                time.sleep(chunk)
                waited += chunk

    # ------------------------------------------------------------------
    # Database status tracking
    # ------------------------------------------------------------------

    def _update_status(
        self,
        status: str,
        last_success: Optional[datetime] = None,
        last_error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write the worker's current state to the ``worker_status`` table."""
        try:
            with session_scope() as session:
                ws = (
                    session.query(WorkerStatus)
                    .filter(WorkerStatus.worker_name == self.name)
                    .first()
                )
                if ws is None:
                    ws = WorkerStatus(worker_name=self.name)
                    session.add(ws)

                ws.status = status
                ws.last_run = _utcnow()
                ws.next_scheduled_run = _utcnow() + timedelta(
                    seconds=self.interval_seconds
                )

                if last_success is not None:
                    ws.last_success = last_success
                if last_error is not None:
                    ws.last_error = last_error
                elif status in ("idle", "running"):
                    ws.last_error = None

                if metadata is not None:
                    # Merge with existing metadata to avoid losing old keys
                    try:
                        existing = json.loads(ws.metadata_json or "{}")
                    except (json.JSONDecodeError, TypeError):
                        existing = {}
                    # Serialize datetimes in metadata values
                    safe_meta = {}
                    for k, v in metadata.items():
                        if isinstance(v, datetime):
                            safe_meta[k] = v.isoformat()
                        elif isinstance(v, (dict, list)):
                            try:
                                json.dumps(v)
                                safe_meta[k] = v
                            except (TypeError, ValueError):
                                safe_meta[k] = str(v)
                        else:
                            safe_meta[k] = v
                    existing.update(safe_meta)
                    ws.metadata_json = json.dumps(existing, default=str)

        except Exception:
            self.logger.warning(
                "[%s] Failed to update worker_status row",
                self.name,
                exc_info=True,
            )

    def update_status(
        self,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Public helper so subclasses can write ad-hoc status updates."""
        self._update_status(status, **kwargs)


def standalone_main(worker_cls, **kwargs):
    """Helper for ``if __name__ == '__main__'`` blocks.

    Initialises the database, instantiates the worker, and runs one-shot
    by default or enters the forever loop if ``--loop`` is passed.
    """
    import sys

    init_db()
    worker = worker_cls(**kwargs)

    if "--loop" in sys.argv:
        import signal as _sig
        import threading

        stop = threading.Event()

        def _handler(signum, frame):
            worker.logger.info("Signal %d received, stopping...", signum)
            stop.set()

        _sig.signal(_sig.SIGINT, _handler)
        _sig.signal(_sig.SIGTERM, _handler)
        worker.run_forever(stop_event=stop)
    else:
        result = worker.run()
        if not result.get("ok"):
            sys.exit(1)
