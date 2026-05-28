"""
Shared CLI runtime: append-only run receipt log and console progress display.

All top-level utilities (``fit_protein_mesh``, ``parameterize_mesh``,
``orientation_sample``, ``visualize_patches``) append to the same log file by
default so successive runs form a chronological receipt.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PACKAGE_DIR.parent
DEFAULT_RUN_LOG = _PROJECT_DIR / "anisotropy_run.log"


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``--log-file``, ``--log-overwrite``, ``--verbose-console``."""
    g = parser.add_argument_group("Run receipt log")
    g.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=f"Append run log here (default: {DEFAULT_RUN_LOG.name} in project root)",
    )
    g.add_argument(
        "--log-overwrite",
        action="store_true",
        help="Truncate the log file at start instead of appending",
    )
    g.add_argument(
        "--verbose-console",
        action="store_true",
        help="Mirror log lines on stdout (default: progress bar only during run)",
    )


class CalculationProgress:
    """
    Console-only progress display.

    Call :meth:`begin` when a calculation **starts**; :meth:`complete` when it
    finishes. Uses ``tqdm`` when installed.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._started = 0
        self._completed = 0
        self._tqdm = None
        if not self.enabled:
            return
        try:
            from tqdm import tqdm  # type: ignore

            self._tqdm = tqdm(
                total=None,
                desc="Initializing",
                unit="calc",
                dynamic_ncols=True,
                bar_format=(
                    "{desc}: |{bar}| {n_fmt}{unit} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
                leave=True,
            )
        except Exception:
            self._tqdm = None

    def set_total(self, total: int | None) -> None:
        if self._tqdm is not None and total is not None and total > 0:
            self._tqdm.reset(total=int(total))

    def begin(self, label: str) -> None:
        if not self.enabled:
            return
        self._started += 1
        text = f"{label} (#{self._started})"
        if len(text) > 76:
            text = text[:73] + "..."
        if self._tqdm is not None:
            self._tqdm.set_description_str(text)
            self._tqdm.refresh()
        else:
            print(f"\r{text}", end="", flush=True)

    def complete(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self._completed += int(n)
        if self._tqdm is not None:
            if self._tqdm.total is None or self._tqdm.total == 0:
                self._tqdm.total = max(self._tqdm.n + int(n), self._completed)
            self._tqdm.update(int(n))
        elif self._tqdm is None:
            print(f"\r  done {self._completed}", end="", flush=True)

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
        elif self.enabled and self._started > 0:
            print()


class RunSession:
    """
    Append-only run receipt + optional verbose console mirror.

    Use as a context manager around ``main()`` body.
    """

    def __init__(
        self,
        utility: str,
        *,
        log_path: Path | None = None,
        overwrite: bool = False,
        verbose_console: bool = False,
        argv: list[str] | None = None,
    ) -> None:
        self.utility = utility
        self.log_path = Path(log_path or DEFAULT_RUN_LOG)
        self.overwrite = bool(overwrite)
        self.verbose_console = bool(verbose_console)
        self.argv = list(argv if argv is not None else sys.argv)
        self.progress = CalculationProgress(enabled=True)
        self._fp = None
        self._opened_at: datetime | None = None

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace, utility: str) -> RunSession:
        log_path = getattr(args, "log_file", None) or DEFAULT_RUN_LOG
        return cls(
            utility,
            log_path=Path(log_path),
            overwrite=bool(getattr(args, "log_overwrite", False)),
            verbose_console=bool(getattr(args, "verbose_console", False)),
        )

    def log(self, message: str) -> None:
        """Write a timestamped line to the receipt log (and console if verbose)."""
        stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        line = f"{stamp}  {message}"
        if self._fp is not None:
            self._fp.write(line + "\n")
            self._fp.flush()
        if self.verbose_console:
            print(message)

    def log_block(self, title: str, body: str) -> None:
        self.log(f"--- {title} ---")
        for row in body.rstrip().splitlines():
            self.log(row)

    def __enter__(self) -> RunSession:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.overwrite else "a"
        self._fp = self.log_path.open(mode, encoding="utf-8")
        self._opened_at = datetime.now(timezone.utc)
        sep = "=" * 72
        self._fp.write(f"\n{sep}\n")
        self.log(f"UTILITY: {self.utility}")
        self.log(f"PID: {os.getpid()}")
        self.log(f"CWD: {Path.cwd()}")
        self.log(f"LOG: {self.log_path.resolve()}")
        self.log(f"CMD: {' '.join(self.argv)}")
        if self.overwrite:
            self.log("LOG_MODE: overwrite (truncated at start)")
        else:
            self.log("LOG_MODE: append")
        self._fp.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.log(f"STATUS: FAILED ({exc_type.__name__}: {exc_val})")
            self.log_block("TRACEBACK", traceback.format_exc())
        else:
            self.log("STATUS: OK")
        if self._opened_at is not None:
            elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
            self.log(f"ELAPSED_S: {elapsed:.2f}")
        self.progress.close()
        if self._fp is not None:
            self._fp.write("\n")
            self._fp.flush()
            self._fp.close()
            self._fp = None
        if exc_type is None:
            print(f"Done. Run receipt appended to {self.log_path.resolve()}")
        else:
            print(
                f"Failed ({exc_type.__name__}). Details in {self.log_path.resolve()}",
                file=sys.stderr,
            )
        return False


@contextmanager
def task_step(
    run: RunSession,
    label: str,
    *,
    steps: int = 1,
) -> Iterator[None]:
    """Context manager: ``begin`` on enter, ``complete`` on exit."""
    run.progress.begin(label)
    try:
        yield
    finally:
        run.progress.complete(steps)


def run_main(
    utility: str,
    args: argparse.Namespace,
    main_fn: Callable[[argparse.Namespace, RunSession], None],
) -> None:
    """Standard entry wrapper for CLI scripts."""
    with RunSession.from_cli_args(args, utility) as run:
        main_fn(args, run)
