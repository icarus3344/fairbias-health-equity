"""Execution guardrail enforcing data isolation, zero-network, and subprocess containment for FairBias R1A-R1."""

import builtins
import datetime
import hashlib
import io
import json
import os
import pathlib
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Enforce no bytecode generation and no user site
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"


class GuardViolationError(PermissionError):
    """Raised when an operation violates FairBias R1 guard constraints."""
    pass


class FairBiasR1Guard:
    """Installed before test runner or project imports to enforce strict containment."""

    _instance: Optional["FairBiasR1Guard"] = None

    def __init__(self, output_dir: Union[str, pathlib.Path], repo_root: Optional[pathlib.Path] = None):
        self.output_dir = pathlib.Path(output_dir).resolve()
        self.repo_root = (repo_root or pathlib.Path(__file__).resolve().parents[1]).resolve()
        self.events: List[Dict[str, Any]] = []
        self.is_installed = False

        # Metrics and budgets
        self.total_events = 0
        self.denied_events = 0
        # Metrics and budgets
        self.total_events = 0
        self.denied_events = 0
        self.allowed_events = 0
        self.truncated_events = 0
        self.max_memory_events = 10000
        self.max_disk_bytes = 50 * 1024 * 1024  # 50 MB
        self.written_bytes = 0

        # Health and expected denial tracking (R2-03, R7-02)
        self.log_healthy = True
        self.log_errors: List[str] = []
        self.disk_events_lost = 0
        self.memory_window_dropped = 0
        self.unexpected_denials: List[Dict[str, Any]] = []
        self.expected_denial_patterns: List[Dict[str, Any]] = []
        self.consumed_expected_denials: List[Dict[str, Any]] = []
        self._closing_log_internally = False
        self._internal_log_operation = False

        # Route temporary directory strictly into output_dir
        self.tmp_dir = (self.output_dir / "tmp").resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(self.tmp_dir)
        os.environ["TEMP"] = str(self.tmp_dir)
        os.environ["TMP"] = str(self.tmp_dir)
        import tempfile
        tempfile.tempdir = str(self.tmp_dir)

        # Pre-open log file descriptor before installing audit hooks to prevent open recursion
        self.log_file = (self.output_dir / "guard_events.jsonl").resolve()
        self._log_fd = os.open(
            str(self.log_file),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        self.fd_registry: Dict[int, Dict[str, Any]] = {
            0: {"path": pathlib.Path("/dev/stdin"), "mode": "r", "origin": "system"},
            1: {"path": pathlib.Path("/dev/stdout"), "mode": "w", "origin": "system"},
            2: {"path": pathlib.Path("/dev/stderr"), "mode": "w", "origin": "system"},
            self._log_fd: {"path": self.log_file, "mode": "w", "origin": "logger"},
        }
        self.registered_fds = set(self.fd_registry.keys())

        # Runtime platform roots: only Framework Python installation and system libraries
        py_base = pathlib.Path(sys.base_prefix).resolve()
        self.allowed_read_roots = [
            self.output_dir,
            py_base,
            pathlib.Path("/usr/lib").resolve(),
            pathlib.Path("/System/Library").resolve(),
        ]

        # Explicit whitelisted platform files (e.g. system timezone and devices)
        self.allowed_platform_files = {
            pathlib.Path("/dev/null").resolve(),
            pathlib.Path("/dev/urandom").resolve(),
            pathlib.Path("/dev/random").resolve(),
            pathlib.Path("/private/var/db/timezone/tz/2026c.1.0/zoneinfo/UTC").resolve(),
            pathlib.Path("/usr/share/zoneinfo/UTC").resolve(),
        }

        # Explicit whitelisted repo configurations
        self.allowed_repo_configs = {
            (self.repo_root / "configs" / "nhis" / "features.json").resolve(),
            (self.repo_root / "configs" / "nhis" / "study.json").resolve(),
        }

        # Forbidden targets
        self.forbidden_repo_dirs = [
            (self.repo_root / "data").resolve(),
            (self.repo_root / "docs" / "releases").resolve(),
            (self.repo_root / "docs" / "plans").resolve(),
            (self.repo_root / "outputs").resolve(),
            (self.repo_root / "artifacts").resolve(),
        ]
        self.forbidden_filenames = {
            "data_compas.csv",
            "data_credit_card.csv",
        }

        # Save original builtins/methods
        self._orig_builtin_open = builtins.open
        self._orig_io_open = io.open
        self._orig_path_open = pathlib.Path.open
        self._orig_os_open = os.open
        self._orig_os_close = os.close
        self._orig_socket_connect = socket.socket.connect
        self._orig_socket_connect_ex = socket.socket.connect_ex
        self._orig_popen = subprocess.Popen

    def close_log_sink(self) -> None:
        """Safely close log file descriptor."""
        if hasattr(self, "_log_fd") and self._log_fd is not None:
            self._closing_log_internally = True
            self._internal_log_operation = True
            try:
                self._orig_os_close(self._log_fd)
            except Exception as exc:
                self.log_healthy = False
                self.log_errors.append(f"Failed to close log sink: {exc}")
            finally:
                if self._log_fd in self.fd_registry:
                    del self.fd_registry[self._log_fd]
                self.registered_fds = set(self.fd_registry.keys())
                self._log_fd = None
                self._internal_log_operation = False
                self._closing_log_internally = False

    def expect_denial(
        self,
        action: Optional[str] = None,
        target_pattern: Optional[str] = None,
        reason_pattern: Optional[str] = None,
        count: int = 1,
        match_mode: str = "exact",
    ) -> Any:
        """Context manager to register short-lived expected denial declarations.

        match_mode controls target matching:
        - "exact": exact string or normalized path equality (default). Subpaths and cross-directory basenames do NOT match.
        - "directory": target is inside or equal to target_pattern directory.
        - "basename": os.path.basename(target) == target_pattern.
        """
        guard = self

        class ExpectedDenialContext:
            def __init__(self) -> None:
                self.entry = {
                    "action": action,
                    "target_pattern": target_pattern,
                    "reason_pattern": reason_pattern,
                    "expected": count,
                    "remaining": count,
                    "match_mode": match_mode,
                }

            def __enter__(self) -> "ExpectedDenialContext":
                guard.expected_denial_patterns.append(self.entry)
                return self

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
                if self.entry in guard.expected_denial_patterns:
                    guard.expected_denial_patterns.remove(self.entry)
                if self.entry["remaining"] != 0:
                    raise AssertionError(
                        f"Expected denial contract violated: expected {self.entry['expected']} denials "
                        f"for action={self.entry['action']}, target_pattern={self.entry['target_pattern']}, "
                        f"but remaining count is {self.entry['remaining']}."
                    )
                return False

        return ExpectedDenialContext()

    def log_event(self, action: str, target: str, decision: str, reason: str) -> None:
        self.total_events += 1
        if decision == "DENIED":
            self.denied_events += 1
            # Check against active expected denial declarations
            matched = False
            for exp in self.expected_denial_patterns:
                if exp.get("remaining", 0) > 0:
                    act = exp.get("action")
                    act_match = (
                        act is None
                        or act == action
                        or (act in ("os.unlink", "os.remove") and action in ("os.unlink", "os.remove"))
                    )
                    t_pat = exp.get("target_pattern")
                    m_mode = exp.get("match_mode", "exact")
                    if t_pat is None:
                        tgt_match = True
                    else:
                        p_str = str(t_pat)
                        t_str = str(target)
                        p_norm = os.path.normpath(p_str)
                        t_norm = os.path.normpath(t_str)
                        if m_mode == "directory":
                            tgt_match = (
                                t_norm == p_norm
                                or t_norm.startswith(f"{p_norm.rstrip('/')}/")
                            )
                        elif m_mode == "basename":
                            tgt_match = (os.path.basename(t_str) == p_str or os.path.basename(t_norm) == p_str)
                        elif m_mode == "prefix":
                            tgt_match = t_str.startswith(p_str)
                        elif m_mode == "exact":
                            tgt_match = (
                                p_str == t_str
                                or p_norm == t_norm
                                or (not os.path.isabs(p_str) and os.path.normpath(os.path.join(str(self.repo_root), p_str)) == t_norm)
                            )
                        else:
                            tgt_match = False
                    r_pat = exp.get("reason_pattern")
                    rsn_match = (
                        r_pat is None
                        or (str(r_pat) == str(reason))
                    )
                    if act_match and tgt_match and rsn_match:
                        exp["remaining"] -= 1
                        matched = True
                        self.consumed_expected_denials.append({
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "action": action,
                            "target": str(target),
                            "decision": "DENIED",
                            "reason": reason,
                            "expected_action": exp.get("action"),
                            "expected_target_pattern": exp.get("target_pattern"),
                            "expected_reason_pattern": exp.get("reason_pattern"),
                            "match_mode": exp.get("match_mode", "exact"),
                        })
                        break
            if not matched:
                self.unexpected_denials.append({
                    "action": action,
                    "target": str(target),
                    "reason": reason,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
        else:
            self.allowed_events += 1

        rec = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": action,
            "target": str(target),
            "decision": decision,
            "reason": reason,
        }

        if len(self.events) < self.max_memory_events:
            self.events.append(rec)
        else:
            self.memory_window_dropped += 1
            self.truncated_events += 1

        # Disk write with strict error handling and budget enforcement
        if self._log_fd is None:
            self.log_healthy = False
            self.log_errors.append("Log event attempted while log descriptor is closed")
            self.disk_events_lost += 1
        elif self.written_bytes >= self.max_disk_bytes:
            self.log_healthy = False
            self.log_errors.append(f"Disk budget exhausted: {self.written_bytes} >= {self.max_disk_bytes}")
            self.disk_events_lost += 1
            self.truncated_events += 1
        else:
            try:
                line = json.dumps(rec) + "\n"
                b = line.encode("utf-8")
                if self.written_bytes + len(b) > self.max_disk_bytes:
                    self.log_healthy = False
                    self.log_errors.append(f"Disk budget exceeded by write: {self.written_bytes + len(b)} > {self.max_disk_bytes}")
                    self.disk_events_lost += 1
                    self.truncated_events += 1
                else:
                    self._internal_log_operation = True
                    try:
                        written = os.write(self._log_fd, b)
                    finally:
                        self._internal_log_operation = False
                    if written < len(b):
                        self.log_healthy = False
                        self.log_errors.append(f"Partial disk write: {written}/{len(b)} bytes written")
                        self.disk_events_lost += 1
                    else:
                        self.written_bytes += written
            except Exception as exc:
                self.log_healthy = False
                self.log_errors.append(f"Disk write exception: {exc}")
                self.disk_events_lost += 1

    def check_path_access(
        self,
        path: Union[str, bytes, os.PathLike, int],
        mode: str = "r",
        flags: int = 0,
        dir_fd: Optional[int] = None,
    ) -> pathlib.Path:
        """Check if file access to path is permitted under guard policy."""
        # Check non-default dir_fd
        if dir_fd is not None and dir_fd != -1:
            msg = f"Non-default dir_fd ({dir_fd}) blocked by FairBias R1 guard"
            self.log_event("dir_fd_access", f"dir_fd={dir_fd}", "DENIED", msg)
            raise GuardViolationError(msg)

        # 1. Check file descriptors
        if isinstance(path, int):
            if path in self.fd_registry:
                reg_info = self.fd_registry[path]
                if path == self._log_fd or reg_info.get("origin") == "logger":
                    if not getattr(self, "_internal_log_operation", False):
                        msg = f"Direct access to guard log descriptor {path} blocked: log sink is protected"
                        self.log_event("log_fd_access", f"fd={path}", "DENIED", msg)
                        raise GuardViolationError(msg)
                # Check write on read-only registered fd
                is_w = any(m in mode for m in ("w", "a", "x", "+")) or bool(
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)
                )
                if is_w and reg_info.get("mode") == "r":
                    msg = f"Write attempted on read-only registered fd {path}"
                    self.log_event("fd_write_violation", str(path), "DENIED", msg)
                    raise GuardViolationError(msg)
                return reg_info["path"]
            msg = f"Unregistered file descriptor {path} denied by guard"
            self.log_event("unregistered_fd", str(path), "DENIED", msg)
            raise GuardViolationError(msg)

        try:
            p_str = os.fsdecode(path)
            p = pathlib.Path(p_str).resolve()
        except Exception as exc:
            self.log_event("path_resolve", str(path), "DENIED", f"Path resolution failed: {exc}")
            raise GuardViolationError(f"Guard rejected unresolvable path: {path}") from exc

        # Check write permissions
        is_write = any(m in mode for m in ("w", "a", "x", "+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)
        )

        # Guard log sink protection: external code cannot truncate, overwrite, or mutate the log file
        if p == self.log_file:
            if is_write:
                msg = f"Direct write/truncate access to guard log file {p} blocked: log sink is protected"
                self.log_event("file_write", str(p), "DENIED", msg)
                raise GuardViolationError(msg)
            return p

        if is_write:
            if p == pathlib.Path("/dev/null").resolve():
                self.log_event("file_write", str(p), "ALLOWED", "Write target is /dev/null")
                return p
            try:
                p.relative_to(self.output_dir)
                self.log_event("file_write", str(p), "ALLOWED", "Write target is within output_dir")
                return p
            except ValueError:
                msg = f"Write access to {p} blocked: writes permitted strictly within output_dir ({self.output_dir})"
                self.log_event("file_write", str(p), "DENIED", msg)
                raise GuardViolationError(msg)

        # Check read permissions: check forbidden rules first
        p_name_lower = p.name.lower()
        if p_name_lower in self.forbidden_filenames:
            msg = f"Read access to baseline dataset {p.name} blocked by guard"
            self.log_event("file_read", str(p), "DENIED", msg)
            raise GuardViolationError(msg)

        if p.suffix.lower() == ".parquet":
            msg = f"Read access to parquet file {p} blocked by guard"
            self.log_event("file_read", str(p), "DENIED", msg)
            raise GuardViolationError(msg)

        for f_dir in self.forbidden_repo_dirs:
            try:
                p.relative_to(f_dir)
                msg = f"Read access to forbidden directory {f_dir.name} ({p}) blocked by guard"
                self.log_event("file_read", str(p), "DENIED", msg)
                raise GuardViolationError(msg)
            except ValueError:
                pass

        # Forbidden runs/ directory check (except our current output_dir)
        runs_dir = (self.repo_root / "runs").resolve()
        try:
            p.relative_to(runs_dir)
            try:
                p.relative_to(self.output_dir)
            except ValueError:
                msg = f"Read access to prior/external run artifacts ({p}) blocked by guard"
                self.log_event("file_read", str(p), "DENIED", msg)
                raise GuardViolationError(msg)
        except ValueError:
            pass

        # Check explicit platform devices and files
        if p in self.allowed_platform_files:
            self.log_event("file_read", str(p), "ALLOWED", "Read target in allowed_platform_files")
            return p

        # Disallow arbitrary /dev/ paths
        if str(p).startswith("/dev/"):
            msg = f"Read access to arbitrary device {p} blocked by guard"
            self.log_event("file_read", str(p), "DENIED", msg)
            raise GuardViolationError(msg)

        # Check output_dir
        try:
            p.relative_to(self.output_dir)
            self.log_event("file_read", str(p), "ALLOWED", "Read target within output_dir")
            return p
        except ValueError:
            pass

        # Check allowed platform roots (stdlib, framework, system libs)
        for r_root in self.allowed_read_roots:
            try:
                p.relative_to(r_root)
                self.log_event("file_read", str(p), "ALLOWED", f"Read target within {r_root}")
                return p
            except ValueError:
                pass

        # Check explicit config files
        if p in self.allowed_repo_configs:
            self.log_event("file_read", str(p), "ALLOWED", "Read target in allowed_repo_configs")
            return p

        # Check repo source directories: strict code-only whitelist
        try:
            p.relative_to(self.repo_root / "src")
            if p.suffix == ".py":
                self.log_event("file_read", str(p), "ALLOWED", "Allowed source in src/")
                return p
            msg = f"Non-code or bytecode file in src/ ({p}) blocked by guard: strictly .py source files permitted"
            self.log_event("file_read", str(p), "DENIED", msg)
            raise GuardViolationError(msg)
        except ValueError:
            pass

        try:
            p.relative_to(self.repo_root / "scripts")
            allowed_script_stems = ("_fairbias_r1_guard", "run_fairbias_r1_guarded_tests")
            if p.suffix == ".py" and any(p.stem.startswith(stem) for stem in allowed_script_stems):
                self.log_event("file_read", str(p), "ALLOWED", "Allowed script in scripts/")
                return p
            msg = f"Unapproved script, bytecode, or non-code file in scripts/ ({p}) blocked by guard"
            self.log_event("file_read", str(p), "DENIED", msg)
            raise GuardViolationError(msg)
        except ValueError:
            pass

        try:
            p.relative_to(self.repo_root / "tests" / "synthetic")
            if p.suffix == ".py":
                self.log_event("file_read", str(p), "ALLOWED", "Allowed test code in tests/synthetic/")
                return p
            msg = f"Non-code or bytecode file in tests/synthetic/ ({p}) blocked by guard"
            self.log_event("file_read", str(p), "DENIED", msg)
            raise GuardViolationError(msg)
        except ValueError:
            pass

        # Default deny for all other files
        msg = f"Read access to {p} blocked: target is not in FairBias R1 allowlist"
        self.log_event("file_read", str(p), "DENIED", msg)
        raise GuardViolationError(msg)

    def install(self) -> None:
        """Install audit hook and function interceptors."""
        if self.is_installed:
            return

        guard = self

        def check_dir_fd(dir_fd: Any, event_name: str) -> None:
            if dir_fd is not None and dir_fd != -1:
                msg = f"Non-default dir_fd ({dir_fd}) blocked by FairBias R1 guard"
                guard.log_event(event_name, f"dir_fd={dir_fd}", "DENIED", msg)
                raise GuardViolationError(msg)

        def verify_mutation_target(target: Any, event_name: str) -> None:
            if isinstance(target, int):
                if target not in guard.fd_registry:
                    msg = f"Mutation on unregistered integer fd {target} blocked by guard"
                    guard.log_event(event_name, str(target), "DENIED", msg)
                    raise GuardViolationError(msg)
                reg_info = guard.fd_registry[target]
                if reg_info.get("mode") != "w":
                    msg = f"Mutation on read-only fd {target} blocked by guard"
                    guard.log_event(event_name, str(target), "DENIED", msg)
                    raise GuardViolationError(msg)
                target_path = reg_info["path"]
                if target_path == guard.log_file:
                    msg = f"Direct mutation on guard log fd {target} blocked by guard"
                    guard.log_event(event_name, str(target), "DENIED", msg)
                    raise GuardViolationError(msg)
                try:
                    target_path.relative_to(guard.output_dir)
                except ValueError:
                    msg = f"Mutation on fd {target} outside output_dir blocked by guard"
                    guard.log_event(event_name, str(target), "DENIED", msg)
                    raise GuardViolationError(msg)
            else:
                try:
                    p_res = pathlib.Path(os.fsdecode(target)).resolve()
                    if p_res == guard.log_file:
                        msg = f"Direct mutation of guard log file {p_res} blocked"
                        guard.log_event(event_name, str(target), "DENIED", msg)
                        raise GuardViolationError(msg)
                    p_res.relative_to(guard.output_dir)
                except Exception:
                    guard.log_event(event_name, str(target), "DENIED", f"File operation {event_name} outside output_dir blocked")
                    raise GuardViolationError(f"File operation {event_name} outside output_dir blocked: {target}")

        # 1. Audit hook covering network, subprocess, mutating file actions, and open
        def audit_hook(event: str, args: Tuple[Any, ...]) -> None:
            # Network events
            if event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo"):
                guard.log_event(event, str(args), "DENIED", f"Network operation {event} blocked by guard")
                raise GuardViolationError(f"Network operation {event} blocked by FairBias R1 guard")

            # Subprocess and execution events
            if (
                event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.fork", "os.forkpty", "os.exec")
                or event.startswith("os.exec")
            ):
                guard.log_event(event, str(args), "DENIED", f"Process execution {event} blocked by guard")
                raise GuardViolationError(f"Subprocess/exec call {event} blocked by FairBias R1 guard")

            # File mutation events outside output_dir
            if event in ("os.remove", "os.rmdir", "os.unlink"):
                dir_fd = args[1] if len(args) > 1 else None
                check_dir_fd(dir_fd, event)
                verify_mutation_target(args[0], event)

            if event in ("os.mkdir", "os.chmod"):
                dir_fd = args[2] if len(args) > 2 else None
                check_dir_fd(dir_fd, event)
                verify_mutation_target(args[0], event)

            if event == "os.utime":
                dir_fd = args[3] if len(args) > 3 else None
                check_dir_fd(dir_fd, event)
                verify_mutation_target(args[0], event)

            if event == "os.truncate":
                verify_mutation_target(args[0], event)

            if event in ("os.rename", "os.replace", "os.link"):
                src_dir_fd = args[2] if len(args) > 2 else None
                dst_dir_fd = args[3] if len(args) > 3 else None
                check_dir_fd(src_dir_fd, event)
                check_dir_fd(dst_dir_fd, event)
                for p_arg in args[:2]:
                    verify_mutation_target(p_arg, event)

            if event == "os.symlink":
                dir_fd = args[2] if len(args) > 2 else None
                check_dir_fd(dir_fd, event)
                for p_arg in args[:2]:
                    verify_mutation_target(p_arg, event)

            # Open events
            if event == "open":
                path = args[0]
                if path == guard._log_fd:
                    if not getattr(guard, "_internal_log_operation", False):
                        msg = f"Reopening guard log file descriptor {path} blocked: log sink is protected"
                        guard.log_event("log_fd_reopen", f"fd={path}", "DENIED", msg)
                        raise GuardViolationError(msg)
                    return
                mode = args[1] if len(args) > 1 else "r"
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                guard.check_path_access(path, mode=str(mode), flags=flags)

        try:
            sys.addaudithook(audit_hook)
        except Exception as exc:
            raise GuardViolationError(f"Failed to install audit hook: {exc}") from exc

        # 2. Intercept builtins.open
        def guarded_builtin_open(*args: Any, **kwargs: Any) -> Any:
            path = args[0] if args else kwargs.get("file")
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            guard.check_path_access(path, mode=str(mode))
            return guard._orig_builtin_open(*args, **kwargs)

        builtins.open = guarded_builtin_open

        # 3. Intercept io.open
        def guarded_io_open(*args: Any, **kwargs: Any) -> Any:
            path = args[0] if args else kwargs.get("file")
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            guard.check_path_access(path, mode=str(mode))
            return guard._orig_io_open(*args, **kwargs)

        io.open = guarded_io_open

        # 4. Intercept Path.open
        def guarded_path_open(self_path: pathlib.Path, *args: Any, **kwargs: Any) -> Any:
            mode = args[0] if args else kwargs.get("mode", "r")
            guard.check_path_access(self_path, mode=str(mode))
            return guard._orig_path_open(self_path, *args, **kwargs)

        pathlib.Path.open = guarded_path_open

        # 5. Intercept os.open
        def guarded_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            mode = "r"
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
                mode = "w"
            dir_fd = kwargs.get("dir_fd")
            resolved_p = guard.check_path_access(path, mode=mode, flags=flags, dir_fd=dir_fd)
            fd = guard._orig_os_open(path, flags, *args, **kwargs)
            guard.fd_registry[fd] = {"path": resolved_p, "mode": mode, "origin": "os.open"}
            guard.registered_fds.add(fd)
            return fd

        os.open = guarded_os_open

        # 6. Intercept os.close to maintain fd registry
        def guarded_os_close(fd: int) -> None:
            if fd == guard._log_fd and not getattr(guard, "_closing_log_internally", False):
                raise GuardViolationError("External closing of guard log descriptor blocked")
            if fd in guard.fd_registry and guard.fd_registry[fd].get("origin") != "logger":
                del guard.fd_registry[fd]
                guard.registered_fds = set(guard.fd_registry.keys())
            return guard._orig_os_close(fd)

        os.close = guarded_os_close

        # 6. Intercept socket.socket.connect / connect_ex
        def guarded_socket_connect(s_self: Any, address: Any) -> None:
            guard.log_event("socket.connect", str(address), "DENIED", "Network connection blocked by socket monkeypatch")
            raise GuardViolationError(f"Network connect to {address} blocked by FairBias R1 guard")

        def guarded_socket_connect_ex(s_self: Any, address: Any) -> int:
            guard.log_event("socket.connect_ex", str(address), "DENIED", "Network connection blocked by socket monkeypatch")
            raise GuardViolationError(f"Network connect_ex to {address} blocked by FairBias R1 guard")

        socket.socket.connect = guarded_socket_connect
        socket.socket.connect_ex = guarded_socket_connect_ex

        # 7. Intercept subprocess.Popen
        def guarded_popen(*args: Any, **kwargs: Any) -> Any:
            cmd = args[0] if args else kwargs.get("args")
            guard.log_event("subprocess.Popen", str(cmd), "DENIED", "Subprocess execution blocked by Popen monkeypatch")
            raise GuardViolationError(f"Subprocess Popen({cmd}) blocked by FairBias R1 guard")

        subprocess.Popen = guarded_popen

        self.is_installed = True
        FairBiasR1Guard._instance = self

    @classmethod
    def get_instance(cls) -> Optional["FairBiasR1Guard"]:
        return cls._instance


def run_sentinel_tests(guard: FairBiasR1Guard) -> Dict[str, Any]:
    """Execute synthetic security traps against guard routes and verify fail-closed rejections.

    Strict verification rules:
    - Never touch real baseline CSVs or real parquet files.
    - Only GuardViolationError (with recorded denial in guard) counts as PASSED_BLOCKED.
    - ConnectionRefused, FileNotFoundError, or generic OSError are NOT counted as blocked.
    """
    results: Dict[str, str] = {}
    sentinel_dir = guard.output_dir / "synthetic_sentinel_targets"
    sentinel_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sentinel: builtins.open blocked on synthetic forbidden CSV
    probe_1 = guard.repo_root / "scratch" / "synthetic_sentinel_forbidden.csv"
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_read", target_pattern=str(probe_1), count=1):
            with builtins.open(probe_1, "r"):
                pass
        results["sentinel_builtin_open"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_builtin_open"] = "PASSED_BLOCKED"
        else:
            results["sentinel_builtin_open"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_builtin_open"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 2. Sentinel: io.open blocked on synthetic parquet path
    probe_2 = guard.repo_root / "data" / "synthetic_probe.parquet"
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_read", target_pattern=str(probe_2), count=1):
            with io.open(probe_2, "r"):
                pass
        results["sentinel_io_open"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_io_open"] = "PASSED_BLOCKED"
        else:
            results["sentinel_io_open"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_io_open"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 3. Sentinel: Path.open blocked on release directory
    probe_3 = guard.repo_root / "docs" / "releases" / "synthetic_probe.json"
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_read", target_pattern=str(probe_3), count=1):
            with probe_3.open("r"):
                pass
        results["sentinel_path_open"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_path_open"] = "PASSED_BLOCKED"
        else:
            results["sentinel_path_open"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_path_open"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 4. Sentinel: os.open blocked on non-code file in scripts/
    probe_4 = guard.repo_root / "scripts" / "synthetic_not_source.csv"
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_read", target_pattern=str(probe_4), count=1):
            fd = os.open(str(probe_4), os.O_RDONLY)
            os.close(fd)
        results["sentinel_os_open"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_os_open"] = "PASSED_BLOCKED"
        else:
            results["sentinel_os_open"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_os_open"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 5. Sentinel: os.open with O_TRUNC write outside output_dir
    probe_5 = guard.repo_root / "scratch" / "synthetic_probe_trunc.txt"
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_write", target_pattern=str(probe_5), count=1):
            fd = os.open(str(probe_5), os.O_WRONLY | os.O_TRUNC)
            os.close(fd)
        results["sentinel_os_open_w_trunc"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_os_open_w_trunc"] = "PASSED_BLOCKED"
        else:
            results["sentinel_os_open_w_trunc"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_os_open_w_trunc"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 6. Sentinel: path traversal escape attempt
    probe_6 = (guard.output_dir / ".." / ".." / "scratch" / "synthetic_traversal.csv").resolve()
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_read", target_pattern=str(probe_6), count=1):
            with builtins.open(probe_6, "r"):
                pass
        results["sentinel_path_traversal"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_path_traversal"] = "PASSED_BLOCKED"
        else:
            results["sentinel_path_traversal"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_path_traversal"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 7. Sentinel: arbitrary integer file descriptor
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="unregistered_fd", target_pattern="99999", count=1):
            guard.check_path_access(99999)
        results["sentinel_unregistered_fd"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_unregistered_fd"] = "PASSED_BLOCKED"
        else:
            results["sentinel_unregistered_fd"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_unregistered_fd"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 8. Sentinel: /dev/fd path bypass attempt
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_read", target_pattern="/dev/fd", match_mode="directory", count=1):
            guard.check_path_access("/dev/fd/99999")
        results["sentinel_dev_fd"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_dev_fd"] = "PASSED_BLOCKED"
        else:
            results["sentinel_dev_fd"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_dev_fd"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 9. Sentinel: Network socket connect attempt
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="socket.connect", count=1):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", 8080))
        results["sentinel_network_connect"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_network_connect"] = "PASSED_BLOCKED"
        else:
            results["sentinel_network_connect"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_network_connect"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 10. Sentinel: Subprocess Popen attempt
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="subprocess.Popen", count=1):
            subprocess.Popen(["echo", "probe"])
        results["sentinel_subprocess"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_subprocess"] = "PASSED_BLOCKED"
        else:
            results["sentinel_subprocess"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_subprocess"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 11. Sentinel: Write outside output_dir attempt
    probe_11 = guard.repo_root / "scratch" / "synthetic_probe_outside.txt"
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="file_write", target_pattern=str(probe_11), count=1):
            with builtins.open(probe_11, "w") as f:
                f.write("probe")
        results["sentinel_write_outside"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_write_outside"] = "PASSED_BLOCKED"
        else:
            results["sentinel_write_outside"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_write_outside"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 12. Sentinel: Symlink escape attempt
    link_target = guard.repo_root / "scratch" / "synthetic_target.csv"
    link_path = sentinel_dir / "escape_symlink.csv"
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="os.symlink", count=1):
            os.symlink(link_target, link_path)
        results["sentinel_symlink_escape"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        if guard.denied_events > initial_denials:
            results["sentinel_symlink_escape"] = "PASSED_BLOCKED"
        else:
            results["sentinel_symlink_escape"] = "FAILED_NO_DENIAL_LOGGED"
    except Exception as exc:
        results["sentinel_symlink_escape"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"

    # 13. Sentinel: Relative unlink with non-default dir_fd
    probe_13_file = sentinel_dir / "sentinel_dirfd_victim.txt"
    probe_13_file.write_text("Artificial sentinel; content must remain intact")
    orig_hash = hashlib.sha256(probe_13_file.read_bytes()).hexdigest()
    dir_fd = os.open(str(sentinel_dir), os.O_RDONLY | os.O_DIRECTORY)
    initial_denials = guard.denied_events
    try:
        with guard.expect_denial(action="os.unlink", target_pattern=f"dir_fd={dir_fd}", count=1):
            os.unlink("sentinel_dirfd_victim.txt", dir_fd=dir_fd)
        results["sentinel_dir_fd_unlink"] = "FAILED_NOT_BLOCKED"
    except GuardViolationError:
        file_intact = probe_13_file.exists() and hashlib.sha256(probe_13_file.read_bytes()).hexdigest() == orig_hash
        if guard.denied_events > initial_denials and file_intact:
            results["sentinel_dir_fd_unlink"] = "PASSED_BLOCKED"
        else:
            results["sentinel_dir_fd_unlink"] = "FAILED_MUTATED_OR_NO_DENIAL"
    except Exception as exc:
        results["sentinel_dir_fd_unlink"] = f"FAILED_WRONG_EXCEPTION: {type(exc).__name__}: {exc}"
    finally:
        os.close(dir_fd)

    return results
