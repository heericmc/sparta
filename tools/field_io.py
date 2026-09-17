"""Read complete frames from an append-only SPARTA grid dump."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np


class FieldFormatError(ValueError):
    """A completed frame is not a valid SPARTA grid dump frame."""


@dataclass(frozen=True)
class Frame:
    timestep: int
    bounds: tuple[tuple[float, float], ...]
    columns: tuple[str, ...]
    data: np.ndarray
    raw: str


@dataclass(frozen=True)
class FrameSet:
    frames: tuple[Frame, ...]
    ignored_incomplete_tail: bool


def _incomplete(path: Path, message: str, final: bool) -> None:
    if final:
        raise EOFError(message)
    raise FieldFormatError(f"{path}: {message}")


def _parse_frame(path: Path, raw: str, final: bool) -> tuple[Frame, bool]:
    partial_tail = False
    if final and not raw.endswith("\n"):
        # Before the full TIMESTEP line exists, the partial header belongs to
        # the preceding frame's slice.  Remove only a genuine prefix; arbitrary
        # ``ITEM: ...`` text remains a malformed completed frame.
        line_start = raw.rfind("\n") + 1
        final_line = raw[line_start:]
        header = "ITEM: TIMESTEP"
        if final_line and header.startswith(final_line):
            raw = raw[:line_start]
            partial_tail = True
        elif final_line.startswith("ITEM:"):
            # This is not a possible start of the next valid frame.  Let the
            # completed frame validation report it rather than hiding damage.
            pass
        else:
            # A short final cell value can look syntactically valid only after
            # conversion to float.  Its missing terminator is the evidence
            # that it remains an append in progress, so stop before validating
            # its row width or numeric contents.
            _incomplete(path, "frame ends in an unterminated final line", final)
    lines = raw.splitlines()
    if len(lines) < 2 or lines[0] != "ITEM: TIMESTEP":
        _incomplete(path, "missing ITEM: TIMESTEP header", final)
    try:
        step = int(lines[1])
    except ValueError as exc:
        raise FieldFormatError(f"{path}: invalid timestep {lines[1]!r}") from exc

    def header(name: str) -> int:
        try:
            return next(i for i, line in enumerate(lines) if line.startswith(name))
        except StopIteration:
            _incomplete(path, f"frame {step} is missing {name}", final)
            raise AssertionError("unreachable")

    ni = header("ITEM: NUMBER OF CELLS")
    bi = header("ITEM: BOX BOUNDS")
    ci = header("ITEM: CELLS")
    if not (ni < bi < ci):
        raise FieldFormatError(f"{path}: frame {step} has out-of-order headers")
    if ni + 1 >= len(lines):
        _incomplete(path, f"frame {step} lacks its cell count", final)
    try:
        ncells = int(lines[ni + 1])
    except ValueError as exc:
        raise FieldFormatError(f"{path}: frame {step} has invalid cell count") from exc
    if ncells < 0:
        raise FieldFormatError(f"{path}: frame {step} has negative cell count")

    bounds = []
    for line in lines[bi + 1:ci]:
        values = line.split()
        if len(values) != 2:
            raise FieldFormatError(f"{path}: frame {step} has malformed box bounds")
        try:
            bounds.append((float(values[0]), float(values[1])))
        except ValueError as exc:
            raise FieldFormatError(f"{path}: frame {step} has nonnumeric box bounds") from exc
    if not bounds:
        _incomplete(path, f"frame {step} has no box bounds", final)

    columns = tuple(lines[ci].split()[2:])
    if not columns:
        _incomplete(path, f"frame {step} has no cell columns", final)
    row_indices = [i for i, line in enumerate(lines[ci + 1:], ci + 1) if line.strip()]
    rows = [lines[i].split() for i in row_indices]
    if len(rows) < ncells:
        _incomplete(path, f"frame {step} declares {ncells} cells but has {len(rows)} rows", final)
    if len(rows) != ncells:
        raise FieldFormatError(f"{path}: frame {step} declares {ncells} cells but has {len(rows)} rows")
    if any(len(row) != len(columns) for row in rows):
        raise FieldFormatError(f"{path}: frame {step} has malformed cell rows")
    try:
        data = np.asarray(rows, dtype=float)
    except ValueError as exc:
        raise FieldFormatError(f"{path}: frame {step} has nonnumeric cell data") from exc
    return Frame(step, tuple(bounds), columns, data, raw), partial_tail


def read_complete_frames(path: str | Path, tail_bytes: int | None = None) -> FrameSet:
    """Return complete frames, ignoring only an incomplete final frame.

    If tail_bytes is given and smaller than the file's size, only that many
    bytes from the end of the file are read/decoded, and the (possibly
    truncated) first frame in that window is discarded. For a large
    multi-GB append-only dump where only a recent fraction of frames is
    actually needed (e.g. average_tail's frac), this avoids reading,
    decoding, and then re-slicing the *entire* file into memory -- which for
    a several-GB file multiplies far past its on-disk size (full read +
    decoded str + one substring copy per frame) and can exhaust available
    RAM even when the caller only wanted the last 25% of frames.
    """
    source = Path(path)
    # Take one finite snapshot rather than letting a concurrently appending
    # writer extend this read past the state we are about to validate.
    try:
        size = source.stat().st_size
        with source.open("rb") as stream:
            if tail_bytes is not None and tail_bytes < size:
                stream.seek(size - tail_bytes)
                text = stream.read().decode("utf-8", errors="ignore")
                first = re.search(r"(?m)^ITEM: TIMESTEP[ \t]*\r?$", text)
                text = text[first.start():] if first else ""
            else:
                text = stream.read(size).decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise FieldFormatError(f"cannot read {source}: {exc}") from exc
    starts = [match.start() for match in re.finditer(r"(?m)^ITEM: TIMESTEP[ \t]*\r?$", text)]
    if not starts:
        raise FieldFormatError(f"{source}: no ITEM: TIMESTEP frame")
    frames: list[Frame] = []
    ignored_tail = False
    for index, start in enumerate(starts):
        raw = text[start:starts[index + 1] if index + 1 < len(starts) else len(text)]
        final = index == len(starts) - 1
        try:
            frame, partial_tail = _parse_frame(source, raw, final)
            frames.append(frame)
            ignored_tail |= partial_tail
        except EOFError:
            ignored_tail = True
    if not frames:
        raise FieldFormatError(f"{source}: no complete frame")
    return FrameSet(tuple(frames), ignored_tail)


def select_frame(frames: tuple[Frame, ...], *, timestep: int | None = None,
                 until_step: int | None = None, until_ms: float | None = None,
                 dt: float | None = None) -> Frame:
    """Select one exact or latest complete frame within an inclusive bound."""
    if timestep is not None:
        for frame in frames:
            if frame.timestep == timestep:
                return frame
        raise FieldFormatError(f"requested timestep {timestep} is not complete; "
                               f"available {[f.timestep for f in frames]}")
    limit = until_step
    if until_ms is not None:
        if dt is None:
            raise FieldFormatError("--until-ms needs DT from manifest.json or --dt")
        if dt <= 0:
            raise FieldFormatError("DT must be positive")
        ms_limit = until_ms * 1e-3
        eligible = [f for f in frames if f.timestep * dt <= ms_limit + abs(ms_limit) * 1e-12]
    else:
        eligible = [f for f in frames if limit is None or f.timestep <= limit]
    if not eligible:
        bound = f"{until_ms} ms" if until_ms is not None else str(until_step)
        raise FieldFormatError(f"no complete frame at or before {bound}")
    return eligible[-1]


def manifest_info(run_dir: str | Path) -> tuple[str | None, float | None]:
    """Return observed manifest status and a positive DT when recorded."""
    manifest = Path(run_dir) / "manifest.json"
    if not manifest.is_file():
        return None, None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A writer may be updating the receipt. Step selection needs no
        # metadata; --dt can supply time conversion until a later read succeeds.
        return "unreadable", None
    status = data.get("status")
    controls = data.get("runtime_controls", {})
    command = data.get("command", [])
    recorded = [command[i + 2] for i in range(len(command) - 2)
                if command[i:i + 2] == ["-var", "DT"]]
    try:
        dt = float(recorded[-1] if recorded else controls["DT"])
    except (KeyError, TypeError, ValueError):
        return status, None
    return status, dt if dt > 0 else None


def recorded_dt(run_dir: str | Path) -> float | None:
    """Return DT from a live manifest or a converter's frozen-field receipt."""
    _, dt = manifest_info(run_dir)
    if dt is not None:
        return dt
    provenance = Path(run_dir) / "provenance.json"
    if not provenance.is_file():
        return None
    try:
        value = float(json.loads(provenance.read_text(encoding="utf-8")).get("source_dt_s"))
    except (OSError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def resolved_dt(run_dir: str | Path, explicit_dt: float | None = None) -> float:
    """Use explicit DT or the run manifest; reject unknown time conversion."""
    if explicit_dt is not None:
        if explicit_dt <= 0:
            raise FieldFormatError("--dt must be positive")
        return explicit_dt
    dt = recorded_dt(run_dir)
    if dt is None:
        raise FieldFormatError("--until-ms needs DT from manifest.json or --dt")
    return dt


def write_frame(frame: Frame, path: str | Path) -> None:
    Path(path).write_bytes(frame.raw.encode("utf-8"))
