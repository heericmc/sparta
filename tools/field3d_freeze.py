#!/usr/bin/env python3
"""Freeze one complete frame from a 3D SPARTA grid dump for
ParticleTracing3D.jl.

Unlike field2tracer.py (the 2D converter), this does NOT reformat data or
remap axes -- ParticleTracing3D.jl reads the raw SPARTA Cartesian grid dump
format directly (id xc yc zc xlo ylo zlo xhi yhi zhi ... , the same 9-line
ITEM: header SPARTA itself writes), and reads cell_fluor3d.surf directly
too (no cell.surfs-style conversion needed for 3D). The ONLY thing needed
before the tracer can read field.grid is picking ONE frame out of the
raw file's multiple append-only snapshots -- field.grid accumulates one
block per `dump grid` interval over the whole run, and reading it with a
simple fixed-header-line parser (skipto=10, as ParticleTracing3D.jl does)
is only correct for a SINGLE frame, not the whole multi-frame file.

This reuses field_io.py's frame parsing directly rather than
reimplementing frame-boundary detection (What counts as a complete vs.
truncated trailing frame, from a run that may still be in progress, is
exactly the kind of thing worth not getting wrong twice.)

Usage: python tools/field3d_freeze.py RUN_DIR --out OUTPUT_FILE [SELECTION]
  (same --timestep / --until-step / --until-ms / --dt selection flags as
  field2tracer.py; default is the latest complete frame)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_io import FieldFormatError, manifest_info, read_complete_frames, resolved_dt, select_frame, write_frame  # noqa: E402


def freeze(rundir, out_path, dumpname="field.grid", timestep=None,
           until_step=None, until_ms=None, dt=None):
    path = os.path.join(rundir, dumpname)
    frames = read_complete_frames(path)
    source_status, manifest_dt = manifest_info(rundir)
    explicit_dt = resolved_dt(rundir, dt) if dt is not None else None
    selection_dt = (resolved_dt(rundir, dt) if until_ms is not None
                    else explicit_dt if explicit_dt is not None else manifest_dt)
    frame = select_frame(frames.frames, timestep=timestep, until_step=until_step,
                          until_ms=until_ms, dt=selection_dt)
    write_frame(frame, out_path)
    return frame.timestep, len(frame.data), len(frames.frames), frames.ignored_incomplete_tail


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rundir")
    p.add_argument("--out", required=True)
    p.add_argument("--dump", default="field.grid")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--timestep", type=int)
    selection.add_argument("--until-step", type=int)
    selection.add_argument("--until-ms", type=float)
    p.add_argument("--dt", type=float)
    a = p.parse_args()
    try:
        step, ncells, nframes, ignored_tail = freeze(
            a.rundir, a.out, a.dump, a.timestep, a.until_step, a.until_ms, a.dt)
    except FieldFormatError as exc:
        sys.exit(str(exc))
    print(f"wrote {a.out}: timestep {step}, {ncells} cells (of {nframes} frame(s) in "
          f"{a.dump}{', trailing incomplete frame ignored' if ignored_tail else ''})")


if __name__ == "__main__":
    main()
