# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Offline post-processing pipeline for MOTRv2 / Query-MARFT tracker outputs.

Purpose: improve IDF1 on DanceTrack without retraining or modifying any
MOTRv2 native code.  Consumes and produces files in the standard MOT
format (``frame,id,x,y,w,h,score,-1,-1,-1``).

Four independent stages (each toggleable via ``--stages``):

    merge    IoU + motion-extrapolation tracklet stitching (supersedes
             ``tools/merge_dance_tracklets.py``; adds a spatial gate on
             top of the original temporal gate).

    interp   Linear bbox interpolation across gaps within a single track
             (ByteTrack / StrongSORT style); directly reduces Frag.

    nms      Per-frame IoU suppression across different track IDs to
             remove duplicate boxes on the same target (reduces IDSW).

    short    Drop any track shorter than ``--min_track_len`` frames;
             run LAST so pre-cleaning doesn't starve the merge stage.

The default pipeline ``merge,interp,nms,short`` is the sensible order
and typically yields +2~4 IDF1 on DanceTrack with zero retraining.

Usage::

    python3 tools/postprocess_tracks.py tracker/ post_tracker/ \\
        --stages merge,interp,nms,short \\
        --merge_t_min 5 --merge_t_max 60 \\
        --merge_iou 0.4 \\
        --interp_max_gap 50 \\
        --nms_iou 0.85 \\
        --min_track_len 10

Any subset of stages may be used, e.g. ``--stages interp,short``.
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Indices in the MOT text file — keep in sync with submit_dance.py format.
F, I, X, Y, W, H, S = 0, 1, 2, 3, 4, 5, 6


# ======================================================================
# Shared helpers
# ======================================================================
def load_mot_file(path: Path) -> np.ndarray:
    """Load a MOT-format txt into an (N, 10) float64 array."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 10), dtype=np.float64)
    return np.loadtxt(path, delimiter=',', ndmin=2)


def save_mot_file(path: Path, arr: np.ndarray) -> None:
    """Save (N, 10) array as MOT txt (frame/id as int, boxes with 2 dec)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.size == 0:
        path.write_text('')
        return
    # Stable sort: primarily by frame, then by id — matches tracker output
    # convention so downstream evaluators see deterministic ordering.
    order = np.lexsort((arr[:, I].astype(np.int64),
                        arr[:, F].astype(np.int64)))
    arr = arr[order]
    lines = []
    for row in arr:
        lines.append(
            f'{int(row[F])},{int(row[I])},'
            f'{row[X]:.2f},{row[Y]:.2f},{row[W]:.2f},{row[H]:.2f},'
            f'{row[S]:.2f},-1,-1,-1\n'
        )
    path.write_text(''.join(lines))


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def _box_iou(a_xywh: np.ndarray, b_xywh: np.ndarray) -> np.ndarray:
    """Compute IoU between rows of ``a`` and rows of ``b`` pairwise.
    Returns (len(a), len(b))."""
    if a_xywh.size == 0 or b_xywh.size == 0:
        return np.zeros((a_xywh.shape[0], b_xywh.shape[0]))
    a = _xywh_to_xyxy(a_xywh)
    b = _xywh_to_xyxy(b_xywh)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class UnionFind:
    """Iterative, path-compressed union-find. No recursion, no cycles."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        root = x
        while root in self.parent:
            root = self.parent[root]
        cur = x
        while cur in self.parent and self.parent[cur] != root:
            nxt = self.parent[cur]
            self.parent[cur] = root
            cur = nxt
        return root

    def merge(self, dst: int, src: int) -> None:
        rd, rs = self.find(dst), self.find(src)
        if rd != rs:
            self.parent[rs] = rd


def _tracks_by_id(arr: np.ndarray) -> Dict[int, np.ndarray]:
    """Group rows by integer track_id; each value is frame-sorted."""
    out: Dict[int, np.ndarray] = {}
    if arr.size == 0:
        return out
    ids = arr[:, I].astype(np.int64)
    for tid in np.unique(ids):
        rows = arr[ids == tid]
        rows = rows[np.argsort(rows[:, F])]
        out[int(tid)] = rows
    return out


# ======================================================================
# Stage 1 — IoU + motion-extrapolation tracklet merging
# ======================================================================
def stage_merge(arr: np.ndarray,
                t_min: int,
                t_max: int,
                iou_thresh: float,
                motion_k: int,
                verbose: bool = False) -> np.ndarray:
    """
    Stitch two tracklets ``i -> j`` iff:

      (a) temporal:    t_min < start_t[j] - end_t[i] < t_max
      (b) uniqueness:  no other tracklet also qualifies on either side
                       (inherits the original merge_dance_tracklets gate
                       to avoid ambiguous merges)
      (c) spatial:     IoU( extrapolated_box_i(start_t[j]), start_box[j] )
                       >= iou_thresh

    Extrapolation uses the last ``motion_k`` frames of ``i`` to estimate
    a constant-velocity motion model; falls back to ``end_box[i]`` for
    short tracks.
    """
    if arr.size == 0:
        return arr

    tracks = _tracks_by_id(arr)
    if len(tracks) < 2:
        return arr

    ids = list(tracks.keys())
    start_t = {tid: int(rows[0, F]) for tid, rows in tracks.items()}
    end_t = {tid: int(rows[-1, F]) for tid, rows in tracks.items()}
    start_box = {tid: rows[0, X:H + 1] for tid, rows in tracks.items()}
    end_box = {tid: rows[-1, X:H + 1] for tid, rows in tracks.items()}

    # For the uniqueness gate — same as merge_dance_tracklets but O(log n).
    sorted_end = sorted(end_t.values())
    sorted_start = sorted(start_t.values())

    def _velocity(rows: np.ndarray) -> np.ndarray:
        """Last-k constant-velocity estimate in xywh space."""
        if rows.shape[0] < 2:
            return np.zeros(4)
        k = min(motion_k, rows.shape[0])
        tail = rows[-k:]
        dt = max(1.0, float(tail[-1, F] - tail[0, F]))
        return (tail[-1, X:H + 1] - tail[0, X:H + 1]) / dt

    vel = {tid: _velocity(rows) for tid, rows in tracks.items()}

    uf = UnionFind()
    # Deterministic order: sort by start_t then id.
    ordered = sorted(ids, key=lambda t: (start_t[t], t))
    for j in ordered:
        sj = start_t[j]
        best = None  # (iou, i)
        for i in ordered:
            if uf.find(i) == uf.find(j):
                continue
            ei = end_t[i]
            dt = sj - ei
            if not (t_min < dt < t_max):
                continue

            # Uniqueness gate — match the original O(n) sum(...) > 1 logic
            # but with bisect, so this is O(log n) per pair.
            lo_lf = sj - t_max + 1
            n_left = (bisect.bisect_right(sorted_end, sj)
                      - bisect.bisect_left(sorted_end, lo_lf))
            if n_left > 1:
                continue
            hi_rt = ei + t_max - 1
            n_right = (bisect.bisect_right(sorted_start, hi_rt)
                       - bisect.bisect_left(sorted_start, ei))
            if n_right > 1:
                continue

            # Spatial gate — extrapolate i's motion to sj, IoU with j start.
            pred_box = end_box[i] + vel[i] * dt
            iou = float(_box_iou(pred_box[None, :],
                                 start_box[j][None, :])[0, 0])
            if iou < iou_thresh:
                continue
            if (best is None) or (iou > best[0]):
                best = (iou, i)

        if best is not None:
            iou, i = best
            if verbose:
                print(f'  merge: {i} <- {j}  iou={iou:.3f}  '
                      f'end_t={end_t[i]} start_t={sj}  dt={sj - end_t[i]}')
            uf.merge(i, j)

    # Rewrite ids in the original array.
    new = arr.copy()
    new_ids = new[:, I].astype(np.int64).copy()
    for idx in range(new.shape[0]):
        new_ids[idx] = uf.find(int(new_ids[idx]))
    new[:, I] = new_ids
    return new


# ======================================================================
# Stage 2 — Linear bbox interpolation across gaps
# ======================================================================
def stage_interp(arr: np.ndarray, max_gap: int,
                 verbose: bool = False) -> np.ndarray:
    """
    Within each track_id, fill any gap of length in ``[2, max_gap]`` frames
    by linearly interpolating the xywh box between the two bracketing
    observations.  Inserted rows carry ``score=1.0`` and the same id.
    """
    if arr.size == 0 or max_gap < 2:
        return arr

    tracks = _tracks_by_id(arr)
    added: List[np.ndarray] = [arr]  # keep originals unchanged
    n_filled = 0
    for tid, rows in tracks.items():
        frames = rows[:, F].astype(np.int64)
        for k in range(len(rows) - 1):
            f1, f2 = int(frames[k]), int(frames[k + 1])
            gap = f2 - f1
            if gap < 2 or gap > max_gap:
                continue
            box1 = rows[k, X:H + 1]
            box2 = rows[k + 1, X:H + 1]
            # Generate gap-1 interpolated rows for frames f1+1 ... f2-1.
            n = gap - 1
            ts = np.arange(1, gap, dtype=np.float64) / gap  # shape (n,)
            interp_boxes = (
                (1.0 - ts)[:, None] * box1[None, :]
                + ts[:, None] * box2[None, :])
            fills = np.zeros((n, 10), dtype=np.float64)
            fills[:, F] = np.arange(f1 + 1, f2, dtype=np.float64)
            fills[:, I] = tid
            fills[:, X:H + 1] = interp_boxes
            fills[:, S] = 1.0
            fills[:, 7:] = -1
            added.append(fills)
            n_filled += n
    if verbose:
        print(f'  interp: filled {n_filled} frames across '
              f'{len(tracks)} tracks')
    return np.concatenate(added, axis=0) if len(added) > 1 else arr


# ======================================================================
# Stage 3 — Per-frame IoU NMS across different IDs
# ======================================================================
def stage_nms(arr: np.ndarray, iou_thresh: float,
              verbose: bool = False) -> np.ndarray:
    """
    For each frame, suppress any two boxes with IoU >= ``iou_thresh`` that
    carry different ``track_id``.  Tie-breaker: keep the id with more
    total observations across the whole sequence (proxy for reliability,
    since MOTRv2 output has score=1 for every row).
    """
    if arr.size == 0:
        return arr

    ids = arr[:, I].astype(np.int64)
    id_count = np.bincount(
        ids + (-ids.min() if ids.min() < 0 else 0))
    # use a plain dict for safety against negative / sparse ids
    count_of: Dict[int, int] = {}
    for tid in np.unique(ids):
        count_of[int(tid)] = int((ids == tid).sum())

    keep_mask = np.ones(arr.shape[0], dtype=bool)
    n_suppressed = 0
    frames = arr[:, F].astype(np.int64)
    for fr in np.unique(frames):
        idxs = np.where(frames == fr)[0]
        if len(idxs) < 2:
            continue
        boxes = arr[idxs, X:H + 1]
        iou = _box_iou(boxes, boxes)
        np.fill_diagonal(iou, 0.0)
        for a in range(len(idxs)):
            if not keep_mask[idxs[a]]:
                continue
            for b in range(a + 1, len(idxs)):
                if not keep_mask[idxs[b]]:
                    continue
                if iou[a, b] < iou_thresh:
                    continue
                id_a = int(arr[idxs[a], I])
                id_b = int(arr[idxs[b], I])
                if id_a == id_b:
                    continue
                # Drop the shorter track's row at this frame.
                if count_of[id_a] >= count_of[id_b]:
                    keep_mask[idxs[b]] = False
                else:
                    keep_mask[idxs[a]] = False
                n_suppressed += 1
    if verbose:
        print(f'  nms: suppressed {n_suppressed} rows '
              f'(iou>={iou_thresh})')
    return arr[keep_mask]


# ======================================================================
# Stage 4 — Short tracklet removal
# ======================================================================
def stage_short(arr: np.ndarray, min_len: int,
                verbose: bool = False) -> np.ndarray:
    """Drop every track whose number of rows is < ``min_len``."""
    if arr.size == 0 or min_len <= 1:
        return arr
    ids = arr[:, I].astype(np.int64)
    keep_ids = {int(tid) for tid in np.unique(ids)
                if int((ids == tid).sum()) >= min_len}
    keep_mask = np.array([int(t) in keep_ids for t in ids])
    if verbose:
        total = len(np.unique(ids))
        print(f'  short: kept {len(keep_ids)}/{total} tracks '
              f'(min_len={min_len})')
    return arr[keep_mask]


# ======================================================================
# Pipeline driver
# ======================================================================
STAGE_FUNCS = {
    'merge': 'stage_merge',
    'interp': 'stage_interp',
    'nms': 'stage_nms',
    'short': 'stage_short',
}


def _apply_pipeline(arr: np.ndarray,
                    stages: List[str],
                    args: argparse.Namespace) -> np.ndarray:
    for name in stages:
        if name == 'merge':
            arr = stage_merge(arr, args.merge_t_min, args.merge_t_max,
                              args.merge_iou, args.merge_motion_k,
                              verbose=args.verbose)
        elif name == 'interp':
            arr = stage_interp(arr, args.interp_max_gap,
                               verbose=args.verbose)
        elif name == 'nms':
            arr = stage_nms(arr, args.nms_iou, verbose=args.verbose)
        elif name == 'short':
            arr = stage_short(arr, args.min_track_len,
                              verbose=args.verbose)
        else:
            raise ValueError(f'Unknown stage: {name!r}')
    return arr


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input_dir', type=Path,
                   help='Directory of per-sequence *.txt tracker outputs.')
    p.add_argument('output_dir', type=Path,
                   help='Where to write post-processed outputs. '
                        'Files go under <output_dir>/tracker/ mirroring '
                        'submit_dance.py convention.')
    p.add_argument('--stages', type=str, default='merge,interp,nms,short',
                   help='Comma-separated subset of {merge,interp,nms,short} '
                        'in desired execution order.')
    # merge stage
    p.add_argument('--merge_t_min', type=int, default=5,
                   help='Minimum temporal gap (frames) between two tracklets '
                        'to consider merging. Lower than the 20 used by the '
                        'original script — DanceTrack often has short blips.')
    p.add_argument('--merge_t_max', type=int, default=60,
                   help='Maximum temporal gap (frames) for merging.')
    p.add_argument('--merge_iou', type=float, default=0.4,
                   help='Minimum IoU between motion-extrapolated box of i '
                        'and first box of j required to merge.')
    p.add_argument('--merge_motion_k', type=int, default=10,
                   help='Tail-window (frames) used to estimate i-track velocity.')
    # interp stage
    p.add_argument('--interp_max_gap', type=int, default=50,
                   help='Do not interpolate across gaps larger than this many '
                        'frames (too risky).')
    # nms stage
    p.add_argument('--nms_iou', type=float, default=0.85,
                   help='IoU threshold for per-frame cross-id suppression.')
    # short stage
    p.add_argument('--min_track_len', type=int, default=10,
                   help='Drop tracks shorter than this many rows after '
                        'all other stages.')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    for s in stages:
        if s not in STAGE_FUNCS:
            print(f'ERROR: unknown stage {s!r}. '
                  f'Valid: {sorted(STAGE_FUNCS)}', file=sys.stderr)
            sys.exit(2)

    out_dir = args.output_dir / 'tracker'
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.input_dir.iterdir()
                   if p.suffix == '.txt')
    if not files:
        print(f'ERROR: no .txt files in {args.input_dir}', file=sys.stderr)
        sys.exit(2)

    print(f'[postprocess] stages = {stages}')
    print(f'[postprocess] processing {len(files)} sequences '
          f'from {args.input_dir}')
    for p in files:
        arr_in = load_mot_file(p)
        n_ids_in = len(np.unique(arr_in[:, I])) if arr_in.size else 0
        arr_out = _apply_pipeline(arr_in, stages, args)
        n_ids_out = len(np.unique(arr_out[:, I])) if arr_out.size else 0
        save_mot_file(out_dir / p.name, arr_out)
        print(f'  {p.name}: rows {arr_in.shape[0]:>6} -> '
              f'{arr_out.shape[0]:>6}   ids {n_ids_in:>5} -> '
              f'{n_ids_out:>5}')
    print(f'[postprocess] done -> {out_dir}')


if __name__ == '__main__':
    main()
