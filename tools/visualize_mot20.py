# ------------------------------------------------------------------------
# Copyright (c) 2026 JoshuaWenHIT. All Rights Reserved.
# ------------------------------------------------------------------------
"""MOT20 tracking-result visualiser (analog of ``tools/visualize.py``).

Mirrors the CLI of ``tools/visualize.py`` (single-file *or* batch mode) but
is tuned for the MOT20 regime:

  * 6-digit frame filenames (``000001.jpg``).  The digit-extraction helper
    in the DanceTrack version actually already handles this, but we make
    it explicit and robust here.
  * Frame rate auto-detected from ``seqinfo.ini`` (MOT20 ships 25 fps for
    train and 25/30 for test sequences).  Falls back to the ``--fps`` CLI
    flag, then to 25.
  * Dense crowds: defaults to thinner rectangles (1 px), smaller ID label,
    and a golden-ratio HSV palette so 100+ IDs/frame remain visually
    distinguishable.  Tiny boxes auto-suppress the ID text so they stay
    readable.
  * Optional ``--min-score`` filter to drop low-confidence rows (useful
    when visualising raw inference output before post-processing).

Expected on-disk layout (matches ``submit_mot20.py`` output and the MOT
Challenge convention)::

    <track_dir>/                           <images_root_dir>/
        MOT20-04.txt                          MOT20-04/
        MOT20-06.txt                              img1/000001.jpg ...
        ...                                       seqinfo.ini

Usage::

    # Batch:
    python3 tools/visualize_mot20.py --batch \\
        --track-dir         exp/mot20/run1/tracker \\
        --images-root-dir   /path/to/MOT20/test \\
        --output-root-dir   exp/mot20/run1/vis \\
        --save-video

    # Single file:
    python3 tools/visualize_mot20.py \\
        --track-path        exp/mot20/run1/tracker/MOT20-04.txt \\
        --images-dir        /path/to/MOT20/test/MOT20-04/img1 \\
        --output-root-dir   exp/mot20/run1/vis \\
        --save-video
"""

from __future__ import annotations

import argparse
import configparser
import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Tracking-result parsing
# ----------------------------------------------------------------------
def parse_tracking_results(track_path: str,
                           min_score: float = 0.0
                           ) -> Dict[int, np.ndarray]:
    """Read a MOT-format result txt and group rows per frame.

    Returns ``{frame_id: np.ndarray[N, 6]}`` with columns
    ``[track_id, x, y, w, h, score]``.
    """
    columns = ['frame', 'track_id', 'x', 'y', 'w', 'h',
               'conf', 'col8', 'col9', 'col10']
    df = pd.read_csv(track_path, header=None, names=columns)
    if min_score > 0:
        df = df[df['conf'] >= min_score]

    out: Dict[int, np.ndarray] = {}
    for frame_id, group in df.groupby('frame'):
        out[int(frame_id)] = group[
            ['track_id', 'x', 'y', 'w', 'h', 'conf']
        ].values
    return out


# ----------------------------------------------------------------------
# Sequence metadata helpers
# ----------------------------------------------------------------------
def _read_seqinfo(images_dir: str) -> Tuple[Optional[int], str]:
    """Return ``(frame_rate, im_ext)`` from ``seqinfo.ini`` if available.

    ``images_dir`` is expected to point at ``.../<seq>/img1``; we look one
    level up for ``seqinfo.ini``.
    """
    ini_path = Path(images_dir).parent / 'seqinfo.ini'
    if not ini_path.exists():
        return None, '.jpg'
    cfg = configparser.ConfigParser()
    try:
        cfg.read(ini_path)
        section = 'Sequence' if cfg.has_section('Sequence') else cfg.sections()[0]
        fps = cfg.getint(section, 'frameRate', fallback=None)
        ext = cfg.get(section, 'imExt', fallback='.jpg')
        return fps, ext
    except Exception:
        return None, '.jpg'


# ----------------------------------------------------------------------
# Color palette
# ----------------------------------------------------------------------
def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Golden-ratio HSV hashing → BGR colour with good separation.

    Using simple modular arithmetic on raw IDs (the DanceTrack version)
    quickly collides into similar colours in MOT20's high-ID regime.
    """
    if track_id < 0:
        return 128, 128, 128
    # 0.61803398875 = 1 / phi
    h = (track_id * 0.61803398875) % 1.0
    hsv = np.uint8([[[int(h * 179), 220, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ----------------------------------------------------------------------
# Single-sequence rendering
# ----------------------------------------------------------------------
def visualize_single_track(images_dir: str,
                           track_data: Dict[int, np.ndarray],
                           output_dir: str,
                           save_video: bool = True,
                           save_frames: bool = True,
                           video_fps: Optional[int] = None,
                           default_fps: int = 25,
                           line_thickness: int = 1,
                           font_scale: float = 0.4,
                           min_label_size: int = 25) -> None:
    """Render bbox + ID overlay for one MOT20 sequence."""
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    detected_fps, im_ext = _read_seqinfo(images_dir)
    fps = video_fps or detected_fps or default_fps

    valid_exts = ('.jpg', '.png', '.jpeg', '.bmp')
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)],
        key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0,
    )
    if not image_files:
        print(f'[vis-mot20] WARN: no images found under {images_dir}')
        return

    video_writer = None
    video_path = None
    if save_video:
        first = cv2.imread(os.path.join(images_dir, image_files[0]))
        if first is None:
            print(f'[vis-mot20] WARN: cannot read first image in {images_dir}')
            return
        h, w = first.shape[:2]
        video_path = os.path.join(output_dir, 'tracking_visualization.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    n_frames = len(image_files)
    for k, img_file in enumerate(image_files):
        digits = ''.join(filter(str.isdigit, img_file))
        if not digits:
            continue
        frame_id = int(digits)
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f'[vis-mot20] WARN: cannot read {img_path}, skipping')
            continue

        if frame_id in track_data:
            for row in track_data[frame_id]:
                track_id, x, y, w_box, h_box = row[:5]
                track_id = int(track_id)
                x, y = int(round(x)), int(round(y))
                w_box, h_box = int(round(w_box)), int(round(h_box))
                if w_box <= 0 or h_box <= 0:
                    continue
                color = _color_for_id(track_id)
                cv2.rectangle(img, (x, y), (x + w_box, y + h_box),
                              color, line_thickness)
                # Skip the label tag for very small boxes — it would
                # cover the whole detection in a dense crowd.
                if min(w_box, h_box) >= min_label_size:
                    label = f'{track_id}'
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    y_text = max(y - 2, th + 2)
                    cv2.rectangle(img,
                                  (x, y_text - th - 2),
                                  (x + tw + 4, y_text + 2),
                                  color, -1)
                    cv2.putText(img, label, (x + 2, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (255, 255, 255), 1, cv2.LINE_AA)

        if save_frames:
            cv2.imwrite(os.path.join(output_dir, img_file), img)
        if video_writer is not None:
            video_writer.write(img)

        if (k + 1) % 200 == 0 or (k + 1) == n_frames:
            print(f'  [{k + 1}/{n_frames}] frames rendered '
                  f'(fps={fps}, ext={im_ext})')

    if video_writer is not None:
        video_writer.release()
        print(f'[vis-mot20] video saved to {video_path}')


# ----------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------
def batch_process_tracks(track_dir: str,
                         images_root_dir: str,
                         output_root_dir: str,
                         save_video: bool = True,
                         save_frames: bool = True,
                         video_fps: Optional[int] = None,
                         min_score: float = 0.0,
                         line_thickness: int = 1,
                         font_scale: float = 0.4,
                         min_label_size: int = 25) -> None:
    track_files = sorted(glob.glob(os.path.join(track_dir, '*.txt')))
    if not track_files:
        print(f'[vis-mot20] no .txt files under {track_dir}')
        return
    print(f'[vis-mot20] found {len(track_files)} sequences, batch start ...')

    for idx, track_path in enumerate(track_files, 1):
        seq = os.path.splitext(os.path.basename(track_path))[0]
        # Standard MOT layout: <root>/<seq>/img1/
        images_dir = os.path.join(images_root_dir, seq, 'img1')
        if not os.path.exists(images_dir):
            print(f'[vis-mot20] [{idx}/{len(track_files)}] {seq}: '
                  f'no img dir {images_dir}, skipped')
            continue
        output_dir = os.path.join(output_root_dir, seq)
        print(f'\n[vis-mot20] [{idx}/{len(track_files)}] {seq}')
        print(f'  images : {images_dir}')
        print(f'  output : {output_dir}')
        try:
            track_data = parse_tracking_results(track_path, min_score=min_score)
            visualize_single_track(
                images_dir=images_dir,
                track_data=track_data,
                output_dir=output_dir,
                save_video=save_video,
                save_frames=save_frames,
                video_fps=video_fps,
                line_thickness=line_thickness,
                font_scale=font_scale,
                min_label_size=min_label_size,
            )
            print(f'  ok')
        except Exception as exc:  # noqa: BLE001 - keep batch resilient
            print(f'  FAIL: {exc}')

    print(f'\n[vis-mot20] all done; results under {output_root_dir}')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='MOT20 tracking-result batch / single-file visualiser '
                    '(analog of tools/visualize.py).')
    # Mode flags
    p.add_argument('--batch', action='store_true',
                   help='Batch mode (process every .txt under --track-dir).')
    p.add_argument('--track-dir',
                   help='Batch: directory with per-sequence .txt files.')
    p.add_argument('--images-root-dir',
                   help='Batch: root containing <seq>/img1/ folders.')
    p.add_argument('--track-path',
                   help='Single: path to one .txt result file.')
    p.add_argument('--images-dir',
                   help='Single: path to a single <seq>/img1 folder.')

    # Output / encoding
    p.add_argument('--output-root-dir', default='mot20_vis',
                   help='Root output directory.')
    p.add_argument('--save-video', action='store_true',
                   help='Encode an mp4 per sequence.')
    p.add_argument('--no-save-frames', action='store_true',
                   help='Skip writing per-frame jpgs (saves disk; videos still produced).')
    p.add_argument('--fps', type=int, default=None,
                   help='Override video frame rate. Default: read seqinfo.ini, '
                        'falling back to 25.')

    # Rendering tuning (MOT20 defaults differ from DanceTrack).
    p.add_argument('--line-thickness', type=int, default=1,
                   help='Bounding-box line thickness in px. Default: 1.')
    p.add_argument('--font-scale', type=float, default=0.4,
                   help='ID label font scale. Default: 0.4.')
    p.add_argument('--min-label-size', type=int, default=25,
                   help='Boxes whose min(w, h) is below this skip the ID label '
                        'to avoid covering the detection. Default: 25 px.')
    p.add_argument('--min-score', type=float, default=0.0,
                   help='Drop rows whose score column is below this. Default: 0.')

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    save_frames = not args.no_save_frames

    if args.batch:
        if not args.track_dir or not args.images_root_dir:
            raise SystemExit('--batch requires --track-dir and --images-root-dir')
        batch_process_tracks(
            track_dir=args.track_dir,
            images_root_dir=args.images_root_dir,
            output_root_dir=args.output_root_dir,
            save_video=args.save_video,
            save_frames=save_frames,
            video_fps=args.fps,
            min_score=args.min_score,
            line_thickness=args.line_thickness,
            font_scale=args.font_scale,
            min_label_size=args.min_label_size,
        )
    else:
        if not args.track_path or not args.images_dir:
            raise SystemExit('Single mode requires --track-path and --images-dir')
        out_dir = os.path.join(args.output_root_dir, 'single_result')
        track_data = parse_tracking_results(args.track_path,
                                            min_score=args.min_score)
        visualize_single_track(
            images_dir=args.images_dir,
            track_data=track_data,
            output_dir=out_dir,
            save_video=args.save_video,
            save_frames=save_frames,
            video_fps=args.fps,
            line_thickness=args.line_thickness,
            font_scale=args.font_scale,
            min_label_size=args.min_label_size,
        )
        print(f'[vis-mot20] single-file done; results saved to {out_dir}')


if __name__ == '__main__':
    main()
