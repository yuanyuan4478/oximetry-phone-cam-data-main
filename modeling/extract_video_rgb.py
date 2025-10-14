"""Extract per-frame RGB statistics from raw Hoffman et al. (2022) videos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
OUTPUT_DIR = ROOT / "data" / "ppg-from-raw"


def iter_videos(side: str) -> Iterable[Tuple[str, Path]]:
    side_dir = RAW_DIR / side
    if not side_dir.exists():
        raise FileNotFoundError(f"Missing raw video directory: {side_dir}")
    for video_path in sorted(side_dir.glob("*.mp4")):
        subject_id = video_path.name.split("-")[0]
        yield subject_id, video_path


def compute_frame_statistics(video_path: Path, crop_ratio: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    stats: list[np.ndarray] = []
    success, frame = capture.read()
    if not success:
        capture.release()
        raise RuntimeError(f"No frames read from {video_path}")

    height, width, _ = frame.shape
    y0 = int((1.0 - crop_ratio) / 2.0 * height)
    y1 = int(height - y0)
    x0 = int((1.0 - crop_ratio) / 2.0 * width)
    x1 = int(width - x0)

    while success:
        roi = frame[y0:y1, x0:x1]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        means = rgb.mean(axis=(0, 1), dtype=np.float64)
        stats.append(means)
        success, frame = capture.read()

    capture.release()
    if not stats:
        raise RuntimeError(f"No statistics collected for {video_path}")
    return np.vstack(stats)


def process_side(side: str, crop_ratio: float, overwrite: bool) -> None:
    output_side_dir = OUTPUT_DIR / side
    output_side_dir.mkdir(parents=True, exist_ok=True)

    for subject_id, video_path in iter_videos(side):
        output_path = output_side_dir / f"{subject_id}.csv"
        if output_path.exists() and not overwrite:
            continue
        frame_stats = compute_frame_statistics(video_path, crop_ratio)
        df = pd.DataFrame(frame_stats, columns=["R", "G", "B"])
        df.to_csv(output_path, index=False, float_format="%.8f")
        print(f"Processed {video_path.name} -> {output_path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop",
        type=float,
        default=0.6,
        help="Central crop ratio (0-1) applied before averaging RGB (default: 0.6)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate CSVs even if output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crop_ratio = float(np.clip(args.crop, 0.05, 1.0))
    for side in ("Left", "Right"):
        process_side(side, crop_ratio, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
