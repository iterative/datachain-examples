import os
import sys
from typing import Dict

import cv2
import shutil


def get_video_metadata(video_path: str) -> Dict[str, float]:
    """
    Extract metadata from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        Dict[str, float]: Dictionary containing video metadata.

    Raises:
        SystemExit: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    duration = frame_count / fps if fps > 0 else 0

    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "codec": codec
    }

    cap.release()

    return metadata


def split_video_to_frames(
    video_path: str,
    output_dir: str,
    prefix: str = "frame",
    step: int = 1,
    rewrite: bool = True
) -> None:
    """
    Split a video into frames and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output frames.
        prefix (str, optional): Prefix for frame filenames. Defaults to "frame".
        step (int, optional): Step size for frame extraction. Defaults to 1.
        rewrite (bool, optional): Whether to overwrite existing output directory. Defaults to True.

    Raises:
        SystemExit: If the video file cannot be opened or if the step size is invalid.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(video_path))[0].split('_')[0]
    video_output_dir = os.path.join(output_dir, video_name)

    if os.path.exists(video_output_dir):
        if rewrite:
            shutil.rmtree(video_output_dir)
        else:
            print(f"Subdirectory already exists: {video_output_dir}")
            sys.exit(1)

    os.makedirs(video_output_dir)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not isinstance(step, int) or step <= 0 or step >= fps:
        print("Error: Step must be an integer greater than 0 and less than the video's FPS.")
        sys.exit(1)

    frame_number = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % step == 0:
            frame_filename = os.path.join(video_output_dir, f"{prefix}_{frame_number:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        frame_number += 1

    cap.release()

    print(f"Saved {saved_frames} frames from {video_path} to {video_output_dir}")
