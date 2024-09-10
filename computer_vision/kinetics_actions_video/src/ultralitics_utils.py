import numpy as np
import torch
import ultralytics as ur
from PIL import Image

from ultralytics.engine.results import Boxes, Keypoints, Results


def normalize_to_pixel(value: float, dim_size: int) -> int:
    """
    Convert a normalized coordinate to pixel value.

    Args:
        value (float): Normalized coordinate value (0-1).
        dim_size (int): Dimension size (width or height).

    Returns:
        int: Pixel coordinate value.
    """
    return int(value * dim_size)


def normalize_bbox(data: dict, height: int, width: int) -> list[float]:
    """
    Normalize bounding box coordinates to pixel values.

    Args:
        data (dict): Dictionary containing bounding box data.
        height (int): Image height.
        width (int): Image width.

    Returns:
        list[float]: Normalized bounding box coordinates and metadata.
    """
    return [
        normalize_to_pixel(data['boxes']['x1'], width),
        normalize_to_pixel(data['boxes']['y1'], height),
        normalize_to_pixel(data['boxes']['x2'], width),
        normalize_to_pixel(data['boxes']['y2'], height),
        data['confidence'],
        data['cls']
    ]


def process_annotation(pose_annotation) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Process a single pose annotation.

    Args:
        pose_annotation: Pose annotation object.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Processed boxes and keypoints data.
    """
    data = pose_annotation.dict()
    orig_shape = tuple(data['orig_shape'])
    height, width = orig_shape

    keypoints_data = torch.tensor([
        [normalize_to_pixel(x, width),
         normalize_to_pixel(y, height),
         v]
        for x, y, v in zip(
            data['keypoints']['x'],
            data['keypoints']['y'],
            data['keypoints']['visible']
        )
    ])
    keypoints_data = keypoints_data.unsqueeze(0)  # Add batch dimension: Shape: (1, 17, 3)
    boxes_data = torch.tensor([normalize_bbox(data, height, width)])

    return boxes_data, keypoints_data


def extract_yolo_results(detections: list) -> tuple[Boxes, Keypoints]:
    """
    Extract YOLO results from a list of detections.

    Args:
        detections (list): list of detection objects.

    Returns:
        tuple[Boxes, Keypoints]: Processed boxes and keypoints.
    """
    all_boxes_data = []
    all_keypoints_data = []

    for pose in detections:
        boxes, keypoints = process_annotation(pose)
        all_boxes_data.append(boxes)
        all_keypoints_data.append(keypoints)

    if all_boxes_data:
        all_boxes_data = torch.cat(all_boxes_data, dim=0)
    if all_keypoints_data:
        all_keypoints_data = torch.cat(all_keypoints_data, dim=0)

    orig_shape = tuple(detections[0].orig_shape)
    boxes = Boxes(all_boxes_data, orig_shape)
    keypoints = Keypoints(all_keypoints_data, orig_shape)

    return boxes, keypoints


def visualize_ultralytics_results(results: Results, scale: float = 1.0) -> Image.Image:
    """
    Visualize Ultralytics Results object.

    Args:
        results (Results): Results object from Ultralytics model.
        scale (float): Scale factor for resizing the image. Default is 1.0.

    Returns:
        Image.Image: Visualized and resized image.
    """
    im_bgr = results.plot(
        font_size=20,
        kpt_radius=5,
    )

    im_rgb = Image.fromarray(im_bgr[..., ::-1])

    orig_height, orig_width = results.orig_shape
    new_size = (int(orig_width * scale), int(orig_height * scale))

    im_rgb = im_rgb.resize(new_size, Image.LANCZOS)
    return im_rgb


def fetch_frame_ids(dc_pose) -> list[str]:
    """
    Fetch frame IDs for a given video based on pose confidence.

    Args:
        dc_pose: DataChain pose object.

    Returns:
        list[str]: list of frame IDs.
    """
    return list(dc_pose.distinct('frame.frame_id').collect('frame.frame_id'))


def process_frame2results(frame_file, pose_detections: list) -> Results:
    """
    Process a single frame to prepare for plotting.

    Args:
        frame_file: Frame file object.
        pose_detections (list): list of pose detections.

    Returns:
        Results: Processed results for plotting.
    """
    img_file_path = frame_file.get_path()
    img_pil = Image.open(img_file_path)
    rgb_array = np.asarray(img_pil)
    if rgb_array.ndim == 3 and rgb_array.shape[2] == 3:
        bgr_array = rgb_array[:, :, ::-1]  # RGB to BGR conversion
    else:
        bgr_array = rgb_array  # Handle grayscale images

    boxes, keypoints = extract_yolo_results(pose_detections)

    return Results(
        bgr_array,
        path=frame_file.get_path(),
        names={0: 'person'},
        boxes=boxes.data,
        keypoints=keypoints.data,
    )
