import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import time
from typing import Any, Dict, List, Tuple, Union
from ultralytics import SAM, YOLO

from utilities import get_filetype

logger = logging.getLogger("zenodo-toolbox")


def detect_persons(filepath: Path, config: Dict[str, Any], bbox_model: YOLO) -> Union[List[Dict[str, float]], bool]:
    """
    Detect persons in an image using a YOLO model and return bounding box data.

    Args:
        filepath: Path to the image file.
        config: Configuration dictionary.
        bbox_model: YOLO model for person detection.

    Returns:
        [0] List of dictionaries containing bounding box data for detected persons, or False if no persons detected.
    """
    data_out = []
    threshold = config["person_masker"]["threshold"]
    results = bbox_model(filepath)
    boxes = results[0].boxes
    boxes_ls = [int(i) for i in boxes.cls]
    if 0 in boxes_ls:  # continue, if person (class_id=0) detected
        logger.info(f"Persons above Threshold detected: {boxes_ls.count(0)}")
        idx = 0
        for class_id in boxes.cls:
            class_id = int(class_id)
            conf = float(boxes.data[idx][4])
            if class_id == 0 and conf > threshold:
                x0 = float(boxes.data[idx][0])
                y0 = float(boxes.data[idx][1])
                x1 = float(boxes.data[idx][2])
                y1 = float(boxes.data[idx][3])
                data = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "conf_bbox": conf}
                data_out.append(data)
            idx += 1
        if config["person_masker"]["write_bbox_images"]:
            target_path = Path(
                f"./{config['person_masker']['paths']['bbox_images']}/{filepath.stem}/{filepath.stem}_bboxes.jpg"
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            results[0].save(filename=str(target_path))
    else:
        return False

    return data_out


def load_person_masker_models(config: Dict[str, Any]) -> Tuple[YOLO, SAM]:
    """
    Load person detection and segmentation models based on configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        [0] Tuple containing the loaded YOLO and SAM models.
    """
    # Set Devices
    bbox_device = config["person_masker"]["bbox_device"].lower()
    if bbox_device not in ["cpu", "cuda"]:
        logger.warning(f"Invalid device '{bbox_device}'. Defaulting to 'cpu'.")
        bbox_device = "cpu"

    segmentation_device = config["person_masker"]["segmentation_device"].lower()
    if segmentation_device not in ["cpu", "cuda"]:
        logger.warning(f"Invalid device '{segmentation_device}'. Defaulting to 'cpu'.")
        segmentation_device = "cpu"

    # Load Models
    bbox_model = YOLO(config["person_masker"]["bbox_model"])
    bbox_model.to(device=bbox_device)

    segmentation_model = SAM(config["person_masker"]["segmentation_model"])
    segmentation_model.to(device=segmentation_device)

    logger.info(f"Loaded Detection Model: {config['person_masker']['bbox_model']}")
    logger.info(f"Loaded Segmentation Model: {config['person_masker']['segmentation_model']}")
    return bbox_model, segmentation_model


def mask_persons(
    bbox_model: YOLO,
    segmentation_model: SAM,
    config: Dict[str, Any],
    filepaths: List[Union[str, Path]] = [],
    process_directory: bool = False,
) -> List[str]:
    """
    Detect and mask persons in images using bounding box and segmentation models.

    Args:
        bbox_model: YOLO model for person detection.
        segmentation_model: SAM model for person segmentation.
        config: Configuration dictionary.
        filepaths: List of image file paths to process.
        process_directory: Flag to process entire directory instead of specific files.

    Returns:
        [0] List of paths to masked image files.
    """

    detection_results = {}

    if not process_directory:
        out_filepaths = []
        for filepath in filepaths:
            filepath = Path(filepath) if isinstance(filepath, str) else filepath
            if get_filetype(filepath) != "image":
                return []
            if not filepath or not filepath.suffix.lower() in config["person_masker"]["process_extensions"]:
                return []

            # detect & get bounding box + confidence data
            detection_data = detect_persons(filepath, config, bbox_model)
            if detection_data:
                detection_results[str(filepath)] = detection_data
                masked_image_path = segment_persons(filepath, config, detection_data, segmentation_model)
                out_filepaths.append(masked_image_path)
                # logger.info(f"Masked {masked_persons_qua} Persons on Image {filepath.name} ...")

                processing_log_path = Path(
                    f'{config["person_masker"]["paths"]["processing_log"]}/{filepath.stem}.json'
                )
                processing_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(processing_log_path, "w") as log_file:
                    json.dump(detection_results, log_file, indent=4)

        return out_filepaths

    else:
        target_directory = Path(config["person_masker"]["paths"]["input_images"])
        total_files = sum(1 for _ in target_directory.glob(f"*{config['allowed_extensions']}"))
        files_with_detections = 0
        logger.info(f"Starting Person Detection and Segmentation for Directory: {str(target_directory)}")
        logger.info(f"Total Files: {total_files}")
        logger.info(f"Included Extensions: {config['allowed_extensions']}")

        results = {}

        image_ct = 0
        max_files = config["max_files"]
        for file_path in target_directory.iterdir():
            if max_files != -1 and image_ct >= max_files:
                break
            if file_path.suffix.lower() in config["allowed_extensions"]:
                image_path = str(file_path)
                logger.info(f"Processing {file_path.stem}.jpg ... ({image_ct}/{total_files})")
                # detect & get bounding box + confidence data
                detection_data = detect_persons(image_path, bbox_model)
                if detection_data:
                    results[image_path] = detection_data
                    segment_persons(image_path, detection_data, segmentation_model)
                    files_with_detections += 1
                image_ct += 1

                processing_log_path = Path(
                    f'{config["person_masker"]["paths"]["processing_log"]}/{filepath.stem}.json'
                )
                processing_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(processing_log_path, "w") as log_file:
                    json.dump(detection_results, log_file, indent=4)

        logger.info("Detection, Segmentation and Masking completed.")
        logger.info(f"Files with Persons detected: {files_with_detections}/{total_files}")


def segment_persons(
    filepath: Path, config: Dict[str, Any], bbox_data: List[Dict[str, float]], segmentation_model: SAM
) -> Path:
    """
    Segment and mask detected persons in an image.

    Args:
        filepath: Path to the image file.
        config: Configuration dictionary.
        bbox_data: List of bounding box data for detected persons.
        segmentation_model: SAM model for person segmentation.

    Returns:
        [0] Path to the masked image file.
    """
    start_time = time.time()
    print(f"Starting segmentation for {filepath}")
    image = Image.open(filepath)
    image_array = np.array(image)

    # Skip blurring for black and white images if blur_bw is False
    if not config["person_masker"]["blur_blackwhite"] and image.mode == "L":
        return

    combined_mask = np.zeros_like(image_array[:, :, 0], dtype=bool)

    ct = 0
    for p in bbox_data:
        bboxes = [p["x0"], p["y0"], p["x1"], p["y1"]]
        results = segmentation_model.predict(filepath, bboxes=bboxes)
        mask = results[0].masks.data.cpu().numpy().squeeze().astype(bool)
        combined_mask |= mask
        if config["person_masker"]["write_segmented_images"]:
            segment_output_path = Path(
                f'{config["person_masker"]["paths"]["segmented_images"]}/{filepath.stem}/{filepath.stem}_{ct}.jpg'
            )
            segment_output_path.parent.mkdir(parents=True, exist_ok=True)
            results[0].save(filename=str(segment_output_path))
        ct += 1

    masked_image = np.copy(image_array)
    masked_image[combined_mask] = 0  # Set pixels where the mask is True to black (0)

    masked_image_pil = Image.fromarray(masked_image)
    masked_image_resized = masked_image_pil.resize(image.size)

    masked_output_path = Path(f'{config["person_masker"]["paths"]["masked_images"]}/{filepath.stem}_masked.jpg')
    masked_output_path.parent.mkdir(parents=True, exist_ok=True)
    masked_image_resized.save(str(masked_output_path))
    logger.debug(f"Success: {masked_output_path}")

    total_time = time.time() - start_time
    print(f"Total time for {filepath}: {total_time:.2f} seconds")
    return Path(masked_output_path)
