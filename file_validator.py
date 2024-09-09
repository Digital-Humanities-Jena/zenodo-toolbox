import json
import logging
from pathlib import Path
from PIL import Image
from typing import List, Union
import wave

import magic
import mutagen
import PyPDF2

logger = logging.getLogger("zenodo-toolbox")


def validate_audio(filepath: Union[str, Path], errors: List[str], file_type: str) -> None:
    """
    Validates an audio file and appends any errors to the provided list.

    Args:
        filepath: Path to the audio file.
        errors: List to append error messages to.
        file_type: MIME type of the audio file.

    Returns:
        None
    """
    if file_type == "audio/x-wav":
        try:
            with wave.open(str(filepath), "rb") as wav_file:
                wav_file.getparams()
            # If we get here, the WAV file is valid
        except wave.Error as e:
            errors.append(f"Invalid WAV file {filepath}: {str(e)}")
    else:
        try:
            mutagen.File(filepath)
            # If we get here, the audio file is valid
        except mutagen.MutagenError as e:
            errors.append(f"Invalid audio file {filepath}: {str(e)}")


def validate_file(filepath: Union[str, Path]) -> List[str]:
    """
    Validates a file based on its type and performs specific checks.

    Args:
        filepath: Path to the file to be validated.

    Returns:
        A list of error messages. An empty list indicates no errors.

    Raises:
        Exception: If an unexpected error occurs during validation.
    """
    errors = []
    filepath = Path(filepath)

    if not filepath.exists():
        errors.append(f"File does not exist: {filepath}")
        return errors

    mime = magic.Magic(mime=True)
    file_type = mime.from_file(str(filepath))

    try:
        if file_type.startswith("image"):
            validate_image(filepath, errors)
        elif file_type == "application/pdf":
            validate_pdf(filepath, errors)
        elif file_type == "application/json":
            validate_json(filepath, errors)
        elif file_type.startswith("audio"):
            validate_audio(filepath, errors, file_type)
        else:
            # For other file types, just check if it's readable
            validate_generic(filepath, errors)

    except Exception as e:
        errors.append(f"Error validating {filepath}: {str(e)}")

    return errors


def validate_generic(filepath: Union[str, Path], errors: List[str]) -> None:
    """
    Attempts to read the first 1KB of a file to validate its readability.

    Args:
        filepath: The path to the file to be validated.
        errors: A list to store error messages if the file cannot be read.

    Returns:
        None

    Raises:
        No exceptions are raised directly by this function.
    """
    try:
        with open(filepath, "rb") as f:
            f.read(1024)  # Read first 1KB
        # If we get here, the file is readable
    except Exception as e:
        errors.append(f"Unable to read file {filepath}: {str(e)}")


def validate_image(filepath: Union[str, Path], errors: List[str]) -> None:
    """
    Validates an image file and appends any errors to the provided list.

    Args:
        filepath: Path to the image file to be validated.
        errors: List to which error messages will be appended.

    Returns:
        None

    Raises:
        No exceptions are raised; errors are appended to the 'errors' list.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        # If we get here, the image is valid
    except Exception as e:
        errors.append(f"Invalid image file {filepath}: {str(e)}")


def validate_json(filepath: Union[str, Path], errors: List[str]) -> None:
    """
    Validates a JSON file and appends any errors to the provided list.

    Args:
        filepath: Path to the JSON file to be validated.
        errors: List to which error messages will be appended.

    Returns:
        None
    """
    try:
        with open(filepath, "r") as f:
            json.load(f)
        # If we get here, the JSON is valid
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON file {filepath}: {str(e)}")


def validate_pdf(filepath: Union[str, Path], errors: List[str]) -> None:
    """
    Validates a PDF file and appends any errors to the provided list.

    Args:
        filepath: The path to the PDF file to validate.
        errors: A list to which error messages will be appended.

    Returns:
        None
    """
    try:
        with open(filepath, "rb") as f:
            PyPDF2.PdfReader(f)
        # If we get here, the PDF is valid
    except Exception as e:
        errors.append(f"Invalid PDF file {filepath}: {str(e)}")
