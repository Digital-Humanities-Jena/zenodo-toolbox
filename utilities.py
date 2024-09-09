from contextlib import contextmanager
from datetime import datetime, timezone
from ftfy import fix_encoding
from html import unescape
import io
import json
import logging
from lxml import etree
import mimetypes
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import re
import requests
import string
import sys
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union
from urllib.parse import urlparse
import yaml

logger = logging.getLogger("zenodo-toolbox")


def add_filetables_to_description(description: str, files_data: Dict[str, str]) -> str:
    """
    Adds formatted tables for main files and thumbnails to a given description.

    Args:
        description: The original description text.
        files_data: A dictionary mapping filenames to their direct links.

    Returns:
        [0] An updated description string with added file tables.

    Notes:
        filedata structure: {"filename": "directlink"}
    """
    # Define Thumbnail Identifiers and Variables
    thumbnails_available = False
    thumbnail_identifiers = ["perspective_1", "perspective_2", "perspective_3", "perspective_4", "perspective_top"]
    thumbnail_sizes = ["512x512", "256x256", "128x128"]
    thumbnail_default_size = "1000x1000"
    thumbnail_extensions = [".png"]
    changelog_identifier = "<br><br><u>Changelog</u>:"

    main_files = []
    thumbnail_files = []
    updated_description = description

    # Clean description from existing tables if detected
    if "<table>" in description:
        description = clean_description(description)

    # Classify & Assign available Files
    for filename, directlink in files_data.items():
        file_ext = Path(filename).suffix
        thumbnail = False

        # Handle Thumbnails
        if file_ext in thumbnail_extensions:
            for perspective in thumbnail_identifiers:
                if perspective in filename:
                    thumbnail = True
                    size_header = ""
                    size_found = False
                    for size in thumbnail_sizes:
                        if size in filename:
                            size_header = size
                            size_found = True
                            break
                    if not size_found:
                        size_header = thumbnail_default_size
                    perspective_header = perspective.replace("_", " ").title()
                    thumbnail_files.append(
                        {
                            "size": size_header,
                            "perspective": perspective_header,
                            "link": directlink,
                        }
                    )

        # Handle Main Files
        if not thumbnail:
            main_files.append({"ext": file_ext, "filename": filename, "link": directlink})

    thumbnails_available = True if thumbnail_files else False

    # Create Table "Main Files" and Populate it
    table_main = "<br><br><u>Main Files:</u><br><table id='main'><tr><th>Filename</th>"

    main_extensions = set([i["ext"] for i in main_files])
    for ext in main_extensions:
        table_main += f"<th>{ext}</th>"
    table_main += "</tr>"
    for filedata in main_files:
        table_main += f"<tr><td><a href='{filedata['link']}'>{filedata['filename']}</a></td>"
        for ext in main_extensions:
            if filedata["ext"] == ext:
                table_main += f"<td><a href='{filedata['link']}'>Link</a></td>"
            else:
                table_main += "<td></td>"
        table_main += "</tr>"
    table_main += "</table><br><br>"

    # Create Table "Thumbnails" (if available) and populate it
    if thumbnails_available:
        table_thumbnails = "<u>Thumbnails:</u><br><table id='thumbnails'><tr><th>Perspective</th>"
        all_sizes = [thumbnail_default_size] + thumbnail_sizes

        # Add Column Headers for each Size
        for size_header in all_sizes:
            table_thumbnails += f"<th>{size_header}</th>"
        table_thumbnails += "</tr>"

        # Mapping thumbnail data to their corresponding perspective and size
        perspectives = [s.replace("_", " ").title() for s in thumbnail_identifiers]
        for perspective in perspectives:
            # Start a new row with the perspective name
            table_thumbnails += f"<tr><td>{perspective}</td>"

            # Check each size for current perspective
            for size_header in all_sizes:
                thumbnail_found = False

                # Search for thumbnail matching the current size and perspective
                for thumbnail_data in thumbnail_files:
                    if thumbnail_data["size"] == size_header and thumbnail_data["perspective"] == perspective:
                        # Add link to thumbnail if found
                        table_thumbnails += f"<td><a href='{thumbnail_data['link']}'>Link</a></td>"
                        thumbnail_found = True
                        break

                # If no matching thumbnail, add an empty cell
                if not thumbnail_found:
                    table_thumbnails += "<td></td>"

            # Close the row
            table_thumbnails += "</tr>"

        # Close the table
        table_thumbnails += "</table>"

        # If a Changelog is found, insert the tables right before it. Its addition/update shall be handled before this function.
        idx_changelog = description.find(changelog_identifier)
        description_text = description[:idx_changelog]
        changelog_section = description[idx_changelog:]
        if idx_changelog == -1:
            updated_description = description + table_main + table_thumbnails
        else:
            updated_description = description_text + table_main + table_thumbnails + changelog_section

    else:
        idx_changelog = description.find(changelog_identifier)
        description_text = description[:idx_changelog]
        changelog_section = description[idx_changelog:]
        if idx_changelog == -1:
            updated_description = description + table_main
        else:
            updated_description = description_text + table_main + changelog_section

    return updated_description


def append_image_metadata_to_description(
    description: str, image_metadata: Dict[str, Union[str, int, float, Dict[str, str]]], add_exif: bool = False
) -> str:
    """
    Appends image metadata and optional EXIF data to a given description.

    Args:
        description: The original description text.
        image_metadata: A dictionary containing image metadata.
        add_exif: Flag to include EXIF data if available.

    Returns:
        [0] (str) The updated description with appended metadata.
    """
    # Add image metadata
    description += f"<br><br><u>Image-Metadata:</u>"
    description += f"<br>Filename: {image_metadata['filename']}"
    description += f"<br>Image Dimensions: {image_metadata['width']}x{image_metadata['height']}"
    description += f"<br>Megapixels: {image_metadata['megapixels']} MP"
    description += f"<br>Filesize: {image_metadata['filesize']}<br>"

    # Add EXIF data
    if add_exif and image_metadata.get("exif", {}):
        description += "<br><u>EXIF Data:</u>"
        for key, value in image_metadata["exif"].items():
            description += f"<br>{key}: {value}"

    return description


def append_to_csv(data: List[Dict[str, Union[str, int, float]]], filename: str) -> None:
    """
    Appends data to a CSV file, creating the file if it doesn't exist.

    Args:
        data: A list of dictionaries containing the data to be appended.
        filename: The name of the CSV file to append to or create.

    Returns:
        None
    """
    df = pd.DataFrame(data)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="a", header=False, index=False)


def append_to_json(data: Any, filename: str) -> None:
    """
    Appends data to a JSON file, creating the file if it doesn't exist.

    Args:
        data: The data to append to the JSON file.
        filename: The name of the JSON file.

    Returns:
        [0] None

    Raises:
        JSONDecodeError: If the existing file contains invalid JSON.
    """
    try:
        with open(filename, "r+") as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(filename, "w") as file:
            json.dump([data], file, indent=4)


def clean_description(description: str) -> str:
    description = remove_tables_by_soup(description)
    # description = remove_table_headers(description)
    return description


def clean_string(input_str: str) -> str:
    if isinstance(input_str, str):
        corrected_str = correct_encoding(input_str)
        return corrected_str
    else:
        return str(input_str)


def construct_zenodo_data(received_data: Dict[str, Any], uploaded_file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs a structured Zenodo data dictionary from received data and uploaded file data.

    Args:
        received_data: Dictionary containing metadata and links from Zenodo.
        uploaded_file_data: Dictionary containing information about the uploaded file.

    Returns:
        [0] (dict): A structured dictionary containing combined Zenodo metadata, file information, and links.
    """
    zenodo_data = {
        "doi": str(received_data["metadata"]["prereserve_doi"]["doi"]),
        "id": str(received_data["id"]),
        "title": str(received_data["title"]),
        "created": str(received_data["created"]),
        "modified": str(received_data["modified"]),
        "record_id": str(received_data["record_id"]),
        "conceptrecid": str(received_data["record_id"]),
        "file": {
            "created": uploaded_file_data["created"],
            "updated": uploaded_file_data["updated"],
            "version_id": uploaded_file_data["version_id"],
            "filename": uploaded_file_data["key"],
            "filesize": uploaded_file_data["size"],
            "mimetype": uploaded_file_data["mimetype"],
            "checksum": uploaded_file_data["checksum"],
            "links": {
                "self": uploaded_file_data["links"]["self"],
                "version": uploaded_file_data["links"]["version"],
                "uploads": uploaded_file_data["links"]["uploads"],
            },
        },
        "links": {
            "file": uploaded_file_data["links"]["self"],
            "self": str(received_data["links"]["self"]),
            "html": str(received_data["links"]["html"]),
            "badge": str(received_data["links"]["badge"]),
            "files": str(received_data["links"]["files"]),
            "bucket": str(received_data["links"]["bucket"]),
            "newversion": str(received_data["links"]["newversion"]),
            "registerconceptdoi": str(received_data["links"]["registerconceptdoi"]),
            "publish": str(received_data["links"]["publish"]),
            "discard": str(received_data["links"]["discard"]),
        },
        "metadata": received_data["metadata"],
        "submitted": received_data["submitted"],
    }
    return zenodo_data


def correct_encoding(input_str: str) -> str:
    """
    Attempts to correct encoding issues in the input string by removing non-printable characters
    and trying different encoding/decoding methods.

    Args:
        input_str: The input string to be processed.

    Returns:
        The corrected string after encoding/decoding attempts.
    """
    # remove non-printable / binary characters and special characters afterwards
    if bool(re.search(r"[^\x20-\x7E]", input_str)):
        # input_str = re.sub(r"[^\x20-\x7E]", "", input_str)
        # input_str = re.sub(r"[^a-zA-Z0-9\s.,-]", "", input_str)
        input_str = re.sub(r"[^\w\s.,-]", "", input_str, flags=re.UNICODE)

    try:
        corrected_str = input_str.encode("windows-1252").decode("utf-8")
        return corrected_str
    except UnicodeDecodeError:
        try:
            corrected_str = input_str.encode("latin1").decode("utf-8")
            return corrected_str
        except UnicodeDecodeError:
            pass

    return input_str


def decode_exif_string(value: Union[str, bytes]) -> str:
    """
    Decodes EXIF string values from bytes to UTF-8 or Latin-1 encoded strings.

    Args:
        value: The EXIF string value to decode.

    Returns:
        The decoded string value.
    """
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin1")  # fallback
    return value


def download_image(image_url: str, target_dir: str) -> Tuple[Optional[str], bool]:
    """
    Downloads an image from a given URL to a specified directory.

    Args:
        image_url: The URL of the image to download.
        target_dir: The directory where the image will be saved.

    Returns:
        [0] The file path of the downloaded image, or None if download failed.
        [1] A boolean indicating whether the file already existed.

    Raises:
        requests.exceptions.RequestException: If there's an error with the HTTP request.
        Exception: For any other unexpected errors during the download process.
    """
    file_path = None
    file_path_existed = False

    try:
        head_response = requests.head(image_url)
        if head_response.status_code == 200 and "Content-Length" in head_response.headers:
            content_length = int(head_response.headers["Content-Length"])
            file_name = os.path.basename(urlparse(image_url).path)
            file_path = os.path.join(target_dir, file_name)

            if os.path.exists(file_path):
                existing_file_size = os.path.getsize(file_path)
                if existing_file_size == content_length:
                    file_path_existed = True
                    return file_path, file_path_existed

            logger.info(f"Downloading Image... {image_url}")
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                file_path_existed = False
                return file_path, file_path_existed
    except requests.exceptions.RequestException as e:
        logger.critical(f"Request Exception (download_image) ({image_url}): {e}")
        file_path = None
    except Exception as e:
        logger.critical(f"Exception (download_image) ({image_url}): {e}")
        file_path = None

    return file_path, file_path_existed


def extract_obj_metrics(obj_filepath: str) -> Dict[str, float]:
    """
    Extracts metrics from an OBJ file.

    Args:
        obj_filepath: Path to the OBJ file.

    Returns:
        [0] A dictionary containing the following metrics:
            - length: Length of the object.
            - width: Width of the object.
            - height: Height of the object.
            - points: Number of points in the object.
            - vertices: Number of vertices in the object.
            - primitives: Number of primitives in the object.

    Raises:
        Exception: If an error occurs while reading or processing the file.
    """
    metrics = {"length": 0.0, "width": 0.0, "height": 0.0, "points": 0, "vertices": 0, "primitives": 0}

    try:
        with open(obj_filepath, "r") as file:
            lines = file.readlines()

        for line in lines:
            if "Bounds:" in line:
                # Extract bounds and calculate dimensions
                bounds = line.split("Bounds: ")[1].strip("[]\n")
                # Split bounds into min and max by " to "
                min_bounds_str, max_bounds_str = bounds.split(" to ")
                # Convert min and max bounds strings into lists of floats
                min_bounds = [float(x) for x in min_bounds_str.strip("[]").split(", ")]
                max_bounds = [float(x) for x in max_bounds_str.strip("[]").split(", ")]
                # Calculate dimensions
                metrics["length"] = max_bounds[0] - min_bounds[0]
                metrics["width"] = max_bounds[1] - min_bounds[1]
                metrics["height"] = max_bounds[2] - min_bounds[2]
            elif "points" in line:
                metrics["points"] = int(line.split(" ")[1])
            elif "vertices" in line:
                metrics["vertices"] = int(line.split(" ")[1])
            elif "primitives" in line:
                metrics["primitives"] = int(line.split(" ")[1])

    except Exception as e:
        print(f"An error occurred: {e}")

    return metrics


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extracts timestamp from filename and converts it to a datetime object.

    Args:
        filename: The filename containing the timestamp.

    Returns:
        [0] Datetime object representing the extracted timestamp, or None if no match is found.
    """
    match = re.search(r"(\d{4})-(\w{3})-(\d{2})_(\d{2})h(\d{2})m(\d{2})s", filename)
    if match:
        return datetime.strptime(match.group(0), "%Y-%b-%d_%Hh%Mm%Ss")
    return None


def extract_url(description: str) -> Optional[str]:
    """
    Extracts the 4DCity URL from a given description string.

    Args:
        description: A string containing the description with a 4DCity URL.

    Returns:
        [0] The extracted 4DCity URL as a string, or None if the URL is not found.

    """
    # This extracts the 4DCity URL from the description in zenodo_metadata (without anchor tags)
    start = description.find("<u>4DCity URL</u>: ") + len("<u>4DCity URL</u>: ")
    end = description.find(".jpg", start) + 4
    if start > len("<u>4DCity URL</u>: ") - 1 and end > start:
        url = unescape(description[start:end])
        return url
    else:
        return None


def extract_url_from_anchor(description: str) -> Optional[str]:
    """
    Extracts the 4DCity URL from an anchor tag in the given description.

    Args:
        description: The HTML-formatted string containing the anchor tag.

    Returns:
        [0] The extracted and unescaped URL if found, None otherwise.
    """
    # This extracts the 4DCity URL by the anchor tag in the description
    pattern = r'<u>4DCity URL</u>: <a href="([^"]+)">'
    match = re.search(pattern, description)
    if match:
        url = unescape(match.group(1))
        return url
    else:
        return None


def get_base_dir() -> Union[str, bytes]:
    """
    Determines the base directory for the application.

    This function checks if the application is running inside a Docker container
    and returns the appropriate base directory path.

    Returns:
        [0] The base directory path as a string or bytes object.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if os.path.exists("/.dockerenv"):
        base_dir = "/app"
    else:
        base_dir = script_dir

    return base_dir


def get_changelog_from_description(description: str) -> str:
    """
    Extracts the changelog section from a given description string.

    Args:
        description: The full description text containing the changelog.

    Returns:
        The extracted changelog text, or an empty string if no changelog is found.

    Note:
        This function removes existing tables from the description before
        extracting the changelog to prevent issues from previous operations.
    """
    # delete existing tables, in order to not mess everything up if something went wrong previously
    if "<table>" in description:
        description = clean_description(description)

    changelog_start = description.find("<u>Changelog</u>:")

    if changelog_start == -1:
        return ""

    changelog = description[changelog_start:]
    return changelog.strip()


def get_filetype(filepath: Union[str, Path]) -> Literal["image", "audio", "video", "json", "document", "other"]:
    """
    Determines the file type based on the file extension and MIME type.

    Args:
        filepath: The path to the file, either as a string or Path object.

    Returns:
        A string indicating the file type: "image", "audio", "video", "json", "document", or "other".
    """
    # Convert to Path object if it's a string
    filepath = Path(filepath) if isinstance(filepath, str) else filepath

    # Get the file extension
    file_extension = filepath.suffix.lower()

    # Get the MIME type
    mime_type, _ = mimetypes.guess_type(str(filepath))

    # Define file type mappings
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    audio_extensions = {".mp3", ".wav", ".ogg", ".flac", ".aac"}
    video_extensions = {".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm"}
    document_extensions = {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"}

    # Check file type based on extension and MIME type
    if file_extension in image_extensions or (mime_type and mime_type.startswith("image/")):
        return "image"
    elif file_extension in audio_extensions or (mime_type and mime_type.startswith("audio/")):
        return "audio"
    elif file_extension in video_extensions or (mime_type and mime_type.startswith("video/")):
        return "video"
    elif file_extension == ".json" or mime_type == "application/json":
        return "json"
    elif file_extension in document_extensions or (mime_type and mime_type.startswith("application/")):
        return "document"
    else:
        return "other"


def get_image_metadata(
    image_path: Union[str, Path], remove_address: bool = True, remove_mail: bool = True
) -> Optional[Dict[str, Union[str, Dict[str, str]]]]:
    """
    Extracts metadata from an image file.

    Args:
        image_path: Path to the image file.
        remove_address: Flag to remove address information from metadata.
        remove_mail: Flag to remove email information from metadata.

    Returns:
        [0] (dict) A dictionary containing image metadata including filename, filesize, dimensions,
            megapixels, and EXIF data. Returns None if the image file doesn't exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    outdata = {}
    filename = image_path.name
    filesize = image_path.stat().st_size / 1024  # Bytes to Kilobytes
    with Image.open(image_path) as img:
        width, height = img.size
        megapixels = (width * height) / 1_000_000
        exif_data = img._getexif()
        exif_decoded_data = {}
        if exif_data:
            exif = {
                Image.ExifTags.TAGS.get(k, k): decode_exif_string(v)
                for k, v in exif_data.items()
                if k in Image.ExifTags.TAGS
            }
            for key, value in exif.items():
                if str(value) == "{}":
                    continue
                # check if MakerNote value is useful:
                if key == "MakerNote" and not is_readable_string(value):
                    continue
                elif key == "Copyright" or key == "Artist":
                    value = fix_encoding(str(value))
                    previous_value = value
                    if remove_address or remove_mail:
                        # remove personal information like the personal address, not recommended to publish that
                        value = remove_personal_information(value, remove_address, remove_mail)
                        if not value == previous_value:
                            value += " [stripped personal information]"
                exif_decoded_data[key] = fix_encoding(str(value))

    outdata = {
        "filename": filename,
        "filesize": f"{filesize:.2f} KB",
        "width": width,
        "height": height,
        "megapixels": f"{megapixels:.2f}",
        "exif": exif_decoded_data,
    }

    return outdata


def get_isodate(input_date: Union[str, float, int]) -> str:
    """
    Convert various date formats to ISO 8601 date (YYYY-MM-DD).
    Handles ISO, EXIF, and common date formats with/without time and timezone.

    Args:
        input_date: Date string, Unix timestamp, or other supported format.

    Returns:
        ISO 8601 formatted date string (YYYY-MM-DD).

    Raises:
        ValueError: If the input date format is not recognized.
    """
    date_formats = [
        # ISO formats
        "%Y-%m-%dT%H:%M:%S.%f%z",  # w/ milliseconds & timezone
        "%Y-%m-%dT%H:%M:%S%z",  # w/o milliseconds but w/ timezone
        "%Y-%m-%dT%H:%M:%S.%f",  # w/ milliseconds but w/o timezone
        "%Y-%m-%dT%H:%M:%S",  # w/o milliseconds & w/o timezone
        "%Y-%m-%d",  # date only
        # EXIF formats
        "%Y:%m:%d %H:%M:%S",  # standard EXIF format
        "%Y:%m:%d",  # EXIF date only
        # Common formats
        "%d/%m/%Y %H:%M:%S",  # dd/mm/yyyy HH:MM:SS
        "%d/%m/%Y",  # dd/mm/yyyy
        "%m/%d/%Y %H:%M:%S",  # mm/dd/yyyy HH:MM:SS
        "%m/%d/%Y",  # mm/dd/yyyy
        "%d.%m.%Y %H:%M:%S",  # dd.mm.yyyy HH:MM:SS
        "%d.%m.%Y",  # dd.mm.yyyy
        "%Y/%m/%d %H:%M:%S",  # yyyy/mm/dd HH:MM:SS
        "%Y/%m/%d",  # yyyy/mm/dd
    ]

    # Try to parse with explicit formats
    for fmt in date_formats:
        try:
            dt = datetime.strptime(input_date, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Try to parse timezone abbreviations
    try:
        # Remove timezone abbreviation if present
        cleaned_date = re.sub(r"\s[A-Z]{3,4}$", "", input_date)
        dt = datetime.fromisoformat(cleaned_date)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Try to parse Unix timestamp
    try:
        dt = datetime.fromtimestamp(float(input_date), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass

    raise ValueError(f"Time data '{input_date}' does not match any known format.")


def get_response_errors(response: requests.models.Response) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extracts error information from an HTTP response.

    Args:
        response: The HTTP response object to analyze.

    Returns:
        [0] (List[str]): A list of error messages extracted from the response.
        [1] (Dict[str, Any]): The complete error data parsed from the response JSON.

    Raises:
        json.JSONDecodeError: If the response body cannot be parsed as JSON.
        Exception: For any unexpected errors during processing.
    """
    errors = []
    error_data = {}

    if response.status_code in [400, 403, 404, 405, 409, 415, 429]:
        try:
            error_data = response.json()
            if isinstance(error_data, dict) and "errors" in error_data:
                errors = error_data["errors"]
            else:
                errors.append("Unexpected error format in response")
        except json.JSONDecodeError:
            errors.append("Unable to parse JSON from response")
        except Exception as e:
            errors.append(f"An unexpected error occurred: {str(e)}")

    return errors, error_data


def get_thumbnails(file_links: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Extracts and organizes thumbnail links for 3D model perspectives.

    Args:
        file_links: List of dictionaries containing file information.

    Returns:
        [0] (dict): Organized thumbnail links by perspective and resolution.
            Keys are perspective types, values are nested dicts of resolutions and links.

    Notes:
        Intended to use for target_type '3dmodels'.
    """
    thumbnails_dict = {}
    perspective_keys = ["perspective_1", "perspective_2", "perspective_3", "perspective_4", "perspective_top"]
    for file in file_links:
        filename = file["filename"]
        filelink = file["links"]["download"].replace("/draft", "")
        if not "perspective" in filename:
            continue
        for key in perspective_keys:
            if key in filename:
                if key not in thumbnails_dict:
                    thumbnails_dict[key] = {}

                resolution = filename.split("_")[-1].split(".")[0]
                thumbnails_dict[key][resolution] = filelink
    sorted_thumbnails_dict = {
        k: {
            ("1000x1000" if res in ["1", "2", "3", "4", "top"] else res): link
            for res, link in sorted(
                v.items(), key=lambda item: (item[0] in ["1", "2", "3", "4", "top"], item[0]), reverse=True
            )
        }
        for k, v in sorted(thumbnails_dict.items())
    }

    return sorted_thumbnails_dict if sorted_thumbnails_dict else {}


def increment_version(input_version: str, level: int) -> str:
    """
    Increments a specific part of a version string.

    This function takes a version string in the format "X.Y.Z" and increments
    the specified part of the version based on the provided level. It also
    handles cases where the version string contains suffixes (e.g., "1.2.3b").

    Args:
        input_version (str): The version string to be incremented, formatted as "X.Y.Z".
        level (int): The level to increment, where 1 refers to the last part (Z),
                     2 refers to the middle part (Y), and so on.

    Returns:
        str: The incremented version string.

    Example:
        >>> increment_version("1.2.3", 1)
        '1.2.4'
        >>> increment_version("1.2.3", 2)
        '1.3.3'
        >>> increment_version("1.2.3", 3)
        '2.2.3'
        >>> increment_version("0.0.2b", 1)
        '0.0.3'
    """
    import re

    # Separate numerical part and suffix
    match = re.match(r"(\d+)\.(\d+)\.(\d+)([a-z]*)", input_version, re.I)
    if not match:
        raise ValueError("Invalid version format")

    version_parts = list(map(int, match.groups()[:3]))
    suffix = match.groups()[3]

    # Increment the specified part
    version_parts[-level] += 1

    # Reset parts to the right of the incremented part to zero
    for i in range(-level + 1, 0):
        version_parts[i] = 0

    # Reassemble the version string without the suffix
    new_version = ".".join(map(str, version_parts))

    return new_version


def identify_dates_in_exif(exif_decoded_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and processes date-related information from EXIF metadata.

    Args:
        exif_decoded_data: Decoded EXIF data from an image.

    Returns:
        [0] (dict) A dictionary containing:
            - Extracted date information with standardized keys
            - An "ISO" object with ISO 8601 formatted dates
            - An "earliest_date" field with the earliest date found
    """
    outdata = {}
    date_tags = {
        "DateTimeOriginal": "original_time",
        "DateTimeDigitized": "digitized_time",
        "DateTime": "last_modified",
        "CreateDate": "create_date",
        "GPSDateStamp": "gps_date",
        "GPSTimeStamp": "gps_time",
        "SubSecTime": "subsec_time",
        "SubSecTimeOriginal": "subsec_time_original",
        "SubSecTimeDigitized": "subsec_time_digitized",
        "OffsetTime": "offset_time",
        "OffsetTimeOriginal": "offset_time_original",
        "OffsetTimeDigitized": "offset_time_digitized",
    }

    for exif_tag, output_key in date_tags.items():
        if exif_tag in exif_decoded_data:
            outdata[output_key] = exif_decoded_data[exif_tag]

    # Special handling for IPTC DateTimeCreated
    if "DateTimeCreated" in exif_decoded_data:
        outdata["iptc_date_created"] = exif_decoded_data["DateTimeCreated"]

    # Create "ISO" object with all values converted to ISO
    iso_dates = {}
    for key, value in outdata.items():
        try:
            iso_dates[key] = get_isodate(str(value))
        except ValueError:
            # Skip if conversion fails
            continue
    outdata["ISO"] = iso_dates

    # Create "earliest" date (ISO) for easier subsequent handling
    if iso_dates:
        earliest_date = min([i for i in iso_dates.values() if i])
        outdata["earliest_date"] = earliest_date

    return outdata


def is_edm_available(files_responses: List[Dict]) -> bool:
    """
    Check if any file in the files_responses ends with "_edm.xml".

    Args:
    files_responses (List[Dict]): List of file response dictionaries from Zenodo.

    Returns:
    bool: True if a file ending with "_edm.xml" is found, False otherwise.
    """
    return any(file["filename"].endswith("_edm.xml") for file in files_responses)


def is_metsmods_available(files_responses: List[Dict]) -> bool:
    """
    Check if any file in the files_responses ends with "_metsmods.xml".

    Args:
    files_responses (List[Dict]): List of file response dictionaries from Zenodo.

    Returns:
    bool: True if a file ending with "_metsmods.xml" is found, False otherwise.
    """
    return any(file["filename"].endswith("_metsmods.xml") for file in files_responses)


def is_readable_string(value: Union[bytes, str]) -> bool:
    """
    Determines if a given value is a readable string.

    For bytes, it checks if the proportion of non-printable characters is below a threshold.
    For strings, it verifies if all characters are printable.

    Args:
        value: The input to be checked for readability.

    Returns:
        True if the input is considered readable, False otherwise.

    Notes:
        Designed for MakerNote (EXIF) to check if a string is actually useful
    """
    threshold = 0.5
    if isinstance(value, bytes):
        printable_bytes = bytes([x for x in range(32, 127)])
        printable_part = value.translate(None, printable_bytes)
        return len(printable_part) / len(value) < threshold
    elif isinstance(value, str):
        return all(char in string.printable for char in value)
    return False


def load_config(config_path: str, replace_asterisk_vars: bool = True) -> Dict[str, Any]:
    """
    Load and process a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.
        replace_asterisk_vars: Flag to replace *variable references in strings.

    Returns:
        [0] (dict): Processed configuration dictionary containing key-value pairs
        from the YAML file, with environment variables and asterisk references
        replaced if specified.
    """
    # Load the YAML file as a string
    with open(config_path, "r") as file:
        yaml_str = file.read()

    # Replace YAML anchors with their corresponding values and get anchor mappings
    yaml_str, anchors = replace_yaml_anchors(yaml_str)

    # Parse the modified YAML string
    config = yaml.safe_load(yaml_str)

    # Replace environment variables
    config = replace_env_vars(config)

    # Replace asterisk variables if enabled
    if replace_asterisk_vars:
        config = replace_asterisk_references(config, anchors)

    return config


def printJSON(object: Dict[str, Any]) -> None:
    """
    Print a JSON representation of the given dictionary with indentation.

    Args:
        object: The dictionary to be printed as JSON.

    Returns:
        [0] None
    """
    print(json.dumps(object, indent=2, ensure_ascii=False))


def replace_yaml_anchors(yaml_str: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace YAML anchors with their corresponding values in a YAML string.

    Args:
        yaml_str: The input YAML string containing anchors.

    Returns:
        [0] The YAML string with anchor declarations removed.
        [1] A dictionary mapping anchor names to their corresponding values.
    """
    anchor_pattern = re.compile(r"&(\w+)\s+(.*)")

    anchors = {}
    lines = yaml_str.split("\n")

    # First pass: collect anchors
    for i, line in enumerate(lines):
        match = anchor_pattern.search(line)
        if match:
            anchor_name, value = match.groups()
            anchors[anchor_name] = value.strip('"')
            lines[i] = line.replace(f"&{anchor_name} ", "")  # Remove anchor declaration

    return "\n".join(lines), anchors


def replace_env_vars(data: Union[Dict[str, Any], List[Any], str, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    """
    Recursively replace environment variables in the input data.

    Args:
        data: The input data to process.

    Returns:
        The processed data with environment variables replaced:
        [0] If input is a dictionary, returns a new dictionary with replaced values.
        [1] If input is a list, returns a new list with replaced values.
        [2] If input is a string, returns the string with environment variables expanded.
        [3] For any other type, returns the input unchanged.
    """
    if isinstance(data, dict):
        return {k: replace_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_env_vars(i) for i in data]
    elif isinstance(data, str):
        return os.path.expandvars(data)
    return data


def replace_asterisk_references(config: Dict[str, Any], anchors: Dict[str, str]) -> Dict[str, Any]:
    """
    Recursively replace asterisk-prefixed variable references in a nested configuration dictionary.

    Args:
        config: The configuration dictionary to process.
        anchors: A dictionary of anchor variables and their corresponding values.

    Returns:
        [0] (dict): A new configuration dictionary with all asterisk references replaced.
    """

    def replace_in_string(s: str) -> str:
        for key, value in anchors.items():
            s = s.replace(f"*{key}", value)
        return s

    def process_paths(paths: Dict[str, Any]) -> Dict[str, Any]:
        processed_paths = {}
        for key, value in paths.items():
            if isinstance(value, str):
                processed_paths[key] = replace_in_string(value)
            elif isinstance(value, dict):
                processed_paths[key] = process_paths(value)
            elif isinstance(value, list):
                processed_paths[key] = [replace_in_string(item) if isinstance(item, str) else item for item in value]
            else:
                processed_paths[key] = value
        return processed_paths

    processed_config = {}
    for section, data in config.items():
        if isinstance(data, dict):
            processed_config[section] = process_paths(data)
        else:
            processed_config[section] = data

    return processed_config


def load_json(json_path: Path) -> Union[Dict[str, Any], List[Any]]:
    """
    Load and parse a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        [0] Parsed JSON data.

    Raises:
        JSONDecodeError: If the file contains invalid JSON.
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the user lacks permission to read the file.
    """
    data = {}
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def merge_csvs(target_directory: Path, remove_draft: bool = True) -> str:
    """
    Merges CSV files in the target directory and optionally removes draft content.

    Args:
        target_directory: Directory containing CSV files to merge.
        remove_draft: Flag to remove draft content from CSV files.

    Returns:
        Path of the merged CSV file as a string.
    """
    csv_files = list(target_directory.glob("*.csv"))
    dataframes = []
    latest_timestamp = None

    for csv_file in csv_files:
        if remove_draft:
            # eventually change column_name here if needed
            remove_url_segment_from_csv(csv_file, column_name="glb")
        df = pd.read_csv(csv_file)
        dataframes.append(df)

        # Extract the timestamp from the filename
        timestamp = extract_timestamp_from_filename(csv_file.stem)
        if timestamp and (latest_timestamp is None or timestamp > latest_timestamp):
            latest_timestamp = timestamp

    merged_df = pd.concat(dataframes, ignore_index=True)

    # Use the latest timestamp for the output filename
    if latest_timestamp:
        output_filename = f"(merged) glb_3dmodels_{latest_timestamp.strftime('%Y-%b-%d_%Hh%Mm%Ss')}.csv"
    else:
        output_filename = "(merged) glb_3dmodels.csv"

    output_path = target_directory / output_filename

    merged_df.to_csv(output_path, index=False)

    return str(output_path)


def remove_personal_information(input_str: str, remove_address: bool = True, remove_mail: bool = True) -> str:
    """
    Removes personal information from the input string.

    Args:
        input_str: String containing potential personal information.
        remove_address: Flag to remove street addresses.
        remove_mail: Flag to remove email addresses.

    Returns:
        Cleaned string with personal information removed.
    """
    # Replace "\\n" with a comma and a space before processing
    preprocessed_str = input_str.replace("\\n", ", ")

    if remove_address:
        street_pattern = r"\b[\w\s]+(?:\b\w*\.?\s*\d+\b)[\w\s,]*"
        preprocessed_str = re.sub(street_pattern, "", preprocessed_str, flags=re.IGNORECASE).strip()

    if remove_mail:
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        preprocessed_str = re.sub(email_pattern, "", preprocessed_str)

    # Remove trailing commas and whitespaces
    cleaned_str = re.sub(r"[, ]+$", "", preprocessed_str)

    return cleaned_str


def remove_tables_by_soup(description: str) -> str:
    """
    Removes tables and cleans up HTML content in the description.

    Args:
        description: HTML string containing description content.

    Returns:
        Cleaned HTML string with tables removed and formatting adjusted.
    """
    # Remove tables (assuming they've already been removed in this case)
    description = re.sub(r"<table>.*?</table>", "", description, flags=re.DOTALL)

    # Remove specific headers
    description = re.sub(r"<u>(Main Files:|Thumbnails:)</u>", "", description)

    # Clean up the description
    description = re.sub(r"&nbsp;", " ", description)
    description = re.sub(r"\s+", " ", description)

    # Standardize all br tags to <br/>
    description = re.sub(r"<br\s*/?\s*>", "<br/>", description)

    # Split the description into non-changelog and changelog parts
    parts = re.split(r"(<u>Changelog</u>:.*)", description, maxsplit=1, flags=re.DOTALL)

    if len(parts) > 1:
        non_changelog, changelog = parts[0], parts[1]
    else:
        non_changelog, changelog = parts[0], ""

    # Handle the non-changelog section
    non_changelog = re.sub(r"(<br/>){3,}", "<br/><br/>", non_changelog)

    # Handle the changelog section
    if changelog:
        # Limit <br/> tags to one between entries in the changelog section
        changelog = re.sub(r"(<br/>)+", "<br/>", changelog)
        # Ensure there's only one <br/> before each hyphen
        changelog = re.sub(r"<br/>\s*-", "<br/>- ", changelog)

    # Combine the parts back together
    description = non_changelog + changelog

    return description.strip()


def remove_table_headers(description: str) -> str:
    """
    Remove specific table headers and excessive line breaks from the description.

    Args:
        description: The input description text.

    Returns:
        The description with specified table headers and excessive line breaks removed.
    """
    # Define the patterns to remove
    patterns = [
        r"<br>(?:<br>)*<u>Main Files:?</u>",
        r"<br>(?:<br>)*<u>Thumbnails:?</u>",
    ]

    # Remove each pattern from the description
    for pattern in patterns:
        description = re.sub(pattern, "", description, flags=re.IGNORECASE)

    # Remove occurrences of more than two consecutive <br> tags
    description = re.sub(r"(<br>){3,}", "<br><br>", description)

    return description.strip()


def remove_tables_from_description(description: str) -> str:
    """
    Remove tables and clean up formatting in the description.

    Args:
        description: The input description text.

    Returns:
        The cleaned description with tables removed and formatting adjusted.
    """
    # delete existing main and thumbnail tables. applying different cases for inconsistent linebreak cases
    pattern_0 = re.compile(r"<p><span>Main Files:</span></p>.*?(?=<br><br><br><u>Changelog</u>)", re.DOTALL)
    pattern_1 = re.compile(r"<br><br><u>Main Files:</u><br>.*?(?=<br><br><br><u>Changelog</u>)", re.DOTALL)
    pattern_2 = re.compile(r"<br><u>Main Files:</u><br>.*?(?=<br><br><br><u>Changelog</u>)", re.DOTALL)
    pattern_3 = re.compile(r"<u>Main Files:</u>.*?(?=<u>Changelog</u>)", re.DOTALL)
    description = re.sub(pattern_0, "", description)
    description = re.sub(pattern_1, "", description)
    description = re.sub(pattern_2, "", description)
    description = re.sub(pattern_3, "", description)
    # in case there is no Changelog, apply the same to clean the Metadata from anything appearing after those tables.
    if not "<u>Changelog</u>" in description:
        pattern_0b = re.compile(r"<p><span>Main Files:</span></p>.*", re.DOTALL)
        pattern_1b = re.compile(r"<br><br><u>Main Files:</u><br>.", re.DOTALL)
        pattern_2b = re.compile(r"<br><u>Main Files:</u><br>.*", re.DOTALL)
        pattern_3b = re.compile(r"<u>Main Files:</u>.*", re.DOTALL)
        description = re.sub(pattern_0b, "", description)
        description = re.sub(pattern_1b, "", description)
        description = re.sub(pattern_2b, "", description)
        description = re.sub(pattern_3b, "", description)
    # additionally clean trash
    if "\n<p>&nbsp;</p>\n" in description:
        description = description.replace("\n<p>&nbsp;</p>\n", "")
    if "<p>" in description or "</p>" in description:
        description = description.replace("<p>", "")
        description = description.replace("</p>", "<br>")

    if "<table>" in description:
        # Pattern to match table structures, including nested tables
        table_pattern = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

        # Remove all tables
        while re.search(table_pattern, description):
            description = re.sub(table_pattern, "", description)

    return description


def remove_url_segment_from_csv(csv_path: Path, column_name: str) -> None:
    """
    Remove '/draft/' segment from URLs in a specified column of a CSV file.

    Args:
        csv_path: Path to the CSV file.
        column_name: Name of the column containing URLs.

    Returns:
        None
    """
    df = pd.read_csv(csv_path)

    if column_name in df.columns:
        df[column_name] = df[column_name].str.replace("/draft/", "/", regex=False)

    df.to_csv(csv_path, index=False)


def resize_image(filepath: Union[str, Path], config: Dict[str, Any]) -> Path:
    """
    Resize an image based on configuration settings.

    Args:
        filepath: Path to the input image file.
        config: Configuration dictionary containing resize settings.

    Returns:
        Path to the resized image file.
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    outpath = Path(f'{config["paths"]["output"]["resized_images"]}/{filepath.stem}_resized{filepath.suffix}')
    use_ratio = config["settings"]["image_resize"]["use_ratio"]

    with Image.open(filepath) as img:
        width, height = img.size

        if use_ratio:
            ratio = config["settings"]["image_resize"]["ratio"]
            new_width = int(width * ratio)
            new_height = int(height * ratio)
        else:
            max_dimension_value = config["settings"]["image_resize"]["max_dimension_value"]
            outpath = Path(
                f'{config["paths"]["output"]["resized_images"]}/{filepath.stem}_{max_dimension_value}px{filepath.suffix}'
            )
            # Calculate the scaling factor
            scale = max_dimension_value / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)

        # Resize the image using LANCZOS resampling for high-quality downscaling
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        outpath.parent.mkdir(parents=True, exist_ok=True)
        resized_img.save(outpath, quality=95, optimize=True)

    return outpath


def search_file(
    filename: Union[str, Path],
    search_dir: Union[str, Path] = "",
    extensions: List[str] = [],
    search_subdirectories: bool = False,
    config: dict = {},
) -> Optional[Path]:
    """
    Search for a file with the given filename and allowed extensions in the specified directory.
    Optionally searches in subdirectories. Ignores case for file extensions.
    Returns the file path if found, None otherwise.

    Args:
        filename (Union[str, Path]): The filename to search for.
        search_dir (Union[str, Path]): The directory to search in. Defaults to current directory.
        extensions (List[str]): List of allowed file extensions.
        search_subdirectories (bool): Whether to search in subdirectories. Defaults to False.
        config (dict): Configuration dictionary. If provided, overrides search_dir and extensions.

    Returns:
        [0] (Path): The path of the found file, or None if not found.
    """
    base_filename = Path(filename).stem

    if config:
        search_dir = Path(config["paths"]["input"]["images"])
        extensions = config["misc"]["image_extensions"]
    else:
        search_dir = Path(search_dir) if search_dir else Path.cwd()

    # Ensure extensions start with a dot
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

    # Define the search pattern
    pattern = f"{base_filename}.*" if extensions else base_filename

    # Define the search method based on whether to include subdirectories
    search_method = search_dir.rglob if search_subdirectories else search_dir.glob

    for file in search_method(pattern):
        if not extensions or file.suffix.lower() in [ext.lower() for ext in extensions]:
            return file

    return None


def setup_logger(config: Dict[str, Any], name: str) -> logging.Logger:
    """
    Set up a logger with file and console handlers based on the provided configuration.

    Args:
        config: Configuration dictionary containing logging settings.
        name: Name of the logger.

    Returns:
        Configured logger object.
    """
    now = datetime.now().strftime(config["logging"]["now_dateformat"])
    log_file_path = Path(
        config["logging"]["log_file_pattern"].format(
            projectTitle=config["project_title"], shortTitle=config["short_title"], now=now
        )
    )

    log_level = getattr(logging, config["logging"]["level"])
    log_format = config["logging"]["log_format"]

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)

    # Ensure the parent directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler setup
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler setup
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


@contextmanager
def silent_stdout() -> Generator[io.StringIO, None, None]:
    """
    Context manager to temporarily redirect stdout to a StringIO object.

    Yields:
        StringIO object containing the captured stdout.
    """
    new_target = io.StringIO()
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def update_config(key_path: List[str], new_value: Any) -> bool:
    """
    Update a specific value in the config.yml file based on the provided key path.

    Args:
        key_path: List of keys representing the path to the value in the config.
        new_value: New value to be set.

    Returns:
        True if update was successful, False otherwise.

    Notes:
        Beware: This resets the formatting of the configuration file and removes comments
    """
    try:
        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)

        temp = config
        for key in key_path[:-1]:
            if key in temp:
                temp = temp[key]
            else:
                return False

        if key_path[-1] in temp:
            temp[key_path[-1]] = new_value
        else:
            return False

        with open("config.yml", "w") as file:
            yaml.safe_dump(config, file, default_flow_style=False)

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def update_metadata_path_in_config(key: str, new_value: str) -> None:
    """
    Update the value of the 'zenodo_metadata_json' key in the config.yml file.

    Args:
        key: Key to be updated (should be 'zenodo_metadata_json').
        new_value: New value to be set for the key.

    Notes:
        Specifically updates the value for the unique key "zenodo_metadata_json".
    """
    file_path = "config.yml"
    with open(file_path, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}:"):
            lines[i] = f'    {key}: "{new_value}"\n'  # 4 Spaces (= 2 Tabs) for this level
            break

    with open(file_path, "w") as file:
        file.writelines(lines)


def update_changelog_in_description(
    record_data: Dict[str, Any], description: str, new_version: str, changelog: str
) -> str:
    """
    Updates the changelog in the description with a new version and changelog entry.

    Args:
        record_data: Dictionary containing record metadata.
        description: Current description text.
        new_version: New version number to be added.
        changelog: Changelog text for the new version.

    Returns:
        Updated description text with the new changelog entry.
    """
    doi = record_data["metadata"]["prereserve_doi"]["doi"]
    version_doi_link = f"https://doi.org/{doi}"

    # handle simple addition if this is the first changelog
    if not "<u>Changelog</u>:" in description:
        description += "<br><br><u>Changelog</u>: <br>"

    # delete existing tables, in order to not mess everything up if something went wrong previously
    if "<table>" in description:
        description = clean_description(description)

    description += f'<br>- v<a href="{version_doi_link}">{new_version}</a>: {changelog}'

    updated_description = description
    return updated_description


def update_description(
    record_data: Dict[str, Any], uploaded_files_data: Dict[str, Any], new_version: str, changelog: str
) -> str:
    """
    Updates the description with new changelog and file tables.

    Args:
        record_data: Dictionary containing record metadata and existing files.
        uploaded_files_data: Dictionary containing data for newly uploaded files.
        new_version: New version number to be added to the changelog.
        changelog: Changelog text for the new version.

    Returns:
        Updated description text with new changelog and file tables.
    """
    # add existing and uploaded filedata to files for description filetables
    files = {i["filename"]: i["links"]["download"].replace("/draft", "") for i in record_data["files"]}
    files.update({i["filename"]: i["links"]["download"] for i in uploaded_files_data})
    old_description = record_data["metadata"]["description"]
    updated_description = update_changelog_in_description(record_data, old_description, new_version, changelog)
    new_description = add_filetables_to_description(updated_description, files)
    return new_description


def validate_edm_xml(xsd_path: Union[str, Path], xml_path: Union[str, Path] = None, xml_string: str = "") -> List[str]:
    """
    Validates an XML file or string against an EDM (Europeana Data Model) XSD schema.

    Args:
        xsd_path: Path to the XSD schema file.
        xml_path: Path to the XML file to validate (optional).
        xml_string: XML content as a string to validate (optional).

    Returns:
        [0] (str) Validation status message.
        [1:] (str) Error messages, if any.

    Raises:
        ValueError: If neither xml_path nor xml_string is provided.
    """
    errors = []

    try:
        # Parse the XSD file
        xmlschema_doc = etree.parse(str(xsd_path))
        xmlschema = etree.XMLSchema(xmlschema_doc)

        # Parse the XML
        if xml_path:
            xml_doc = etree.parse(str(xml_path))
        elif xml_string:
            xml_doc = etree.fromstring(xml_string.encode("utf-8"))
        else:
            raise ValueError("Either xml_path or xml_string must be provided")

        # Validate the XML against the schema
        is_valid = xmlschema.validate(xml_doc)

        if is_valid:
            errors.append("The XML is valid according to the provided EDM schema.")
        else:
            errors.append("The XML is not valid according to the provided EDM schema.")
            errors.append("Validation errors:")
            for error in xmlschema.error_log:
                errors.append(f"  Line {error.line}: {error.message}")

    except etree.XMLSyntaxError as e:
        errors.append(f"XML parsing error: {str(e)}")
    except etree.XMLSchemaParseError as e:
        errors.append(f"XSD parsing error: {str(e)}")
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"An error occurred: {str(e)}")

    return errors


def validate_metsmods(
    mets_xsd_path: Union[str, Path],
    mods_xsd_path: Union[str, Path],
    xml_path: Union[str, Path] = None,
    xml_string: str = "",
) -> List[str]:
    """
    Validates an XML file or string against METS and MODS schemas.

    Args:
        mets_xsd_path: Path to the METS XSD schema file.
        mods_xsd_path: Path to the MODS XSD schema file.
        xml_path: Path to the XML file to validate (optional).
        xml_string: XML content as a string to validate (optional).

    Returns:
        [0] (str) METS validation status message.
        [1:] (str) METS error messages, if any.
        [...] (str) MODS section validation status and error messages, if any.

    Raises:
        ValueError: If neither xml_path nor xml_string is provided.
    """
    results = []

    try:
        # Parse the XSD files
        mets_xmlschema_doc = etree.parse(str(mets_xsd_path))
        mods_xmlschema_doc = etree.parse(str(mods_xsd_path))

        # Create schema objects
        mets_xmlschema = etree.XMLSchema(mets_xmlschema_doc)
        mods_xmlschema = etree.XMLSchema(mods_xmlschema_doc)

        # Parse the XML
        if xml_path:
            xml_doc = etree.parse(str(xml_path))
        elif xml_string:
            xml_doc = etree.fromstring(xml_string.encode("utf-8"))
        else:
            raise ValueError("Either xml_path or xml_string must be provided")

        # Validate against METS schema
        is_valid_mets = mets_xmlschema.validate(xml_doc)
        if is_valid_mets:
            results.append("The XML is valid according to the METS schema.")
        else:
            results.append("The XML is not valid according to the METS schema.")
            results.append("METS validation errors:")
            for error in mets_xmlschema.error_log:
                results.append(f"  Line {error.line}: {error.message}")

        # Validate MODS sections
        mods_sections = xml_doc.xpath("//mods:mods", namespaces={"mods": "http://www.loc.gov/mods/v3"})
        if mods_sections:
            for i, mods_section in enumerate(mods_sections, 1):
                is_valid_mods = mods_xmlschema.validate(mods_section)
                if is_valid_mods:
                    results.append(f"MODS section {i} is valid.")
                else:
                    results.append(f"MODS section {i} is not valid.")
                    results.append(f"MODS section {i} validation errors:")
                    for error in mods_xmlschema.error_log:
                        results.append(f"  Line {error.line}: {error.message}")
        else:
            results.append("No MODS sections found in the XML.")

    except etree.XMLSyntaxError as e:
        results.append(f"XML parsing error: {str(e)}")
    except etree.XMLSchemaParseError as e:
        results.append(f"XSD parsing error: {str(e)}")
    except ValueError as e:
        results.append(str(e))
    except Exception as e:
        results.append(f"An error occurred: {str(e)}")

    return results


def validate_zenodo_metadata(zenodo_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Validates Zenodo metadata against a set of predefined rules and constraints.

    This function checks the structure, required fields, data types, and specific
    content requirements for Zenodo metadata. It performs comprehensive validation
    including checks for upload types, publication types, date formats, creator
    information, access rights, and more.

    Args:
        zenodo_metadata: The Zenodo metadata to be validated.

    Returns:
        A list of error messages. An empty list indicates no validation errors.
    """
    errors = []

    # Check if the "metadata" key exists and extract the actual metadata
    if "metadata" not in zenodo_metadata:
        errors.append("Missing required 'metadata' key")
        return errors
    metadata = zenodo_metadata["metadata"]

    required_fields = ["upload_type", "publication_date", "title", "creators", "description", "access_right"]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")

    valid_datatypes = {
        "upload_type": str,
        "publication_type": str,
        "image_type": str,
        "publication_date": str,
        "title": str,
        "creators": list,  # list of dicts
        "description": str,
        "access_right": str,
        "license": str,
        "embargo_date": str,
        "access_conditions": str,
        "doi": str,
        "prereserve_doi": dict,
        "keywords": list,  # list of strings
        "notes": str,
        "related_identifiers": list,  # list of dicts
        "contributors": list,  # list of dicts
        "references": list,  # list of strings
        "communities": list,  # list of dicts
        "grants": list,  # list of dicts
        "journal_title": str,
        "journal_volume": str,
        "journal_issue": str,
        "journal_pages": str,
        "conference_title": str,
        "conference_acronym": str,
        "conference_dates": str,
        "conference_place": str,
        "conference_url": str,
        "conference_session": str,
        "conference_session_part": str,
        "imprint_publisher": str,
        "imprint_isbn": str,
        "imprint_place": str,
        "partof_title": str,
        "partof_pages": str,
        "thesis_supervisors": list,  # list of dicts
        "thesis_university": str,
        "subjects": list,  # list of dicts
        "version": str,
        "language": str,
        "locations": list,  # list of dicts
        "dates": list,  # list of dicts
        "method": str,
    }

    # Check for unknown fields
    for field in metadata:
        if field not in valid_datatypes:
            errors.append(f"Unknown field: {field}")

    # Check datatypes
    for field, expected_type in valid_datatypes.items():
        if field in metadata:
            value = metadata[field]
            if not isinstance(value, expected_type):
                errors.append(f"{field} must be of type {expected_type.__name__}, not {type(value).__name__}")
            else:
                # Additional checks for list types
                if expected_type == list:
                    if field in ["keywords", "references"]:
                        if not all(isinstance(item, str) for item in value):
                            errors.append(f"All items in {field} must be strings")
                    elif field in [
                        "creators",
                        "contributors",
                        "related_identifiers",
                        "communities",
                        "grants",
                        "thesis_supervisors",
                        "subjects",
                        "locations",
                        "dates",
                    ]:
                        if not all(isinstance(item, dict) for item in value):
                            errors.append(f"All items in {field} must be dictionaries")
                        else:
                            # Additional checks for specific list of dicts
                            if field == "creators" or field == "contributors":
                                for item in value:
                                    if "name" not in item:
                                        errors.append(f"Each item in {field} must have a 'name' key")
                            elif field == "related_identifiers":
                                for item in value:
                                    if "identifier" not in item or "relation" not in item:
                                        errors.append(
                                            f"Each item in {field} must have 'identifier' and 'relation' keys"
                                        )
                            elif field == "communities":
                                for item in value:
                                    if "identifier" not in item:
                                        errors.append(f"Each item in {field} must have an 'identifier' key")
                            elif field == "grants":
                                for item in value:
                                    if "id" not in item:
                                        errors.append(f"Each item in {field} must have an 'id' key")
                            elif field == "subjects":
                                for item in value:
                                    if "term" not in item or "identifier" not in item:
                                        errors.append(f"Each item in {field} must have 'term' and 'identifier' keys")
                            elif field == "locations":
                                for item in value:
                                    if "place" not in item:
                                        errors.append(f"Each item in {field} must have a 'place' key")
                            elif field == "dates":
                                for item in value:
                                    if "type" not in item or ("start" not in item and "end" not in item):
                                        errors.append(
                                            f"Each item in {field} must have a 'type' key and at least one of 'start' or 'end' keys"
                                        )

    valid_upload_types = [
        "publication",
        "poster",
        "presentation",
        "dataset",
        "image",
        "video",
        "software",
        "lesson",
        "physicalobject",
        "other",
    ]
    if "upload_type" in metadata and metadata["upload_type"] not in valid_upload_types:
        errors.append(f"Invalid upload_type: {metadata['upload_type']}")

    # Validate publication_type if upload_type is 'publication'
    if metadata.get("upload_type") == "publication":
        if "publication_type" not in metadata:
            errors.append("Missing publication_type for upload_type 'publication'")
        else:
            valid_pub_types = [
                "annotationcollection",
                "book",
                "section",
                "conferencepaper",
                "datamanagementplan",
                "article",
                "patent",
                "preprint",
                "deliverable",
                "milestone",
                "proposal",
                "report",
                "softwaredocumentation",
                "taxonomictreatment",
                "technicalnote",
                "thesis",
                "workingpaper",
                "other",
            ]
            if metadata["publication_type"] not in valid_pub_types:
                errors.append(f"Invalid publication_type: {metadata['publication_type']}")

    # Validate image_type if upload_type is 'image'
    if metadata.get("upload_type") == "image":
        if "image_type" not in metadata:
            errors.append("Missing image_type for upload_type 'image'")
        else:
            valid_image_types = ["figure", "plot", "drawing", "diagram", "photo", "other"]
            if metadata["image_type"] not in valid_image_types:
                errors.append(f"Invalid image_type: {metadata['image_type']}")

    # Validate publication_date
    if "publication_date" in metadata:
        try:
            datetime.strptime(metadata["publication_date"], "%Y-%m-%d")
        except ValueError:
            errors.append("Invalid publication_date format. Use YYYY-MM-DD.")

    # Validate creators
    if "creators" in metadata:
        if not isinstance(metadata["creators"], list):
            errors.append("creators must be a list of objects")
        else:
            for creator in metadata["creators"]:
                if "name" not in creator:
                    errors.append("Each creator must have a 'name'")
                if "orcid" in creator and not re.match(r"\d{4}-\d{4}-\d{4}-\d{3}[0-9X]", creator["orcid"]):
                    errors.append(f"Invalid ORCID format for creator: {creator['name']}")

    # Validate access_right
    valid_access_rights = ["open", "embargoed", "restricted", "closed"]
    if "access_right" in metadata and metadata["access_right"] not in valid_access_rights:
        errors.append(f"Invalid access_right: {metadata['access_right']}")

    # Validate license if access_right is 'open' or 'embargoed'
    if metadata.get("access_right") in ["open", "embargoed"]:
        if "license" not in metadata:
            errors.append("Missing license for open or embargoed access_right")

    # Validate embargo_date if access_right is 'embargoed'
    if metadata.get("access_right") == "embargoed":
        if "embargo_date" not in metadata:
            errors.append("Missing embargo_date for embargoed access_right")
        else:
            try:
                embargo_date = datetime.strptime(metadata["embargo_date"], "%Y-%m-%d")
                if embargo_date <= datetime.now():
                    errors.append("embargo_date must be in the future")
            except ValueError:
                errors.append("Invalid embargo_date format. Use YYYY-MM-DD.")

    # Validate access_conditions if access_right is 'restricted'
    if metadata.get("access_right") == "restricted":
        if "access_conditions" not in metadata:
            errors.append("Missing access_conditions for restricted access_right")

    return errors


def write_json(data: Union[Dict[str, Any], List[Any]], json_path: Union[str, Path]) -> bool:
    """
    Write data to a JSON file with UTF-8 encoding and formatted indentation.

    Args:
    - data: The data to write to the JSON file.
    - json_path: The path to the JSON file.

    Returns:
    [0] (bool) True if the write operation was successful, False otherwise.

    Raises:
    - Exception: If an error occurs during the file writing process.
    """
    try:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"(write_json) An error occurred while writing to the file: {e}")
        return False
