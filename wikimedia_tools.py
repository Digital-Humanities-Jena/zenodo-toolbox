import requests
import json
import time
import csv
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from requests.exceptions import RequestException
from urllib.parse import urlencode

logger = logging.getLogger("zenodo-toolbox")

ApiParams = Dict[str, str]
ImageData = List[Tuple[str, Optional[float], Optional[float]]]


class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.tokens = max_calls
        self.last_check = time.time()

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            self.wait_for_token()
            return func(*args, **kwargs)

        return wrapper

    def wait_for_token(self):
        now = time.time()
        time_passed = now - self.last_check
        self.last_check = now

        self.tokens += time_passed * (self.max_calls / self.period)
        if self.tokens > self.max_calls:
            self.tokens = self.max_calls

        if self.tokens < 1:
            sleep_time = (1 - self.tokens) * (self.period / self.max_calls)
            time.sleep(sleep_time)
            self.tokens = 0
        else:
            self.tokens -= 1


rate_limiter = RateLimiter(max_calls=50, period=1)  # 50 calls per second


@rate_limiter
def api_call(url: str, params: Dict[str, str]) -> requests.Response:
    return requests.get(url, params=params, timeout=10)


def append_to_csv(data: List[Tuple[str, Optional[float], Optional[float]]], filename: str):
    try:
        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Original Filename", "Latitude", "Longitude"])
            writer.writerows(data)
    except IOError as e:
        logger.error(f"Error writing to CSV file: {e}")


def extract_image_info(
    metadata: Dict[str, Any], original_filename: str
) -> Tuple[Optional[float], Optional[float], str]:
    pages = metadata.get("query", {}).get("pages", {})
    if not pages:
        logger.warning("No pages found in metadata")
        return None, None, original_filename

    page_id = next(iter(pages))
    page_data = pages[page_id]

    # Extract coordinates
    coordinates = page_data.get("coordinates", [])
    if coordinates:
        latitude = coordinates[0].get("lat")
        longitude = coordinates[0].get("lon")
    else:
        imageinfo = page_data.get("imageinfo", [{}])[0]
        extmetadata = imageinfo.get("extmetadata", {})
        latitude = extmetadata.get("GPSLatitude", {}).get("value")
        longitude = extmetadata.get("GPSLongitude", {}).get("value")

    logger.debug(f"Extracted info: lat={latitude}, lon={longitude}, filename={original_filename}")

    return (float(latitude) if latitude else 0.0, float(longitude) if longitude else 0.0, original_filename)


def fetch_images(params: ApiParams, api_url: str) -> List[str]:
    all_images = []
    while True:
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract image titles, remove "File:" prefix, and replace spaces with underscores
            images = [
                item["title"].replace("File:", "").replace(" ", "_") for item in data["query"]["categorymembers"]
            ]
            all_images.extend(images)

            if "continue" not in data:
                break

            params.update(data["continue"])
            time.sleep(1)  # Delay to respect rate limits
        except RequestException as e:
            logger.error(f"Error fetching images: {e}")
            time.sleep(5)  # Wait longer before retrying
        except json.JSONDecodeError:
            logger.error("Error decoding JSON response")
            break

    return all_images


def get_api_params(author: str, limit: int = 500) -> Tuple[ApiParams, str]:
    """
    Generate API parameters for querying images by author.

    Args:
        author (str): The username of the author.
        limit (int): The number of results to return per query (default 500).

    Returns:
        Tuple[ApiParams, str]: A tuple containing the API parameters and the API URL.
    """
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:Photographs by User:{author}",
        "cmlimit": str(limit),
        "cmtype": "file",
        "continue": "",
    }
    return params, api_url


def get_image_coordinates(image_title: str) -> Dict[str, Optional[Tuple[float, float]]]:
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": image_title,
        "prop": "coordinates|wbentity",
        "wbptterms": "alias",
    }
    try:
        response = api_call(api_url, params)
        response.raise_for_status()
        data = response.json()
        pages = data["query"]["pages"]
        page_id = next(iter(pages))
        page_data = pages[page_id]

        result = {"camera_location": None, "object_location": None}

        # Get camera location
        if "coordinates" in page_data:
            coords = page_data["coordinates"][0]
            result["camera_location"] = (coords["lat"], coords["lon"])

        # Get object location
        if "wbentity" in page_data and "claims" in page_data["wbentity"]:
            claims = page_data["wbentity"]["claims"]
            if "P1259" in claims:  # P1259 is the property for coordinates
                coord_claim = claims["P1259"][0]
                if "mainsnak" in coord_claim and "datavalue" in coord_claim["mainsnak"]:
                    coord_value = coord_claim["mainsnak"]["datavalue"]["value"]
                    result["object_location"] = (coord_value["latitude"], coord_value["longitude"])

        return result
    except RequestException as e:
        logger.error(f"Error fetching data for {image_title}: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error processing data for {image_title}: {e}")

    return {"camera_location": None, "object_location": None}


def get_image_metadata(image_title: str) -> Dict[str, Any]:
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "imageinfo|coordinates",
        "iiprop": "extmetadata",
        "titles": f"File:{image_title}",  # Add "File:" prefix here
        "format": "json",
    }
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"API response for {image_title}: {json.dumps(data, indent=2)}")
        return data
    except requests.RequestException as e:
        logger.error(f"Error fetching metadata for {image_title}: {e}")
        return {}


def load_processed_images(filename: str) -> Set[str]:
    """
    Load the set of already processed image titles from a CSV file.

    Args:
        filename (str): The name of the CSV file.

    Returns:
        Set[str]: A set of processed image titles.
    """
    processed_images = set()
    try:
        with open(filename, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            processed_images = {row[0] for row in reader}
    except FileNotFoundError:
        logger.info(f"No existing file found: {filename}")
    return processed_images


def params_to_url(api_url: str, params: Dict[str, str]) -> str:
    """
    Convert API URL and parameters to a single URL string.

    Args:
        api_url (str): The base API URL.
        params (Dict[str, str]): The API parameters.

    Returns:
        str: A complete URL string that can be used in a browser.
    """
    return f"{api_url}?{urlencode(params)}"


def process_images(images: List[str]) -> List[Tuple[str, Optional[float], Optional[float]]]:
    results = []
    for image in images:
        metadata = get_image_metadata(image)
        lat, lon, original_filename = extract_image_info(metadata)
        if original_filename:
            results.append((original_filename, lat, lon))
        else:
            logger.warning(f"Could not extract information for image: {image}")
    return results


def main():
    author = "Author"  # enter Author here
    csv_filename = "image_coordinates.csv"

    processed_images = load_processed_images(csv_filename)
    logger.info(f"Found {len(processed_images)} already processed images")

    params, api_url = get_api_params(author)
    images = fetch_images(params, api_url)

    logger.info(f"Total images found: {len(images)}")

    results = []
    for i, image in enumerate(images, 1):
        logger.debug(f"Processing image {i}: {image}")
        if image in processed_images:
            logger.info(f"Skipping already processed image: {image}")
            continue

        try:
            metadata = get_image_metadata(image)
            logger.debug(f"Metadata retrieved for {image}")
            lat, lon, original_filename = extract_image_info(metadata, image)
            logger.debug(f"Extracted info: lat={lat}, lon={lon}, filename={original_filename}")
            results.append((original_filename, lat, lon))
        except Exception as e:
            logger.error(f"Error processing image {image}: {e}")

        if i % 100 == 0:
            logger.info(f"Processed {i} images")
            append_to_csv(results, csv_filename)
            results = []

    if results:
        append_to_csv(results, csv_filename)

    logger.info("Processing complete. Results saved to image_coordinates.csv")


if __name__ == "__main__":
    main()
