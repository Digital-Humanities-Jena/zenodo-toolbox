from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

import logging
import re
import time
from typing import Any, Dict, List, Iterator, Union

logger = logging.getLogger("zenodo-toolbox")


def reduce_string_levels(input_str: str, split_by_hyphens: bool = False) -> Iterator[str]:
    """
    Generates progressively reduced versions of the input string.

    Args:
        input_str: The input string to be processed.
        split_by_hyphens: If True, splits the string by both spaces and hyphens.

    Returns:
        An iterator yielding progressively reduced versions of the input string.

    Example:
        >>> list(reduce_string_levels("a b c", False))
        ['a b c', 'b c', 'c']
        >>> list(reduce_string_levels("a-b c", True))
        ['a-b c', 'b c', 'c']
    """
    if split_by_hyphens:
        words = re.split(r"[ -]", input_str)
    else:
        words = input_str.split()

    for i in range(len(words)):
        yield " ".join(words[i:])


def retrieve_coordinates(
    input_data: Dict[str, str], custom_config: Dict[str, Any]
) -> List[Dict[str, Union[float, str]]]:
    """
    Retrieves geographical coordinates based on input data and custom configuration.

    Args:
        input_data: Dictionary containing location information.
        custom_config: Dictionary with configuration settings for geocoding.

    Returns:
        [0] Dictionary containing latitude, longitude, place name, and description.
    """
    config = custom_config if custom_config else {}

    location_details = []
    geolocator = Nominatim(user_agent=config["geolocator"]["user_agent"], timeout=config["geolocator"]["timeout"])

    basevalues = config.get("column_basevalues", {})

    country_bv = basevalues.get("country", "country")
    sublocation_bv = basevalues.get("sublocation", "sublocation")
    location_bv = basevalues.get("location", "location")
    province_bv = basevalues.get("province", "province")

    country = input_data.get(country_bv, "").replace(".", "")
    state = input_data.get(province_bv, "").replace(".", "")
    location = input_data.get(location_bv, "").replace(".", "")
    sublocation = input_data.get(sublocation_bv, "").replace(".", "")
    description = ""
    structured_query = {}

    detailed_structured_query = config["geolocator"]["detailed_structured_query"]
    reducer_split_by_hyphens = config["geolocator"]["reducer_split_by_hyphens"]

    if detailed_structured_query == True:
        # not really tested /- ignore this for now
        if "," in state:
            splitted_state = state.split(",", 1)
            state = splitted_state[0].strip()
            province = splitted_state[1].strip()

            structured_query = {
                "country": country,
                "state": state,
                "county": province,
                "city": location,
                "query": sublocation,
            }

            description = f"{country}/{state}/{province}/{location}/{sublocation}"

        else:
            structured_query = {
                "country": country,
                "state": state,
                "location": location,
                "query": sublocation,
            }

            description = f"{country}/{state}/{location}/{sublocation}"
    else:
        structured_query = {"country": country, "city": location, "street": sublocation}

    # print(f"Structured Query: {structured_query}")

    if "," in state:
        splitted_state = state.split(",", 1)
        state = splitted_state[0].strip()
        province = splitted_state[1].strip()
        description = f"{country}/{state}/{province}/{location}/{sublocation}"
    else:
        description = f"{country}/{state}/{location}/{sublocation}"

    attempts = 0
    max_attempts = config["geolocator"]["max_attempts"]
    backoff_factor = config["geolocator"]["backoff_factor"]
    while attempts < max_attempts:
        try:
            location_data = geolocator.geocode(structured_query)
            if location_data:
                location_details.append(
                    {
                        "lat": location_data.latitude,
                        "lon": location_data.longitude,
                        "place": location,
                        "description": description,
                    }
                )
                return location_details
            attempts = max_attempts
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Geocoding failed on attempt {attempts + 1}: {e}")
            time.sleep(backoff_factor**attempts)
            attempts += 1

    if not location_details:
        print("Reducing Sublocation Details...")
        for reduced_sublocation in reduce_string_levels(sublocation, reducer_split_by_hyphens):
            structured_query["street"] = reduced_sublocation

            try:
                location_data = geolocator.geocode(structured_query)
                if location_data:
                    location_details.append(
                        {
                            "lat": location_data.latitude,
                            "lon": location_data.longitude,
                            "place": location,
                            "description": description,
                        }
                    )
                    return location_details
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                print(f"Failed during reduction attempt: {e}")
                break

            time.sleep(config["geolocator"]["sleep_time"])

    if not location_details:
        location_details = [
            {"lat": 0.0, "lon": 0.0, "place": location if location != "" else "Unknown", "description": description}
        ]
        print("No results found after reducing Sublocation.")
        return location_details
