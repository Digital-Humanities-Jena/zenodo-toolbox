from datetime import datetime
import logging
import numpy as np
import openpyxl
import pandas as pd
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Union

from utilities import (
    add_filetables_to_description,
    append_image_metadata_to_description,
)

logger = logging.getLogger("zenodo-toolbox")


def clean_author_string(
    input_str: str, custom_config: Dict[str, Dict[str, Union[bool, List[str]]]] = {}
) -> Union[str, List[str]]:
    """
    Clean and format author string based on provided configuration.

    Args:
        input_str: The input author string to be cleaned.
        custom_config: Custom configuration dictionary for cleaning options.

    Returns:
        Either a cleaned author string or a list of author strings if split_authors is True and multiple authors are found.
    """
    config = custom_config if custom_config else {}

    add_space_after_colon = config["settings"]["add_space_after_colon_in_author"]
    capitalize_patterns = config["misc"]["capitalize_author_substrings"]
    split_authors = config["settings"]["split_authors"]
    split_chars = config["misc"]["split_characters"]["author"]

    if add_space_after_colon:
        input_str = re.sub(r"(:)(\S)", r"\1 \2", input_str)

    if capitalize_patterns:
        pattern = r"^(" + "|".join(re.escape(p) for p in capitalize_patterns) + r")"
        match = re.match(pattern, input_str, re.IGNORECASE)

        if match:
            prefix = match.group(1)
            capitalized_prefix = prefix[0].upper() + prefix[1:].lower()
            input_str = capitalized_prefix + input_str[len(prefix) :]

    if split_authors:
        split_pattern = "|".join(re.escape(char) for char in split_chars)
        authors = [author.strip() for author in re.split(split_pattern, input_str) if author.strip()]

        if len(authors) > 1:
            return authors

    return input_str


def clean_description_text(text: str) -> str:
    """
    Clean and format the input text by removing unnecessary commas, extra spaces, and ensuring proper spacing after periods.

    Args:
        text: The input string to be cleaned.

    Returns:
        The cleaned and formatted string.
    """
    # Remove comma if it immediately follows a period
    text = re.sub(r"\.,", ".", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # Ensure there's a space after each period (except at the end of the string)
    text = re.sub(r"\.(?!$|\s)", ". ", text)
    return text.strip()


def clean_copyright_string(input_str: str, custom_config: Dict[str, Any] = {}) -> str:
    """
    Cleans a copyright string based on default and custom configurations.

    Args:
        input_str: The input copyright string to be cleaned.
        custom_config: A dictionary containing custom configuration options.

    Returns:
        The cleaned copyright string.
    """
    config = custom_config if custom_config else {}

    remove_cpright_symbol = config["settings"]["remove_copyright_symbol"]
    remove_mail_from_cpright = config["settings"]["remove_mail_from_copyright"]

    if remove_cpright_symbol:
        input_str = input_str.replace("©", "").strip()
    if remove_mail_from_cpright:
        input_str = re.sub(r",?\s*[^,\s]+@[^,\s]+(?:\.[^,\s]+)*", "", input_str).rstrip(".").rstrip("-").strip()

    return input_str


def construct_description(
    row_data: Dict[str, Any], files_data: Dict[str, Any] = {}, custom_config: Dict[str, Any] = {}
) -> str:
    """
    Constructs a description string based on provided data and configuration.

    Args:
        row_data: Dictionary containing data for description construction.
        files_data: Optional dictionary containing file-related data.
        custom_config: Optional dictionary for custom configuration settings.

    Returns:
        [0] (str) Constructed description string.
    """
    config = custom_config if custom_config else {}
    description_parts: List[str] = []
    description_constructor = config.get("description_constructor", {})

    add_filetables = config["settings"].get("add_filetables_to_description", False)

    # Sort the lines based on their numeric suffix
    sorted_lines = sorted(description_constructor.items(), key=lambda x: int(x[0].split("_")[1]))

    # Dynamically process each line
    for line, template in sorted_lines:
        try:
            # Extract the field name from the template
            field_name = template.split("{")[1].split("}")[0]

            if field_name in row_data:
                content = row_data[field_name]

                # Handle list content
                if isinstance(content, list):
                    if field_name == "keywords":
                        content = ", ".join(str(item) for item in content)
                    elif field_name == "description":
                        content = " ".join(str(item) for item in content)
                        content = clean_description_text(content)

                # Format the template with the content
                formatted_line = template.format(**{field_name: content})
                description_parts.append(formatted_line)
        except (IndexError, KeyError) as e:
            # If there's an error in template format or missing key, skip this line
            print(f"Warning: Error processing {line}: {e}")

    # Join all parts and strip any trailing whitespace
    description: str = "".join(description_parts).rstrip()

    if add_filetables and files_data:
        description = add_filetables_to_description(description, files_data)

    return description


def construct_title(row_data: Dict[str, Any], custom_config: Dict[str, Union[str, Dict[str, str]]] = {}) -> str:
    """
    Constructs a title string from row data based on a custom configuration.

    Args:
        row_data: Dictionary containing data for title construction.
        custom_config: Optional configuration dictionary for title construction.

    Returns:
        [0] (str) Constructed title string.
    """
    config = custom_config if custom_config else {}
    title_parts: List[str] = []
    title_constructor = config.get("title_constructor", {})

    # Dynamically process each position
    for pos, value in sorted(title_constructor.items()):
        if pos.startswith("pos_"):
            if value in row_data:
                content = row_data[value]
                if isinstance(content, list):
                    content = " ".join(content)
                title_parts.append(str(content).strip())

    # Join the parts with the specified separator, ensuring spaces
    separator: str = f" {title_constructor.get('separator', '-')} "
    title: str = separator.join(filter(bool, title_parts))

    return title.strip()


def construct_zenodo_metadata(
    row_data: Dict[str, Any],
    image_metadata: Dict[str, Any] = {},
    dates_data: Dict[str, str] = {},
    locations_data: Dict[str, Any] = {},
    files_data: Dict[str, Any] = {},
    custom_config: Dict[str, Any] = {},
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs Zenodo metadata from various input data sources.

    Args:
        row_data: Data for a single row.
        image_metadata: Metadata for associated images.
        dates_data: Date-related information.
        locations_data: Location-related information.
        files_data: Information about associated files.
        custom_config: Custom configuration settings.

    Returns:
        [0] A dictionary containing Zenodo metadata with nested structure.
    """
    config = custom_config if custom_config else {}
    metadata_settings = config["zenodo_metadata"]
    zenodo_metadata_out = {}

    basevalues = config.get("column_basevalues", {})
    keywords_bv = basevalues.get("keywords", "keywords")

    upload_type = metadata_settings["upload_type"]
    image_type = metadata_settings["image_type"]
    publication_date = datetime.now().strftime("%Y-%m-%d")
    title = construct_title(row_data, config)
    description = construct_description(row_data, files_data=files_data, custom_config=config)
    if config["settings"]["add_image_metadata_to_description"] and image_metadata:
        add_exif = config["settings"]["add_exif_to_description"]
        description = append_image_metadata_to_description(description, image_metadata, add_exif)
    creators = metadata_settings["creators"]
    notes = ""
    imprint_publisher = "Zenodo"
    access_right = "open"
    license = "cc-by-4.0"
    keywords = row_data[keywords_bv]
    contributors = metadata_settings["contributors"]
    communities = []
    grants = []
    version = metadata_settings["initial_version"]
    locations = locations_data

    dates = []
    if dates_data and dates_data.get("earliest_date", ""):
        dates = [
            {
                "start": dates_data["earliest_date"],
                "end": dates_data["earliest_date"],
                "type": "Collected",
            }
        ]

    # check if title already exists and append a numerus currens in that case (check again then)

    zenodo_metadata_out = {
        "metadata": {
            "upload_type": upload_type,
            "image_type": image_type,
            "publication_date": publication_date,
            "title": title,
            "creators": creators,
            "description": description,
            "notes": notes,
            "imprint_publisher": imprint_publisher,
            "access_right": access_right,
            "license": license,
            "keywords": keywords,
            "notes": notes,
            "contributors": contributors,
            "communities": communities,
            "grants": grants,
            "version": version,
            "locations": locations,
            "dates": dates,
        }
    }

    if not dates:
        del zenodo_metadata_out["metadata"]["dates"]
    if not locations:
        del zenodo_metadata_out["metadata"]["locations"]
    if not communities:
        del zenodo_metadata_out["metadata"]["communities"]
    if not grants:
        del zenodo_metadata_out["metadata"]["grants"]

    return zenodo_metadata_out


def convert_description_to_list(description_str: str, custom_config: Dict[str, Any] = {}) -> Union[str, List[str]]:
    """
    Converts a description string into a list of sentences based on custom configuration.

    Args:
        description_str: The input description string to be converted.
        custom_config: Custom configuration dictionary for splitting rules.

    Returns:
        [0] The original string if no split occurred or the input is invalid.
        [1] A list of split sentences based on the configuration rules.

    """
    config = custom_config if custom_config else {}

    if not description_str or not isinstance(description_str, str):
        return description_str  # Return the input as is if it's not a non-empty string

    split_chars = config["misc"]["split_characters"]["description"]
    exceptions = config["misc"]["split_exceptions"]["description"]

    # Sort exceptions by length (longest first) to avoid partial matches
    exceptions = sorted(exceptions, key=len, reverse=True)

    # Function to check if a string ends with any exception
    def ends_with_exception(s):
        return any(s.endswith(ex) for ex in exceptions)

    sentences = []
    current_sentence = ""

    # Iterate through each character in the description
    for i, char in enumerate(description_str):
        current_sentence += char

        # Check if the current character is a split character
        if char in split_chars:
            # If the current sentence doesn't end with an exception, it's a complete sentence
            if not ends_with_exception(current_sentence.strip()):
                sentences.append(current_sentence.strip())
                current_sentence = ""

        # If it's the last character, add the remaining sentence
        elif i == len(description_str) - 1:
            sentences.append(current_sentence.strip())

    # Remove any empty sentences
    sentences = [sentence for sentence in sentences if sentence]
    cleaned_sentences = []
    for char in split_chars:
        for sentence in sentences:
            cleaned_sentences.append(sentence.rstrip(char))

    # Return the original string if no split occurred
    if len(cleaned_sentences) <= 1:
        return description_str.strip()

    return cleaned_sentences


def convert_keywords_to_list(
    keywords_str: str, custom_config: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
) -> List[str]:
    """
    Converts a string of keywords into a sorted list of unique keywords.

    Args:
        keywords_str: A string containing keywords.
        custom_config: A configuration dictionary with split characters.

    Returns:
        [0] A sorted list of unique keywords.
    """
    config = custom_config if custom_config else {}

    split_chars = config["misc"]["split_characters"]["keywords"]
    unique_items = set()

    # Start with the first split character
    groups = keywords_str.split(split_chars[0])

    for group in groups:
        # If there's a second split character, split again
        if len(split_chars) > 1:
            items = group.split(split_chars[1])
        else:
            items = [group]

        stripped_items = [item.strip() for item in items if item.strip()]
        unique_items.update(stripped_items)

    return sorted(list(unique_items))


def excel_to_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Converts an Excel file to a pandas DataFrame.

    Args:
        file_path: Path to the Excel file.

    Returns:
        [0] A pandas DataFrame containing the data from the Excel file.

    Raises:
        FileNotFoundError: If the specified Excel file is not found.
        Exception: For other errors encountered while loading the Excel file.
    """
    logger.info(f"Converting Excel File to Dataframe ... ({file_path})")
    try:
        workbook = openpyxl.load_workbook(file_path)
    except FileNotFoundError:
        logger.critical(f"Error: Excel File not found: {file_path} ")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error with Loading Excel File: {e}")
    sheet = workbook.active

    data_rows = []
    for row in sheet.iter_rows(values_only=True):
        data_rows.append(row)

    df = pd.DataFrame(data_rows[1:], columns=data_rows[0])
    return df


def get_mapped_entry(dataframe: pd.DataFrame, idx: int, custom_config: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Extract and process a single row from a DataFrame based on custom configuration.

    Args:
        dataframe: Input DataFrame containing the data.
        idx: Index of the row to process.
        custom_config: Configuration dictionary for customizing the mapping process.

    Returns:
        [0] Processed dictionary containing mapped and cleaned data from the specified row.
    """
    config = custom_config if custom_config else {}

    column_mapping = config.get("column_mapping", {})
    basevalues = config.get("column_basevalues", {})
    split_keywords = config.get("settings", {}).get("split_keywords", False)
    split_description = config.get("settings", {}).get("split_description", False)

    author_bv = basevalues.get("author", "author")
    copyright_bv = basevalues.get("copyright", "copyright")
    description_bv = basevalues.get("description", "description")
    keywords_bv = basevalues.get("keywords", "keywords")

    # Use a dictionary comprehension to create the output dictionary
    mapped_data = {column_mapping[col]: dataframe[col].iloc[idx] for col in dataframe.columns if col in column_mapping}

    if author_bv in mapped_data and pd.notna(mapped_data[author_bv]):
        mapped_data[author_bv] = clean_author_string(mapped_data[author_bv], config)
        if author_bv != "author" and "author" in mapped_data:
            del mapped_data["author"]

    if copyright_bv in mapped_data and pd.notna(mapped_data[copyright_bv]):
        mapped_data[copyright_bv] = clean_copyright_string(mapped_data[copyright_bv], config)
        if copyright_bv != "copyright" and "copyright" in mapped_data:
            del mapped_data["copyright"]

    if split_description and description_bv in mapped_data and pd.notna(mapped_data[description_bv]):
        mapped_data[description_bv] = convert_description_to_list(mapped_data[description_bv], config)
        if description_bv != "description" and "description" in mapped_data:
            del mapped_data["description"]

    if split_keywords and keywords_bv in mapped_data and pd.notna(mapped_data[keywords_bv]):
        mapped_data[keywords_bv] = convert_keywords_to_list(mapped_data[keywords_bv], config)
        if keywords_bv != "keywords" and "keywords" in mapped_data:
            del mapped_data["keywords"]

    # Handle potential missing values
    for key, value in mapped_data.items():
        if isinstance(value, (pd.Series, np.ndarray)):
            # For array-like values, replace NaN with None
            mapped_data[key] = [None if pd.isna(v) else v for v in value]
        elif isinstance(value, list):
            # For list values, strip whitespace from string elements
            mapped_data[key] = [v.strip() if isinstance(v, str) else v for v in value]
        elif pd.isna(value):
            mapped_data[key] = None
        elif isinstance(value, str):
            # Strip whitespace from string values
            mapped_data[key] = value.strip()

    return mapped_data
