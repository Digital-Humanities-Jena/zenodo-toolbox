from collections import defaultdict
from datetime import datetime
import logging
import numpy as np
import openpyxl
import os
import pandas as pd
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from utilities import add_filetables_to_description, append_image_metadata_to_description

logger = logging.getLogger("zenodo-toolbox")


def check_header_convergence(
    directory: Union[str, Path],
    reference_file: Optional[Union[str, Path]] = None,
    common_headers: Optional[List[str]] = None,
    print_all_headers: bool = False,
) -> None:
    """
    Checks header convergence for Excel files in the specified directory.

    Args:
        directory: Path to the directory containing Excel files.
        reference_file: Path to the Excel file that determines the common headers.
        common_headers: List of strings defining the common headers.
        print_all_headers: If True, prints all column headers for each Excel file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified directory or reference file is not found.
        ValueError: If both reference_file and common_headers are provided or if neither is provided.
    """
    if (reference_file is not None) and (common_headers is not None):
        raise ValueError("Either reference_file or common_headers should be provided, not both.")

    if (reference_file is None) and (common_headers is None):
        raise ValueError("Either reference_file or common_headers must be provided.")

    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    excel_files = sorted([f for f in directory_path.glob("*.xlsx") if f.is_file()])
    if not excel_files:
        logger.warning(f"No Excel files found in directory: {directory}")
        return

    headers_dict: Dict[str, List[str]] = {}
    non_matching_headers_info = []
    all_headers_info = []

    # Determine common headers
    if reference_file:
        ref_file_path = Path(reference_file)
        if not ref_file_path.is_absolute():
            ref_file_path = Path(directory) / ref_file_path

        if not ref_file_path.exists():
            raise FileNotFoundError(f"Reference file not found: {ref_file_path}")

        try:
            ref_df = excel_to_dataframe(ref_file_path)
            common_headers = ref_df.columns.tolist()
        except FileNotFoundError:
            logger.critical(f"Error: Reference Excel File not found: {ref_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing reference file {ref_file_path.name}: {e}")
            raise

    print(f"Common headers determined: {common_headers}")

    # Process all Excel files
    for file_path in excel_files:
        try:
            df = excel_to_dataframe(file_path)
            headers = df.columns.tolist()
            headers_dict[file_path.name] = headers

            if set(headers) != set(common_headers):
                missing_headers = set(common_headers) - set(headers)
                added_headers = set(headers) - set(common_headers)
                non_matching_headers_info.append((file_path.name, missing_headers, added_headers))

            if print_all_headers:
                all_headers_info.append((file_path.name, headers))

        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")

    if not headers_dict:
        logger.warning("No valid Excel files were processed.")
        return

    # Print sorted non-matching headers information
    if non_matching_headers_info:
        print("\nFiles with non-matching headers:")
        for filename, missing, added in sorted(non_matching_headers_info):
            print(f"\nHeaders in {filename} do not match the common headers.")
            if missing:
                print(f"  Missing headers: {', '.join(sorted(missing))}")
            if added:
                print(f"  Added headers: {', '.join(sorted(added))}")

    # Print all headers if requested
    if print_all_headers:
        print("\nAll headers:")
        for filename, headers in sorted(all_headers_info):
            print(f"Headers in {filename}:")
            print(headers)
            print()

    # Check for files with non-converging headers
    non_converging_files = [
        filename for filename, headers in headers_dict.items() if set(headers) != set(common_headers)
    ]

    if non_converging_files:
        print("\nFiles with non-converging headers:")
        for filename in sorted(non_converging_files):
            print(f"- {filename}")
    else:
        print("\nAll files have converging headers.")


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


def construct_title(row_data: Dict[str, Any], custom_config: Dict[str, Union[str, Dict[str, Any]]] = {}) -> str:
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
    commands = title_constructor.get("commands", [])

    # Check if we need to remove extensions
    remove_ext = any(cmd.get("remove_ext", False) for cmd in commands)

    # Dynamically process each position
    for pos, value in sorted(title_constructor.items()):
        if pos.startswith("pos_"):
            if value in row_data:
                content = row_data[value]
                if isinstance(content, list):
                    content = " ".join(map(str, content))
                else:
                    content = str(content)

                # Remove extension if command is set and content looks like a file path
                if remove_ext:
                    try:
                        path = Path(content)
                        if path.suffix:
                            content = str(path.with_suffix(""))
                    except Exception:
                        # If it's not a valid path, keep the original content
                        pass

                title_parts.append(content.strip())

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
        [1] A list of unique split sentences based on the configuration rules.

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

    # Remove any empty sentences and duplicates while maintaining order
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        for char in split_chars:
            cleaned_sentence = cleaned_sentence.rstrip(char)
        if cleaned_sentence and cleaned_sentence not in seen:
            unique_sentences.append(cleaned_sentence)
            seen.add(cleaned_sentence)

    # Return the original string if no split occurred
    if len(unique_sentences) <= 1:
        return description_str.strip()

    return unique_sentences


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


def handle_duplicates_in_df(df: pd.DataFrame, filename_header: str, delete_divergent: bool = False) -> pd.DataFrame:
    """
    Identifies duplicates in the dataframe based on the filename_header,
    compares their data, and handles them accordingly.

    Args:
        df (pd.DataFrame): The input dataframe.
        filename_header (str): The column name containing filenames.
        delete_divergent (bool): If True, deletes divergent duplicates keeping the row with more data.

    Returns:
        pd.DataFrame: The dataframe with duplicates handled.
    """
    # Identify duplicates
    duplicates = df[df.duplicated(subset=[filename_header], keep=False)]

    if duplicates.empty:
        print("No duplicates found.")
        return df

    # Group duplicates
    grouped = duplicates.groupby(filename_header)

    convergent_duplicates = []
    divergent_duplicates = []

    for filename, group in grouped:
        if len(group) > 1:
            is_convergent = True
            divergent_fields = {}

            # Compare all rows in the group
            first_row = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                for column in df.columns:
                    if column != filename_header and first_row[column] != row[column]:
                        is_convergent = False
                        if column not in divergent_fields:
                            divergent_fields[column] = set()
                        divergent_fields[column].add(first_row[column])
                        divergent_fields[column].add(row[column])

            if is_convergent:
                convergent_duplicates.append(filename)
            else:
                divergent_duplicates.append((filename, divergent_fields))

    # Print divergent duplicates
    if divergent_duplicates:
        print("Divergent duplicates found:")
        for filename, fields in divergent_duplicates:
            print(f"\nFilename: {filename}")
            for field, values in fields.items():
                print(f"  {field}: {values}")

    # Remove convergent duplicates
    df = df[~df[filename_header].isin(convergent_duplicates) | ~df.duplicated(subset=[filename_header], keep="first")]

    # Handle divergent duplicates if requested
    if delete_divergent:
        for filename, _ in divergent_duplicates:
            duplicate_rows = df[df[filename_header] == filename]
            row_to_keep = duplicate_rows.loc[duplicate_rows.notna().sum(axis=1).idxmax()]
            df = pd.concat([df[df[filename_header] != filename], pd.DataFrame([row_to_keep])], ignore_index=True)

    print(f"\nRemoved {len(convergent_duplicates)} convergent duplicates.")
    if delete_divergent:
        print(f"Removed {len(divergent_duplicates)} divergent duplicates, keeping rows with more data.")

    return df


def identify_and_delete_duplicate_files(
    target_directory: str,
    df_filenames: Set[str],
    file_extensions: List[str] = None,
    identifier_pattern: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[str, str]]]:
    """
    Identifies duplicate files in the target directory based on a regex identifier pattern,
    keeps the largest file, renames it if necessary, and deletes smaller duplicates.

    Args:
        target_directory (str): Path to the directory containing the files.
        df_filenames (Set[str]): Set of filenames referenced in the dataframe.
        file_extensions (List[str], optional): List of file extensions to consider. If None, all files are considered.
        identifier_pattern (str, optional): Regex pattern to identify duplicates. If None, full filename is used.

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, Tuple[str, str]]]:
            - A dictionary of deleted files, where keys are identifiers and values are lists of deleted file paths.
            - A dictionary of kept files, where keys are identifiers and values are tuples of (original name, final name).
    """
    target_path = Path(target_directory)

    # Get all files in the target directory
    if file_extensions:
        all_files = [f for ext in file_extensions for f in target_path.glob(f"*{ext}")]
    else:
        all_files = list(target_path.glob("*.*"))

    # Group files by identifier
    file_groups = defaultdict(list)
    for file_path in all_files:
        if identifier_pattern:
            match = re.search(identifier_pattern, file_path.name)
            if match:
                key = match.group()
            else:
                continue  # Skip files that don't match the identifier pattern
        else:
            key = file_path.stem  # Use full filename (without extension) if no identifier is provided

        file_groups[key].append(file_path)

    # Identify duplicates
    duplicates = {key: paths for key, paths in file_groups.items() if len(paths) > 1}

    deleted_files = defaultdict(list)
    kept_files = {}
    for key, paths in duplicates.items():
        # Sort paths by file size (largest first)
        sorted_paths = sorted(paths, key=lambda p: p.stat().st_size, reverse=True)

        # Keep the largest file
        kept_file = sorted_paths[0]
        original_name = kept_file.name

        # Check if the kept file needs to be renamed
        df_matching_files = [f for f in df_filenames if key in f]
        if df_matching_files:
            new_name = df_matching_files[0]
            if kept_file.name != new_name:
                new_path = kept_file.parent / new_name
                if new_path.exists():
                    # If the new name already exists, it's probably a smaller duplicate
                    # We'll delete it and then rename our kept file
                    os.remove(new_path)
                    deleted_files[key].append(str(new_path))
                    print(f"Deleted existing smaller file: {new_path}")
                kept_file.rename(new_path)
                print(f"Renamed {kept_file.name} to {new_name}")
                kept_file = new_path

        kept_files[key] = (original_name, kept_file.name)

        # Delete the smaller duplicates
        for file_path in sorted_paths[1:]:
            if file_path != kept_file:  # Ensure we're not deleting the file we want to keep
                try:
                    os.remove(file_path)
                    deleted_files[key].append(str(file_path))
                    print(f"Deleted duplicate: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    # Print summary
    print(f"Found {len(duplicates)} sets of duplicate files.")
    total_deleted = sum(len(files) for files in deleted_files.values())
    print(f"{total_deleted} files were deleted.")

    return dict(deleted_files), kept_files


def merge_excel_to_df(directory: Union[str, Path], add_source: bool = True) -> pd.DataFrame:
    """
    Merges all Excel files in the specified directory into a single DataFrame.

    Args:
        directory: Path to the directory containing Excel files.
        add_source: If True, adds a 'Source_File' column to identify the source of each row.

    Returns:
        A pandas DataFrame containing the merged data from all Excel files.

    Raises:
        FileNotFoundError: If the specified directory is not found.
        ValueError: If no Excel files are found in the directory.
    """
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    excel_files = [f for f in directory_path.glob("*.xlsx") if f.is_file()]
    if not excel_files:
        raise ValueError(f"No Excel files found in directory: {directory}")

    merged_df = pd.DataFrame()

    for file_path in excel_files:
        try:
            df = excel_to_dataframe(file_path)

            # Add a column to identify the source file if requested
            if add_source:
                df["Source_File"] = file_path.name

            # Append the data to the merged DataFrame
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df], ignore_index=True)

            logger.info(f"Successfully merged data from: {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")

    if merged_df.empty:
        raise ValueError("No data could be merged from the Excel files.")

    logger.info(f"Successfully merged data from {len(excel_files)} Excel files.")

    return merged_df


def rename_files_by_identifier(
    target_directory: str,
    df_filenames: Set[str],
    file_extensions: List[str] = None,
    identifier_pattern: Optional[str] = None,
) -> Dict[str, tuple]:
    """
    Renames files in the target directory to match filenames in the dataframe,
    based on a common identifier pattern.

    Args:
        target_directory (str): Path to the directory containing the files.
        df_filenames (Set[str]): Set of filenames referenced in the dataframe.
        file_extensions (List[str], optional): List of file extensions to consider. If None, all files are considered.
        identifier_pattern (str, optional): Regex pattern to identify files. If None, full filename is used.

    Returns:
        Dict[str, tuple]: A dictionary of renamed files, where keys are identifiers
                          and values are tuples of (original name, new name).
    """
    target_path = Path(target_directory)

    # Get all files in the target directory
    if file_extensions:
        all_files = [f for ext in file_extensions for f in target_path.glob(f"*{ext}")]
    else:
        all_files = list(target_path.glob("*.*"))

    # Group files by identifier
    file_groups = {}
    for file_path in all_files:
        if identifier_pattern:
            match = re.search(identifier_pattern, file_path.name)
            if match:
                key = match.group()
            else:
                continue  # Skip files that don't match the identifier pattern
        else:
            key = file_path.stem  # Use full filename (without extension) if no identifier is provided

        if key not in file_groups:
            file_groups[key] = file_path

    renamed_files = {}

    for identifier, file_path in file_groups.items():
        # Find matching filename in df_filenames
        matching_df_filename = next((f for f in df_filenames if identifier in f), None)

        if matching_df_filename:
            # Check if the file needs to be renamed
            if file_path.name != matching_df_filename:
                new_path = file_path.parent / matching_df_filename

                # Check if the new filename already exists
                if not new_path.exists():
                    try:
                        file_path.rename(new_path)
                        renamed_files[identifier] = (file_path.name, new_path.name)
                        print(f"Renamed: {file_path.name} -> {new_path.name}")
                    except Exception as e:
                        print(f"Error renaming {file_path.name}: {e}")
                else:
                    print(f"Cannot rename {file_path.name} to {matching_df_filename}: File already exists")

    # Print summary
    print(f"\nRenamed {len(renamed_files)} files.")

    return renamed_files
