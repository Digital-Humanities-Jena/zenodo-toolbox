from datetime import datetime
import logging
import os
from pathlib import Path
import requests
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from db_tools import sql_append_to_str, sql_get, sql_insert, sql_update, upsert_operation, upsert_published_data
from rate_limiter import RateLimiterParallel
from utilities import (
    get_changelog_from_description,
    get_response_errors,
    get_thumbnails,
    is_edm_available,
    is_metsmods_available,
    load_config,
    increment_version,
    update_description,
)

zenodo_config = load_config("Configs/zenodo.yaml", False)
USE_SANDBOX = zenodo_config["main"]["use_sandbox"]
ZENODO_BASE_URL = "https://sandbox.zenodo.org" if USE_SANDBOX else "https://zenodo.org"
if zenodo_config["main"]["use_env_api_key"]:
    ZENODO_API_KEY = os.environ.get("ZENODO_SANDBOX_API_KEY") if USE_SANDBOX else os.environ.get("ZENODO_API_KEY")
else:
    ZENODO_API_KEY = (
        zenodo_config["preferences"]["zenodo_sandbox_api_key"]
        if USE_SANDBOX
        else zenodo_config["preferences"]["zenodo_api_key"]
    )
HEADERS = {"Content-Type": "application/json"}
PARAMS = {"access_token": ZENODO_API_KEY}

rate_per_hour = zenodo_config["rates"]["per_hour"]
rate_per_min = zenodo_config["rates"]["per_minute"]
rate_limiter_zenodo = RateLimiterParallel(
    rate_per_min, rate_per_hour, db_path=os.path.join(os.environ.get("TMPDIR", "/tmp"), "rate_limiter.db")
)

logger = logging.getLogger("zenodo-toolbox")


def create_new_version(
    record_data: Dict[str, Any], discard_existing_drafts: bool = True, db_connection: sqlite3.Connection = None
) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Creates a new version of a Zenodo record.

    Args:
        record_data: Dictionary containing record information.
        discard_existing_drafts: Flag to discard existing drafts if creation fails.
        db_connection: Optional SQLite database connection.

    Returns:
        [0] A dictionary with operation status, response code, and message.
        [1] A dictionary with the new version's data (empty if creation fails).

    This function attempts to create a new version of a Zenodo record. If the
    creation fails and discard_existing_drafts is True, it will attempt to
    discard existing drafts and retry. The function handles rate limiting and
    updates the database if a connection is provided.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}
    concept_recid = record_data["conceptrecid"]

    link_newversion = record_data["links"]["newversion"]
    rate_limiter_zenodo.wait_for_rate_limit()
    r_newversion = requests.post(link_newversion, params=PARAMS)
    rate_limiter_zenodo.record_request()
    if not r_newversion.status_code == 201:
        if discard_existing_drafts:
            # Discard existing Drafts
            discard_msg, discard_data = discard_draft(
                concept_recid=concept_recid, db_connection=db_connection, record_data=record_data
            )
            if not discard_msg["success"]:
                return_msg = discard_msg
                return return_msg, return_data
            else:
                return create_new_version(record_data, discard_existing_drafts, db_connection)
        else:
            return_msg = {"success": False, "response": r_newversion.status_code, "text": r_newversion.text}
            return return_msg, return_data
    else:
        return_msg = {"success": True, "response": r_newversion.status_code, "text": r_newversion.text}
        return_data = r_newversion.json()
        if db_connection:
            upsert_operation(db_connection, "create_version", return_data)
        return return_msg, return_data


def create_record(
    zenodo_metadata: dict, db_connection: sqlite3.Connection = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Creates a new record on Zenodo using the provided metadata.

    Args:
        zenodo_metadata: A dictionary containing the metadata for the new record.
        db_connection: An optional SQLite database connection.

    Returns:
        [0] return_msg: Status information about the operation.
        [1] return_data: The created record data or error information.
    """
    return_msg = {"success": False, "response": 0, "text": "", "errors": []}
    return_data = {}

    url = f"{ZENODO_BASE_URL}/api/deposit/depositions"

    try:
        rate_limiter_zenodo.wait_for_rate_limit()
        r = requests.post(url, params=PARAMS, json=zenodo_metadata)
        rate_limiter_zenodo.record_request()

        if r.status_code == 201:
            return_data = r.json()
            return_msg.update(
                {"success": True, "response": r.status_code, "text": "Zenodo Record created successfully."}
            )
            if db_connection:
                upsert_operation(db_connection, "new_record_created", return_data)
        else:
            errors, error_data = get_response_errors(r)
            return_msg.update({"response": r.status_code, "text": "Failed to create Zenodo Record.", "errors": errors})
            return_data = error_data

        # rate_limiter_zenodo.record_request()

    except requests.exceptions.RequestException as e:
        return_msg["text"] = f"Error creating Zenodo record: {str(e)}"
        if hasattr(e, "response") and e.response is not None:
            return_msg["response"] = e.response.status_code
            try:
                return_data = e.response.json()
            except ValueError:
                return_data = {"error": e.response.text}
        else:
            return_data = {"error": str(e)}

    return return_msg, return_data


def delete_files_in_draft(filedata: List[Dict[str, Any]]) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Deletes files from a draft Zenodo record.

    Args:
        filedata: A list of dictionaries containing file information.

    Returns:
        [0] return_msg: Status information about the operation.
        [1] return_data: Additional data about the deletion process.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}

    for data in filedata:
        # --- old API ---
        # deposition_id = data["deposition_id"]
        # file_crumb = data["links"]["self"].split("files/")[1].split("/")[0]
        # delete_file_link = f"{ZENODO_BASE_URL}/api/deposit/depositions/{deposition_id}/files/{file_crumb}"
        # --- --- ---
        rate_limiter_zenodo.wait_for_rate_limit()
        r_deletefile = requests.delete(
            data["links"]["self"],
            params=PARAMS,
        )
        rate_limiter_zenodo.record_request()
        if r_deletefile.status_code == 204:
            return_msg = {"success": True, "response": 204, "text": r_deletefile.text}
        else:
            # Sometimes, Zenodo returns a 500. This may or may not delete the file. Handle return_msg.
            if r_deletefile.status_code == 500:
                return_msg = {"success": True, "response": 500, "text": r_deletefile.text}
            else:
                return_msg = {"success": False, "response": r_deletefile.status_code, "text": r_deletefile.text}
                return return_msg, return_data

    return return_msg, return_data


def delete_file_in_deposition(
    record_data: Dict[str, Any], filename: str
) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Deletes a specific file from a Zenodo deposition.

    Args:
        record_data: A dictionary containing the deposition record data.
        filename: The name of the file to be deleted.

    Returns:
        [0] return_msg: Status information about the operation.
        [1] return_data: Additional data about the deletion process.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}
    detected_existing_file = False

    for filedata in record_data["files"]:
        existing_filename = filedata["filename"]
        if existing_filename == filename:
            detected_existing_file = True
            existing_file_link = filedata["links"]["self"]
            rate_limiter_zenodo.wait_for_rate_limit()
            r_deletefile = requests.delete(
                existing_file_link,
                params=PARAMS,
            )
            rate_limiter_zenodo.record_request()
            if r_deletefile.status_code == 204:
                return_msg = {"success": True, "response": 204, "text": r_deletefile.text}
            else:
                return_msg = {"success": False, "response": r_deletefile.status_code, "text": r_deletefile.text}

    if not detected_existing_file:
        return_msg = {"success": True, "response": 200, "text": "No conflicting files detected in deposition."}

    return return_msg, return_data


def discard_draft(
    discard_link: str = "",
    concept_recid: str = "",
    db_connection: Optional[sqlite3.Connection] = None,
    record_data: Dict[str, Any] = {},
) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Discards a draft record on Zenodo using either a discard link or concept record ID.

    Args:
        discard_link: URL to discard the draft.
        concept_recid: Concept record ID to identify and discard the draft.
        db_connection: Optional SQLite database connection.
        record_data: Dictionary containing record data.

    Returns:
        [0] return_msg: Status information about the operation.
        [1] return_data: retrieved data dictionary.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}

    if discard_link:
        rate_limiter_zenodo.wait_for_rate_limit()
        r_discard = requests.post(discard_link, params=PARAMS)
        rate_limiter_zenodo.record_request()
        return_msg.update({"response": r_discard.status_code, "text": r_discard.text})
        if r_discard.status_code == 204:
            return_msg = {"success": True, "response": r_discard.status_code, "text": r_discard.text}
        else:
            return_msg["success"] = False

    # Retrieve Draft for Concept Record ID and Discard Draft if identified
    elif concept_recid:
        retrieval_msg, retrieval_data = retrieve_by_concept_recid(concept_recid=concept_recid, all_versions=True)
        if retrieval_msg["success"] and type(retrieval_data) == list:
            print("Retrieval completed.")
            draft_msg, draft_data = identify_draft(retrieval_data)
            if draft_msg["success"]:
                print("Draft completed.")
                discard_link = draft_data["links"]["discard"]
                rate_limiter_zenodo.wait_for_rate_limit()
                r_discard = requests.post(discard_link, params=PARAMS)
                rate_limiter_zenodo.record_request()
                return_msg.update({"response": r_discard.status_code, "text": r_discard.text})
                if r_discard.status_code == 204:
                    print("Discard completed.")
                    return_msg = {"success": True, "response": r_discard.status_code, "text": r_discard.text}
                    latest_msg, latest_data = identify_latest_record(record_data_ls=retrieval_data, ignore_drafts=True)
                    return_data = latest_data
                else:
                    return_msg["success"] = False

    if return_msg["success"] and db_connection and record_data:
        upsert_operation(db_connection, "discard", record_data)

    return return_msg, return_data


def identify_draft(record_data_ls: List[Dict[str, Any]]) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Identifies a draft record from a list of record data.

    Args:
        record_data_ls: List of dictionaries containing record data.

    Returns:
        [0] return_msg: status message dictionary
        [1] return_data: identified draft data dictionary.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}
    draft_identified = False
    for data in record_data_ls:
        if not data["submitted"]:
            draft_identified = True
            return_msg["success"] = True
            return_data = data
            break
    if not draft_identified:
        return_msg["response"] = 404
        return_msg["text"] = "No Draft identified in Records."

    return (return_msg, return_data) if draft_identified else (return_msg, {})


def identify_latest_record(
    record_data_ls: List[Dict[str, Any]], ignore_drafts: bool = True
) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Identifies the latest record from a list of record data.

    Args:
        record_data_ls: List of dictionaries containing record data.
        ignore_drafts: Boolean flag to ignore draft records.

    Returns:
        [0] return_msg: status message dictionary
        [1] return_data: latest record data dictionary.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}
    latest_record = None

    for data in record_data_ls:
        if ignore_drafts and not data["submitted"]:
            continue
        if latest_record is None or data["created"] > latest_record["created"]:
            latest_record = data

    if latest_record:
        return_msg = {"success": True, "response": 200, "text": latest_record["conceptrecid"]}
        return_data = latest_record
    else:
        return_msg["text"] = "No suitable Records provided. Draft pending?"

    return return_msg, return_data


def publish_record(
    record_data: Dict[str, Any],
    db_connection: Optional[sqlite3.Connection] = None,
    additional_data: Dict[str, Any] = {},
) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Publishes a record to Zenodo and updates the local database.

    Args:
        record_data: A dictionary containing record information, including a publish link.
        db_connection: An optional SQLite database connection for storing record data.
        additional_data: An optional dictionary with extra data to be processed and stored.

    Returns:
        [0] Dictionary with publish operation status (success, response code, and message).
        [1] Dictionary with the published record data or original record data if publishing fails.

    The function handles rate limiting, publishes the record, processes additional data
    (including thumbnail generation), and updates the local database if a connection is provided.
    It also handles various HTTP response scenarios, including gateway timeouts (504).
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}

    publish_link = record_data["links"]["publish"]
    rate_limiter_zenodo.wait_for_rate_limit()
    r_publish = requests.post(publish_link, params=PARAMS)
    rate_limiter_zenodo.record_request()
    if r_publish.status_code not in [200, 201, 202, 204]:
        if r_publish.status_code == 504:
            return_msg["response"] = 504
            return_msg["text"] += f" || 504: {record_data['conceptrecid']}"
            return_data = record_data

            if db_connection:
                if additional_data:
                    filedata = additional_data["filedata"]
                    thumbnails_data = get_thumbnails(filedata)
                    additional_data.update(
                        {
                            "thumbnails_data": thumbnails_data,
                            "edm_available": is_edm_available(filedata),
                            "metsmods_available": is_metsmods_available(filedata),
                            "thumbnails_available": True if thumbnails_data else False,
                        }
                    )
                upsert_operation(db_connection, "published", return_data)
                upsert_published_data(db_connection, return_data, additional_data)
        else:
            return_msg["response"] = r_publish.status_code
            return_msg["text"] = r_publish.text
    else:
        return_msg = {"success": True, "response": r_publish.status_code, "text": r_publish.text}
        return_data = r_publish.json()

        if db_connection:
            if additional_data:
                filedata = additional_data["filedata"]
                thumbnails_data = get_thumbnails(filedata)
                additional_data.update(
                    {
                        "thumbnails_data": thumbnails_data,
                        "edm_available": is_edm_available(filedata),
                        "metsmods_available": is_metsmods_available(filedata),
                        "thumbnails_available": True if thumbnails_data else False,
                    }
                )
            upsert_operation(db_connection, "publish", return_data)
            upsert_published_data(db_connection, return_data, additional_data)

    return return_msg, return_data


def retrieve_by_concept_recid(
    concept_recid: str, all_versions: bool
) -> Tuple[Dict[str, Union[bool, int, str]], Union[Dict, List[Dict]]]:
    """
    Retrieves Zenodo record(s) based on a concept record ID.

    Args:
        concept_recid: The concept record ID to search for.
        all_versions: If True, retrieves all versions of the record; if False, retrieves only the latest version.

    Returns:
        [0] Dictionary with status information (success, response code, and text).
        [1] The retrieved data as either a single dictionary (latest version) or a list of dictionaries (all versions).

    Note:
        Uses rate limiting when making requests to the Zenodo API.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}

    query_string = f"conceptrecid:{concept_recid}"
    rate_limiter_zenodo.wait_for_rate_limit()
    r = requests.get(
        f"{ZENODO_BASE_URL}/api/deposit/depositions",
        params={
            **PARAMS,
            "q": query_string,
            "all_versions": all_versions,
        },
        timeout=120,
    )
    rate_limiter_zenodo.record_request()
    return_msg.update({"response": r.status_code, "text": r.text})
    if r.status_code in [200, 201, 202] and r.json():
        return_msg["success"] = True
        return_data = r.json()  # be aware this is an array of objects if all_versions = True
    else:
        return_msg["success"] = False

    return return_msg, return_data


def update_metadata(
    record_data: Dict[str, Any], new_metadata: Dict[str, Any], db_connection: Optional[sqlite3.Connection] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Updates the metadata of a Zenodo record and optionally stores the result in a database.

    Args:
        record_data: A dictionary containing the record's data, including the 'links' key with 'latest_draft' URL.
        new_metadata: A dictionary with the new metadata to be applied to the record.
        db_connection: An optional SQLite database connection for storing the update result.

    Returns:
        [0] (dict) Status message with keys 'success', 'response', and 'text'.
        [1] (dict) The updated record data from Zenodo's response.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}

    rate_limiter_zenodo.wait_for_rate_limit()
    r_newmetadata = requests.put(record_data["links"]["latest_draft"], json=new_metadata, params=PARAMS)
    rate_limiter_zenodo.record_request()
    return_msg.update({"response": r_newmetadata.status_code, "text": r_newmetadata.text})
    if r_newmetadata.status_code not in [200, 201, 202]:
        return_msg["success"] = False
        return return_msg, return_data
    else:
        return_msg["success"] = True
        return_data = r_newmetadata.json()
        if db_connection:
            upsert_operation(db_connection, "push_metadata", return_data)
        return return_msg, return_data


def update_record(
    latest_data: dict = {},
    filepaths: list = [],
    replace_existing_files: bool = True,
    replace_description: str = "",
    changelog: str = "Updated Record.",
    debug_mode: bool = False,
    db_connection: sqlite3.Connection = None,
    additional_data: dict = {},
) -> tuple[dict, dict]:
    """
    Updates a record on Zenodo by creating a new version, uploading files, updating metadata, and publishing.

    Args:
        latest_data: Dictionary containing the latest record data.
        filepaths: List of file paths to upload.
        replace_existing_files: Whether to replace existing files.
        replace_description: New description to replace the existing one.
        changelog: Changelog message for the update.
        debug_mode: If True, stops before publishing.
        db_connection: SQLite database connection.
        additional_data: Additional data to include in the record.

    Returns:
        [0] (dict): Status message containing success, response code, and text.
        [1] (dict): Updated record data or empty dict if update fails.
    """
    return_msg = {"success": False, "response": 0, "text": ""}
    return_data = {}

    if latest_data:
        # Create new Record Version and assign Variables if successful
        created_msg, created_data = create_new_version(
            record_data=latest_data, discard_existing_drafts=True, db_connection=db_connection
        )
        if not created_msg["success"]:
            return_msg = created_msg
            return_msg["text"] += " || (-> create_new_version)"
            return return_msg, return_data

        # Upload Files into new Deposition
        uploaded_files_msg, uploaded_files_data = upload_files_into_deposition(
            record_data=created_data,
            filepaths=filepaths,
            replace_existing=replace_existing_files,
            db_connection=db_connection,
        )
        if not uploaded_files_msg["success"]:
            discard_draft(
                discard_link=created_data["links"]["discard"],
                concept_recid="",
                db_connection=db_connection,
                record_data=created_data,
            )
            return_msg = {"success": False, "response": 404, "text": "File Upload failed. Record Version Discarded."}
            return return_msg, {}

        # Update Metadata and Description
        metadata = {"metadata": created_data["metadata"]}
        metadata["metadata"]["publication_date"] = datetime.now().strftime("%Y-%m-%d")
        old_version = latest_data["metadata"]["version"]
        new_version = increment_version(old_version, 1)
        metadata["metadata"]["version"] = new_version
        if replace_description:
            old_description = created_data["metadata"]["description"]
            created_data["metadata"][
                "description"
            ] = f"{replace_description}<br><br>{get_changelog_from_description(old_description)}"
        new_description = update_description(created_data, uploaded_files_data, new_version, changelog)
        metadata["metadata"]["description"] = new_description
        metadata_msg, metadata_data = update_metadata(created_data, metadata, db_connection=db_connection)

        # Publish new Record Version
        if not debug_mode:
            additional_data["filedata"] = uploaded_files_data
            published_msg, published_data = publish_record(
                created_data, db_connection=db_connection, additional_data=additional_data
            )
            if published_msg["success"]:
                return_msg = published_msg
                return_data = published_data
        else:
            print(f"Debug Mode active. Publishing stopped.")
            return_msg = {
                "success": True,
                "response": 200,
                "text": "Debug Mode active. Everything until publishing went fine.",
            }
            return_data = metadata_data

    return return_msg, return_data


def update_sql_by_record_data(record_data: Dict[str, Any]) -> Tuple[Dict[str, Union[bool, int, str]], Dict[str, Any]]:
    """
    Updates multiple SQL tables with data from a Zenodo record.

    This function processes the provided record data and updates various SQL tables
    including 'mainfiles', 'links', 'records', 'states', 'descriptions', and 'versions'.
    It handles file information, links, metadata, and version details.

    Args:
        record_data: A dictionary containing the Zenodo record data.

    Returns:
        [0] A dictionary with keys:
            - 'success': Boolean indicating overall success of operations.
            - 'response': Integer response code (200 for success, 0 for failure).
            - 'text': String containing error messages, if any.
        [1] An empty dictionary for potential future use.
    """
    return_msg = {"success": True, "response": 200, "text": ""}
    return_data = {}

    # table: mainfiles
    # Check if Files or Links already exist and Update or Insert
    mainfiles_sql_data = sql_get("mainfiles", "concept_recid IN", [str(record_data["conceptrecid"])])["data"]
    mainfiles_filenames = [i["filename"] for i in mainfiles_sql_data]

    record_filenames = []
    record_directlinks = {}
    for rec_filedata in record_data["files"]:
        record_filenames.append(rec_filedata["filename"])
        record_directlinks[rec_filedata["filename"]] = rec_filedata["links"]["download"].replace("/draft", "")

    for filedata in mainfiles_sql_data:
        concept_recid = filedata["concept_recid"]
        filename = filedata["filename"]
        file_extension = Path(filename).suffix
        direct_link = record_directlinks[filename]
        if filename in record_filenames:
            # Update Filelinks
            try:
                sql_update("mainfiles", {"direct_link": direct_link}, "concept_recid = ", [concept_recid])
            except Exception as e:
                error_msg = f"Error updating DB table (mainfiles): {str(e)}. Value: 'direct_link': {direct_link} || "
                return_msg = {"success": False, "response": 0}
                return_msg["text"] += error_msg
                print(error_msg)
                pass

    for rec_filedata in record_data["files"]:
        filename = rec_filedata["filename"]
        file_extension = Path(filename).suffix
        direct_link = record_directlinks[filename]
        if not filename in mainfiles_filenames:
            # Insert Filedata
            values = {
                "concept_recid": concept_recid,
                "title": record_data["metadata"]["title"],
                "subset": mainfiles_sql_data[0].get("subset", "") if mainfiles_sql_data else "",
                "filetype": file_extension,
                "filename": filename,
                "direct_link": direct_link,
                "source": filedata["source"],
            }
            try:
                sql_insert("mainfiles", values)
            except Exception as e:
                error_msg = f"Error inserting into DB table (mainfiles): {str(e)}. Values: {values} || "
                return_msg = {"success": False, "response": 0}
                return_msg["text"] += error_msg
                pass

    # table: links
    sql_data_links = {
        "self": record_data["links"]["self"],
        "html": record_data["links"]["html"],
        "doi": record_data["links"]["doi"],
        "concept_doi": record_data["links"]["parent_doi"],
        "files": record_data["links"]["files"],
        "publish": record_data["links"]["publish"],
        "edit": record_data["links"]["edit"],
        "discard": record_data["links"]["discard"],
        "new_version": record_data["links"]["newversion"],
    }
    try:
        sql_update("links", sql_data_links, "concept_recid = ", [record_data["conceptrecid"]])
    except Exception as e:
        error_msg = f"Error inserting into DB table (links): {str(e)}. Values: {sql_data_links} || "
        return_msg = {"success": False, "response": 0}
        return_msg["text"] += error_msg
        pass

    # table: records
    sql_data_records = {
        "latest_recid": record_data["id"],
        "latest_doi": record_data["doi"],
        "latest_version": record_data["metadata"]["version"],
        "latest_update": record_data["metadata"]["publication_date"],
    }
    try:
        sql_update("records", sql_data_records, "concept_recid = ", [record_data["conceptrecid"]])
    except Exception as e:
        error_msg = f"Error writing to DB table (records): {str(e)}. Values: {sql_data_records} || "
        return_msg = {"success": False, "response": 0}
        return_msg["text"] += error_msg
        pass

    # table: states
    sql_data_states = {
        "latest_version": record_data["metadata"]["version"],
        "edm_available": True,
        "metsmods_available": True,
    }
    try:
        sql_update("states", sql_data_states, "concept_recid = ", [record_data["conceptrecid"]])
    except Exception as e:
        error_msg = f"Error writing to DB table (states): {str(e)}. Values: {sql_data_states} || "
        return_msg = {"success": False, "response": 0}
        return_msg["text"] += error_msg
        pass

    # table: descriptions
    sql_data_descriptions = {
        "latest_version": record_data["metadata"]["version"],
        "description": record_data["metadata"]["description"],
    }
    try:
        sql_update("descriptions", sql_data_descriptions, "concept_recid = ", [record_data["conceptrecid"]])
    except Exception as e:
        error_msg = f"Error writing to DB table (descriptions): {str(e)}. Values: {sql_data_descriptions} || "
        return_msg = {"success": False, "response": 0}
        return_msg["text"] += error_msg
        pass

    # table: versions
    try:
        response = sql_append_to_str(
            "versions", "concept_recid = ", str(record_data["conceptrecid"]), "record_ids", str(record_data["id"])
        )
        if "ignored_existing" in response.keys():
            error_msg = f"RecordID {str(record_data['id'])} was already present in versions table || "
            return_msg["success"] = True
            return_msg["text"] += error_msg
    except Exception as e:
        error_msg = f"Error writing to DB table (versions): {str(e)}. Column Header: record_ids. Value: {str(record_data['id'])} || "
        return_msg = {"success": False, "response": 0}
        return_msg["text"] += error_msg
        pass

    return return_msg, return_data


def upload_files_into_deposition(
    record_data: Dict[str, Any],
    filepaths: Union[str, List[str], Path, List[Path]],
    replace_existing: bool = False,
    db_connection: Optional[sqlite3.Connection] = None,
) -> Tuple[Dict[str, Union[bool, int, str]], Union[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Uploads files to a Zenodo deposition and optionally replaces existing files.

    Args:
        record_data: Information about the Zenodo record.
        filepaths: Path(s) to the file(s) to be uploaded.
        replace_existing: Whether to replace existing files with the same name.
        db_connection: Optional SQLite database connection for storing upload data.

    Returns:
        [0] Message containing upload status, response code, and any error text.
        [1] List of uploaded file data or empty dict if upload failed.
    """
    return_msg = {"success": True, "response": 0, "text": ""}
    return_data = []

    uploaded_files_data = []

    files_deposit_url = f'{record_data["links"]["files"]}?access_token={ZENODO_API_KEY}'  # deprecated

    record_id = record_data["id"]
    bucket_url = record_data["links"]["bucket"]

    combined_file_data = []  # includes source of file in objects

    for filepath_str in filepaths:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"File {filepath_str} does not exist! Skipping whole process for this Record...")
            return_msg = {"success": False, "response": 404, "text": "File is missing. Skipped!"}
            return return_msg, return_data

        filename = str(filepath.name)
        direct_link = f"{ZENODO_BASE_URL}/api/records/{record_id}/files/{filename}/content"

        if replace_existing:
            delete_msg, delete_data = delete_file_in_deposition(record_data=record_data, filename=filename)
            if not delete_msg["success"]:
                print(f"Could not delete existing File {filename} in ConceptRecID {record_data['conceptrecid']}")
                return delete_msg, delete_data
            elif delete_msg["success"]:
                if delete_msg["response"] == 204:
                    print(f"Replaced existing File {filename}")
                if delete_msg["response"] == 200:
                    print(f"Uploading File {filename} ...")

        # --- deprecated: ---
        # upload_data = {"name": filepath.name}
        # files = {"file": open(filepath, "rb")}
        # r_fileupload = requests.post(files_deposit_url, data=upload_data, files=files) # deprecated
        # --- --- ---
        rate_limiter_zenodo.wait_for_rate_limit()
        with open(filepath, "rb") as file:
            r_fileupload = requests.put(f"{bucket_url}/{filename}", data=file, params=PARAMS)
        rate_limiter_zenodo.record_request()

        if r_fileupload.status_code not in [200, 201, 202, 204]:
            if r_fileupload.status_code == 504:
                return_msg["response"] = 504
                return_msg["text"] += f" || 504: {filepath_str}"
                return_data = record_data
                uploaded_files_data.append(
                    {
                        "deposition_id": str(record_data["id"]),
                        "filename": filename,
                        "filesource": Path(filepath_str).resolve().__str__(),
                        "links": {"download": direct_link},
                    }
                )
                continue
            elif r_fileupload.status_code == 400 and replace_existing:
                print("Filename already existing! Attempting to delete the existing one...")
                draft_msg, draft_data = retrieve_by_concept_recid(record_data["conceptrecid"], all_versions=True)
                if draft_msg["success"] and draft_data:
                    latest_draft_data = identify_latest_record(draft_data, ignore_drafts=False)[1]
                    delete_msg, delete_data = delete_file_in_deposition(
                        record_data=latest_draft_data, filename=filepath.name
                    )
                    if delete_msg["success"] and not latest_draft_data["files"]:
                        print(
                            "Retrieved Files list is empty. Use local file responses to delete existing files with similar filenames from an initial record draft."
                        )
                    if not delete_msg["success"]:
                        print(
                            f"Could not delete existing File {filepath.name} in ConceptRecID {record_data['conceptrecid']}"
                        )
                        return delete_msg, delete_data
                else:
                    print(f"Could not retrieve Draft for ConceptRecID {record_data['conceptrecid']}")
            else:
                return_msg = {"success": False, "reponse": r_fileupload.status_code, "text": r_fileupload.text}
                return return_msg, {}
        else:
            return_msg = {"success": True, "response": r_fileupload.status_code, "text": r_fileupload.text}
            data = r_fileupload.json()
            if not "links" in data:
                data["links"] = {"download": direct_link}
            else:
                data["links"].update({"download": direct_link})
            # --- deprecated ---
            # data["links"]["download"] = data["links"]["download"].replace("/draft", "")
            # --- --- ---
            data["deposition_id"] = str(record_data["id"])
            data["filename"] = filename
            data["file_source"] = Path(filepath_str).resolve().__str__()
            uploaded_files_data.append(data)

    if uploaded_files_data:
        if db_connection:
            upsert_operation(db_connection, "upload_files", record_data, uploaded_files_data)
        return return_msg, uploaded_files_data
    else:
        return return_msg, return_data
