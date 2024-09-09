from contextlib import closing
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import requests
import sqlite3
from tabulate import tabulate
from typing import Any, Dict, List, Union

from utilities import load_config

logger = logging.getLogger("zenodo-toolbox")
db_config = load_config("Configs/db_config.yaml")

# --- WIP ---
REMOTE_API_KEY = os.environ.get("REMOTE_API_KEY", "")  # set API key for remote application
REMOTE_HEADERS = {"Content-Type": "application/json", "API-Key": REMOTE_API_KEY}
# --- ---


def add_unique_constraint(cursor: sqlite3.Cursor, table_name: str, columns: List[str]) -> None:
    """
    Adds a unique constraint to the specified table for the given columns.

    Args:
        cursor: SQLite database cursor.
        table_name: Name of the table to add the constraint to.
        columns: List of column names to include in the unique constraint.

    Returns:
        None
    """
    constraint_name = f"unique_{table_name}_{'_'.join(columns)}"
    columns_str = ", ".join(columns)
    sql = f"CREATE UNIQUE INDEX IF NOT EXISTS {constraint_name} ON {table_name} ({columns_str})"
    cursor.execute(sql)


def clear_operations_by_status(
    connection: sqlite3.Connection, status: Union[str, List[str]], processed_concept_recids: List[str] = []
) -> bool:
    """
    Deletes records from the operations table based on specified status(es) and optional concept_recids.

    Args:
        connection: SQLite database connection object.
        status: Single status string or list of status strings to match.
        processed_concept_recids: List of concept_recids to filter deletion (optional).

    Returns:
        [0] (bool) True if operation was successful, False otherwise.
    """
    try:
        cursor = connection.cursor()

        # Convert status to a list if it's a single string
        if isinstance(status, str):
            status = [status]

        # Prepare the status condition
        status_condition = " OR ".join(["status = ?"] * len(status))

        if processed_concept_recids:
            # Convert the list to a comma-separated string for the SQL query
            concept_recids_str = ",".join(["?"] * len(processed_concept_recids))

            # Delete records that match the status(es) and are in the processed list
            query = f"""
                DELETE FROM operations
                WHERE ({status_condition}) AND concept_recid IN ({concept_recids_str})
                """
            cursor.execute(query, status + processed_concept_recids)
        else:
            # Delete all records that match the status(es) if the list is empty
            query = f"""
                DELETE FROM operations
                WHERE {status_condition}
                """
            cursor.execute(query, status)

        connection.commit()
        logger.debug(f"Successfully cleared {cursor.rowcount} entries with status {status} from operations table.")
        return True

    except sqlite3.Error as e:
        logger.critical(f"Database error: {e}")
        logger.critical(f"Clearing entries with status {status} from operations table failed! Performing Rollback...")
        connection.rollback()
        return False
    except Exception as e:
        logger.critical(f"Error in clear_operations_by_status: {e}")
        logger.critical(f"Clearing entries with status {status} from operations table failed! Performing Rollback...")
        connection.rollback()
        return False


def create_db(custom_cfg: Dict[str, Any] = {}) -> Union[sqlite3.Connection, None]:
    """
    Creates a SQLite database based on the provided configuration.

    Args:
        custom_cfg: Custom configuration dictionary to override default settings.

    Returns:
        [0] SQLite connection object if the database is successfully created.
        [1] None if an error occurs or local database usage is disabled.

    Raises:
        sqlite3.Error: If there's an error while creating the database.
    """
    db_config = custom_cfg if custom_cfg else db_config
    if db_config["use_local_db"]:
        local_db_path = Path(db_config["local_db_path"])
        local_db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating Local SQLite Database ...")

        try:
            conn = sqlite3.connect(str(local_db_path))
            conn.execute("PRAGMA foreign_keys = ON")

            with closing(conn.cursor()) as cursor:
                for table in db_config["db_structures"]["tables"]:
                    table_name = list(table.keys())[0]
                    columns = table[table_name]

                    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
                    create_table_sql += ",\n".join(
                        [f"{col_name} {col_type}" for col_name, col_type in columns.items()]
                    )

                    if "foreign_keys" in table:
                        for fk in table["foreign_keys"]:
                            for fk_col, reference in fk.items():
                                create_table_sql += f",\nFOREIGN KEY ({fk_col}) REFERENCES {reference}"

                    create_table_sql += "\n)"

                    cursor.execute(create_table_sql)

                    if "indexes" in table:
                        for index_col in table["indexes"]:
                            index_name = f"idx_{table_name}_{index_col}"
                            cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({index_col})")

                    if "updated_at" in columns:
                        trigger_sql = f"""
                        CREATE TRIGGER IF NOT EXISTS update_{table_name}_timestamp 
                        AFTER UPDATE ON {table_name}
                        BEGIN
                            UPDATE {table_name} SET updated_at = CURRENT_TIMESTAMP
                            WHERE id = NEW.id;
                        END;
                        """
                        cursor.execute(trigger_sql)

                    # Add UNIQUE constraints
                    if "unique_constraints" in table:
                        for constraint in table["unique_constraints"]:
                            add_unique_constraint(cursor, table_name, constraint["columns"])
                    elif "UNIQUE" in columns.get("concept_recid", ""):
                        add_unique_constraint(cursor, table_name, ["concept_recid"])

            conn.commit()
            logger.info(f"Database created successfully at {local_db_path}")
            return conn

        except sqlite3.Error as e:
            logger.critical(f"Error creating database: {e}")
            return None
    else:
        logger.info("Local database usage is disabled in the configuration.")
        return None


def get_row(
    db_connection: sqlite3.Connection, table: str, find_in_column: str, value: Any, print_result: bool = False
) -> Dict[str, Any]:
    """
    Retrieves a single row from a SQLite database table based on a specified column and value.

    Args:
        db_connection: SQLite database connection object.
        table: Name of the table to query.
        find_in_column: Name of the column to search in.
        value: Value to search for in the specified column.
        print_result: If True, prints the result in a tabulated format.

    Returns:
        [0] A dictionary containing column names as keys and row values as values.
            Returns an empty dictionary if no row is found or an error occurs.

    Raises:
        ValueError: If the specified column is not found in the table.
    """
    try:
        cursor = db_connection.cursor()

        # Fetch column names
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [column[1] for column in cursor.fetchall()]

        # Check if the specified column exists
        if find_in_column not in columns:
            raise ValueError(f"Column '{find_in_column}' not found in table '{table}'")

        # Execute the query
        query = f"SELECT * FROM {table} WHERE {find_in_column} = ?"
        cursor.execute(query, (value,))
        row = cursor.fetchone()

        if row:
            result = dict(zip(columns, row))

            if print_result:
                # Format the data for tabulate
                table_data = [[k, v] for k, v in result.items()]
                print(tabulate(table_data, headers=["Column", "Value"], tablefmt="pretty"))

            return result
        else:
            if print_result:
                print(f"No row found in table '{table}' with {find_in_column} = {value}")
            return {}

    except sqlite3.Error as e:
        logger.error(f"Database error in get_row: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error in get_row: {e}")
        return {}


def initialize_db(custom_cfg: Dict[str, Union[bool, str]] = {}) -> Union[sqlite3.Connection, None]:
    """
    Initialize or connect to a SQLite database based on configuration.

    Args:
        custom_cfg: Custom configuration dictionary to override default settings.

    Returns:
        [0] SQLite connection object if successful, None otherwise.

    Raises:
        sqlite3.Error: If there's an error connecting to the existing database.
    """
    db_config = custom_cfg if custom_cfg else db_config
    if db_config["use_local_db"]:
        local_db_path = Path(db_config["local_db_path"])
        if not local_db_path.exists():
            return create_db(custom_cfg)
        else:
            try:
                conn = sqlite3.connect(str(local_db_path))
                conn.execute("PRAGMA foreign_keys = ON")
                logger.info(f"Connected to existing database at {local_db_path}")
                return conn
            except sqlite3.Error as e:
                logger.critical(f"Error connecting to existing database: {e}")
                return None
    else:
        logger.info("Local database usage is disabled in the configuration.")
        return None


def print_table(connection: sqlite3.Connection, table: str, concept_recid: Union[str, int] = "") -> None:
    """
    Prints a formatted table of data from a specified SQLite table.

    Args:
        connection: SQLite database connection object.
        table: Name of the table to query.
        concept_recid: Optional concept record ID to filter results.

    Returns:
        None

    Raises:
        sqlite3.Error: If a database error occurs.
    """
    try:
        cursor = connection.cursor()

        if concept_recid:
            # If concept_recid is provided, select only the matching row
            cursor.execute(f"SELECT * FROM {table} WHERE concept_recid = ?", (concept_recid,))
        else:
            # If no concept_recid is provided, select all rows
            cursor.execute(f"SELECT * FROM {table}")

        rows = cursor.fetchall()

        if not rows:
            print(f"No entries found{' for the given concept_recid' if concept_recid else ''}.")
            return

        column_names = [description[0] for description in cursor.description]

        # Use tabulate to create a formatted table
        table_output = tabulate(rows, headers=column_names, tablefmt="pretty")
        print(table_output)

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")


def set_row(
    db_connection: sqlite3.Connection,
    table: str,
    new_data: Dict[str, Any],
    where_column: str,
    where_value: Any,
    print_result: bool = False,
) -> bool:
    """
    Updates a row in the specified SQLite table based on a where condition.

    Args:
        db_connection: SQLite database connection object.
        table: Name of the table to update.
        new_data: Dictionary containing column names and their new values.
        where_column: Name of the column to use in the WHERE clause.
        where_value: Value to match in the WHERE clause.
        print_result: If True, prints the update result.

    Returns:
        True if a row was updated, False otherwise.

    Raises:
        ValueError: If the where_column doesn't exist or no valid columns to update.
        sqlite3.Error: If a database error occurs.
    """
    try:
        cursor = db_connection.cursor()

        # Fetch column names
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [column[1] for column in cursor.fetchall()]

        # Check if the specified where_column exists
        if where_column not in columns:
            raise ValueError(f"Column '{where_column}' not found in table '{table}'")

        # Filter row_data to include only existing columns
        valid_data = {k: v for k, v in new_data.items() if k in columns}

        if not valid_data:
            raise ValueError("No valid columns to update")

        # Construct the UPDATE query
        set_clause = ", ".join([f"{k} = ?" for k in valid_data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where_column} = ?"

        # Execute the query
        cursor.execute(query, list(valid_data.values()) + [where_value])
        db_connection.commit()

        if cursor.rowcount > 0:
            if print_result:
                print(f"Successfully updated row in table '{table}' where {where_column} = {where_value}")
                # Format the data for tabulate
                table_data = [[k, v] for k, v in valid_data.items()]
                print(tabulate(table_data, headers=["Column", "New Value"], tablefmt="pretty"))
            return True
        else:
            if print_result:
                print(f"No row found in table '{table}' where {where_column} = {where_value}")
            return False

    except sqlite3.Error as e:
        logger.error(f"Database error in set_row: {e}")
        return False
    except Exception as e:
        logger.error(f"Error in set_row: {e}")
        return False


def sql_append_to_str(
    table: str, condition: str, target: str, column_header: str, str_to_append: str, ignore_existing: bool = True
) -> Dict[str, Union[bool, str]]:
    """
    Appends a string to an existing string value in a SQL database table.

    Args:
        table: Name of the SQL table.
        condition: SQL condition for selecting the target row.
        target: Value to match in the condition.
        column_header: Name of the column to update.
        str_to_append: String to append to the existing value.
        ignore_existing: If True, skip appending if str_to_append already exists.

    Returns:
        [0] A dictionary containing the update response or ignored status.

    Raises:
        ValueError: If no data is found for the given condition and target.

    Example:
        sql_append_to_str("versions", "concept_recid = ", "57297", "record_ids", "123456")
        -- before: 79116,78846 || after: 79116,78846,123456
    """
    current_data = sql_get(table, condition, [target])

    if not current_data["data"]:
        raise ValueError(f"No data found for condition: {condition} ({target})")
    else:
        current_str_value = current_data["data"][0][column_header]

    if str_to_append in current_str_value and ignore_existing:
        return {"ignored_existing": True}
    else:
        updated_record_ids = f"{current_str_value},{str_to_append}"

    # Update the field with the new string
    update_values = {column_header: updated_record_ids}
    update_response = sql_update(table, update_values, condition, [target])

    return update_response


def sql_delete(table: str, condition: str, targets: List[str]) -> Dict[str, Any]:
    """
    Executes a SQL DELETE operation on a specified table with given conditions.

    Args:
        table: Name of the table to delete from.
        condition: SQL condition string (without the values).
        targets: List of target values to be included in the condition.

    Returns:
        [0] (dict): JSON response from the DELETE request.

    Example:
        sql_delete("states", "concept_recid IN", ["000test", "001test", "002test"])
    """
    delete_url = "https://target-server/jena-zenodo/api/delete"
    targets_str = ", ".join([f"'{element}'" for element in targets])
    payload = {"table": table, "condition": f"{condition} ({targets_str})"}

    r_delete = requests.delete(delete_url, headers=REMOTE_HEADERS, data=json.dumps(payload))

    return r_delete.json()


def sql_get(table: str, condition: str, targets: List[str]) -> Dict[str, Any]:
    """
    Performs a SQL-like GET request to retrieve data from a specified table.

    Args:
        table: The name of the table to query.
        condition: The SQL condition to apply (e.g., "column_name IN").
        targets: A list of target values to include in the condition.

    Returns:
        [0] (dict): A dictionary containing the JSON response from the GET request.
            The 'data' key in this dictionary typically holds the retrieved records.

    Example:
        sql_get("states", "concept_recid IN", ["000test", "001test", "002test"])['data']
    """
    get_url = "https://target-server/jena-zenodo/api/get"
    targets_str = ", ".join([f"'{element}'" for element in targets])
    payload = {"table": table, "condition": f"{condition} ({targets_str})"}

    r_get = requests.get(get_url, headers=REMOTE_HEADERS, params=payload)

    return r_get.json()


def sql_insert(table: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inserts data into a specified SQL table.

    Args:
        table: Name of the table to insert data into.
        values: Dictionary containing column names as keys and corresponding values to be inserted.

    Returns:
        [0] (dict): JSON response from the insert operation.

    Example:
        sql_insert("states", {"concept_recid": "000test", "title": "test", "type": "testmodel", "latest_version": "test.test.0", "thumbnails_available": False, "edm_available": False, "metsmods_available": False})
    """
    insert_url = "https://target-server/api/insert"
    payload = {
        "table": table,
        "values": values,
    }

    r_insert = requests.post(insert_url, headers=REMOTE_HEADERS, data=json.dumps(payload))

    return r_insert.json()


def sql_update(table: str, values: Dict[str, Any], condition: str, targets: List[str]) -> Dict[str, Any]:
    """
    Performs an SQL UPDATE operation on a specified table.

    Args:
        table: Name of the table to update.
        values: Dictionary of column-value pairs to update.
        condition: SQL condition string without the target values.
        targets: List of target values to be included in the condition.

    Returns:
        [0] (dict): JSON response from the update request.

    Example:
        sql_update("states", {"latest_version": "44.44.44"}, "concept_recid = ", ["000test"])
    """
    update_url = "https://target-server/api/update"
    targets_str = ", ".join([f"'{element}'" for element in targets])
    payload = {"table": table, "values": values, "condition": f"{condition} ({targets_str})"}

    r_update = requests.post(update_url, headers=REMOTE_HEADERS, data=json.dumps(payload))

    return r_update.json()


def upsert_data(
    connection: sqlite3.Connection,
    table_name: str,
    data: Dict[str, Any],
    primary_key: str,
    primary_key_value: Union[str, int],
) -> bool:
    """
    Upserts data into a SQLite table, handling JSON, DATETIME, and BOOLEAN types.

    Args:
        connection: SQLite database connection.
        table_name: Name of the target table.
        data: Dictionary containing column-value pairs to upsert.
        primary_key: Name of the primary key column.
        primary_key_value: Value of the primary key for the record.

    Returns:
        True if the upsert operation was successful, False otherwise.
    """
    try:
        cursor = connection.cursor()

        # Get the column names and types from the table
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()
        columns = {col[1]: col[2] for col in table_info}

        # Prepare the data for insertion/update
        prepared_data = {}
        for key, value in data.items():
            if key in columns:
                if columns[key].startswith("JSON"):
                    prepared_data[key] = json.dumps(value, ensure_ascii=False)
                elif columns[key].startswith("DATETIME"):
                    if isinstance(value, str):
                        prepared_data[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    elif isinstance(value, datetime):
                        prepared_data[key] = value
                elif columns[key].startswith("BOOLEAN"):
                    prepared_data[key] = 1 if value else 0
                else:
                    prepared_data[key] = value

        # Ensure the primary key is in the prepared data
        prepared_data[primary_key] = primary_key_value

        # Start a transaction
        connection.execute("BEGIN TRANSACTION")

        try:
            # Attempt to insert the record
            columns_clause = ", ".join(prepared_data.keys())
            placeholders = ", ".join(["?" for _ in prepared_data])
            cursor.execute(
                f"""
                INSERT INTO {table_name} 
                ({columns_clause})
                VALUES ({placeholders})
                """,
                list(prepared_data.values()),
            )
        except sqlite3.IntegrityError:
            # If insert fails, attempt to update
            set_clause = ", ".join([f"{k} = ?" for k in prepared_data.keys() if k != primary_key])
            values = [v for k, v in prepared_data.items() if k != primary_key]
            values.append(primary_key_value)

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET {set_clause}
                WHERE {primary_key} = ?
                """,
                values,
            )

        connection.commit()
        logger.debug(f"Data upserted successfully into table {table_name}.")
        return True

    except sqlite3.Error as e:
        connection.rollback()
        logger.critical(f"Database error: {e}")
        logger.critical(f"DB Upsert Operation failed! Performing Rollback...")
        return False
    except Exception as e:
        connection.rollback()
        logger.critical(f"Error in upsert_data: {e}")
        logger.critical(f"DB Upsert Operation failed! Performing Rollback...")
        return False


def upsert_operation(
    connection: sqlite3.Connection,
    operation_type: str,
    zenodo_response_data: Dict[str, Any],
    zenodo_file_data: List[Dict[str, Any]] = [],
) -> bool:
    """
    Upserts an operation record in the database based on Zenodo response data.

    Args:
        connection: SQLite database connection.
        operation_type: Type of operation being performed.
        zenodo_response_data: Response data from Zenodo API.
        zenodo_file_data: List of file data associated with the Zenodo record.

    Returns:
        True if the upsert operation was successful, False otherwise.
    """
    try:
        cursor = connection.cursor()
        data = zenodo_response_data
        concept_recid = data["conceptrecid"]
        recid = data["id"]
        links = json.dumps(data["links"])
        files = json.dumps(zenodo_file_data, ensure_ascii=False)
        initiated = datetime.fromisoformat(data["created"].replace("Z", "+00:00"))
        modified = datetime.fromisoformat(data["modified"].replace("Z", "+00:00"))
        if operation_type == "create_record" or "create_version":
            doi = data["metadata"]["prereserve_doi"]["doi"]
        else:
            doi = data["doi"]

        if operation_type == "publish":
            status = "published"
        elif operation_type == "discard":
            status = "discarded"
        elif operation_type == "push_metadata":
            existing_row_data = get_row(connection, "operations", "recid", data["id"])
            if existing_row_data:
                if existing_row_data["status"] == "published" or existing_row_data["status"] == "inprogress":
                    status = "inprogress"
                else:
                    status = "draft"
            else:
                status = "draft"
        else:
            status = "draft"

        # Check if a record with this concept_recid already exists
        cursor.execute(
            """
            SELECT id FROM operations WHERE concept_recid = ?
        """,
            (concept_recid,),
        )
        existing_record = cursor.fetchone()

        if existing_record:
            # Update existing record
            cursor.execute(
                """
                UPDATE operations
                SET recid = ?, operation = ?, status = ?, initiated = ?, 
                    links = ?, files = ?, doi = ?, modified = ?
                WHERE concept_recid = ?
            """,
                (recid, operation_type, status, initiated, links, files, doi, modified, concept_recid),
            )
        else:
            # Insert new record
            cursor.execute(
                """
                INSERT INTO operations 
                (concept_recid, recid, operation, status, initiated, links, files, doi, modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (concept_recid, recid, operation_type, status, initiated, links, files, doi, modified),
            )

        connection.commit()

        # # Additional Operations
        # if operation_type == "new_version_created":
        #     update_value_in_db(db_connection, "records", {"status": status, "recid": recid, "doi": doi, "modified": modified})

        logger.debug(f"Operation {operation_type} for concept_recid {concept_recid} upserted successfully.")
        return True

    except sqlite3.Error as e:
        logger.critical(f"Database error: {e}")
        logger.critical(f"DB Upsert Operation failed! Performing Rollback...")
        connection.rollback()
        return False
    except Exception as e:
        logger.critical(f"Error in upsert_db: {e}")
        logger.critical(f"DB Upsert Operation failed! Performing Rollback...")
        connection.rollback()
        return False


def upsert_published_data(
    connection: sqlite3.Connection, zenodo_response_data: Dict[str, Any], additional_data: Dict[str, Any]
) -> bool:
    """
    Upserts published data into various SQLite database tables based on Zenodo response and additional data.

    Args:
        connection: SQLite database connection object.
        zenodo_response_data: Dictionary containing Zenodo API response data.
        additional_data: Dictionary containing supplementary data for the upsert operation.

    Returns:
        True if the upsert operation was successful, False otherwise.
    """
    try:
        current_timestamp = datetime.now().isoformat()
        cursor = connection.cursor()
        data = zenodo_response_data

        concept_recid = data["conceptrecid"]
        concept_doi = data["conceptdoi"]
        recid = data["id"]
        title = data["title"]
        doi = data["doi"]
        version = data["metadata"]["version"]
        access_right = data["metadata"]["access_right"]
        license_ = data["metadata"].get("license", "")
        owner_id = data["owner"]
        changelogs = json.dumps({})

        filedata = additional_data.get("filedata", [])
        subset = additional_data.get("subset", "")
        changelogs = json.dumps(additional_data.get("changelogs", {}), ensure_ascii=False, indent=4)
        thumbnails_data = additional_data.get("thumbnails_data", {})
        type_ = additional_data.get("type", "")
        # TODO: handle empty values by get_row

        modified = datetime.fromisoformat(data["modified"].replace("Z", "+00:00"))

        # Fetch existing all_recids
        cursor.execute("SELECT all_recids FROM records WHERE concept_recid = ?", (concept_recid,))
        result = cursor.fetchone()
        existing_all_recids = result[0] if result else ""

        # Update all_recids
        all_recids_list = existing_all_recids.split(",") if existing_all_recids else []
        if str(recid) not in all_recids_list:
            all_recids_list.append(str(recid))
        all_recids = ",".join(all_recids_list)

        # -- Table: records --
        cursor.execute(
            """
            INSERT INTO records (concept_recid, concept_doi, type, title, subset, recid, doi, version, 
                                 access_right, license, owner_id, all_recids, changelogs, modified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(concept_recid) DO UPDATE SET
                concept_doi = excluded.concept_doi,
                type = excluded.type,
                title = excluded.title,
                subset = excluded.subset,
                recid = excluded.recid,
                doi = excluded.doi,
                version = excluded.version,
                access_right = excluded.access_right,
                license = excluded.license,
                owner_id = excluded.owner_id,
                all_recids = excluded.all_recids,
                changelogs = excluded.changelogs,
                modified = excluded.modified
            """,
            (
                concept_recid,
                concept_doi,
                type_,
                title,
                subset,
                recid,
                doi,
                version,
                access_right,
                license_,
                owner_id,
                all_recids,
                changelogs,
                modified,
            ),
        )

        # -- Table: links --
        cursor.execute(
            """
            INSERT INTO links (concept_recid, recid, self, html, doi, concept_doi, files, bucket, publish, edit, discard, new_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(concept_recid) DO UPDATE SET
                recid = excluded.recid,
                self = excluded.self,
                html = excluded.html,
                doi = excluded.doi,
                concept_doi = excluded.concept_doi,
                files = excluded.files,
                bucket = excluded.bucket,
                publish = excluded.publish,
                edit = excluded.edit,
                discard = excluded.discard,
                new_version = excluded.new_version
            """,
            (
                concept_recid,
                recid,
                data["links"]["self"],
                data["links"]["html"],
                data["links"]["doi"],
                data["links"]["parent_doi"],
                data["links"]["files"],
                data["links"]["bucket"],
                data["links"]["publish"],
                data["links"]["edit"],
                data["links"]["discard"],
                data["links"]["newversion"],
            ),
        )

        # -- Table: mainfiles --
        for file in filedata:
            filename = file["filename"]
            filetype = Path(filename).suffix[1:].upper()
            direct_link = file["links"]["download"]
            file_source = file["file_source"]

            cursor.execute(
                """
                INSERT INTO mainfiles (concept_recid, recid, filetype, filename, direct_link, file_source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(concept_recid, filename) DO UPDATE SET
                    recid = excluded.recid,
                    filetype = excluded.filetype,
                    direct_link = excluded.direct_link,
                    file_source = excluded.file_source
                """,
                (concept_recid, recid, filetype, filename, direct_link, file_source),
            )

        # -- Table: states --
        cursor.execute(
            """
            INSERT INTO states (concept_recid, recid, thumbnails_available, edm_available, metsmods_available)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(concept_recid) DO UPDATE SET
                recid = excluded.recid,
                thumbnails_available = excluded.thumbnails_available,
                edm_available = excluded.edm_available,
                metsmods_available = excluded.metsmods_available
            """,
            (
                concept_recid,
                recid,
                additional_data.get("thumbnails_available", False),
                additional_data.get("edm_available", False),
                additional_data.get("metsmods_available", False),
            ),
        )

        # -- Table: thumbnails --
        cursor.execute(
            """
            INSERT INTO thumbnails (concept_recid, recid, perspective, res_1000x1000, res_512x512, res_256x256, res_128x128)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(concept_recid) DO UPDATE SET
                recid = excluded.recid,
                perspective = excluded.perspective,
                res_1000x1000 = excluded.res_1000x1000,
                res_512x512 = excluded.res_512x512,
                res_256x256 = excluded.res_256x256,
                res_128x128 = excluded.res_128x128
            """,
            (
                concept_recid,
                recid,
                thumbnails_data.get("perspective", ""),
                thumbnails_data.get("res_1000x1000", ""),
                thumbnails_data.get("res_512x512", ""),
                thumbnails_data.get("res_256x256", ""),
                thumbnails_data.get("res_128x128", ""),
            ),
        )

        # -- Table: responses --
        cursor.execute(
            """
            INSERT INTO responses (concept_recid, recid, data)
            VALUES (?, ?, ?)
            ON CONFLICT(concept_recid) DO UPDATE SET
                recid = excluded.recid,
                data = excluded.data
            """,
            (concept_recid, recid, json.dumps(data, ensure_ascii=False, indent=4)),
        )

        connection.commit()
        return True

    except sqlite3.Error as e:
        logger.critical(f"Database error: {e}")
        logger.critical(f"DB Upsert Operation failed! Performing Rollback...")
        connection.rollback()
        return False
    except Exception as e:
        logger.critical(f"Error in upsert_db: {e}")
        logger.critical(f"DB Upsert Operation failed! Performing Rollback...")
        connection.rollback()
        return False
