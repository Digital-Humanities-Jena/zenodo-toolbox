# Zenodo Toolbox
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub issues](https://img.shields.io/github/issues/Digital-Humanities-Jena/zenodo-toolbox)](https://github.com/Digital-Humanities-Jena/zenodo-toolbox/issues)
[![GitHub stars](https://img.shields.io/github/stars/Digital-Humanities-Jena/zenodo-toolbox)](https://github.com/Digital-Humanities-Jena/zenodo-toolbox/stargazers)

## Overview

Zenodo Toolbox is a comprehensive set of tools designed to facilitate interaction with Zenodo, a general-purpose open-access repository developed under the European OpenAIRE program and operated by CERN. This toolbox provides a range of functionalities for creating, managing, and publishing records on Zenodo, as well as performing various data processing operations.

**Warning: Please be advised that this repository is still undergoing development. It is intended that a stable version will be released, but the date or timeframe is not yet determined.**

## Features

- Zenodo record creation and management
- File upload and publication
- Database setup and management
- Version handling and record updates
- Community management and restriction handling
- Custom configuration options
- Image processing operations
- Excel file processing and batch operations
- 3D model operations
- XML generation (EDM and METS/MODS examples)

# Table of Contents

- [Zenodo Toolbox](#zenodo-toolbox)
  - [Overview](#overview)
  - [Features](#features)
- [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Tutorials](#tutorials)
  - [Advanced Features](#advanced-features)
    - [Database Integration](#database-integration)
    - [Version Control](#version-control)
    - [Community Management](#community-management)
    - [Custom Configurations](#custom-configurations)
    - [Image Processing and Analysis](#image-processing-and-analysis)
    - [Excel Processing and Batch Operations](#excel-processing-and-batch-operations)
    - [3D Model Handling](#3d-model-handling)
    - [XML Generation for Linked Data](#xml-generation-for-linked-data)
  - [Contributing](#contributing)
  - [License](#license)

## Prerequisites

- **Python ≥ 3.8**
- (optional) **Blender** (if 3D model operations are intended)

Tested on Linux & Mac OS.

## Installation

1. Clone the repository:
```python
git clone https://github.com/Digital-Humanities-Jena/zenodo-toolbox.git
```


2. Create and activate a conda environment:
```python
conda create --name zenodo-toolbox python=3.10
conda activate zenodo-toolbox
```

3. Install required packages:
```python
pip install -r requirements.txt
```


4. Download required models:
- [Ultralytics YOLOv10x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt)
- [Meta SAM2 (Segment-Anything Model 2)](https://github.com/ultralytics/assets/releases/download/v8.2.0/sam2_t.pt)
- Optional: [Other Models (\@ Ultralytics)](https://docs.ultralytics.com/models/)

Place these models in the `./Models` directory.

## Usage
Explore the functionalities by following the provided Tutorials (Jupyter Notebooks).

Each tutorial is designed to cover specific aspects of the Zenodo Toolbox, providing practical examples and explanations to help users effectively utilize the toolbox's features.

## Tutorials

00. [Setup Zenodo](./Tutorials/00_setup_zenodo.ipynb)
    - Create a Zenodo account and obtain API tokens
    - Set up environment variables for API access
    - Configure the Zenodo client for sandbox and production environments
    - Test the connection to Zenodo API
01. [Create Records](./Tutorials/01_create_record.ipynb)
    - Initialize a new Zenodo deposit
    - Add metadata to the deposit (title, description, authors, etc.)
    - Set the publication date and access rights
    - Add keywords and additional descriptions
    - Specify related identifiers and references
    - Save the deposit as a draft
02. [Upload Files & Publish](./Tutorials/02_upload_files_publish.ipynb)
    - Upload files to an existing Zenodo deposit
    - Manage file metadata (e.g., file type, description)
    - Update deposit metadata if needed
    - Publish the deposit, making it publicly available
    - Retrieve and display the published record's information
    - Handle potential errors during the upload and publishing process
03. [Database Setup](./Tutorials/03_setup_database.ipynb)
    - Initialize and configure the SQLite database
    - Create tables for storing Zenodo record information
    - Implement functions for database operations (insert, update, delete)
    - Demonstrate basic querying and data retrieval
04. [Record Updates & Version Handling](./Tutorials/04_handle_versions.ipynb)
    - Update existing Zenodo records
    - Handle different versions of records
    - Implement version control for Zenodo deposits
    - Demonstrate how to create new versions of existing records
05. [Communities & Restrictions](./Tutorials/05_communities_and_restrictions.ipynb)
    - Manage Zenodo communities
    - Set up and handle access restrictions for records
    - Implement functions to add records to specific communities
    - Demonstrate how to apply and remove restrictions on deposits
06. [Custom Configurations](./Tutorials/06_custom_configurations.ipynb)
    - Create and manage custom configuration files
    - Implement functions to load and apply custom settings
    - Demonstrate how to use configuration files for different projects or workflows
07. [Image Operations](./Tutorials/07_image_operations.ipynb)
    - Perform various image processing operations
    - Utilize YOLOv10 for object detection in images
    - Apply SAM2 for image segmentation and masking
    - Generate image metadata including dimensions, file size, and detected objects
    - Create and manipulate image thumbnails
08. [Excel Processing, Batch & Advanced Operations](./Tutorials/08_process_excel_file.ipynb)
    - Process Excel files to extract metadata for Zenodo records
    - Perform batch operations on multiple records
    - Implement advanced filtering and data manipulation techniques
    - Generate reports and summaries from Excel data
09. [3D Model Operations](./Tutorials/09_model_operations.ipynb)
    - Process 3D model files (e.g., OBJ, STL)
    - Generate thumbnails and previews for 3D models
    - Extract metadata from 3D model files
    - Perform basic transformations on 3D models
10.  [Generate XMLs: EDM and METS/MODS Examples](./Tutorials/10_linked_data_xmls.ipynb)
     - Create Europeana Data Model (EDM) XML files
     - Generate METS/MODS XML files for digital library metadata
     - Implement functions to convert Zenodo metadata to standardized XML formats
     - Demonstrate the creation of linked data representations

**Upcoming**:
<ul><ul>
<li> Integrity Tools (DB Synchronization, Response Validation)</li>
<li>Rate Limiting & Maximum Process Parallelization</li>
<li>Database Access Interface w/ Query & Export Options</li>
<li>Advanced Logging Functionalities</li>
<li>Updates for the Database Tutorial regarding external DBs & Export Functions</li>
<li>Audio Tools (ASR w/ Timestamps, Noise Reduction, Metadata Extraction, Conversion)</li>
</ul></ul>

## Advanced Features

### Database Integration

The Zenodo Toolbox includes a robust database integration feature, allowing users to:

- Set up and manage a local SQLite database for storing Zenodo record information
- Perform CRUD (Create, Read, Update, Delete) operations on the database
- Synchronize local database entries with Zenodo records
- Execute complex queries for data analysis and reporting

This feature enhances the toolbox's capability to manage large numbers of records efficiently and provides a local cache for faster operations.

### Version Control

The toolbox implements sophisticated version control mechanisms for Zenodo records:

- Create new versions of existing records
- Track changes between versions
- Manage metadata updates across different versions
- Ensure consistency and traceability of record evolution over time

### Community Management

Users can effectively manage Zenodo communities using the toolbox:

- Add records to specific communities
- Remove records from communities
- Apply and manage access restrictions on deposits
- Facilitate collaboration and sharing within defined groups

### Custom Configurations

The Zenodo Toolbox supports custom configurations, allowing users to:

- Create project-specific configuration files
- Define default settings for different workflows
- Easily switch between different configuration profiles
- Streamline repetitive tasks by pre-configuring common operations

### Image Processing and Analysis

The Zenodo Toolbox includes advanced image processing capabilities:

- Object detection using YOLOv10x, allowing for automatic identification and labeling of objects in images
- Image segmentation and masking with SAM2, enabling precise isolation of image elements
- Automatic generation of image metadata, including dimensions, file size, and detected objects
- Creation and manipulation of image thumbnails for efficient preview generation

These features allow for sophisticated image analysis and metadata enrichment, enhancing the quality and searchability of image-based Zenodo records.

### Excel Processing and Batch Operations

The toolbox provides robust Excel processing capabilities:

- Extract metadata from structured Excel files to create or update Zenodo records
- Perform batch operations on multiple records simultaneously
- Implement advanced filtering and data manipulation techniques
- Generate comprehensive reports and summaries from Excel data

These features streamline the process of managing large datasets and creating multiple Zenodo records from structured data sources.

### 3D Model Handling

For 3D model files, the Zenodo Toolbox offers specialized functionality:

- Process common 3D model formats such as OBJ and STL
- Generate 2D thumbnails and previews for 3D models
- Extract metadata specific to 3D models, including vertex count, face count, and bounding box dimensions
- Perform basic transformations on 3D models, such as scaling and rotation

These capabilities extend the toolbox's utility to researchers and institutions working with 3D digital assets.

### XML Generation for Linked Data

The toolbox includes functions for generating standardized XML formats:

- Create Europeana Data Model (EDM) XML files, facilitating integration with Europeana collections
- Generate METS/MODS XML files, a standard format for digital library metadata
- Convert Zenodo metadata to these standardized XML formats, enhancing interoperability
- Support the creation of linked data representations, improving discoverability and data reuse

These features enable users to create rich, interoperable metadata that adheres to established standards in digital libraries and cultural heritage institutions.

## Contributing

Contributions to the Zenodo Toolbox are welcome. Please feel free to submit pull requests, create issues or suggest improvements.

## License

GNU General Public License v3.0 (GPL-3.0)

This license is suitable for academic and open-source projects, ensuring that derivative works remain open source while allowing for wide use and modification.