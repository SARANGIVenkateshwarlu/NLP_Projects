Summary Data ingestion-part 1

In this session, the goal is to convert a Jupyter Notebook-based Hugging Face text summarization project into a modular, production-ready end-to-end pipeline.
✅ Objective

Transform the existing notebook into a scalable project structure with clearly defined modules such as:

    Data Ingestion
    Data Transformation
    Model Training
    Model Evaluation
    Model Deployment

The focus of this video is on Data Ingestion, the first component.
🔁 Overall Project Structure Plan

To build a production-ready workflow, the following elements are introduced:

    config.yaml – Stores configuration details (paths, URLs, artifact locations).
    params.yaml – Stores model hyperparameters (used later in training).
    Config Entity (Data Classes) – Defines structured configuration objects.
    Configuration Manager – Reads YAML files and prepares configurations.
    Components – Individual modules like:
        Data Ingestion
        Data Transformation
        Model Trainer
    Pipelines –
        Training Pipeline
        Prediction Pipeline
    APIs & Frontend – For training and batch predictions.

The project will ultimately be fully automated and production-scalable.
📦 Data Ingestion Module
Step 1: Create Notebook for Data Ingestion

A new notebook (01_data_ingestion.ipynb) is created to implement ingestion logic before converting it into a .py file.
⚙️ Step 2: Update config.yaml

The configuration defines:

    artifacts_root
    data_ingestion:
        root_dir
        source_URL
        local_data_file
        unzip_dir

These specify:

    Where outputs (artifacts) will be stored
    Where to download the dataset from
    Where to save the zip file
    Where to extract the data

Artifacts are essentially the outputs of a module.
🧱 Step 3: Create Data Class (Config Entity)

A DataIngestionConfig class is created using @dataclass.

It defines structured fields:

    root_dir
    source_url
    local_data_file
    unzip_dir

This ensures clean, structured configuration handling.
🛠 Step 4: Configuration Manager

A ConfigurationManager class is created to:

    Read config.yaml
    Read params.yaml
    Create artifact root directory
    Return DataIngestionConfig

It:

    Uses helper functions (read_yaml, create_directories)
    Extracts data ingestion settings
    Prepares config object for the component

This separates configuration logic from component logic — a key industry practice.
🔄 Step 5: Data Ingestion Component

A DataIngestion class is created with:
1️⃣ download_file()

    Checks if the dataset zip already exists
    If not, downloads it from GitHub using urllib
    Logs status messages

2️⃣ extract_zip_file()

    Creates unzip directory
    Extracts zip contents using zipfile

▶️ Step 6: Execute the Flow

Execution flow:

    Initialize ConfigurationManager
    Get DataIngestionConfig
    Initialize DataIngestion with config
    Call:
        download_file()
        extract_zip_file()

Result:

    artifacts/ folder created
    data_ingestion/ subfolder created
    data.zip downloaded
    Dataset extracted
    train.csv, test.csv, validation.csv available

✅ Data ingestion is successfully completed.