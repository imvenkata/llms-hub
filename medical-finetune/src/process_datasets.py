import os
import logging
import pandas as pd
from datasets import Dataset, DatasetDict
from dataset_utils import (
    Llama3InstructDataset,
    MistralInstructDataset,
    GemmaInstructDataset,
)
import certifi

# Set the SSL certificate file path
os.environ["SSL_CERT_FILE"] = certifi.where()

REMOVE_COLUMNS = ["source", "focus_area"]
RENAME_COLUMNS = {"question": "input", "answer": "output"}
INSTRUCTION = "Answer the question truthfully, you are a medical professional."
DATASETS_PATHS = [
    "/Users/venkata.medabala/Projects/llms-hub/medical-finetune/data/raw/medical_meadow_wikidoc.csv",
    "/Users/venkata.medabala/Projects/llms-hub/medical-finetune/data/raw/medquad.csv",
]

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_MAPPING = {
    "llama3": Llama3InstructDataset,
    "mistral": MistralInstructDataset,
    "gemma": GemmaInstructDataset,
}


def process_dataset(dataset_path: str, model: str) -> pd.DataFrame:
    """
    Process the dataset to be in the format required by the model.
    """
    logger.info(f"Processing dataset: {dataset_path} for model {model}")

    if model not in MODEL_MAPPING:
        raise ValueError(
            f"Model '{model}' not supported. Available models: {', '.join(MODEL_MAPPING.keys())}"
        )

    dataset_class = MODEL_MAPPING[model]
    dataset = dataset_class(dataset_path)

    dataset.drop_columns(REMOVE_COLUMNS)
    logger.info("Columns removed: %s", REMOVE_COLUMNS)

    dataset.rename_columns(RENAME_COLUMNS)
    logger.info("Columns renamed: %s", RENAME_COLUMNS)

    dataset.create_instruction(INSTRUCTION)
    logger.info("Instruction created: %s", INSTRUCTION)

    dataset.drop_bad_rows(["input", "output"])
    logger.info("Bad rows dropped based on columns: %s", ["input", "output"])

    dataset.create_prompt()
    logger.info("Prompt column created")

    return dataset.get_dataset()


def create_huggingface_dataset(dataset: pd.DataFrame) -> DatasetDict:
    """
    Create a Huggingface dataset from the processed dataset.
    """
    logger.info("Creating Huggingface dataset")
    dataset.reset_index(drop=True, inplace=True)
    return {"train": Dataset.from_pandas(dataset)}


def process_all_datasets(
    dataset_paths: list[str], model_names: list[str]
) -> dict[str, pd.DataFrame]:
    """
    Process all datasets for given model names and concatenate them.
    """
    processed_datasets = {model: [] for model in model_names}

    for dataset_path in dataset_paths:
        for model in model_names:
            dataset = process_dataset(dataset_path, model)
            hf_dataset = create_huggingface_dataset(dataset)
            processed_datasets[model].append(hf_dataset)

            if model == "llama3":
                dataset_name = os.path.basename(dataset_path).split(".")[0]
                hf_dataset["train"].push_to_hub(
                    f"{model}_{dataset_name}_instruct_dataset"
                )

    concatenated_datasets = {
        model: pd.concat(
            [ds["train"].to_pandas() for ds in datasets], ignore_index=True
        )
        for model, datasets in processed_datasets.items()
    }
    return concatenated_datasets


def save_and_push_datasets(
    datasets: dict[str, pd.DataFrame],
    base_path: str,
    suffix: str = "",
    push_to_hub: bool = True,
) -> None:
    """
    Save and push datasets to disk and hub.
    """
    for model, dataset in datasets.items():
        save_path = os.path.join(base_path, f"medical_{model}_instruct_dataset{suffix}")
        if model == "llama3":
            dataset.save_to_disk(save_path)
            if push_to_hub:
                dataset.push_to_hub(f"medical_{model}_instruct_dataset{suffix}")


def create_short_datasets(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Create short versions of datasets.
    """
    short_datasets = {}
    for model, dataset in datasets.items():
        short_dataset = pd.concat(
            [dataset.iloc[:1000], dataset.iloc[-5000:-4000]], ignore_index=True
        )
        short_datasets[model] = create_huggingface_dataset(short_dataset)
    return short_datasets


if __name__ == "__main__":
    processed_data_path = (
        "/Users/venkata.medabala/Projects/llms-hub/medical-finetune/data/processed"
    )
    os.makedirs(processed_data_path, exist_ok=True)

    MODEL_NAMES = ["llama3"]

    # Process and concatenate all datasets
    concatenated_datasets = process_all_datasets(DATASETS_PATHS, MODEL_NAMES)

    # Save and push concatenated datasets (uncomment to execute)
    # save_and_push_datasets(concatenated_datasets, processed_data_path)

    # Create short datasets for free colab training (uncomment to execute)
    # short_datasets = create_short_datasets(concatenated_datasets)

    # Save and push short datasets (uncomment to execute)
    # save_and_push_datasets(short_datasets, processed_data_path, suffix="_short")
