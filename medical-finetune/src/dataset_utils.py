from abc import ABC, abstractmethod
import pandas as pd


class InstructDataset(ABC):
    """abstract class for creating instruct datasets"""

    def __inti__(self, dataset_path: str):
        self.dataset = None
        self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str):
        """Load the dataset from the given path

        :param dataset_path: a path to the dataset
        """
        self.dataset = pd.read_csv(dataset_path)

    def rename_columns(self, columns: dict[str, str]):
        """Rename the columns of the dataset

        :param columns: a dictionary of old and new column names
        """
        self.dataset.rename(columns=columns, inplace=True)

    def drop_columns(self, columns: list[str]):
        """Drop the columns from the dataset

        :param columns: a list of column names to drop
        """

        drop_columns = [col for col in columns if col in self.dataset.columns]
        self.dataset.drop(columns=drop_columns, inplace=True)

    def drop_bad_rows(self, columns: list[str]):
        """Drop rows with missing values, duplicates in the specified columns

        :param columns: a list of column names to check for missing values
        """
        self.dataset.dropna(subset=columns, inplace=True)
        self.dataset.drop_duplicates(subset=columns, inplace=True)

    @abstractmethod
    def create_prompt(self) -> None:
        """Create the prompt for the dataset"""
        pass

    def get_dataset(self) -> pd.DataFrame:
        """Return the dataset"""
        return self.dataset


class Llama3InstructDataset(InstructDataset):
    """Class for creating instruct datasets for the LLAMA3 dataset"""

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    def create_prompt(self):
        """Create the prompt for the dataset"""
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"""<|start_header_id|>system<|end_header_id|> {row['instruction']}<|eot_id|><|start_header_id|>user<|end_header_id|> This is the question: {row['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|> {row['output']}<|eot_id|>"""
            prompts.append(prompt)
        self.dataset["prompt"] = prompts


class MistralInstructDataset(InstructDataset):

    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"""<s>[INST] {row['instruction']} This is the question: {row['input']} [/INST] \\n {row['output']}</s>"""
            prompts.append(prompt)
        self.dataset["prompt"] = prompts


class GemmaInstructDataset(InstructDataset):

    def create_prompt(self):
        """
        Create the prompt column in the dataset which will be used for
        """
        prompts = []
        for index, row in self.dataset.iterrows():
            prompt = f"<start_of_turn>user {row['instruction']} This is the question: {row['input']}<end_of_turn> \\n <start_of_turn>model {row['output']}<end_of_turn>model"
            prompts.append(prompt)
        self.dataset["prompt"] = prompts
