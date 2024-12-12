import os
from torch.utils.data import Dataset, DataLoader
import torch


def tokenize_sequence(sequence, vocab):
    """
    Converts a sequence of RNA nucleotides or structure annotations into a list of indices.

    Args:
        sequence (str): The RNA sequence or structure.
        vocab (dict): A dictionary mapping characters to indices.

    Returns:
        list: A list of token indices.
    """
    return [vocab[char.upper()] for char in sequence]


class RNADataset(Dataset):
    def __init__(self, directory, max_len=None):
        """
        Args:
            directory (str): Path to the directory containing RNA structure files.
            max_len (int, optional): Maximum sequence length for padding/truncation.
        """
        self.data = []
        self.max_len = max_len

        # Define vocabularies for sequences and structures
        self.sequence_vocab = {char: idx for idx, char in enumerate("ACGU", start=1)}
        self.structure_vocab = {char: idx for idx, char in enumerate("().", start=1)}

        # Read files from the directory
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as file:
                lines = file.read().strip().split('\n')
                true_structure = lines[0]
                sequence = lines[1]
                predictions = lines[2:]

                # Tokenize the sequence and structures
                tokenized_sequence = tokenize_sequence(sequence, self.sequence_vocab)
                tokenized_true_structure = tokenize_sequence(true_structure, self.structure_vocab)
                tokenized_predictions = [tokenize_sequence(pred, self.structure_vocab) for pred in predictions]

                # Store data point
                self.data.append({
                    "sequence": tokenized_sequence,
                    "true_structure": tokenized_true_structure,
                    "predictions": tokenized_predictions
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Optionally pad/truncate sequences and structures to max_len
        if self.max_len:
            item["sequence"] = self._pad_or_truncate(item["sequence"], self.max_len)
            item["true_structure"] = self._pad_or_truncate(item["true_structure"], self.max_len)
            item["predictions"] = [
                self._pad_or_truncate(pred, self.max_len) for pred in item["predictions"]
            ]

        # Convert lists to tensors
        sequence_tensor = torch.tensor(item["sequence"], dtype=torch.long)
        true_structure_tensor = torch.tensor(item["true_structure"], dtype=torch.long)
        predictions_tensor = torch.tensor(item["predictions"], dtype=torch.long)

        return {
            "sequence": sequence_tensor,
            "true_structure": true_structure_tensor,
            "predictions": predictions_tensor
        }

    def _pad_or_truncate(self, sequence, max_len):
        """Pads or truncates a sequence to the specified length."""
        if len(sequence) < max_len:
            return sequence + [0] * (max_len - len(sequence))
        return sequence[:max_len]


# Example usage
directory = "eterna_data/StructureData/ensemble_train_set"
max_len = 128  # Specify a maximum length for the sequences (if needed)
dataset = RNADataset(directory, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate over the dataloader
for batch in dataloader:
    print("Sequence batch:", batch["sequence"].shape)
    print("True structure batch:", batch["true_structure"].shape)
    print("Predictions batch:", batch["predictions"].shape)
    break
