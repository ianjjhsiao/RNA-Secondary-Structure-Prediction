import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


def tokenize_sequence(sequence, vocab):
    return [vocab[char.upper()] for char in sequence]


class RNADataset(Dataset):
    def __init__(self, directory, max_len=None):
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


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, max_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.fc(x)
        return x


# Evaluation function for structure prediction
def evaluate_file(model, file_path, dataset, max_len):
    """
    Evaluates the model on a single RNA sequence file to predict the structure.

    Args:
        model (nn.Module): The trained Transformer model.
        file_path (str): Path to the RNA structure file.
        dataset (RNADataset): The dataset object for tokenization.
        max_len (int): Maximum sequence length for padding/truncation.

    Returns:
        str: The predicted RNA structure.
    """
    model.eval()
    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n')
        sequence = lines[2]

        tokenized_sequence = dataset._pad_or_truncate(
            tokenize_sequence(sequence, dataset.sequence_vocab), max_len
        )
        input_tensor = torch.tensor([tokenized_sequence], dtype=torch.long)

        with torch.no_grad():
            output = model(input_tensor).argmax(dim=-1).squeeze(0)

        idx_to_char = {idx: char for char, idx in dataset.structure_vocab.items()}
        predicted_structure = ''.join(idx_to_char[idx.item()] for idx in output if idx.item() != 0)

    return predicted_structure


# Training the Transformer
if __name__ == "__main__":
    directory = "eterna_data/StructureData/ensemble_train_set"
    max_len = 128  # Specify a maximum length for the sequences (if needed)
    batch_size = 16
    embed_size = 64
    num_heads = 4
    hidden_dim = 256
    num_layers = 2
    num_epochs = 100
    learning_rate = 0.001

    # Prepare dataset and dataloader
    dataset = RNADataset(directory, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the model, loss, and optimizer
    vocab_size = len(dataset.sequence_vocab) + 1  # +1 for padding index
    model = TransformerModel(vocab_size, embed_size, num_heads, hidden_dim, num_layers, max_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch["sequence"]  # Use true structure as input
            targets = batch["true_structure"]  # Use sequence as target

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    # Example of evaluation
    test_file_path = "eterna_data/StructureData/test_sample.txt"
    predicted_struct = evaluate_file(model, test_file_path, dataset, max_len)
    print(f"Predicted Structure: {predicted_struct}")
