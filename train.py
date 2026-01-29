import torch
import torch.nn as nn
import numpy as np
from model.char_rnn import CharRNN
from tqdm import tqdm

# Load dataset
with open("data/handwriting_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

encoded = np.array([char_to_idx[c] for c in text])

# Hyperparameters
seq_length = 60
epochs = 10
lr = 0.003

model = CharRNN(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("🚀 Training started...")

for epoch in range(epochs):
    hidden = model.init_hidden(1)
    total_loss = 0
    steps = 0

    for i in tqdm(range(0, len(encoded) - seq_length - 1, seq_length)):
        x = encoded[i:i + seq_length]
        y = encoded[i + 1:i + seq_length + 1]

        x = torch.tensor(x).unsqueeze(0)
        y = torch.tensor(y).unsqueeze(0)

        optimizer.zero_grad()
        output, hidden = model(x, hidden.detach())

        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/steps:.4f}")

# Save model
torch.save({
    "model_state": model.state_dict(),
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char
}, "handwriting_rnn.pth")

print("✅ Training complete. Model saved as handwriting_rnn.pth")