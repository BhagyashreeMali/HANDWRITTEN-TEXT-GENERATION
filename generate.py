import torch
from model.char_rnn import CharRNN

checkpoint = torch.load("handwriting_rnn.pth")

char_to_idx = checkpoint["char_to_idx"]
idx_to_char = checkpoint["idx_to_char"]
vocab_size = len(char_to_idx)

model = CharRNN(vocab_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

def generate_text(start_char="L", length=300, temperature=0.5):
    hidden = model.init_hidden(1)
    char = torch.tensor([[char_to_idx[start_char]]])
    result = start_char

    for _ in range(length):
        output, hidden = model(char, hidden)
        logits = output.squeeze() / temperature
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs, 1).item()

        result += idx_to_char[idx]
        char = torch.tensor([[idx]])

    return result

print("\n📝 Generated Handwritten-Like Text:\n")
print(generate_text())