import json
import nltk
import numpy as np
import random
import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import gradio as gr

# Download NLTK resources
nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

# Load intents from JSON file
with open("intents.json", "r") as f:
    intents = json.load(f)

# Tokenization and stemming functions
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Define the Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

all_words = []
tags = []
xy = []

# Prepare training data
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!"]
all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

# Dataset and DataLoader
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize the model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "chatbot_model.pth")

# Gradio chatbot interface
model.eval()  # Ensure the model is in evaluation mode
bot_name = "Shishir's Bot"

def get_response(user_input):
    tokenized_input = tokenize(user_input)
    bag = bag_of_words(tokenized_input, all_words)
    bag = torch.tensor(bag, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(bag)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

    for intent in intents["intents"]:
        if tag == intent["tag"]:
            return random.choice(intent["responses"])

    return "I'm sorry, I didn't understand that. Could you rephrase?"

with gr.Blocks() as interface:
    gr.Markdown("### Chat with Shishir's Bot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message")
    send_btn = gr.Button("Send")

    def chat_interaction(history, user_message):
        if user_message.strip():
            bot_reply = get_response(user_message)
            history.append(("You", user_message))
            history.append((bot_name, bot_reply))
        return history, ""

    send_btn.click(chat_interaction, inputs=[chatbot, msg], outputs=[chatbot, msg])

interface.launch()
