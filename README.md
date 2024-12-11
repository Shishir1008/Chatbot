# Chatbot Project

This repository contains a Python-based chatbot project designed to interact with users using natural language processing (NLP) techniques. The chatbot is trained on intents and provides meaningful responses based on user input.

---

## Features
- Uses PyTorch for training a neural network to classify intents.
- Supports tokenization, stemming, and bag-of-words for text preprocessing.
- Implements a user-friendly interface with Gradio.
- Easy-to-extend JSON-based intents system for adding new patterns and responses.

---

## Demo
Interact with the chatbot via its Gradio-based user interface:

```bash
python chatbot.py
```

---

## Installation

### Prerequisites
Ensure you have Python 3.7 or above installed on your system. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/Shishir1008/Chatbot.git
cd Chatbot
```

---

## Project Structure
```
Chatbot/
├── chatbot.py         # Main Python file containing the chatbot logic
├── intents.json       # JSON file defining the chatbot's intents
├── requirements.txt   # Required Python packages
├── README.md          # Project description and instructions
```

---

## Usage

1. **Train the Model**:
   The training process is embedded within the chatbot script. Simply run the following command to start training and launch the chatbot interface:
   ```bash
   python chatbot.py
   ```

2. **Interact with the Chatbot**:
   Once the Gradio interface launches, type your messages, and the chatbot will respond accordingly.

---

## Adding New Intents

To add new intents, edit the `intents.json` file:

```json
{
  "intents": [
    {
      "tag": "new_tag",
      "patterns": ["User input pattern 1", "Pattern 2"],
      "responses": ["Response 1", "Response 2"]
    }
  ]
}
```

After editing, re-run `chatbot.py` to retrain the model and apply the changes.

---

## Technologies Used
- **Python**: Programming language.
- **PyTorch**: Neural network training.
- **Gradio**: Interface for chatbot interactions.
- **NLTK**: Tokenization and stemming.
- **NumPy**: Data manipulation.

---

## Future Enhancements
- Add more advanced NLP techniques (e.g., embeddings, transformers).
- Integrate with external APIs for dynamic responses.
- Enhance the user interface for improved usability.

---

## Author
**Shishir**

For any questions or collaborations, feel free to reach out through the [repository](https://github.com/Shishir1008/Chatbot.git).

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

