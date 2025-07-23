# Next Word Predictor with LSTM

This project is a Python application that predicts the next word in a sentence using a Long Short-Term Memory (LSTM) neural network. It features a real-time graphical user interface (GUI) built with Tkinter, which suggests the top three most likely words as the user types.

The model can be trained on any custom text corpus provided in `.txt` or `.docx` format.

Next Word Predictor GUI Screenshot
<img width="618" height="305" alt="image" src="https://github.com/user-attachments/assets/e8d33f4f-0731-4d3f-bef3-8ccceece30b8" />



---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Configuration](#1-configuration)
  - [2. Training a New Model](#2-training-a-new-model)
  - [3. Running the Application](#3-running-the-application)
- [Full Python Code](#full-python-code)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Real-time Word Prediction**: Suggests the top 3 most likely next words as you type.
- **LSTM-based Model**: Utilizes a deep learning model to understand sentence context and structure.
- **Custom Training Data**: Train the model on any text corpus from a `.txt` or `.docx` file.
- **Interactive GUI**: A user-friendly and responsive interface built with Tkinter.
- **Model Persistence**: Automatically saves the trained model to disk for quick reuse.
- **Extensible Architecture**: The code is organized into modular classes for data handling, model definition, and the GUI, making it easy to extend.

---

## Project Structure

The project is organized into three main classes within a single Python script:

- `TextDataset`: Handles loading and preprocessing the text corpus. It reads `.txt` and `.docx` files, builds a vocabulary, and converts text into numerical sequences for training.
- `LSTMModel`: Defines the neural network architecture using TensorFlow and Keras. It consists of an embedding layer, multiple LSTM layers, and a dense output layer.
- `NextWordPredictorGUI`: Implements the Tkinter GUI, which captures user input, sends it to the model for prediction, and displays the results.

---

## How It Works

1. **Data Preprocessing**: The `TextDataset` class reads the input text, converts it to lowercase, and splits it into a list of words (tokens). A vocabulary of unique words is created, and each word is mapped to a unique integer index.

2. **Sequence Generation**: The text is transformed into input-output pairs. An input consists of a sequence of words of a fixed length (`SEQUENCE_LENGTH`), and the output is the single word that immediately follows.

3. **Model Training**: If a pre-trained model is not found, the script trains the `LSTMModel`. The model learns to predict the index of the next word by processing the input sequences. Its weights are adjusted through backpropagation to minimize the loss (the difference between predicted and actual next words).

4. **Prediction**: In the GUI, the last `SEQUENCE_LENGTH` words typed by the user are converted into a numerical sequence. This sequence is fed into the trained model, which outputs a probability distribution over the entire vocabulary. The top 3 words with the highest probabilities are identified.

5. **Display**: The top 3 predicted words are displayed as clickable buttons in the GUI. Clicking a button appends that word to the input text, allowing for a continuous and interactive writing experience.

---

## Installation

Follow these steps to set up the project on your local machine.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/next-word-predictor.git
   cd next-word-predictor
   ```

2. **Create and activate a virtual environment (recommended):**
   This isolates the project's dependencies from your system's Python installation.
   ```bash
   # Create the virtual environment
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install tensorflow numpy python-docx
   ```

---

## Usage

### 1. Configuration

Before running the script, you must configure the file paths. Open the Python script and modify the following variables in the `if __name__ == "__main__":` block:

- **`TEXT_FILE_PATH`**: Set this to the absolute or relative path of your training data file (e.g., `"C:/Users/YourUser/Documents/corpus.txt"`).
- **`MODEL_PATH`**: Set this to the path where the trained model should be saved and loaded from (e.g., `"C:/Users/YourUser/Documents/saved_lstm_model.keras"`).

```python
if __name__ == "__main__":
    # --- Configuration ---
    SEQUENCE_LENGTH = 10
    TEXT_FILE_PATH = "path/to/your/training_data.txt"  # <-- IMPORTANT: CHANGE THIS
    MODEL_PATH = "path/to/your/saved_lstm_model.keras" # <-- IMPORTANT: CHANGE THIS
```

### 2. Training a New Model

To train a new model, simply run the script. If the file specified by `MODEL_PATH` does not exist, the application will automatically:

- Load the data from `TEXT_FILE_PATH`.
- Preprocess the data.
- Build and train the LSTM model.
- Save the trained model to the `MODEL_PATH`.

```bash
python your_script_name.py
```

Training may take a significant amount of time depending on the size of your dataset and your hardware (CPU/GPU).

### 3. Running the Application

If a trained model already exists at `MODEL_PATH`, running the script will load the model and launch the GUI directly.

```bash
python your_script_name.py
```

The GUI window will appear. Start typing in the text box, and predictions will appear below it in real-time.

---

## Full Python Code

Here is the complete, self-contained Python script for the application.

```python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from docx import Document
from keras.saving import register_keras_serializable

class TextDataset:
    """
    Handles loading, preprocessing, and preparing text data for the model.
    Supports both .txt and .docx file formats.
    """
    def __init__(self, text_file_path, sequence_length):
        self.sequence_length = sequence_length
        
        if text_file_path.endswith('.docx'):
            text = self.convert_docx_to_text(text_file_path)
        else:
            text = self.read_text_file(text_file_path)
        
        PAD_TOKEN = "<PAD>"
        words = text.split()
        unique_words = sorted(set(words))
        # Insert a padding token for sequences shorter than sequence_length
        unique_words.insert(0, PAD_TOKEN)
        
        self.vocabulary = unique_words
        self.word_to_index = {word: idx for idx, word in enumerate(unique_words)}
        self.index_to_word = {idx: word for idx, word in enumerate(unique_words)}
        self.words = words  # Store all words for training data generation
        
        print(f"Vocabulary size: {len(self.vocabulary)}")

    def prepare_training_data(self):
        """
        Prepares training data by creating input sequences and corresponding target words.
        """
        sequences = []
        next_words = []
        
        for i in range(len(self.words) - self.sequence_length):
            seq = self.words[i:i + self.sequence_length]
            next_word = self.words[i + self.sequence_length]
            
            # Convert words to their corresponding indices
            seq_indices = [self.word_to_index.get(word, 0) for word in seq] # Use 0 for unknown words
            next_word_index = self.word_to_index.get(next_word, 0)
            
            sequences.append(seq_indices)
            next_words.append(next_word_index)
        
        return np.array(sequences), np.array(next_words)

    def read_text_file(self, file_path):
        """Reads a text file with multiple encoding fallbacks."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read().lower()
                print(f"Successfully read file using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
                
        if text is None:
            raise ValueError("Could not read the file with any of the attempted encodings")
        
        return text

    def convert_docx_to_text(self, docx_path):
        """Extracts text from a .docx file and converts it to lowercase."""
        try:
            doc = Document(docx_path)
            return '\n'.join([para.text.lower() for para in doc.paragraphs])
        except Exception as e:
            raise IOError(f"Failed to read DOCX file: {e}")


@register_keras_serializable(package="Custom", name="LSTMModel")
class LSTMModel(tf.keras.Model):
    """
    Custom LSTM model for next word prediction.
    This class defines the model architecture and serialization methods.
    """
    def __init__(self, vocab_size, embedding_dim=100, sequence_length=10, hidden_size=128, num_layers=3, **kwargs):
        super(LSTMModel, self).__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        self.lstm_layers = []
        for i in range(num_layers):
            # The last LSTM layer should not return sequences
            is_last_layer = (i == num_layers - 1)
            self.lstm_layers.append(
                layers.LSTM(hidden_size, return_sequences=not is_last_layer)
            )
            
        self.dense = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        """Defines the forward pass of the model."""
        x = self.embedding(inputs)
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        return self.dense(x)

    def get_config(self):
        """Serializes the model's configuration."""
        config = super(LSTMModel, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a model instance from its configuration."""
        return cls(**config)

class NextWordPredictorGUI:
    """
    Manages the Tkinter-based graphical user interface for the application.
    """
    def __init__(self, root, model_path, dataset):
        self.root = root
        self.root.title("Next Word Predictor")
        self.dataset = dataset
        self.prediction_buttons = []

        try:
            self.model = load_model(
                model_path,
                custom_objects={"LSTMModel": LSTMModel}
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.show_error_message()
            return

        self.create_gui_elements()

    def predict_next_word(self, text):
        """
        Predicts the next word based on the input text.
        Returns the top 3 predictions with their probabilities.
        """
        try:
            if not text.strip():
                return []

            # Tokenize and process the input text
            words = text.lower().strip().split()
            
            # Use the last `sequence_length` words for the prediction
            recent_words = words[-self.dataset.sequence_length:]
            
            # Convert words to indices, using 0 (PAD token) for unknown words
            sequence = [self.dataset.word_to_index.get(word, 0) for word in recent_words]
                
            # Pad the sequence if it is shorter than the required length
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [sequence], maxlen=self.dataset.sequence_length, padding='pre'
            )
                
            # Make a prediction
            predictions = self.model.predict(padded_sequence, verbose=0)
            
            # Get the indices of the top 3 predictions
            top_k = 3
            top_indices = (-predictions).argsort()[:top_k]
            
            # Format results
            result = []
            for idx in top_indices:
                word = self.dataset.index_to_word[idx]
                probability = float(predictions[idx] * 100)
                result.append((word, f"{probability:.1f}%"))
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return []

    def on_key_release(self, event):
        """
        Handles the key release event to update predictions in real-time.
        """
        text = self.text_entry.get()
        predictions = self.predict_next_word(text)
        
        # Clear previous prediction buttons
        for button in self.prediction_buttons:
            button.destroy()
        self.prediction_buttons.clear()
        
        # Create new buttons for the new predictions
        for word, prob in predictions:
            btn = tk.Button(
                self.predictions_frame,
                text=f"{word} ({prob})",
                command=lambda w=word: self.add_word(w)
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.prediction_buttons.append(btn)

    def add_word(self, word):
        """Appends the selected predicted word to the text entry."""
        current_text = self.text_entry.get()
        if current_text and not current_text.endswith(' '):
            new_text = current_text + " " + word
        else:
            new_text = current_text + word
            
        self.text_entry.delete(0, tk.END)
        self.text_entry.insert(0, new_text)
        self.text_entry.icursor(tk.END)  # Move cursor to the end
        self.on_key_release(None)  # Update predictions after adding the word

    def create_gui_elements(self):
        """Initializes and arranges all the GUI widgets."""
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=20, pady=20)

        title_label = tk.Label(
            self.main_frame,
            text="Next Word Predictor",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        input_frame = tk.Frame(self.main_frame)
        input_frame.pack(fill=tk.X, pady=10)

        self.label = tk.Label(
            input_frame,
            text="Type your text here:",
            font=("Arial", 12)
        )
        self.label.pack()

        self.text_entry = tk.Entry(
            input_frame,
            width=50,
            font=("Arial", 12)
        )
        self.text_entry.pack(pady=5)
        self.text_entry.bind("<KeyRelease>", self.on_key_release)

        self.predictions_frame = tk.Frame(self.main_frame)
        self.predictions_frame.pack(fill=tk.X, pady=10)

    def show_error_message(self):
        """Displays an error message in the GUI if the model fails to load."""
        error_label = tk.Label(
            self.root,
            text="Error loading model. Please check the console for details.",
            fg="red",
            font=("Arial", 12)
        )
        error_label.pack(pady=10)

def train_model(model, dataset, epochs=10, batch_size=64):
    """Trains the model on the provided dataset."""
    print("Preparing training data...")
    X, y = dataset.prepare_training_data()
    
    # Add a dimension to y to match the model's output shape if needed
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=-1)

    print("Training model...")
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )
    return model

def create_and_train_model(dataset):
    """Creates a new LSTM model and initiates the training process."""
    print("Creating a new model...")
    model = LSTMModel(
        vocab_size=len(dataset.vocabulary),
        embedding_dim=100,
        sequence_length=dataset.sequence_length,
        hidden_size=128,
        num_layers=3
    )

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return train_model(model, dataset)

if __name__ == "__main__":
    # --- Configuration ---
    SEQUENCE_LENGTH = 10
    # IMPORTANT: Update this path to your training file
    TEXT_FILE_PATH = "path/to/your/training_data.txt" 
    # IMPORTANT: Update this path to where you want to save the model
    MODEL_PATH = "path/to/your/saved_lstm_model.keras" 

    # --- Application Start ---
    try:
        # Check if placeholder paths have been updated
        if "path/to/your" in TEXT_FILE_PATH or "path/to/your" in MODEL_PATH:
            print("="*50)
            print("ERROR: Please update the TEXT_FILE_PATH and MODEL_PATH variables.")
            print("="*50)
            exit()
            
        dataset = TextDataset(TEXT_FILE_PATH, SEQUENCE_LENGTH)
        print(f"Dataset initialized with vocabulary size: {len(dataset.vocabulary)}")

        if not os.path.exists(MODEL_PATH):
            print("Model not found. Training a new model...")
            model = create_and_train_model(dataset)
            print("Saving model...")
            model.save(MODEL_PATH)
            print(f"Model saved successfully to {MODEL_PATH}")
        else:
            print(f"Loading existing model from {MODEL_PATH}...")
        
        root = tk.Tk()
        app = NextWordPredictorGUI(root, MODEL_PATH, dataset)
        root.mainloop()

    except FileNotFoundError:
        print(f"ERROR: The training file was not found at '{TEXT_FILE_PATH}'.")
        print("Please ensure the path is correct.")
    except Exception as e:
        print(f"An unexpected application error occurred: {e}")
```

---

## Model Architecture

The `LSTMModel` is composed of the following layers:

- **Embedding Layer**: Converts word indices into dense vectors of fixed size (`embedding_dim`). This allows the model to learn relationships between words.

- **LSTM Layers**: A stack of `num_layers` LSTM cells that process the sequences of embeddings to capture temporal dependencies and context. `return_sequences=True` is used on all but the last LSTM layer to pass the full sequence to the next layer.

- **Dense Layer**: A fully connected output layer with a softmax activation function. It produces a probability distribution over the entire vocabulary, indicating the likelihood of each word being the next one.

---

## Dependencies

- TensorFlow (>= 2.x)
- NumPy
- python-docx
- Tkinter (included with standard Python distributions)

---

## Contributing

Contributions are welcome! If you would like to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add a new feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
