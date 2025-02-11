# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from docx import Document
from keras.saving import register_keras_serializable

class TextDataset:
    def __init__(self, text_file_path, sequence_length):
        self.sequence_length = sequence_length
        
        if text_file_path.endswith('.docx'):
            text = self.convert_docx_to_text(text_file_path)
        else:
            text = self.read_text_file(text_file_path)
        
        PAD_TOKEN = "<PAD>"
        words = text.split()
        unique_words = sorted(set(words))
        unique_words.insert(0, PAD_TOKEN)
        
        self.vocabulary = unique_words
        self.word_to_index = {word: idx for idx, word in enumerate(unique_words)}
        self.index_to_word = {idx: word for idx, word in enumerate(unique_words)}
        self.words = words  # Store all words for training data generation
        
        print(f"Vocabulary size: {len(self.vocabulary)}")

    def prepare_training_data(self):
        """Prepare training data sequences."""
        sequences = []
        next_words = []
        
        for i in range(0, len(self.words) - self.sequence_length):
            seq = self.words[i:i + self.sequence_length]
            next_word = self.words[i + self.sequence_length]
            
            # Convert words to indices
            seq_indices = [self.word_to_index.get(word, 0) for word in seq]
            next_word_index = self.word_to_index.get(next_word, 0)
            
            sequences.append(seq_indices)
            next_words.append(next_word_index)
        
        return np.array(sequences), np.array(next_words)

    def read_text_file(self, file_path):
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
        doc = Document(docx_path)
        return '\n'.join([para.text.lower() for para in doc.paragraphs])



@register_keras_serializable(package="Custom", name="LSTMModel")
class LSTMModel(tf.keras.Model):
    ...

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
            self.lstm_layers.append(
                layers.LSTM(hidden_size, return_sequences=(i < num_layers - 1))
            )
            
        self.dense = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        return self.dense(x)

    def get_config(self):
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
        return cls(
            vocab_size=config.pop("vocab_size"),
            embedding_dim=config.pop("embedding_dim", 100),
            sequence_length=config.pop("sequence_length", 10),
            hidden_size=config.pop("hidden_size", 128),
            num_layers=config.pop("num_layers", 3),
            **config
        )

class NextWordPredictorGUI:
    def __init__(self, root, model_path, dataset):
        self.root = root
        self.root.title("Next Word Predictor")
        self.dataset = dataset
        self.prediction_buttons = []  # Add this line to store prediction buttons

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
        try:
            if not text.strip():
                return "Waiting for input..."

            # Tokenize input text
            words = text.lower().strip().split()
            sequence = []
            
            # Take last n words where n is sequence_length
            recent_words = words[-self.dataset.sequence_length:]
            
            # Convert words to indices, using 0 (PAD token) for unknown words
            for word in recent_words:
                if word in self.dataset.word_to_index:
                    sequence.append(self.dataset.word_to_index[word])
                else:
                    sequence.append(0)  # PAD token index
                
            # Pad sequence if needed
            while len(sequence) < self.dataset.sequence_length:
                sequence.insert(0, 0)  # Pad with zeros at the beginning
                
            # Make prediction
            input_sequence = np.array([sequence])
            predictions = self.model.predict(input_sequence, verbose=0)
            
            # Handle different prediction shapes
            if len(predictions.shape) == 3:
                predictions = predictions[0, -1, :]  # Take last timestep
            elif len(predictions.shape) == 2:
                predictions = predictions[0]
                
            # Get top 3 predictions
            top_k = 3
            top_indices = (-predictions).argsort()[:top_k]
            
            # Clear previous result
            for button in self.prediction_buttons:
                button.destroy()
            self.prediction_buttons.clear()
            
            # Format results and create buttons
            result = []
            for idx in top_indices:
                if isinstance(idx, np.integer):
                    idx = idx.item()
                word = self.dataset.index_to_word[idx]
                prob = float(predictions[idx] * 100)
                result.append((word, f"{prob:.1f}%"))
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            print(f"Debug info - Text: {text}")
            if 'sequence' in locals():
                print(f"Debug info - Sequence: {sequence}")
            if 'predictions' in locals():
                print(f"Debug info - Predictions shape: {predictions.shape}")
            return []

    def on_key_release(self, event):
        text = self.text_entry.get()
        predictions = self.predict_next_word(text)
        
        # Clear previous buttons
        for button in self.prediction_buttons:
            button.destroy()
        self.prediction_buttons.clear()
        
        # Create new buttons for predictions
        for word, prob in predictions:
            btn = tk.Button(
                self.predictions_frame,
                text=f"{word} ({prob})",
                command=lambda w=word: self.add_word(w)
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.prediction_buttons.append(btn)

    def add_word(self, word):
        """Add the selected word to the text entry"""
        current_text = self.text_entry.get()
        if current_text:
            new_text = current_text + " " + word
        else:
            new_text = word
        self.text_entry.delete(0, tk.END)
        self.text_entry.insert(0, new_text)
        self.text_entry.icursor(tk.END)  # Move cursor to end
        self.on_key_release(None)  # Update predictions

    def create_gui_elements(self):
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
        error_label = tk.Label(
            self.root,
            text="Error loading model. Please check console.",
            fg="red",
            font=("Arial", 12)
        )
        error_label.pack(pady=10)

def train_model(model, dataset, epochs=10, batch_size=64):
    """Train the model on the dataset."""
    print("Preparing training data...")
    X, y = dataset.prepare_training_data()
    
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
    print("Creating new model...")
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
    SEQUENCE_LENGTH = 10
    TEXT_FILE_PATH = "C:\\Users\\hp\\Downloads\\sherlock-holm.es_stories_plain-text_advs.txt"
    MODEL_PATH = "C:\\Users\\hp\\Downloads\\saved_lstm_model.keras"

    try:
        dataset = TextDataset(TEXT_FILE_PATH, SEQUENCE_LENGTH)
        print(f"Dataset initialized with vocabulary size: {len(dataset.vocabulary)}")

        if not os.path.exists(MODEL_PATH):
            print("Training new model...")
            model = create_and_train_model(dataset)
            print("Saving model...")
            model.save(MODEL_PATH)
            print("Model saved successfully!")
        else:
            print("Loading existing model...")
        
        root = tk.Tk()
        app = NextWordPredictorGUI(root, MODEL_PATH, dataset)
        root.mainloop()

    except Exception as e:
        print(f"Application error: {e}")