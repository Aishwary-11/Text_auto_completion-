# Next Word Predictor with LSTM

This project implements a next-word prediction application using a Long Short-Term Memory (LSTM) neural network. It features a graphical user interface (GUI) built with Tkinter that suggests the next word in a sequence as the user types.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
  - [Training a New Model](#training-a-new-model)
  - [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

*   **Real-time Word Prediction**: Suggests the top 3 most likely next words as you type.
*   **LSTM-based Model**: Utilizes a deep learning model for sequence prediction.
*   **Custom Training Data**: Can be trained on any `.txt` or `.docx` file.
*   **Interactive GUI**: A user-friendly interface built with Tkinter.
*   **Model Persistence**: Saves the trained model to disk for later use.
*   **Extensible Architecture**: The model and data handling are modular, allowing for easy extension.

## Project Structure

The project is contained within a single Python script and is organized into the following main classes:

*   `TextDataset`: Handles the loading and preprocessing of the text corpus. It reads `.txt` and `.docx` files, creates a vocabulary, and prepares the data for training.
*   `LSTMModel`: Defines the architecture of the neural network using TensorFlow and Keras. It consists of an embedding layer, multiple LSTM layers, and a dense output layer.
*   `NextWordPredictorGUI`: Implements the Tkinter-based graphical user interface, which allows users to input text and receive real-time word predictions.

## How It Works

1.  **Data Preprocessing**: The `TextDataset` class reads the input text, converts it to lowercase, and splits it into a list of words. A vocabulary of unique words is created, and each word is mapped to a unique index.

2.  **Sequence Generation**: The text is then transformed into sequences of a fixed length (`SEQUENCE_LENGTH`). Each sequence consists of a set of input words and the corresponding next word as the target.

3.  **Model Architecture**: The `LSTMModel` is a neural network designed for sequence data.
    *   **Embedding Layer**: Converts the integer-encoded words into dense vectors of a fixed size.
    *   **LSTM Layers**: Process the sequences of embeddings to capture temporal dependencies.
    *   **Dense Layer**: Outputs a probability distribution over the entire vocabulary for the next word.

4.  **Training**: If a pre-trained model is not found, the script trains the `LSTMModel` on the generated sequences. The model learns to predict the next word in a sequence.

5.  **Prediction**: The `NextWordPredictorGUI` takes the user's input text, preprocesses it into a sequence, and feeds it to the trained model. The model then predicts the top 3 most likely next words, which are displayed as interactive buttons in the GUI.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/next-word-predictor.git
    cd next-word-predictor
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install tensorflow numpy python-docx
    ```

## Usage

### Training a New Model

1.  Place your training data (a `.txt` or `.docx` file) in the project directory.
2.  Update the `TEXT_FILE_PATH` variable in the script to point to your training file.
3.  If a `saved_lstm_model.keras` file exists, delete it to trigger the training process.
4.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```
    The script will preprocess the data, build the model, and start the training process. The trained model will be saved as `saved_lstm_model.keras`.

### Running the Application

Once a model has been trained and saved, you can run the application directly:

```bash
python your_script_name.py
