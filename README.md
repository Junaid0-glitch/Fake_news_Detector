# Fake News Detector

This project is a Fake News Detection system using DistilBERT and PyTorch, with a Streamlit web app for user interaction.

## Features
- Data preprocessing and visualization (Jupyter Notebook)
- Model training using DistilBERT embeddings
- Streamlit app for real-time news classification

## Files
- `Fakke_news_detector.ipynb`: Data analysis, preprocessing, model training
- `app.py`: Streamlit web app for fake news detection
- `News_classifier.pt`: Trained PyTorch model weights
- `tokenizer_distilbert/`: Saved tokenizer files

## Usage
1. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
2. **Run the Streamlit app**
   ```powershell
   streamlit run app.py
   ```
3. **Interact**
   - Paste news text in the app to check if it is Fake or True.

## Training
- See the notebook for data loading, preprocessing, model training, and evaluation steps.

## Model & Tokenizer
- The model and tokenizer are saved after training and loaded in the app for inference.

## Requirements
See `requirements.txt` for all required Python packages.
