import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn

class NewsClassifier(nn.Module):
    def __init__(self):
        super(NewsClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = bert_output.last_hidden_state[:, 0, :]
        return self.classifier(sentence_embeddings)

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("tokenizer_distilbert")
    model = NewsClassifier()
    model.load_state_dict(torch.load("News_classifier.pt", map_location="cpu"))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()
class_names = ["True", "Fake"]

st.title("Fake News Detection App")
st.write("Paste a news article/text below to check if it is **Fake** or **True**.")

news_text = st.text_area("Enter News Text", height=200)

if st.button("Predict"):
    if news_text.strip():

        encoding = tokenizer(news_text, padding="max_length", max_length=200, truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
 
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            prediction = torch.argmax(outputs, dim=1).item()
        result = class_names[prediction]
        st.success(f"This news is **{result}**.")
    else:
        st.warning("Please enter some news text!")
