import torch
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification

stance_labels = {0: "None", 1: "Favor", 2: "Against"}
sentiment_labels = {0: "Neutral", 1: "Positive", 2: "Negative"}
sarcasm_labels = {0: "No", 1: "Yes"}

# Charger les checkpoints des modèles
stance_model_checkpoint = "best-checkpoint_stance.pth"
sentiment_model_checkpoint = "best-checkpoint-Sentiment.pth"
sarcasm_model_checkpoint = "best-checkpoint_Sarcasm.pth"

# Charger les données de test
data = pd.read_csv("PredictionStanceEval.csv")

# Charger le tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02-twitter')

# Charger les modèles
stance_model = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter", num_labels=3)
stance_model.load_state_dict(torch.load(stance_model_checkpoint))
stance_model.eval()

sentiment_model = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter", num_labels=3)
sentiment_model.load_state_dict(torch.load(sentiment_model_checkpoint))
sentiment_model.eval()

sarcasm_model = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter", num_labels=2)
sarcasm_model.load_state_dict(torch.load(sarcasm_model_checkpoint))
sarcasm_model.eval()

# Faire des prédictions
for index, row in data.iterrows():
    tweet = row['text']

    # Tokeniser et préparer les données pour le modèle
    inputs = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Faire des prédictions avec chaque modèle
    stance_logits = stance_model(input_ids, attention_mask=attention_mask)[0]
    sentiment_logits = sentiment_model(input_ids, attention_mask=attention_mask)[0]
    sarcasm_logits = sarcasm_model(input_ids, attention_mask=attention_mask)[0]

    stance_prediction = torch.argmax(stance_logits).item()
    sentiment_prediction = torch.argmax(sentiment_logits).item()
    sarcasm_prediction = torch.argmax(sarcasm_logits).item()
    
    stance_label = stance_labels[stance_prediction]
    sentiment_label = sentiment_labels[sentiment_prediction]  # Correction ici
    sarcasm_label = sarcasm_labels[sarcasm_prediction]  # Correction ici
    
    # Afficher les prédictions
    print(f"Tweet: {tweet}")
    print(f"Prédiction Stance : {stance_label}")
    print(f"Prédiction Sentiment : {sentiment_label}")
    print(f"Prédiction Sarcasme : {sarcasm_label}")
    print("-" * 50)
