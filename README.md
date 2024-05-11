# Mawqif
The Python stanceEval.py file is designed to make predictions on test data using three fine-tuning classification models based on BERT (Bidirectional Encoder Representations from Transformers) for the following tasks:
  - Stance Detection: Determine the position of a tweet in relation to a given topic, with the categories "NONE", "FAVOR" and "AGAINST".
  - Sentiment Analysis: Evaluate the sentiment expressed in a tweet, with the categories "Neutral", "Positive" and "Negative".
  - Sarcasm Detection: Identify whether a tweet is sarcastic or not, with the "No" and "Yes" categories.
Here's how to use this script:

1- Make sure you have installed the necessary libraries (torch, pandas, transformers).
2- Make sure you have the model checkpoints (best-checkpoint_stance.pth, best-checkpoint-Sentiment.pth, best-checkpoint_Sarcasm.pth) as well as the test data file (PredictionStanceEval.csv) in the same directory as your script.
3- Run the script. It will make predictions for each tweet in the test data using all three models. Predictions will be displayed for each tweet, showing the predicted position, sentiment and sarcasm.
