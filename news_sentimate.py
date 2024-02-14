import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from typing import Tuple


device = "cpu"
if torch.cuda.is_available:
    device = "cuda:0"

make_tokens = AutoTokenizer.from_pretrained("ProsusAI/finbert")
my_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
response = ["positive", "negative", "neutral"]

def what_sentiment(news):
    if news:
        tokens = make_tokens(news, return_tensors="pt", padding=True).to(device)
        res = my_model(tokens["input_ids"], attention_mask = tokens["attention_mask"])["logits"]
        res = torch.nn.functional.softmax(torch.sum(res, 0), dim =-1)
        prob =res[torch.argmax(res)]
        sentiment = response[torch.argmax(res)]
        return prob, sentiment
    else:
        return 0, response[-1]    
    
if __name__ == "__main__":
    tensor, sentiment = what_sentiment(['the ceo resigns'])
    print(tensor, sentiment)
       