from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from news_sentimate import what_sentiment
from get_news import get_article_summaries
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from typing import Tuple
import requests

device = "cpu"
if torch.cuda.is_available:
    device = "cuda:0"

make_tokens = AutoTokenizer.from_pretrained("ProsusAI/finbert")
my_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
response = ["positive", "negative", "neutral"]


NEWS_API_KEY ="5S........3P6MR"    
API_KEY = "PK5OH..........WZX"
SECRET = "tg6Lz........................pnoC"
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CONFIG = {
    "API_KEY": API_KEY,
    "API_SECRET": SECRET,
    "PAPER": True
}

def get_news(symbol, start, end):
    url = f"https://newsapi.org/v2/everything?q={symbol}&from={start}&to={end}&apiKey=28af44cf04e4412cbfbd34c33d459603"
    response = requests.get(url)
    data = response.json()
    headlines = [article['title'] for article in data['articles']]
    return headlines

class alpha(Strategy):
    def initialize(self, symbol:str = "TSLA", cash_at_risk:float = 0.1):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=SECRET)
    
    
    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash*self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y%m%d'), three_days_prior.strftime('%Y%m%d')
    
    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()        
        article_summaries = get_article_summaries(NEWS_API_KEY, self.symbol, three_days_prior, today)
        print("article_summaries")
        probability, sentiment = what_sentiment(article_summaries)
        return probability, sentiment 
    
    def trade_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        if cash > last_price:
            if sentiment == "positve" and probability > 0.90:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type = "bracket",
                    take_profit_price=last_price*1.20,
                    stop_loss_price= last_price*0.8
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability>0.90:
                if (self.last_trade == "buy"):
                    self.sell_all()
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.15
                )
                self.submit_order(order) 
                self.last_trade = "sell"            

start_date = datetime(2010, 1,1)
end_date = datetime(2023,1,1)

broker = Alpaca(ALPACA_CONFIG)
strategy = alpha(name = "new1",
                 broker=broker,
                 parameters={"symbol": "SPY",
                             "cash_at_risk":0.5})
top_20_us_stocks_tickers = [
    "MSFT",
    "AAPL",
    "GOOG",
    "AMZN",
    "NVDA",
    "META",
    "BRK-B",
    "LLY",
    "TSLA",
    "AVGO",
    "V",
    "JPM",
    "UNH",
    "WMT",
    "MA",
    "XOM",
    "JNJ",
    "PG",
    "HD",
    "COST"
]
                    
top_us_stocks = top_20_us_stocks_tickers
for symbol in top_us_stocks:
    strategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={"symbol": symbol, "cash_at_risk": 0.1}
    ) 
        