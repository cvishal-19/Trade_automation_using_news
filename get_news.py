import requests

def get_article_summaries(api_key, ticker, time_from, time_to, limit=1):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={time_from}T0000&time_to={time_to}T0000&limit={limit}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    if 'feed' in data:
        feed = data['feed']
        article_summaries = [article['summary'] for article in feed]
        return article_summaries
    else:
        return []