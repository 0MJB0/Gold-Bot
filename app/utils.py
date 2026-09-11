from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from .performance import cached_result

import feedparser
from bs4 import BeautifulSoup
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import yfinance as yf
import datetime
import logging
from plotly.offline import plot
import plotly.graph_objs as go

logger = logging.getLogger(__name__)
GOLD_PRICE_COLUMNS = ['Open', 'Close', 'High', 'Low']

# ========================================== Predict.html Functions  =====================================================
@lru_cache(maxsize=1)
def load_dataset():
    
    df_technical = pd.read_excel("app/Data/1980-2024_Dataset_investing.xlsx")

    df_technical = df_technical.iloc[::-1]

    X_tech = df_technical.drop(columns=['Dates', 'Close'])
    y_tech = df_technical['Close']

    return X_tech, y_tech 

def normalize_and_split_data(X_tech, y_tech):
    
    scaler = MinMaxScaler()
    X_tech_scaled = scaler.fit_transform(X_tech)

    X_tech_train, X_tech_test, y_tech_train, y_tech_test = train_test_split(X_tech_scaled, y_tech, test_size=0.2, shuffle=False)
    
    return scaler, X_tech_train, X_tech_test, y_tech_train, y_tech_test

def create_train_and_predict_data(X_tech_train, y_tech_train, X_tech_test):

    with open('app/Data/technique_model.pkl', 'rb') as f:
        mlp_model = pickle.load(f)
    
    y_tech_pred = mlp_model.predict(X_tech_test)
    
    return mlp_model, y_tech_pred

def calculate_gold_price_trend_for_p(predicted_categories):
        base_score = 0
        for category in predicted_categories:
            if category in [0, 2, 4, 7, 9, 10, 12, 13]:
                base_score += 1
            elif category in [1, 3, 5, 6, 8, 11]:
                base_score -= 1
        if base_score < 0:
            return "down"
        elif base_score > 0:
            return "up"
        else:
            return "same"


def future_trend_prediction(model, X_last, scaler):
    future_trend = model.predict(scaler.transform(X_last))[0]
    return "up" if future_trend > X_last[0, -1] else "down"


def get_last_data(X_tech):
    
    X_last_day = X_tech.iloc[-1].values.reshape(1, -1)
    X_last_week = X_tech.iloc[-7].values.reshape(1, -1)
    X_last_month = X_tech.iloc[-30].values.reshape(1, -1)

    return X_last_day, X_last_week, X_last_month


def future_trend(mlp_model, scaler, X_last_day, X_last_week, X_last_month):
    
    day_trend = future_trend_prediction(mlp_model, X_last_day, scaler)
    week_trend = future_trend_prediction(mlp_model, X_last_week, scaler)
    month_trend = future_trend_prediction(mlp_model, X_last_month, scaler)

    return day_trend, week_trend, month_trend

def final_trend(day_trend, week_trend, month_trend, news_trend):
    
    final_day_trend = day_trend if (day_trend == news_trend) else ("down" if day_trend == "up" else "up")
    final_week_trend = week_trend if (week_trend == news_trend) else ("down" if week_trend == "up" else "up")
    final_month_trend = month_trend if (month_trend == news_trend) else ("down" if month_trend == "up" else "up")

    return final_day_trend, final_week_trend, final_month_trend 


# ========================================== Fundemantal.html Functions  =====================================================


def load_rss_feed(url):
    return get_news_titles_and_urls_from_rss(url)

@lru_cache(maxsize=1)
def load_model_and_vectorizer():
    df = pd.read_excel('app/Data/fundemantal_dataset.xlsx')

    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['Title'].values)

    with open('app/Data/fundemantal_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return vectorizer, model

def get_all_news_titles(rss_urls):
    # Both pages share cached feeds; fetch independent sources concurrently.
    with ThreadPoolExecutor(max_workers=2) as executor:
        feeds = list(executor.map(load_rss_feed, rss_urls))
    items = [item for feed in feeds for item in feed]
    return items, [item['title'] for item in items]


def predict_news_categories_df(news_titles, vectorizer, model):
    return predict_news_categories(pd.DataFrame({'Title': news_titles}), vectorizer, model)

def filter_news_items(news_items):
    return [item for item in news_items if item['title'] and item['description']]


def get_news_titles_from_rss(url):
    return [item['title'] for item in get_news_titles_and_urls_from_rss(url)]


def clean_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

@cached_result(300)
def get_news_titles_and_urls_from_rss(url):
    try:
        response = requests.get(url, timeout=(3, 5))
        response.raise_for_status()
    except requests.RequestException:
        logger.warning('News source unavailable: %s', url)
        return []
    feed = feedparser.parse(response.content)
    return [dict(title=clean_html_tags(entry.get('title', '')),
                 description=clean_html_tags(entry.get('description', '')),
                 link=entry.get('link', ''))
            for entry in feed.entries if entry.get('title')]


def predict_news_categories(df, vectorizer, model):
    if df.empty:
        return pd.DataFrame(columns=['Title', 'Predicted_Category'])
    # Separate text and label columns
    X = df['Title'].values

    # Use TF-IDF vectorization to convert text data to vectors
    X_vectorized = vectorizer.transform(X)

    # Make predictions
    predicted_categories = model.predict(X_vectorized)

    # Save predicted categories and news titles to a DataFrame
    result_df = pd.DataFrame({'Title': X, 'Predicted_Category': predicted_categories})
    return result_df

def calculate_gold_price_trend(predicted_categories):
    base_score = 0  # Base score

    # Calculate score based on predicted categories
    for category in predicted_categories:
        if category in [0, 2, 4, 7, 9, 10, 12, 13]:
            base_score += 1
        elif category in [1, 3, 5, 6, 8, 11]:
            base_score -= 1

    # Gold price prediction
    if base_score < 0:
        base_score_text = "Gold price may decrease"
    elif base_score > 0:
        base_score_text = "Gold price may increase"
    else:
        base_score_text = "Gold price may remain unchanged"
    
    return base_score_text

def get_news_titles_and_descriptions_from_rss(rss_url):
    # Fetch RSS feed
    response = requests.get(rss_url)
    if response.status_code == 200:
        # Parse XML
        soup = BeautifulSoup(response.content, 'xml')
        # Extract news titles and descriptions
        items = soup.find_all('item')
        news_items = []
        for item in items:
            title = item.find('title').text
            description = item.find('description').text
            # Remove HTML tags from description
            description = BeautifulSoup(description, "html.parser").get_text()
            news_items.append({'title': title, 'description': description})
        return news_items
    else:
        return []

    # ========================================== Index.html Functions  =====================================================

def get_date_and_period_params(request):
    period = request.GET.get('period', '10d')
    start_date_str = request.GET.get('start_date', '')
    end_date_str = request.GET.get('end_date', '')

    start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None

    return period, start_date, end_date

@cached_result(60)
def plot_gold_data(gold_data):
    fig = go.Figure()
    for column, color in [('Open', '#929292'), ('Close', '#303030'),
                          ('High', '#65747d'), ('Low', '#9b9389')]:
        fig.add_trace(go.Scatter(
            x=gold_data.index, y=gold_data[column], name=column,
            mode='lines', line=dict(color=color, width=2 if column == 'Close' else 1.5,
                                     dash={'Open': 'dot', 'Close': 'solid', 'High': 'dash', 'Low': 'dashdot'}[column]),
            hovertemplate='%{y:,.2f} USD<extra>' + column + '</extra>',
        ))
    fig.update_layout(
        height=380, margin=dict(l=55, r=20, t=45, b=40),
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Segoe UI, sans-serif', color='#686868', size=11),
        hovermode='x unified', dragmode=False,
        legend=dict(orientation='h', y=1.16, x=0),
        xaxis=dict(showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(gridcolor='#eeeeee', zeroline=False, fixedrange=True, tickprefix='$'),
    )
    return plot(fig, auto_open=False, output_type='div', include_plotlyjs=False,
                config={'responsive': True, 'displayModeBar': False})




@cached_result(60)
def fetch_gold_data(ticker='GC=F', period='10d', interval='1d', start_date=None, end_date=None):
    options = dict(tickers=ticker, interval=interval, auto_adjust=False,
                   progress=False, threads=False, timeout=15)
    if start_date and end_date:
        options.update(start=start_date, end=end_date)
    elif period.endswith('d') and period[:-1].isdigit() and period not in ('1d', '5d'):
        # Custom day periods are not supported by every Yahoo/yfinance version.
        end = datetime.datetime.now(datetime.timezone.utc)
        options.update(start=end - datetime.timedelta(days=int(period[:-1])), end=end)
    else:
        options['period'] = period

    try:
        data = yf.download(**options)
    except Exception:
        logger.exception('Could not fetch gold prices for %s', ticker)
        return pd.DataFrame(columns=GOLD_PRICE_COLUMNS)

    if data is None or data.empty:
        return pd.DataFrame(columns=GOLD_PRICE_COLUMNS)
    # Recent yfinance versions return (Price, Ticker) columns even for one ticker.
    # Select that ticker so templates and Plotly receive scalar price columns.
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level=-1)
    return data.reindex(columns=GOLD_PRICE_COLUMNS).dropna(how='all').sort_index()

def fetch_recent_gold_data(ticker='GC=F', days=10):
    data = fetch_gold_data(ticker=ticker, period=f'{days}d', interval='1d')
    return data.tail(days)[::-1]



@lru_cache(maxsize=1)
def load_technical_prediction_resources():
    features, target = load_dataset()
    scaler, *_ = normalize_and_split_data(features, target)
    with open('app/Data/technique_model.pkl', 'rb') as source:
        model = pickle.load(source)
    return model, scaler, get_last_data(features)
