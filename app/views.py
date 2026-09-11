from django.shortcuts import render
from django.shortcuts import redirect

from .performance import cached_result
from .utils import (
    plot_gold_data, fetch_gold_data, get_date_and_period_params,
    load_model_and_vectorizer, get_all_news_titles, predict_news_categories_df,
    calculate_gold_price_trend, calculate_gold_price_trend_for_p,
    load_technical_prediction_resources, future_trend, final_trend,
)

RSS_URLS = (
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "https://www.theguardian.com/world/rss",
)


@cached_result(300)
def classified_news():
    items, titles = get_all_news_titles(RSS_URLS)
    if not titles:
        return []
    vectorizer, model = load_model_and_vectorizer()
    predictions = predict_news_categories_df(titles, vectorizer, model)
    return list(zip(items, predictions['Title'], predictions['Predicted_Category']))


    # ========================================== Page Render section =====================================================

def fundemantal(request):
    return render(request, 'fundemantal.html', {})

def update_chart(request):
    period = request.GET.get('period', '10d')
    return redirect(f'/?period={period}')

# ========================================== Technical Analysis =============================================

def predict(request):
    model, scaler, last_data = load_technical_prediction_resources()
    categories = [category for _, _, category in classified_news()]
    news_trend = calculate_gold_price_trend_for_p(categories)
    day, week, month = future_trend(model, scaler, *last_data)
    day, week, month = final_trend(day, week, month, news_trend)
    return render(request, 'predict.html', {
        'combined_prediction_for_day': day,
        'combined_prediction_for_week': week,
        'combined_prediction_for_month': month,
        'gold_price_trend': news_trend,
    })


def news(request):
    rows = classified_news()
    return render(request, 'fundemantal.html', {
        'news_list': [row for row in rows if row[0]['description']],
        'gold_price_trend': calculate_gold_price_trend([row[2] for row in rows]),
    })


# Views.py (Home Page) - the function that pulls the last 10 days prices from yf + graph of the data

def index(request):
    # Get date and period parameters from request
    period, start_date, end_date = get_date_and_period_params(request)

    # Fetch gold price data for plotting
    gold_data = fetch_gold_data(period=period, start_date=start_date, end_date=end_date)
    
    # Plot gold price data
    plot_div_gold = plot_gold_data(gold_data)

    # Reuse the chart data so the table does not require another network request.
    table_gold_prices = gold_data.iloc[::-1]

    return render(request, 'index.html', {
        'plot_div_gold': plot_div_gold,
        'period': period,
        'start_date': start_date,
        'end_date': end_date,
        'table_gold_prices': table_gold_prices,
        'summary': gold_data.iloc[-1].to_dict() if not gold_data.empty else None,
        'summary_date': gold_data.index[-1] if not gold_data.empty else None
    })
