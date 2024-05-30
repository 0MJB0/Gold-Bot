from django.shortcuts import render
from django.shortcuts import redirect
import pandas as pd

from .utils import (
    filter_news_items, 
    plot_gold_data,
    fetch_gold_data,
    calculate_gold_price_trend,
    load_model_and_vectorizer,
    get_all_news_titles,
    predict_news_categories_df,
    fetch_recent_gold_data,
    get_date_and_period_params,
    load_dataset,
    normalize_and_split_data,
    create_train_and_predict_data,
    calculate_gold_price_trend_for_p,
    get_last_data,
    future_trend,
    final_trend
)


    # ========================================== Page Render section =====================================================

def fundemantal(request):
    return render(request, 'fundemantal.html', {})

def update_chart(request):
    period = request.GET.get('period', '10d')
    return redirect(f'/?period={period}')

# ========================================== Technical Analysis =============================================

def predict(request):
   
  # Load historical data and get features
    X_tech, y_tech = load_dataset()
    # Normalize the data and split it into training and test sets
    scaler, X_tech_train, X_tech_test, y_tech_train, y_tech_test = normalize_and_split_data(X_tech, y_tech)
    
    # Create and train the MLP Regressor model, and predict on the technical test data
    mlp_model, y_tech_pred = create_train_and_predict_data(X_tech_train, y_tech_train, X_tech_test)
    
    #Load the fundemental dataset and create vectorizer and model
    vectorizer, model = load_model_and_vectorizer()

    rss_urls = [ 
           "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
           "https://www.theguardian.com/world/rss"
    ]
    
    all_news_items, all_news_titles = get_all_news_titles(rss_urls)

    # Predict categories for news titles
    predicted_categories = [model.predict(vectorizer.transform([title]))[0] for title in all_news_titles]

    # Calculate the gold price trend from news
    news_trend = calculate_gold_price_trend_for_p(predicted_categories)

    X_last_day, X_last_week, X_last_month = get_last_data(X_tech)

    day_trend, week_trend, month_trend = future_trend(mlp_model, scaler, X_last_day, X_last_week, X_last_month)

    # Compare results from two datasets to make the final prediction
    final_day_trend, final_week_trend, final_month_trend = final_trend(day_trend, week_trend, month_trend, news_trend)

    # Save prediction results to an Excel file
    output_df = pd.DataFrame({
        '1 day later': [final_day_trend],
        '1 week later': [final_week_trend],
        '1 month later': [final_month_trend]
    })

    output_df.to_excel("future_gold_trend_predictions.xlsx", index=False)

    # Prepare the context for the template
    context = {
        'combined_prediction_for_day': final_day_trend,  # Example: first prediction trend
        'gold_price_trend': news_trend,
        'combined_prediction_for_week': final_week_trend,
        'combined_prediction_for_month': final_month_trend
    }

    return render(request, 'predict.html', context)


# ========================================== Fundemantal Analysis ==========================================


def news(request):
    rss_urls = [
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        "https://www.theguardian.com/world/rss"
    ]

    # Load the model and vectorizer
    vectorizer, model = load_model_and_vectorizer()

    # Get all news titles and items
    all_news_items, all_news_titles = get_all_news_titles(rss_urls)

    # Predict categories using news titles
    predicted_df = predict_news_categories_df(all_news_titles, vectorizer, model)

    # Calculate the gold price trend
    gold_price_trend = calculate_gold_price_trend(predicted_df['Predicted_Category'].values)

    # Filter out empty entries
    filtered_news_list = filter_news_items(all_news_items)

    return render(request, 'fundemantal.html', {
        'news_list': zip(filtered_news_list, predicted_df['Title'].values, predicted_df['Predicted_Category'].values),
        'gold_price_trend': gold_price_trend  
    })

# Views.py (Home Page) - the function that pulls the last 10 days prices from yf + graph of the data

def index(request):
    # Fetch recent gold price data for the table
    recent_gold_prices = fetch_recent_gold_data(days=10)

    # Get date and period parameters from request
    period, start_date, end_date = get_date_and_period_params(request)

    # Fetch gold price data for plotting
    gold_data = fetch_gold_data(period=period, start_date=start_date, end_date=end_date)
    
    # Plot gold price data
    plot_div_gold = plot_gold_data(gold_data)

    # Fetch gold price data for the table below the chart
    table_gold_prices = fetch_gold_data(period=period, start_date=start_date, end_date=end_date)[::-1] if gold_data is not None else []

    return render(request, 'index.html', {
        'recent_gold_prices': recent_gold_prices,
        'plot_div_gold': plot_div_gold,
        'period': period,
        'start_date': start_date,
        'end_date': end_date,
        'table_gold_prices': table_gold_prices
    })
