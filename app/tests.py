from unittest.mock import patch, Mock
import time
import requests
from django.core.cache import cache

import pandas as pd
from django.test import SimpleTestCase

from .utils import fetch_gold_data


class GoldPricesTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        self.prices = pd.DataFrame(
            [[2401.25, 2405.50, 2410.75, 2390.25],
             [2411.25, 2415.50, 2420.75, 2400.25]],
            index=pd.to_datetime(['2026-09-08', '2026-09-09']),
            columns=['Open', 'Close', 'High', 'Low'],
        )

    @patch('app.utils.yf.download')
    def test_multiindex_prices_render_as_numbers_with_one_download(self, download):
        data = self.prices.copy()
        data.columns = pd.MultiIndex.from_product([data.columns, ['GC=F']])
        download.return_value = data
        response = self.client.get('/')
        self.assertContains(response, '<td>2401.25</td>')
        self.assertContains(response, '<td>2415.50</td>')
        self.assertContains(response, '<td>2026-09-09</td>')
        self.assertLess(response.content.index(b'<td>2026-09-09</td>'),
                        response.content.index(b'<td>2026-09-08</td>'))
        download.assert_called_once()
        options = download.call_args.kwargs
        self.assertNotIn('period', options)
        self.assertEqual((options['end'] - options['start']).days, 10)

    @patch('app.utils.yf.download')
    def test_flat_columns_and_standard_period_still_work(self, download):
        download.return_value = self.prices
        pd.testing.assert_frame_equal(fetch_gold_data(period='1mo'), self.prices)
        self.assertEqual(download.call_args.kwargs['period'], '1mo')

    @patch('app.utils.yf.download')
    def test_empty_response_keeps_homepage_available(self, download):
        for result in (None, pd.DataFrame()):
            with self.subTest(result=type(result).__name__):
                download.return_value = result
                self.assertContains(self.client.get('/'),
                                    'No gold prices available for the selected period')

    @patch('app.utils.yf.download', side_effect=TimeoutError('Timed out'))
    def test_download_failure_keeps_homepage_available(self, download):
        with self.assertLogs('app.utils', level='ERROR'):
            self.assertContains(self.client.get('/'),
                                'No gold prices available for the selected period')


    @patch('app.utils.yf.download')
    def test_repeat_navigation_reuses_prices_and_chart(self, download):
        download.return_value = self.prices
        first = self.client.get('/')
        second = self.client.get('/')
        self.assertEqual(first.content, second.content)
        download.assert_called_once()
        self.assertLess(len(first.content), 100000)
        self.assertContains(first, 'js/plotly.min.js')
        self.client.get('/?period=1mo')
        self.assertEqual(download.call_count, 2)

    @patch('app.utils.yf.download')
    def test_prices_refresh_after_expiration(self, download):
        download.return_value = self.prices
        fetch_gold_data()
        with patch('django.core.cache.backends.locmem.time.time', return_value=time.time() + 61):
            fetch_gold_data()
        self.assertEqual(download.call_count, 2)


class NewsPerformanceTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)

    @patch('app.utils.requests.get')
    def test_feeds_download_once_and_share_titles(self, get):
        from .utils import get_all_news_titles
        get.return_value = Mock(content=b'<rss><channel><item><title>Gold news</title><description>Market update</description><link>https://example.com</link></item></channel></rss>')
        first = get_all_news_titles(('https://example.com/a', 'https://example.com/b'))
        second = get_all_news_titles(('https://example.com/a', 'https://example.com/b'))
        self.assertEqual(first, second)
        self.assertEqual(first[1], ['Gold news', 'Gold news'])
        self.assertEqual(get.call_count, 2)
        self.assertTrue(all(call.kwargs['timeout'] == (3, 5) for call in get.call_args_list))

    @patch('app.utils.requests.get', side_effect=requests.Timeout())
    def test_unavailable_feed_has_short_retry_cooldown(self, get):
        from .utils import get_news_titles_and_urls_from_rss
        with self.assertLogs('app.utils', level='WARNING'):
            self.assertEqual(get_news_titles_and_urls_from_rss('https://example.com'), [])
        self.assertEqual(get_news_titles_and_urls_from_rss('https://example.com'), [])
        get.assert_called_once()
        with patch('django.core.cache.backends.locmem.time.time', return_value=time.time() + 16):
            with self.assertLogs('app.utils', level='WARNING'):
                get_news_titles_and_urls_from_rss('https://example.com')
        self.assertEqual(get.call_count, 2)

    @patch('app.views.load_technical_prediction_resources')
    @patch('app.views.future_trend', return_value=('up', 'down', 'up'))
    @patch('app.views.load_model_and_vectorizer')
    @patch('app.views.get_all_news_titles')
    @patch('app.views.predict_news_categories_df')
    def test_news_and_predictions_share_one_classification(self, classify, feeds, resources, trends, technical):
        feeds.return_value = ([{'title': 'Gold', 'description': 'News', 'link': 'https://example.com'}], ['Gold'])
        resources.return_value = (Mock(), Mock())
        technical.return_value = (Mock(), Mock(), (Mock(), Mock(), Mock()))
        classify.return_value = pd.DataFrame({'Title': ['Gold'], 'Predicted_Category': [0]})
        self.assertContains(self.client.get('/news/'), 'News')
        self.assertEqual(self.client.get('/predict/').status_code, 200)
        self.assertEqual(self.client.get('/news/').status_code, 200)
        feeds.assert_called_once()
        classify.assert_called_once()

    @patch('app.views.get_all_news_titles', return_value=([], []))
    @patch('app.views.load_model_and_vectorizer')
    def test_empty_news_skips_classification(self, resources, feeds):
        self.assertContains(self.client.get('/news/'), 'No market news available')
        resources.assert_not_called()
