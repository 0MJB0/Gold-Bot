# Arden Nexus · Gold Bot

A Django dashboard for exploring gold futures prices, market news, and model-based directional signals. A minimalist interface brings price history and research tools into one workspace.

## Features

- **Market overview:** daily open, close, high, and low prices for Yahoo Finance's `GC=F` gold futures symbol.
- **Historical charts:** compare price series with 10-day, one-month, one-year, or custom date filters.
- **Session summaries:** view the latest session within the selected range and browse the price history table.
- **Predictions:** combine a saved technical model with news classification to display daily, weekly, and monthly directional signals.
- **Market news:** read headlines from CNBC and The Guardian with model-generated outlook labels.
- **Responsive interface:** neutral colors, generous spacing, accessible text labels, and SVG line icons.

This is a research application, not a trading bot: it does not place orders or connect to a brokerage account.

## Stack

| Area | Tools |
| --- | --- |
| Application | Python, Django, SQLite |
| Market data | yfinance, pandas |
| Charts | Plotly, served locally |
| News | requests, feedparser, Beautiful Soup |
| Models | scikit-learn, saved pickle models |
| Datasets | Excel files read with openpyxl |
| Interface | Django templates, CSS, inline SVG |

## Local setup

Run the commands from the project root, where `manage.py` is located. Python and Git must be installed. Python 3.14 is used in the current development environment; other versions have not been verified for this setup.

### 1. Clone the project

```bash
git clone https://github.com/0MJB0/Gold-Bot.git
cd Gold-Bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

The existing `requirements.txt` is a historical Conda package listing, **not a pip-compatible requirements file**. Install the application's direct dependencies explicitly:

```bash
python -m pip install --upgrade pip
python -m pip install Django django-heroku pandas scikit-learn yfinance plotly feedparser beautifulsoup4 requests openpyxl
```

These packages are not version-locked. A fresh installation may differ from the development environment. In particular, the bundled scikit-learn models were serialized in an older environment and can produce compatibility warnings or loading errors with different library versions. The historical package listing records scikit-learn 1.3.0; it is not a verified installation recipe for current Python versions.

### 4. Initialize the database and start Django

```bash
python manage.py migrate
python manage.py runserver
```

Open **http://127.0.0.1:8000/**.

Internet access is required to retrieve prices and RSS feeds. No API key is configured by this application. Keep the bundled files in `app/Data/` in place and start Django from the project root, since model and dataset paths are currently relative to that directory.

## Pages

| Path | Purpose |
| --- | --- |
| `/` | Price summary, historical chart, and session table |
| `/predict/` | Technical and news-based directional signals |
| `/news/` | Headlines, summaries, and classified outlooks |
| `/admin/` | Django administration; requires a superuser |

Optional administrator account:

```bash
python manage.py createsuperuser
```

The date filters use calendar days, so a 10-day range can contain fewer than 10 trading sessions. The custom end date follows the data provider's exclusive-end convention. Summary values belong to the latest session **in the selected range**, not necessarily today's session.

## Performance and refresh behavior

| Resource | Cache behavior |
| --- | --- |
| Price downloads | 60 seconds |
| Generated chart HTML | 60 seconds |
| RSS feeds | 300 seconds |
| Classified news shared by both research pages | 300 seconds |
| Empty cached results | 15-second retry cooldown |
| Loaded datasets, models, and prepared technical resources | Kept in memory for the server process lifetime |

The two RSS sources are fetched concurrently, once per source on a cache miss, with connection and read timeouts. The chart and table reuse the same downloaded price data. Opening the prediction page does not write an Excel output file.

Caches are populated on demand: the first request or a request after expiry can still wait for external services. There is no background refresh or live streaming. Feed and classification caches expire independently, so news refreshes can lag by more than one five-minute interval.

The default Django cache is local to each process. Restart the server after replacing models or datasets to clear the in-memory resources. Multiple server workers will maintain separate caches unless a shared backend is configured.

## Project structure

```text
Gold-Bot/
├── manage.py
├── core/
│   ├── settings.py          # Django configuration
│   └── urls.py              # Page routes
├── app/
│   ├── views.py             # Page data and shared news classification
│   ├── utils.py             # Price feeds, RSS parsing, charts, and models
│   ├── performance.py       # Time-limited result caching
│   ├── tests.py             # Rendering, caching, and failure handling tests
│   ├── Data/                # Bundled Excel datasets and saved models
│   ├── templates/           # Shared layout and page templates
│   └── static/
│       ├── css/dashboard.css
│       └── js/plotly.min.js
└── requirements.txt         # Historical Conda environment listing
```

The local SQLite database, virtual environment, Python bytecode, and root-level prediction output are excluded by `.gitignore`.

## Validation

```bash
python manage.py check
python manage.py test app
```

The tests cover price formatting, empty responses, download failures, cache reuse and expiry, RSS retry cooldowns, and shared news classification. External service responses are mocked; passing tests does not verify Yahoo Finance or RSS availability.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `No module named ...` | Activate the environment where dependencies were installed. Use `python -m pip` to install into that interpreter. |
| Price data is unavailable | Check connectivity and retry after the cache cooldown. Yahoo Finance may reject or rate-limit requests. |
| No market news | A source may be unavailable or return an empty feed. Check server logs for `News source unavailable`. |
| Model or dataset not found | Run from the project root and confirm that `app/Data/` contains the bundled files. |
| Pickle compatibility warning or error | Check the Python/scikit-learn environment used to create the model; replacing or retraining it may be necessary. |
| First page load is slower | Model initialization and uncached network requests happen on demand. Subsequent requests reuse cached resources. |
| Old styles remain visible | Hard-refresh the page with Ctrl+F5. |

## Research limitations

Prices describe gold futures in USD, not a local physical-gold retail quote. Technical signals use the bundled historical dataset rather than automatically updating it from the overview's Yahoo feed. Day, week, and month labels refer to the application's existing model logic; they are not guarantees of forecast accuracy. No validated accuracy or profitability claim is made.

For research and educational use only. Not investment advice.

## Deployment status

The checked-in configuration is intended for local development: it contains a development secret key, enables `DEBUG`, and allows all hosts. Before a public deployment, configure a private secret key, disable debug mode, restrict allowed hosts, and set up the database and static-file serving for the target environment. The repository includes legacy `django-heroku` configuration; it is not a complete production deployment setup.
