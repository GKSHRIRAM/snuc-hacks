# Domain Intelligence Collector

A lightweight pipeline to collect intelligence on a domain name, gathering:
1. Reviews (Reddit, G2, Trustpilot, Google Play)
2. Web crawl data (Scrapy, Playwright)
3. Historical snapshots (Wayback Machine CDX)
4. News & mentions (GDELT, CommonCrawl, Google News)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install
```

2. Run the application:
```bash
python main.py
```

3. Open `http://localhost:5000` to access the web interface.
