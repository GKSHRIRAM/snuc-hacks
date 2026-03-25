# MarketLens BI Engine

![MarketLens Dashboard](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-00a393)

**MarketLens** is a multi-source market intelligence pipeline that automatically discovers competitors, analyzes their historical pricing and features, gauges community sentiment, and generates deep market intelligence using LLMs.

## 🌟 Features

- **Automated Discovery**: Enter a startup description, and the engine automatically discovers 5 real competitors in the industry.
- **Deep Wayback Archiving & History**: Scrapes both live pages and 12-month historical snapshots from the Wayback Machine to detect pricing changes and feature alterations.
- **Sentiment & Review Sentinel**: Leverages Reddit analysis and live Trustpilot scraping to score positive/negative community sentiment.
- **AI-Powered Synthesis**: Feeds extracted data into Groq or local Ollama models to generate JSON-structured competitor analysis, pricing vacuums, and market positioning matrices.
- **Rich Dashboard**: A beautiful, premium vanilla JS & Chart.js frontend requiring zero build steps. Includes pricing comparisons, component arrays, and market positioning charts.

## 🌟 Key Advantages & Differentiators

### 1. Solving the Timeseries "Cold Start" Problem
Most competitive intelligence tools require you to set them up and wait months to aggregate historical trend data. **MarketLens eliminates the cold start problem.** By piping competitor URLs through the Wayback Machine's CDX API, we instantly fetch 12-month historical snapshots on *day zero*. You get immediate visibility into how competitors have changed their pricing, tier structures, and headline features over the past year.

### 2. Domain-Level Context, Not Just Company Scraping
We don't just extract data from the top 5 competitors in isolation. MarketLens synthesizes broad domain intelligence—identifying prevailing pricing vacuums across the entire market, overused industry messaging you should avoid, and macro-level feature gaps. This provides actionable "entry opportunities" that examining a single competitor's pricing page could never reveal.

### 3. Review Sentinel: Qualitative Meets Quantitative 
Pricing trend charts aren't enough. We actively correlate quantitative tracking with qualitative community sentiment by scraping live Trustpilot reviews and summarizing Reddit narratives. This helps you understand if a competitor's recent price hike resulted in user backlash or acceptance.

## 🏗️ Architecture

- **`main.py`**: The central orchestrator running a 6-phase async FastAPI pipeline.
- **`tools/`**: Contains specialized scrapers and utility agents (`searxng.py`, `wayback.py`, `firecrawl_extractor.py`, `ddg_chat.py`, `llm_client.py`).
- **`insight_engine.py` & `differ.py`**: Calculates the delta between historical and live snapshots, synthesizing market insights and generating programmatic summaries.
- **`normaliser.py`**: Standardizes data extracted by LLMs into Pydantic-validated structures.
- **`frontend/index.html`**: A static single-page application built with HTML, robust CSS mapping, and local JS. 

## 🚀 Getting Started

Read the **[Setup Guide (setup.md)](setup.md)** for instructions on installing dependencies, setting up your `.env` file, and running the server.

### Quick Start
1. `pip install -r requirements.txt`
2. Add your `.env` configuration (Groq & Firecrawl API keys).
3. `python3 main.py`
4. Open `frontend/index.html` in Chrome/Safari.

## 📊 Phase Execution Flow
1. **Understanding Agent**: Identifies target industry and determines top 5 competitors.
2. **Discovery Agent**: Uses SearxNG/DuckDuckGo to locate actual pricing and feature URLs.
3. **Extraction Engine**: Pulls Live data (Firecrawl/BeautifulSoup), Historical data (Wayback CDX API), and Sentiment data (Trustpilot/Reddit).
4. **Archiver**: Asynchronously pushes the live competitor pages to the Wayback machine for future historical queries.
5. **Normalization & Synthesis**: Evaluates pricing deltas and normalizes the output using large language models into strict JSON.
6. **Insight Generation**: Generates actionable differentiation angles and opportunity gaps. 
