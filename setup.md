# MarketLens BI Engine setup guide

Follow these steps to deploy and run the MarketLens Competitive Intelligence engine.

## 1. Prerequisites
- Python 3.10+
- An API key for [Groq](https://groq.com/) (Free inference)
- An API key for [Firecrawl](https://firecrawl.dev/) (For live web scraping)

*(Optional)* If you wish to use local LLM inference instead of Groq, you will need **Ollama** running locally or on your local network.

## 2. Clone and Setup Environment

Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

## 3. Configuration

Create a `.env` file in the root directory:
```bash
touch .env
```

Add the following inside your `.env` file:
```env
# Primary LLM Provider: 'groq' or 'ollama'
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# If using Ollama (local LLM), uncomment these:
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_MODEL=llama3.1

# Scraper API
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

## 4. Run the Engine

Start the FastAPI application:
```bash
python3 main.py
```
*The API will start on `http://localhost:8000`.*

## 5. View the Dashboard

Open `frontend/index.html` directly in your web browser. 
There is no build process required. It will instantly connect to your local backend and allow you to run market analysis interactively.
