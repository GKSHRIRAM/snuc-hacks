import streamlit as st
import json
from engine.review_engine import ReviewEngine

# Initialize only the Review Engine
ge = ReviewEngine()

st.set_page_config(page_title="Sentilyze: Market Intelligence", layout="wide")

st.title("🔍 Sentilyze: Domain Intelligence")
st.markdown("---")

domain = st.text_input("Enter Company Domain (e.g., nvidia.com or apple.com)")

if st.button("Generate Intelligence"):
    if domain:
        with st.spinner(f'Scraping live data for {domain}...'):
            # Fetch data from our custom scraper
            score, reviews = ge.get_review_data(domain)
            
            # --- UI Display ---
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Trustpilot Sentiment Score", f"{score:.2f}")
                # Color coding for the judge's convenience
                if score > 0.2:
                    st.success("Positive Market Sentiment")
                elif score < -0.2:
                    st.error("Negative Market Sentiment")
                else:
                    st.warning("Neutral/Mixed Market Sentiment")

            with col2:
                if reviews:
                    st.success(f"Successfully extracted {len(reviews)} reviews!")
                    
                    # Prepare the JSON structure
                    report = {
                        "target_domain": domain,
                        "sentiment_score": score,
                        "data_source": "Trustpilot Scraper (BeautifulSoup)",
                        "reviews": reviews
                    }
                    json_string = json.dumps(report, indent=4)
                    
                    # The Download Button
                    st.download_button(
                        label="📥 Download JSON Sentiment Report",
                        data=json_string,
                        file_name=f"{domain}_sentiment.json",
                        mime="application/json"
                    )
                else:
                    st.warning("Score retrieved, but detailed review text was blocked by the host.")

            # Preview Section
            if reviews:
                with st.expander("View Scraped Data Preview"):
                    st.table(reviews[:5]) # Shows first 5 reviews in a clean table
    else:
        st.error("Please enter a valid domain.")