import streamlit as st
from dotenv import load_dotenv
from engine.reddit_engine import RedditEngine
from engine.review_engine import ReviewEngine

load_dotenv() # Reads the .env file

st.set_page_config(page_title="Sentilyze", page_icon="📊")
st.title("Sentilyze: Market Sentiment Tool")

# User Input
domain = st.text_input("Enter a Company or Market Keyword", placeholder="e.g. OpenAI, Nvidia, SaaS")

if st.button("Analyze Now"):
    if not domain:
        st.warning("Please enter a keyword first.")
    else:
        with st.spinner('Gathering cross-platform data...'):
            re = RedditEngine()
            ge = ReviewEngine()
            
            # Run calculations
            red_score = re.get_sentiment(domain)
            rev_score = ge.get_g2_score(domain)
            
            # Display metrics
            st.subheader(f"Analysis for: {domain}")
            c1, c2 = st.columns(2)
            c1.metric("Reddit Pulse", f"{red_score:.2f}")
            c2.metric("Review Score (G2)", f"{rev_score:.2f}")
            
            # Final Sentiment Logic
            avg = (red_score + rev_score) / 2
            if avg > 0.1:
                st.success("Overall Sentiment: **POSITIVE** 🚀")
            elif avg < -0.1:
                st.error("Overall Sentiment: **NEGATIVE** 📉")
            else:
                st.info("Overall Sentiment: **NEUTRAL** ⚖️")