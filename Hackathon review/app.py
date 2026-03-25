import streamlit as st
from dotenv import load_dotenv
from engine import RedditEngine, ReviewEngine

load_dotenv() # Crucial: links your .env keys to the code

st.set_page_config(page_title="Domain Intel", page_icon="🔍")
st.title("🕵️ Domain Intelligence Collector")

target = st.text_input("Enter Company or Product Name (e.g., Postman)")

if st.button("Generate Intelligence Report"):
    if not target:
        st.warning("Please enter a target name.")
    else:
        with st.spinner(f"Scouring the web for {target}..."):
            # Initialize the brains
            red = RedditEngine()
            rev = ReviewEngine()
            
            # Fetch scores
            social_val = red.get_sentiment(target)
            market_val = rev.get_review_score(target)
            
            # Display metrics
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Reddit Sentiment", f"{social_val:.2f}")
            c2.metric("G2 Market Score", f"{market_val:.2f}")
            
            # Final Calculation
            total = (social_val + market_val) / 2
            st.info(f"Overall Market Confidence: {total:.2f}")
