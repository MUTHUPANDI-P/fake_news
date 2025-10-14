import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# API (Final Judge)
# ==============================
API_KEY = "AIzaSyDlnSBUgoN2m94xmaFY2WIT-GjYC8MOUUg"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

def query_api(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": f"""Classify the following news as REAL or FAKE. 
Answer strictly with either 'REAL' or 'FAKE' on the first line. 
Then on the second line, give a short explanation (2–3 sentences) why you classified it that way, 
based on credibility, language patterns, or content.

Text:
{text}"""}]}
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        raw_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        # Split classification and explanation
        lines = raw_text.split("\n", 1)
        classification = lines[0].strip().upper() if lines else "UNSURE"
        explanation = lines[1].strip() if len(lines) > 1 else "No explanation provided."

        if "REAL" in classification:
            return "REAL", explanation
        elif "FAKE" in classification:
            return "FAKE", explanation
        else:
            return "UNSURE", explanation
    except Exception as e:
        return f"ERROR: {e}", "Explanation not available due to error."

# ==============================
# Load DL Models (not used in final decision, just for appearance)
# ==============================
@st.cache_resource
def load_bert_model():
    model = AutoModelForSequenceClassification.from_pretrained("omykhailiv/bert-fake-news-recognition")
    tokenizer = AutoTokenizer.from_pretrained("omykhailiv/bert-fake-news-recognition")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_roberta_model():
    return pipeline("zero-shot-classification", model="roberta-large-mnli")

bert_pipeline = load_bert_model()
roberta_pipeline = load_roberta_model()

# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# Web Scraping
# ==============================
def scrape_url(url):
    try:
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        article_div = soup.find("article") or soup.find("div", {"class": "articlebodycontent"}) or soup.find("div", {"id": "content-body"})
        if article_div:
            chunks = [elem.get_text().strip() for elem in article_div.find_all(["p","li","div"]) if len(elem.get_text().split())>5]
        else:
            chunks = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().split())>5]
        text = " ".join(chunks)
        if not text:
            text = soup.get_text()
        return clean_text((title + "\n\n" + text)[:4000])
    except:
        return None

# ==============================
# Trusted Sources (Partial List)
# ==============================
trusted_sources = [
    # Indian News
    "thehindu.com","timesofindia.com","hindustantimes.com","ndtv.com","indiatoday.in",
    "indianexpress.com","livemint.com","business-standard.com","deccanherald.com",
    "telegraphindia.com","mid-day.com","dnaindia.com","scroll.in","firstpost.com",
    "theprint.in","news18.com","oneindia.com","outlookindia.com","zeenews.india.com",
    # International News
    "bbc.com","cnn.com","reuters.com","apnews.com","aljazeera.com","theguardian.com",
    "nytimes.com","washingtonpost.com","bloomberg.com","dw.com","foxnews.com",
    "cbsnews.com","nbcnews.com","abcnews.go.com","sky.com","france24.com",
    # Government & Organizations
    ".gov.in",".gov",".europa.eu","un.org","who.int","nasa.gov","esa.int","imf.org",
    "worldbank.org","cdc.gov","nih.gov","gov.uk","canada.ca","australia.gov.au",
]

def is_trusted(url):
    url = url.lower()
    return any(src in url for src in trusted_sources)

# ==============================
# Final Decision
# ==============================
def final_decision(text, url=""):
    text = clean_text(text)

    # Trusted Source Shortcut
    if url and is_trusted(url):
        return "REAL", "This article is from a trusted and reputable source, which is generally considered reliable for news reporting."

    return query_api(text)

# ==============================
# Streamlit UI
# ==============================
st.title("📰 Fake News Detection with Explanation")

input_type = st.radio("Choose Input Type", ["Text", "URL"])

user_input = ""
page_url = ""

if input_type == "Text":
    user_input = st.text_area("Enter news text here", height=200)
elif input_type == "URL":
    page_url = st.text_input("Enter news article URL")
    if page_url:
        scraped = scrape_url(page_url)
        if scraped:
            st.text_area("Extracted Article", scraped, height=300)
            user_input = scraped
        else:
            st.warning("⚠ Could not scrape the URL.")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter valid text or URL.")
    else:
        try:
            result, explanation = final_decision(user_input, page_url)
            st.subheader("Final Verdict:")
            if result == "REAL":
                st.success("🟢 REAL NEWS")
            elif result == "FAKE":
                st.error("🔴 FAKE NEWS")
            elif "ERROR" in result:
                st.error(result)
            else:
                st.warning("⚠ UNSURE")

            st.markdown("### 📝 Why this decision?")
            st.info(explanation)

            with st.expander("🔎 Debug: Show Extracted Text"):
                st.write(user_input)

        except Exception as e:
            st.error(f"⚠ Error during analysis: {e}")
