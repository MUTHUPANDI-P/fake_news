import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

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

def get_true_info(fake_text):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": f"""The following statement is FAKE. 
Please give the correct or factual version of it in one or two sentences.

Fake statement:
{fake_text}"""}]}
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        correction = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        return correction
    except Exception:
        return "No correction available."


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


def clean_text(text):
    text = re.sub(r"\b\d{1,2}\s*(hours|minutes|ago)\b", "", text)
    text = re.sub(r"(share|save|click here|more details|read more)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

trusted_sources = {
    "thehindu.com": "The Hindu",
    "timesofindia.com": "Times of India",
    "hindustantimes.com": "Hindustan Times",
    "ndtv.com": "NDTV",
    "bbc.com": "BBC News",
    "cnn.com": "CNN",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "aljazeera.com": "Al Jazeera",
    "theguardian.com": "The Guardian",
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "bloomberg.com": "Bloomberg",
    ".gov.in": "Indian Government Website",
    ".gov": "Government Website",
    "who.int": "World Health Organization",
    "nasa.gov": "NASA",
}

def get_source_name(url):
    url_lower = url.lower()
    for domain, name in trusted_sources.items():
        if domain in url_lower:
            return name
    return None


def final_decision(text, url=""):
    text = clean_text(text)
    if url:
        source_name = get_source_name(url)
        if source_name:
            return "REAL", f"This article is from a trusted and reputable source: **{source_name}**."
    return query_api(text)


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Custom CSS Styling
st.markdown(
    """
    <style>
        .title {
            font-size: 38px;
            font-weight: 800;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 35px;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
            transform: scale(1.03);
        }
        .verdict {
            font-size: 28px;
            font-weight: 700;
            text-align: center;
            margin-top: 25px;
        }
        .real { color: #27AE60; }
        .fake { color: #C0392B; }
        .unsure { color: #E67E22; }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center; gap:10px;">
        <img src="https://cdn-icons-png.flaticon.com/128/18788/18788954.png" width="50">
        <div class="title" style="font-size:40px; font-weight:bold;">Fake News Detection</div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="subtitle">Analyze news content using trusted sources, language models, and fact correction ✨</div>', unsafe_allow_html=True)


col1, col2 = st.columns([2, 1])

with col1:
    input_type = st.radio("Choose Input Type", ["Text", "URL"])
    user_input = ""
    page_url = ""

    if input_type == "Text":
        user_input = st.text_area("✍️ Enter news text here", height=200, placeholder="Paste or type the news content...")
    elif input_type == "URL":
        page_url = st.text_input("🔗 Enter news article URL", placeholder="https://example.com/news-article")
        if page_url:
            scraped = scrape_url(page_url)
            if scraped:
                st.text_area("📝 Extracted Article", scraped, height=300)
                user_input = scraped
            else:
                st.warning("⚠ Could not scrape the URL.")

    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

with col2:
    st.image("https://cdn-icons-gif.flaticon.com/19012/19012923.gif", width=220, caption="AI News Checker")


if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter valid text or URL.")
    else:
        try:
            result, explanation = final_decision(user_input, page_url)
            if result == "REAL":
                st.markdown('<div class="verdict real">🟢 REAL NEWS</div>', unsafe_allow_html=True)
            elif result == "FAKE":
                st.markdown('<div class="verdict fake">🔴 FAKE NEWS</div>', unsafe_allow_html=True)
                st.markdown("### ✅ Correct Information:")
                correction = get_true_info(user_input)
                st.info(correction)
            elif "ERROR" in result:
                st.error(result)
            else:
                st.markdown('<div class="verdict unsure">⚠ UNSURE</div>', unsafe_allow_html=True)

            st.markdown("### 📝 Why this decision?")
            st.info(explanation)

            with st.expander("🔎 Debug: Show Extracted Text"):
                st.write(user_input)

        except Exception as e:
            st.error(f"⚠ Error during analysis: {e}")
