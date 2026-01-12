import streamlit as st
import pandas as pd
import joblib
from youtube_comment_downloader import YoutubeCommentDownloader
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from preprocess import preprocess_comment
import plotly.express as px
import plotly.graph_objects as go


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

#CSS
st.markdown("""
<style>  
            
[data-testid="stSubheader"] {
    color:blue !important;
}
                                  
.stButton > button {
    background-color: #ff0000;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #cc0000;
                       
}
</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("lgb_model.pkl")

model = load_model()

# --------------------------------------------------
# Fetch YouTube Comments (NO API KEY)
# --------------------------------------------------
def fetch_comments(video_url, limit=1000):
    downloader = YoutubeCommentDownloader()
    comments = []

    try:
        for c in downloader.get_comments_from_url(video_url, sort_by=0):
            text = c.get("text")
            if text:
                comments.append(text)
            if len(comments) >= limit:
                break
    except Exception as e:
        st.error(f"Error fetching comments: {e}")

    return comments

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("<h1 style='color: red;'>YouTube Comment Analyzer</h1>", unsafe_allow_html=True)
st.write("Analyze **Positive / Neutral / Negative** sentiment using Machine Learning")

url = st.text_input("🔗 Enter YouTube Video URL")
max_comments = st.slider("Number of comments", 100, 2000, 500, step=100)

analyze = st.button(" Analyze Comments")

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if analyze:
    if not url:
        st.warning("Please enter a YouTube URL")
        st.stop()

    with st.spinner("Fetching comments..."):
        comments = fetch_comments(url, max_comments)

    if len(comments) == 0:
        st.error("No comments fetched.")
        st.stop()

    st.success(f"Fetched {len(comments)} comments")

    df = pd.DataFrame({"comment": comments})

    with st.spinner("Running sentiment analysis..."):
        df["clean_comment"] = df["comment"].apply(preprocess_comment)
        df["sentiment"] = model.predict(df["clean_comment"])


    sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    df["sentiment_label"] = df["sentiment"].map(sentiment_map)

    # --------------------------------------------------
    # TOP: Sentiment Overview
    # --------------------------------------------------
    st.markdown("<h2 style='color: #23aaf2;'>Sentiment Overview</h2>", unsafe_allow_html=True)

    counts_pct = df["sentiment_label"].value_counts(normalize=True) * 100
    avg_sentiment = df["sentiment"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🟢 Positive (%)", f"{counts_pct.get('Positive', 0):.2f}")
    col2.metric("⚪ Neutral (%)", f"{counts_pct.get('Neutral', 0):.2f}")
    col3.metric("🔴 Negative (%)", f"{counts_pct.get('Negative', 0):.2f}")
    col4.metric("📊 Avg Sentiment Score", f"{avg_sentiment:.2f}")


    # --------------------------------------------------
    # MIDDLE
    # --------------------------------------------------
    st.divider()

    st.markdown("<h2 style='color: #23aaf2;'>Sentiment Trend</h2>", unsafe_allow_html=True)

    df["comment_index"] = range(len(df))

    fig_trend = px.line(
        df,
        x="comment_index",
        y="sentiment",
        labels={"comment_index": "Comment Index", "sentiment": "Sentiment Score"},
        template="plotly_white"
    )

    fig_trend.update_traces(
    line=dict(color= '#3DD56D', width=2) 
    )

    fig_trend.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig_trend, use_container_width=True)


    from collections import Counter

    words = " ".join(df["clean_comment"]).split()
    top_words = Counter(words).most_common(15)
    top_words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

    st.markdown("<h2 style='color: #23aaf2;'>Top Keywords</h2>", unsafe_allow_html=True)

    fig_keywords = px.bar(
        top_words_df,
        x="Word",
        y="Frequency",
        template="plotly_white"
    )

    fig_keywords.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig_keywords, use_container_width=True)

    st.markdown("<h2 style='color: #23aaf2;'>Word Cloud</h2>", unsafe_allow_html=True)

    text_for_wc = " ".join(df["clean_comment"])

    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color="white",
        max_words=200,
        collocations=False
    ).generate(text_for_wc)

    fig_wc, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig_wc)


    # --------------------------------------------------
    # BOTTOM
    # --------------------------------------------------
    st.divider()

    st.markdown("<h2 style='color: #23aaf2;'>Most Extreme Comments</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 Top Positive")
        for c in df[df["sentiment"] == 1]["comment"].head(5):
            st.write("•", c)

    with col2:
        st.markdown("### 🔴 Top Negative")
        for c in df[df["sentiment"] == -1]["comment"].head(5):
            st.write("•", c)


    st.markdown("<h2 style='color: #23aaf2;'>Sentiment Vs Comment Length</h2>", unsafe_allow_html=True)

    df["comment_length"] = df["comment"].str.len()

    fig_box = px.box(
    df,
    x="sentiment_label",
    y="comment_length",
    color="sentiment_label",
    template="plotly_white",
    labels={
        "sentiment_label": "Sentiment",
        "comment_length": "Comment Length"
    }
    )

    fig_box.update_traces(marker_opacity=0.6)
    fig_box.update_layout(
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig_box, use_container_width=True)
