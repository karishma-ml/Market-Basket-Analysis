import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# PAGE CONFIG
st.set_page_config(page_title="Market Basket", layout="centered")


# GLOBAL STYLES
st.markdown("""
<style>

/* Make app take full screen */
html, body, [data-testid="stAppViewContainer"], .stApp {
    height: 100%;
    min-height: 100vh;
    margin: 0;
    padding: 0;
}

/* Center Content */
main > div {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 40px;
}

/* Background Image Full Display */
.stApp {
    background-image: url("https://i.imgur.com/KHGVEdO.jpeg");
    background-size: cover;         
    background-repeat: no-repeat;      
    background-position: center center;
    background-attachment: fixed;      
    background-color: white;
}

/* Menu Color Theme */
div[data-testid="stHorizontalBlock"] {
    background-color: rgba(140, 80, 40, 0.7);
    padding: 8px;
    border-radius: 10px;
}

/* Metric cards */
.stMetric {
    background-color: rgba(255, 240, 220, 0.8);
    padding: 10px;
    border-radius: 10px;
}

/* Tabs */
.stTabs [role="tab"] {
    padding: 10px;
    font-size: 16px;
    color: black;
    background-color: #a05e34;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background-color: #7d4525;
    color: white;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    background-color: rgba(255, 255, 255, 0.85);
}

/* Simple card style for charts & results */
.card {
    background-color: rgba(255, 255, 255, 0.78);
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODELS
frequent_itemset = joblib.load("frequent_itemset.joblib")
rules = joblib.load("rules_joblib")
corpus = joblib.load("corpus.pkl")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("Groceries_dataset.csv")
    df["itemDescription"] = df["itemDescription"].str.strip()
    basket = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
    all_items = sorted(df["itemDescription"].unique())
    return df, basket, all_items

df, basket, all_items = load_data()

st.header("🛒 Market Basket Analysis")
st.success("Dataset Loaded Successfully")


# SESSION STATE
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("logged_in", False)

# TOP MENU
page = option_menu(
    None,
    ["Home", "Dataset", "Visuals", "Chatbot", "About"],
    icons=["house", "table", "bar-chart", "robot", "info-circle"],
    orientation="horizontal",
    styles={
        "container": {"padding": "5px"},
        "icon": {"color": "#F0D6A6", "font-size": "20px"},
        "nav-link": {"color": "black"},
        "nav-link-selected": {"background-color": "#7d4525", "color": "white"},
    },
)

# HOME PAGE
if page == "Home":
    st.title("Welcome 😊")
    st.markdown("""
    - Real grocery transactions  
    - Used for Market Basket Analysis  
    - Apriori Algorithm  
    - Frequent Itemsets & Rules  
    """)

# DATASET PAGE
elif page == "Dataset":
    st.title("📂 Dataset")

    tab1, tab2, tab3 = st.tabs(["Preview", "Information", "Summary"])

    with tab1:
        st.dataframe(df.head())

    with tab2:
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

    with tab3:
        st.write(df.describe())

# VISUALS PAGE
elif page == "Visuals":
    st.title("📊 Frequent Itemsets & Association Rules")

    min_sup = st.slider("Minimum Support", 0.0, 1.0, 0.1)
    min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.1)

    # Frequent Itemsets Graph 
    st.subheader("Frequent Itemsets")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    fi = frequent_itemset[frequent_itemset["support"] >= min_sup]

    if not fi.empty:
        top = fi.sort_values("support", ascending=False).head(10)
        x_vals = top["itemsets"].astype(str)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_alpha(0)          # transparent figure
        ax.set_facecolor("none")        # transparent axes

        sns.barplot(x=x_vals, y=top["support"], ax=ax , palette="plasma")
        plt.xticks(rotation=45, ha="right")
        ax.set_ylabel("Support")
        ax.set_title("Top 10 Frequent Itemsets")

        for p in ax.patches:
            h = p.get_height()
            ax.text(p.get_x() + p.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        st.pyplot(fig)
    else:
        st.warning("No itemsets found for selected support.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Association Rules Graph 
    st.subheader("Association Rules")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    fr = rules[
        (rules["support"] >= min_sup) &
        (rules["confidence"] >= min_conf)
    ]

    if not fr.empty:
        fr = fr.copy()
        fr["label"] = fr.apply(lambda r: f"{set(r['antecedents'])} → {set(r['consequents'])}", axis=1)

        # sort by confidence and take top 10
        fr = fr.sort_values("confidence", ascending=False).head(10)

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")

        sns.barplot(x="label", y="confidence", data=fr, ax=ax2 , palette="plasma")
        plt.xticks(rotation=45, ha="right")
        ax2.set_ylabel("Confidence")
        ax2.set_title("Association Rules (Support + Confidence Filter)")

        for p in ax2.patches:
            h = p.get_height()
            ax2.text(p.get_x() + p.get_width()/2, h, f"{h:.2f}",ha="center", va="bottom", fontsize=8)

        st.pyplot(fig2)
    else:
        st.warning("No rules found for selected thresholds.")

    st.markdown('</div>', unsafe_allow_html=True)


# CHATBOT PAGE
elif page == "Chatbot":
    st.title("🤖 Smart Assistant")

    def chatbot_response(user_text):
        text = user_text.lower()
        for q, a in corpus.items():
            words = q.lower().split()[:2]
            if all(w in text for w in words):
                return a
        return corpus.get("default", "Sorry, I don't understand.")

    user_input = st.text_input("Ask about Apriori, Support, Confidence...")

    if user_input:
        reply = chatbot_response(user_input)
        st.session_state.chat_history.append(("You: " + user_input, "Bot: " + reply))

    for u, b in st.session_state.chat_history:
        st.write(u)
        st.write(b)

    # LOGIN
    CREDS = {"director": "dir123", "karishma": "kar123", "nisha": "nis123"}

    if not st.session_state.logged_in:
        st.subheader("🔒 Login Required")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if CREDS.get(u) == p:
                st.session_state.logged_in = True
                st.success(f"Welcome, {u}")
            else:
                st.error("Invalid Credentials")

    # AFTER LOGIN
    if st.session_state.logged_in:
        st.subheader("📊 Authenticated Area")

        tab1, tab2 = st.tabs(["Itemsets & Rules", "Item Pair Checker"])

        # TAB 1: Itemsets & Rules with sliders
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            auth_min_sup = st.slider("Auth Minimum Support", 0.0, 1.0, 0.1)
            auth_min_conf = st.slider("Auth Minimum Confidence", 0.0, 1.0, 0.1)

            st.write(f"### ⭐ Frequent Itemsets (Support ≥ {auth_min_sup:.2f})")
            st.dataframe(frequent_itemset[frequent_itemset["support"] >= auth_min_sup])

            st.write(
                f"### 🔗 Association Rules (Support ≥ {auth_min_sup:.2f}, "
                f"Confidence ≥ {auth_min_conf:.2f})"
            )
            st.dataframe(
                rules[(rules["support"] >= auth_min_sup) &(rules["confidence"] >= auth_min_conf)])

            st.markdown('</div>', unsafe_allow_html=True)

        # TAB 2: Item Pair Checker
        with tab2:
            st.subheader("🔍 Item Pair Probability Checker")
            st.markdown('<div class="card">', unsafe_allow_html=True)

            i1 = st.selectbox("Select Item 1", all_items)
            i2 = st.selectbox("Select Item 2", all_items)

            if st.button("Find Relation"):
                out = rules[(rules["antecedents"] == frozenset([i1])) &(rules["consequents"] == frozenset([i2]))]

                if not out.empty:
                    st.write(f"**Support:** {float(out['support'].iloc[0]):.4f}")
                    st.write(f"**Confidence:** {float(out['confidence'].iloc[0]):.4f}")
                else:
                    st.error("No direct association found 🤷")

            st.markdown('</div>', unsafe_allow_html=True)


# ABOUT PAGE
elif page == "About":
    st.title("ℹ️ About This Application")

    st.markdown("""
    ### 🛒 Market Basket Analysis App  
    This application demonstrates **Association Rule Mining** using the Groceries dataset.

    ### 🔧 Technologies Used  
    - **Python** – Core programming language  
    - **Streamlit** – For building the interactive web application  
    - **HTML & CSS** – For layout, design, and UI enhancements  
    - **Apriori Algorithm** – For mining frequent itemsets  
    - **Pandas & NumPy** – Data manipulation  
    - **Matplotlib & Seaborn** – Data visualization  

    ### 👩‍💻 Developed By  
    **Karishma**

    ### ⭐ Objective  
    To analyze customer shopping patterns and identify items frequently bought together for business insights.
    """)
