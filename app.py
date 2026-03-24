import streamlit as st
import joblib

# 1. ตั้งค่าหน้าเว็บให้เป็นธีมมืดและไอคอนโรงหนัง
st.set_page_config(page_title="Cinema Sentiment AI", page_icon="🍿", layout="centered")

# 2. ใส่ CSS เพื่อตกแต่งธีมโรงหนัง (Cinema Styling)
st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #e50914; /* สีแดง Netflix/Cinema */
        color: white;
        border-radius: 20px;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        background-color: #2b2b2b;
        color: #f1f1f1;
        border: 1px solid #e50914;
    }
    h1 {
        color: #ffd700; /* สีทอง */
        text-align: center;
        text-shadow: 2px 2px #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. โหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load('imdb_sentiment_model.pkl')

model = load_model()

# 4. ส่วนแสดงผล (UI)
st.title("🍿 MOVIE CRITIC AI 🎬")
st.write("---")
st.markdown("<h3 style='text-align: center; color: #ffd700;'>โรงภาพยนตร์วิเคราะห์อารมณ์</h3>", unsafe_allow_html=True)
st.write("ยินดีต้อนรับเข้าสู่โรงหนัง! ลองพิมพ์รีวิวของคุณลงบนตั๋วด้านล่าง แล้วให้ AI ของเราประเมินคะแนนให้ครับ")

# ช่องรับข้อมูล
user_input = st.text_area("🎟️ Your Movie Review (English only):", height=150, placeholder="Write your review here...")

if st.button("📽️ START ANALYSIS"):
    if user_input.strip() == "":
        st.warning("⚠️ โปรดระบุรีวิวหนังก่อน")
    else:
        # วิเคราะห์ผล
        probabilities = model.predict_proba([user_input])[0]
        sentiment_score = probabilities[1] * 10 
        
        st.markdown("---")
        
        # แสดงผลคะแนนแบบโรงหนัง
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="IMDB Score (AI Prediction)", value=f"{sentiment_score:.1f} / 10")
        with col2:
            stars = int(sentiment_score / 2)
            if stars < 1 and sentiment_score > 0.3: stars = 1
            st.write("Rating:")
            st.subheader(f"{'⭐' * stars}{'☆' * (5-stars)}")

        # แถบแสดงอารมณ์
        if sentiment_score >= 7.5:
            st.success("🎉 **Blockbuster!** นี่คือรีวิวระดับ 5 ดาว หนังเรื่องนี้ต้องห้ามพลาด")
        elif sentiment_score >= 4.5:
            st.warning("🍿 **Average Joe.** เป็นหนังที่ดูได้เพลินๆ แต่ยังมีจุดให้ติอยู่บ้าง")
        else:
            st.error("🍅 **Rotten Tomato!** รีวิวนี้บ่นยับ AI พบว่าผู้ชมไม่ประทับใจอย่างแรง")

st.write("---")
st.caption("Developed with ❤️ for Data Science Project | 2026 Cinema AI Version")
