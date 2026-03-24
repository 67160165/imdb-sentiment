import streamlit as st
import joblib

# 1. โหลดโมเดล AI ที่เราเทรนไว้
# (ฟังก์ชัน st.cache_resource จะช่วยให้โหลดโมเดลแค่ครั้งเดียวตอนเปิดเว็บ เว็บจะได้ไม่ช้า)
@st.cache_resource
def load_model():
    model = joblib.load('imdb_sentiment_model.pkl')
    return model

model = load_model()

# 2. ออกแบบหน้าเว็บ (UI)
st.title("🎬 ระบบวิเคราะห์อารมณ์รีวิวภาพยนตร์")
st.markdown("**IMDB Sentiment Analysis Project**")
st.write("โมเดล Machine Learning นี้ สร้างขึ้นเพื่อแยกแยะว่ารีวิวภาพยนตร์ที่คุณพิมพ์ เป็นการชื่นชม (Positive) หรือวิจารณ์ในแง่ลบ (Negative)")

# 3. สร้างช่องกรอกข้อความ (Input Validation ตามเกณฑ์อาจารย์)
user_input = st.text_area("✍️ พิมพ์รีวิวภาพยนตร์ภาษาอังกฤษของคุณที่นี่:", height=150)

# 4. สร้างปุ่มกดสำหรับประมวลผล
if st.button("🔍 วิเคราะห์รีวิว"):
    if user_input.strip() == "":
        st.warning("⚠️ กรุณาพิมพ์ข้อความรีวิวก่อนกดปุ่มวิเคราะห์ครับ")
    else:
        # 1. ให้ AI ทำนายผล
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        
        # 2. คำนวณคะแนน (แปลงความมั่นใจเป็นคะแนน 1-10)
        # probabilities[1] คือความมั่นใจฝั่ง Positive
        sentiment_score = probabilities[1] * 10 

        st.markdown("---")
        st.subheader("📊 ผลการวิเคราะห์จาก AI")

        # 3. แสดง Metric คะแนนตัวเลข
        st.metric(label="คะแนนความประทับใจ (Sentiment Score)", value=f"{sentiment_score:.1f} / 10")

        # 4. แสดงระดับดาว (Star Rating)
        stars = int(sentiment_score / 2) # 10 คะแนน หาร 2 = 5 ดาว
        if stars < 1 and sentiment_score > 0.5: stars = 1 # ปัดขึ้นให้มีอย่างน้อย 1 ดาวถ้าไม่แย่เกินไป
        st.write(f"ระดับความชอบ: {'⭐' * stars}{'☆' * (5-stars)}")

        # 5. แสดงสถานะและคำอธิบาย
        if sentiment_score >= 7.5:
            st.success(f"🌟 **ผลลัพธ์: เชิงบวกมาก (Excellent)**")
            st.write("AI มั่นใจว่านี่คือรีวิวที่ประทับใจสุดๆ")
        elif sentiment_score >= 4.5:
            st.warning(f"😐 **ผลลัพธ์: กลางๆ (Neutral/Fair)**")
            st.write("AI มองว่ารีวิวนี้มีทั้งส่วนดีและส่วนที่เฉยๆ ปนกัน")
        else:
            st.error(f"💔 **ผลลัพธ์: เชิงลบ (Negative)**")
            st.write("AI ตรวจพบความไม่พึงพอใจในเนื้อหารีวิวนี้")
