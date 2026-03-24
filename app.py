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
    # เช็กว่าผู้ใช้พิมพ์ข้อความมาหรือเปล่า (ถ้าว่างเปล่าให้แจ้งเตือน)
    if user_input.strip() == "":
        st.warning("⚠️ กรุณาพิมพ์ข้อความรีวิวก่อนกดปุ่มวิเคราะห์ครับ")
    else:
        # ให้ AI ทำนายผล
        prediction = model.predict([user_input])[0]
        # ขอค่าความมั่นใจ (Probability) เพื่อโชว์ตามเกณฑ์อาจารย์
        probabilities = model.predict_proba([user_input])[0]
        
        st.markdown("---")
        st.subheader("📊 ผลการวิเคราะห์:")
        
        # แสดงผลลัพธ์พร้อมสีสันให้ดูง่าย
        if prediction == 'positive':
            st.success("🌟 **รีวิวนี้เป็น: เชิงบวก (Positive)**")
            st.info(f"ความมั่นใจของโมเดล: {probabilities[1]:.2%}")
        else:
            st.error("💔 **รีวิวนี้เป็น: เชิงลบ (Negative)**")
            st.info(f"ความมั่นใจของโมเดล: {probabilities[0]:.2%}")