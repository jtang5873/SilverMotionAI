import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

# ======================================================
# Init OpenAI
# ======================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ======================================================
# Utility: Convert uploaded image to base64
# ======================================================
def img_to_base64(uploaded_file):
    img = Image.open(uploaded_file)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ======================================================
# GPT-5.1 Vision Diagnosis (Western + TCM)
# ======================================================
def diagnose_with_vision(image_file, symptoms, lang):
    base64_img = img_to_base64(image_file)

    prompt = f"""
You are a senior orthopedic specialist and senior TCM physician.

Analyze the joint image and symptoms:
Symptoms: {symptoms}

Return a dual medical report:

===== Western Medicine =====
1. Likely diagnosis  
2. Visible structural issues in the image  
3. Severity (mild / moderate / severe)  
4. Home care treatment  
5. When to visit a doctor  

===== Traditional Chinese Medicine =====
1. TCM pattern (证型)  
2. Related meridians (经络)  
3. Imbalance explanation (气血/寒湿/肝肾亏虚)  
4. Acupoints (穴位)  
5. Gentle herbal suggestions (elder-safe)  

Return in: {lang}.
"""

    response = client.responses.create(
        model="gpt-5.1-vision",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_image", "image": base64_img}
                ]
            }
        ]
    )

    # NEW: Correct extraction for Responses API
    try:
        output = response.output[0].content[0].text
    except Exception:
        output = "⚠️ Error: Could not extract output_text from GPT-5.1 Vision."

    return output


# ======================================================
# Text Diagnosis (GPT-5.1)
# ======================================================
def diagnose_from_text(symptoms, lang):
    prompt = f"""
You are an orthopedic doctor + TCM doctor.
Patient symptoms: {symptoms}

Provide:

===== Western Medicine =====
• Possible diagnosis  
• Why these symptoms occur  
• Recommended exercises  
• Home care instructions  

===== Traditional Chinese Medicine =====
• TCM pattern  
• Acupoints  
• Daily lifestyle  
• Food therapy  

Language: {lang}
"""

    resp = client.responses.create(model="gpt-5.1", input=prompt)
    return resp.output_text


# ======================================================
# Elder Speaker (summary + TTS)
# ======================================================
def elder_summarize(text, lang):
    if lang.startswith("中文"):
        p = f"将以下内容总结成 3 句简单易懂的老人语言：\n{text}"
    else:
        p = f"Rewrite into 3 sentences for seniors:\n{text}"

    summary_resp = client.responses.create(model="gpt-5.1", input=p)
    summary = summary_resp.output_text

    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=summary,
        format="mp3"
    )

    return summary, audio.read()


# ======================================================
# TCM Remedies (Standalone)
# ======================================================
def generate_tcm(lang):
    if lang.startswith("中文"):
        lp = "用中文回答。"
    else:
        lp = "Respond in English."

    p = f"""
Provide Traditional Chinese Medicine advice for joint pain:
- Patterns (寒湿 / 气滞血瘀 / 肝肾亏虚)
- Meridians & acupoints
- Food therapy
- Lifestyle
- Elder safety notes

{lp}
"""
    resp = client.responses.create(model="gpt-5.1", input=p)
    return resp.output_text


# ======================================================
# Pain Score AI
# ======================================================
def estimate_pain_score(symptoms, lang):
    p = f"""
Based on symptoms:
{symptoms}

Predict:
• Pain score (0–10)
• Mobility impact
• Red flags
• Elder-friendly advice
Language: {lang}
"""
    resp = client.responses.create(model="gpt-5.1", input=p)
    return resp.output_text


# ======================================================
# Basic Motion Tracking (GPT-4o Vision)
# ======================================================
def analyze_motion(image_file, lang):
    base64_img = img_to_base64(image_file)

    p = f"""
You are a physiotherapy expert.
Analyze the image for:

- Joint alignment  
- Knee valgus/varus  
- Posture issues  
- Movement safety  
- Rehab corrections  

Language: {lang}
"""

    r = client.responses.create(
        model="gpt-4.1-vision",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": p},
                    {"type": "input_image", "image": base64_img}
                ]
            }
        ]
    )
    return r.output[0].content[0].text


# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="SilverMotionAI", layout="wide")

st.title("💙 SilverMotionAI — Medical • TCM • Vision AI • Rehab • Motion Tracking")

out_lang = st.sidebar.selectbox("🌍 Output Language", ["English", "中文（简体）"])

tabs = st.tabs([
    "🏠 Home",
    "📸 Image Diagnosis",
    "📝 Text Diagnosis",
    "🔊 Elder Speaker",
    "🌿 TCM Remedies",
    "❤️ Pain Score",
    "🏃 Motion Tracking"
])

# ---------------- Home ----------------
with tabs[0]:
    st.subheader("AI-powered Joint Health Assistant")
    st.write("Choose a function above to begin.")


# ---------------- Image Diagnosis ----------------
with tabs[1]:
    st.header("📸 AI Image Diagnosis — GPT-5.1 Vision")

    img = st.file_uploader("Upload joint image", type=["png", "jpg", "jpeg"])
    sx = st.text_area("Describe symptoms", "")

    if st.button("🔍 Run Diagnosis"):
        if img:
            with st.spinner("Analyzing image..."):
                result = diagnose_with_vision(img, sx, out_lang)
            st.success("Diagnosis Ready ✓")
            st.markdown(result)
        else:
            st.warning("Please upload an image first.")


# ---------------- Text Diagnosis ----------------
with tabs[2]:
    st.header("📝 Text Diagnosis")
    ts = st.text_area("Enter symptoms")
    if st.button("Run Text Diagnosis"):
        st.markdown(diagnose_from_text(ts, out_lang))


# ---------------- Elder Speaker ----------------
with tabs[3]:
    st.header("🔊 Elder Speaker Mode")
    long_text = st.text_area("Paste diagnosis text here")
    if st.button("Generate Elder Summary"):
        s, audio = elder_summarize(long_text, out_lang)
        st.write(s)
        st.audio(audio, format="audio/mp3")


# ---------------- TCM Remedies ----------------
with tabs[4]:
    st.header("🌿 TCM Joint Pain Remedies")
    if st.button("Generate TCM Advice"):
        st.markdown(generate_tcm(out_lang))


# ---------------- Pain Score ----------------
with tabs[5]:
    st.header("❤️ AI Pain Score Estimation")
    psx = st.text_area("Describe pain level / mobility issue")
    if st.button("Estimate Pain Score"):
        st.markdown(estimate_pain_score(psx, out_lang))


# ---------------- Motion Tracking ----------------
with tabs[6]:
    st.header("🏃 Motion Tracking — GPT-4o Vision")
    motion_img = st.file_uploader("Upload posture/movement image", type=["jpg","jpeg","png"])
    if st.button("Analyze Motion"):
        if motion_img:
            with st.spinner("Analyzing posture..."):
                m = analyze_motion(motion_img, out_lang)
            st.markdown(m)
        else:
            st.warning("Upload an image first.")
