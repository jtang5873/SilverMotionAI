import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

# ==========================================================
# Init OpenAI
# ==========================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ==========================================================
# Utility: Convert uploaded image to raw bytes (NEW Vision API)
# ==========================================================
def read_image_bytes(uploaded_file):
    return uploaded_file.read()


# ==========================================================
# GPT-5.1 Vision Diagnosis (Western + TCM)
# ==========================================================
def diagnose_with_vision(image_file, symptoms, lang):
    img_bytes = read_image_bytes(image_file)

    prompt = f"""
You are a senior orthopedic specialist + senior TCM doctor.

Analyze the joint photo + symptoms and produce a dual medical report.

Symptoms: {symptoms}

===== Western Medicine =====
1. Probable diagnosis
2. Visible abnormalities
3. Severity level
4. Home-care recommendations
5. Red flags (when to see a doctor)

===== Traditional Chinese Medicine =====
1. TCM pattern diagnosis (证型)
2. Meridians involved (经络)
3. Imbalance mechanism (气血 / 寒湿 / 肝肾亏虚)
4. Acupoints to massage (穴位)
5. Elder-safe herbal suggestions

Language: {lang}
"""

    response = client.responses.create(
        model="gpt-5.1-vision",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_image", "image": img_bytes},
                ],
            }
        ],
    )

    return response.output_text


# ==========================================================
# GPT-5.1 Text Diagnosis
# ==========================================================
def diagnose_from_text(symptoms, lang):
    prompt = f"""
You are an orthopedic doctor + TCM expert.

Patient symptoms: {symptoms}

Provide:

===== Western Medicine =====
1. Possible diagnosis
2. Why symptoms happen
3. Recommended exercises
4. Home advice

===== TCM =====
1. Pattern diagnosis
2. Acupoints
3. Food therapy
4. Daily lifestyle

Language: {lang}
"""

    resp = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )
    return resp.output_text


# ==========================================================
# Elder Speaker (Summary + TTS)
# ==========================================================
def make_elder_summary(text, lang):
    if lang.startswith("中文"):
        p = f"请把以下医疗内容写成 3 句老人容易理解的简单说明：\n{text}"
    else:
        p = f"Rewrite the following into 3 simple senior-friendly sentences:\n{text}"

    summary_resp = client.responses.create(
        model="gpt-5.1",
        input=p
    )

    summary = summary_resp.output_text

    # TTS
    audio_resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        input=summary,
        voice="alloy",
        format="mp3"
    )

    audio_bytes = audio_resp.read()
    return summary, audio_bytes


# ==========================================================
# TCM Remedies
# ==========================================================
def generate_tcm_recommendation(lang):
    if lang.startswith("中文"):
        lp = "用简体中文回答。"
    else:
        lp = "Respond in English."

    prompt = f"""
Provide a Traditional Chinese Medicine joint-pain guideline.

Include:
- TCM patterns
- Meridians & acupoints
- Food therapy
- Daily routines
- Safety notes

{lp}
"""

    r = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )
    return r.output_text


# ==========================================================
# Pain Score
# ==========================================================
def estimate_pain_score(symptoms, lang):
    prompt = f"""
Estimate pain level (0-10) from symptoms:

Symptoms: {symptoms}

Return:
1. Pain score 0-10
2. Explanation
3. Suggested actions

Language: {lang}
"""
    r = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )
    return r.output_text


# ==========================================================
# Rehab Video Generator (description only)
# ==========================================================
def generate_rehab_routine(symptoms, lang):
    prompt = f"""
You are a physical therapist.

Generate a 3-step rehab routine for the symptoms:

Symptoms: {symptoms}

Return:
- Warm-up
- Main exercise
- Cool-down
- Safety notes

Language: {lang}
"""
    r = client.responses.create(model="gpt-5.1", input=prompt)
    return r.output_text


# ==========================================================
# AI Motion Tracking (Skeleton Extraction)
# ==========================================================
def analyze_motion_with_vision(image_file, lang):
    img_bytes = read_image_bytes(image_file)

    prompt = f"""
Analyze body posture from this image.

Return:
1. Skeleton points (key joints)
2. Posture issues
3. Risk assessment
4. Corrections
5. Senior-friendly explanation

Language: {lang}
"""

    resp = client.responses.create(
        model="gpt-5.1-vision",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_image", "image": img_bytes},
                ],
            }
        ],
    )
    return resp.output_text


# ==========================================================
# UI — Material Theme
# ==========================================================
def apply_ui_theme():
    st.markdown("""
    <style>
    .main { background-color: #F4F9FF !important; }
    h1, h2, h3 { color: #003C71 !important; font-weight: 700; }
    .stButton>button {
        background-color: #1E88E5 !important;
        color: white !important;
        border-radius: 8px !important;
        font-size: 18px !important;
        padding: 10px 22px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #D1E9FF !important;
        color: #003C71 !important;
    }
    </style>
    """, unsafe_allow_html=True)


apply_ui_theme()

# ==========================================================
# Main UI — Tabs
# ==========================================================
st.title("💙 SilverMotion AI — Joint Care Super App")

lang = st.sidebar.selectbox(
    "🌍 Output Language",
    ["English", "中文（简体）"]
)

tabs = st.tabs([
    "🏠 Home",
    "📸 Image Diagnosis",
    "✍️ Text Diagnosis",
    "🔊 Elder Speaker",
    "🌿 TCM Remedies",
    "❤️ Pain Score",
    "🎥 Rehab Generator",
    "🕺 Motion Tracking",
])

tab_home, tab_img, tab_text, tab_elder, tab_tcm, tab_pain, tab_rehab, tab_motion = tabs


# ==========================================================
# Home Tab
# ==========================================================
with tab_home:
    st.header("Welcome to SilverMotion AI")
    st.write("Medical • TCM • Vision AI • Rehab • Motion Tracking • Elder-Friendly")


# ==========================================================
# Image Diagnosis
# ==========================================================
with tab_img:
    st.header("📸 AI Image Diagnosis — GPT-5.1 Vision")

    img = st.file_uploader("Upload a joint photo", type=["jpg", "jpeg", "png"])
    symptoms = st.text_area("Describe symptoms")

    if st.button("🔍 Run Diagnosis"):
        if img:
            result = diagnose_with_vision(img, symptoms, lang)
            st.success("Diagnosis Completed")
            st.write(result)
        else:
            st.error("Please upload an image.")


# ==========================================================
# Text Diagnosis
# ==========================================================
with tab_text:
    st.header("✍️ Text Diagnosis")
    symptoms = st.text_area("Describe symptoms")
    if st.button("Run Text Diagnosis"):
        st.write(diagnose_from_text(symptoms, lang))


# ==========================================================
# Elder Speaker
# ==========================================================
with tab_elder:
    st.header("🔊 Elder Speaker")

    text = st.text_area("Paste medical text to simplify")
    if st.button("Generate Elder Summary"):
        summary, audio_bytes = make_elder_summary(text, lang)
        st.write(summary)
        st.audio(audio_bytes, format="audio/mp3")


# ==========================================================
# TCM Remedies
# ==========================================================
with tab_tcm:
    st.header("🌿 TCM Remedies")
    if st.button("Generate"):
        st.write(generate_tcm_recommendation(lang))


# ==========================================================
# Pain Score
# ==========================================================
with tab_pain:
    st.header("❤️ Pain Score")
    symptoms = st.text_input("Symptoms")
    if st.button("Estimate Pain"):
        st.write(estimate_pain_score(symptoms, lang))


# ==========================================================
# Rehab Generator
# ==========================================================
with tab_rehab:
    st.header("🎥 Rehab Video Generator (Text Guide)")
    symptoms = st.text_area("Rehab symptoms")
    if st.button("Generate Rehab Routine"):
        st.write(generate_rehab_routine(symptoms, lang))


# ==========================================================
# Motion Tracking
# ==========================================================
with tab_motion:
    st.header("🕺 AI Motion Tracking")
    img = st.file_uploader("Upload motion image", type=["jpg", "png"])
    if st.button("Analyze Motion"):
        if img:
            res = analyze_motion_with_vision(img, lang)
            st.write(res)
        else:
            st.error("Upload an image for motion tracking.")
