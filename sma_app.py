import streamlit as st
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64

# ==========================================================
# INIT OPENAI — MUST USE NEW SYNTAX (NO api_key ARG)
# ==========================================================
client = OpenAI()   # Uses Streamlit Secrets automatically


# ==========================================================
# Utility: Convert uploaded image → base64
# ==========================================================
def img_to_base64(uploaded_file):
    img = Image.open(uploaded_file)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ==========================================================
# GPT-5.1 Vision Diagnosis (Image + Symptoms)
# ==========================================================
def diagnose_with_vision(image_file, symptoms, lang):

    base64_img = img_to_base64(image_file)

    prompt_text = f"""
You are a senior orthopedic specialist + senior TCM doctor.

Symptoms: {symptoms}

Return a dual medical report:

===== Western Medicine =====
1. Likely diagnosis
2. Visible structural issues
3. Severity (mild/moderate/severe)
4. Home-care recommendations
5. When to seek medical care

===== TCM =====
1. Pattern diagnosis (证型)
2. Meridians involved
3. Imbalance explanation
4. Acupoints
5. Mild safe herbal suggestions

Language: {lang}
"""

    response = client.responses.create(
        model="gpt-5.1-vision",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_img}"
                    }
                ]
            }
        ]
    )

    return response.output_text


# ==========================================================
# GPT-5.1 Text-only Diagnosis
# ==========================================================
def diagnose_from_text(symptoms, lang):

    prompt = f"""
Patient symptoms: {symptoms}

Give BOTH Western Medicine + TCM report.

Language: {lang}
"""

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )
    return response.output_text


# ==========================================================
# Elder Speaker (Summary + TTS)
# ==========================================================
def create_elder_tts(text, lang):

    if lang.startswith("中文"):
        prompt = f"将以下内容总结为 3 句简单老人语言：\n{text}"
    else:
        prompt = f"Simplify into 3 elderly-friendly sentences:\n{text}"

    summary_resp = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )
    summary = summary_resp.output_text

    # TTS
    audio_resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=summary,
        format="mp3"
    )
    audio_bytes = audio_resp.read()

    return summary, audio_bytes


# ==========================================================
# TCM Remedy Generator
# ==========================================================
def generate_tcm_recommendation(lang):

    if lang.startswith("中文"):
        lang_cmd = "用中文回答。"
    else:
        lang_cmd = "Respond in English."

    prompt = f"""
Provide a safe TCM guideline for elderly joint pain.

Include:
• Pattern types
• Meridians + acupoints
• Food therapy
• Daily habits
• Safety notes

{lang_cmd}
"""

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )
    return response.output_text


# ==========================================================
# Rehab Video Generator (GPT-Video API)
# ==========================================================
def generate_rehab_video(joint_type, lang):

    script_prompt = f"""
Generate a rehabilitation exercise script for {joint_type} joint pain.
Include warm-up, stretching, strengthening.

Language: {lang}
"""

    script_resp = client.responses.create(
        model="gpt-5.1",
        input=script_prompt
    )
    script_text = script_resp.output_text

    # Generate actual video
    video_resp = client.video.generate(
        model="gpt-4o-video",
        prompt=script_text
    )

    vid_bytes = video_resp.read()

    return script_text, vid_bytes


# ==========================================================
# GPT-5.1 Motion Tracking
# ==========================================================
def perform_motion_tracking(uploaded_video, lang):

    video_bytes = uploaded_video.read()

    prompt = f"""
Analyze human motion for rehab quality:

• Joint angle stability  
• Balance  
• Posture correctness  
• Weak points  
• Improvements  

Language: {lang}
"""

    response = client.responses.create(
        model="gpt-5.1-vision",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_file", "input_file": video_bytes}
                ]
            }
        ]
    )

    return response.output_text


# ==========================================================
# Streamlit UI (Single Layer — Tabs Only)
# ==========================================================
st.set_page_config(page_title="SilverMotion AI", layout="wide")

st.sidebar.title("🌍 Output Language")
output_lang = st.sidebar.selectbox("Choose", ["English", "中文（简体）"])

tabs = st.tabs([
    "🏠 Home",
    "📸 Image Diagnosis",
    "📝 Text Diagnosis",
    "🔊 Elder Speaker",
    "🌿 TCM Remedies",
    "❤️ Pain Score",
    "🎥 Rehab Video",
    "🕺 Motion Tracking"
])

# ----------------------------------------------------------
# HOME
# ----------------------------------------------------------
with tabs[0]:
    st.title("💙 SilverMotion AI – Medical • TCM • Vision AI • Rehab • Motion Tracking")
    st.write("One-stop multimodal elder-friendly joint health assistant.")


# ----------------------------------------------------------
# IMAGE DIAGNOSIS
# ----------------------------------------------------------
with tabs[1]:
    st.header("📸 AI Image Diagnosis — GPT-5.1 Vision (Western + TCM)")

    img = st.file_uploader("Upload joint photo", type=["jpg", "png", "jpeg"])
    symptoms = st.text_area("Describe symptoms")

    if st.button("🔍 Run Diagnosis"):
        if img:
            result = diagnose_with_vision(img, symptoms, output_lang)
            st.success(result)
        else:
            st.error("Please upload an image.")


# ----------------------------------------------------------
# TEXT DIAGNOSIS
# ----------------------------------------------------------
with tabs[2]:
    st.header("📝 Text Diagnosis")
    symptoms = st.text_area("Describe your symptoms")

    if st.button("Run Text Diagnosis"):
        if symptoms.strip():
            r = diagnose_from_text(symptoms, output_lang)
            st.success(r)
        else:
            st.error("Enter symptoms first.")


# ----------------------------------------------------------
# ELDER SPEAKER
# ----------------------------------------------------------
with tabs[3]:
    st.header("🔊 Elder Speaker")

    text_in = st.text_area("Paste medical explanation")
    if st.button("Generate Elder Summary + TTS"):
        if text_in.strip():
            summary, audio = create_elder_tts(text_in, output_lang)
            st.write(summary)
            st.audio(audio, format="audio/mp3")
        else:
            st.error("Enter text first.")


# ----------------------------------------------------------
# TCM REMEDIES
# ----------------------------------------------------------
with tabs[4]:
    st.header("🌿 TCM Remedies")
    if st.button("Generate TCM Advice"):
        r = generate_tcm_recommendation(output_lang)
        st.success(r)


# ----------------------------------------------------------
# PAIN SCORE – GPT AI feedback
# ----------------------------------------------------------
with tabs[5]:
    st.header("❤️ Pain Score AI")

    ps = st.slider("Rate your pain level", 0, 10, 5)
    if st.button("Explain My Pain Level"):
        resp = client.responses.create(
            model="gpt-5.1",
            input=f"Pain score: {ps}. Explain in {output_lang}."
        )
        st.success(resp.output_text)


# ----------------------------------------------------------
# REHAB VIDEO
# ----------------------------------------------------------
with tabs[6]:
    st.header("🎥 Rehab Video Generator")

    joint = st.selectbox("Joint", ["Knee", "Shoulder", "Hip", "Neck"])
    if st.button("Generate Rehab Video"):
        script, vbytes = generate_rehab_video(joint, output_lang)
        st.write(script)
        st.video(vbytes)


# ----------------------------------------------------------
# MOTION TRACKING
# ----------------------------------------------------------
with tabs[7]:
    st.header("🕺 Motion Tracking – GPT-5.1 Vision")

    video_file = st.file_uploader("Upload exercise video", type=["mp4", "mov"])

    if st.button("Analyze Motion"):
        if video_file:
            r = perform_motion_tracking(video_file, output_lang)
            st.success(r)
        else:
            st.error("Upload a video first.")
