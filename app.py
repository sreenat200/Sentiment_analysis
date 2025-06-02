import streamlit as st
import os
import tempfile

from Lead_score_conversion import (
    MalayalamTranscriptionPipeline,
    analyze_text,
    save_analysis_to_csv,
    compare_analyses,
    print_analysis_summary
)

st.set_page_config(page_title="Malayalam Audio Analyzer", layout="wide")
st.title("🎙️ Malayalam Audio Sentiment & Intent Analyzer")

uploaded_file = st.file_uploader("Upload Malayalam audio file", type=["mp3", "wav", "aac", "m4a", "flac", "ogg", "wma"])

if uploaded_file is not None:
    tmp_path = None
    transcriber = MalayalamTranscriptionPipeline()

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Transcription
        with st.spinner("🔍 Transcribing audio..."):
            results = transcriber.transcribe_audio(tmp_path)

        if not results or not results.get("raw_transcription"):
            st.error("❌ Transcription failed.")
            st.stop()

        raw_text = results["raw_transcription"]
        st.subheader("📄 English Transcription")
        st.write(raw_text)

        # Translation
        with st.spinner("🌐 Translating to Malayalam..."):
            results = transcriber.translate_to_malayalam(results)
            ml_text = results.get("translated_malayalam", "")

        st.subheader("🌐 Malayalam Translation")
        st.write(ml_text)

        # Sentiment & Intent
        with st.spinner("📊 Running Sentiment & Intent Analysis..."):
            en_analysis = analyze_text(raw_text, "en")
            ml_analysis = analyze_text(ml_text, "ml")
            comparison = compare_analyses(en_analysis, ml_analysis)

        st.success("✅ Analysis complete.")

        # Download CSVs
        st.subheader("📁 Download Results")
        for label, data in [
            ("English Analysis", en_analysis),
            ("Malayalam Analysis", ml_analysis),
            ("Comparison", comparison)
        ]:
            path = save_analysis_to_csv(data, label.lower().replace(" ", "_"))
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {label}",
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime="text/csv"
                    )

        # Summary
        st.subheader("📈 Summary Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🇬🇧 English Analysis")
            print_analysis_summary(en_analysis, "English")
        with col2:
            st.markdown("### 🇮🇳 Malayalam Analysis")
            print_analysis_summary(ml_analysis, "Malayalam")

        # Metrics
        intent_matches = sum(1 for item in comparison if item["intent_match"])
        intent_match_rate = intent_matches / len(comparison) if comparison else 0
        sentiment_diff = sum(item["sentiment_diff"] for item in comparison) / len(comparison) if comparison else 0

        st.markdown(f"### 🔄 Intent Match Rate: `{intent_match_rate:.1%}`")
        st.markdown(f"### 🎭 Avg Sentiment Score Difference: `{sentiment_diff:.2f}`")

        # Lead score
        en_avg = sum(x["sentiment_score"] for x in en_analysis) / len(en_analysis) if en_analysis else 0
        ml_avg = sum(x["sentiment_score"] for x in ml_analysis) / len(ml_analysis) if ml_analysis else 0
        lead_score = int(((en_avg + ml_avg) / 2) * 100)

        st.subheader(f"🔥 Lead Score: `{lead_score}/100`")
        if lead_score >= 70:
            st.success("💡 High interest lead")
        elif lead_score >= 40:
            st.info("🧐 Moderate interest lead")
        else:
            st.warning("❄️ Low interest lead")

    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")

    finally:
        transcriber.cleanup()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
