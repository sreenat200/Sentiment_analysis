import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from pydub import AudioSegment
import zipfile
import json
import re
from io import BytesIO
import shutil
import base64
import logging
from main_python import (
    analyze_text,
    compare_analyses,
    generate_analysis_pdf,
    save_analysis_to_csv,
    MalayalamTranscriptionPipeline,
    split_into_sentences
)
from gdrive_utils import upload_to_gdrive, search_gdrive_files, download_from_gdrive

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Lead Scoring System",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""<style>
    .main {
        background-color: #202427;
        color: #DCD9D9;
        margin-top: 0;
        padding-top: 0;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
        margin-top: -0px; /* Adjusted to move title further upward */
    }
    .logo-text {
        font-size: 32px;
        font-weight: bold;
        color: #ffffff;
    }
    .glow-logo {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #A1C4FD, #C2E9FB, #E0F7FA);
        color: #000000;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 50%;
        font-size: 24px;
        font-weight: bold;
        animation: glow 2s ease-in-out infinite;
    }
    @keyframes glow {
        0% {
            box-shadow: 0 0 8px 2px rgba(0, 191, 255, 0.3);
            transform: scale(1);
        }
        50% {
            box-shadow: 0 0 25px 12px rgba(135, 206, 250, 0.85);
            transform: scale(1.06);
        }
        100% {
            box-shadow: 0 0 8px 2px rgba(0, 191, 255, 0.3);
            transform: scale(1);
        }
    }
    .stApp h1 {
        color: #ffffff !important;
        font-size: 2rem !important;
        animation: slideUpFade 1.5s ease-out;
        margin-top: 0;
    }
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .stText, .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput, .stCheckbox {
        color: #e0e0e0 !important;
    }
    .stButton>button {
        background-color: #D22B2B;
        color: #ffffff;
        border-radius: 8px;
        padding: 8px 16px;
        transition: transform 0.2s, background-color 0.2s;
        width: auto;
        max-width: 180px;
        display: inline-block;
        text-align: center;
        font-size: 14px;
        border: none;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #B02424;
    }
    .clear-search-button>button {
        background-color: transparent;
        color: #D22B2B;
        border: 1px solid #D22B2B;
        border-radius: 8px;
        padding: 8px 16px;
        transition: transform 0.2s, background-color 0.2s;
        width: auto;
        max-width: 180px;
        display: inline-block;
        text-align: center;
        font-size: 14px;
    }
    .clear-search-button>button:hover {
        transform: scale(1.05);
        background-color: rgba(178, 36, 36, 0.1);
        color: #B02424;
        border-color: #B02424;
    }
    .dashboard-analysis-button>button {
        background-color: #D22B2B;
        color: #ffffff;
        border-radius: 4px;
        padding: 3px 6px;
        transition: transform 0.2s, background-color 0.2s;
        width: auto;
        max-width: 80px;
        display: inline-block;
        text-align: center;
        font-size: 10px;
        border: none;
    }
    .dashboard-analysis-button>button:hover {
        transform: scale(1.05);
        background-color: #B02424;
    }
    .score-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        text-align: center;
    }
    .high-score {
        background-color: #4CBB17;
        color: #ffffff;
    }
    .medium-score {
        background-color: #f1c40f;
        color: #2c3e50;
    }
    .low-score {
        background-color: #e74c3c;
        color: #ffffff;
    }
    .lead-score, .intent-score {
        font-size: 2rem;
        font-weight: bold;
    }
    .error-message {
        background-color: #e74c3c;
        color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CBB17, #D22B2B); 
    }
    .stProgress {
        margin-top: 40px;
    }
    .dashboard-header {
        background-color: transparent;
        color: #ffffff;
        padding: 6px;
        margin-bottom: 4px;
        text-align: left;
        font-size: 2rem;
        font-weight: normal;
    }
    .dashboard-section {
        margin-top: 5px;
    }
    .pdf-viewer-button>button {
        background-color: #D22B2B;
        color: #ffffff;
        border-radius: 4px;
        padding: 4px 8px;
        transition: transform 0.2s, background-color 0.2s;
        width: auto;
        max-width: 100px;
        display: inline-block;
        text-align: center;
        font-size: 12px;
        border: none;
    }
    .pdf-viewer-button>button:hover {
        transform: scale(1.05);
        background-color: #B02424;
    }
    .expander-content {
        background-color: transparent;
        padding: 10px;
        border: 1px solid #FFFFFF;
        border-radius: 5px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    .emotion-display {
        background-color: #2c3e50;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .emotion-text {
        color: white;
        margin: 0;
        font-size: 1.2rem;
    }
    .search-container {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-bottom: 4px;
    }
    .emotion-block {
        background-color: transparent;
        border: 1px solid #ffffff;
        color: #ffffff;
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 1.2rem;
        text-align: center;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    @media (max-width: 768px) {
        .stApp h1 { font-size: 1.5rem !important; }
        .score-card { padding: 10px; }
        .lead-score, .intent-score { font-size: 1.5rem; }
        .stButton>button, .clear-search-button>button {
            padding: 6px 12px;
            font-size: 12px;
            max-width: 150px;
        }
        .dashboard-analysis-button>button,
        .pdf-viewer-button>button {
            padding: 3px 6px;
            font-size: 10px;
            max-width: 80px;
        }
        .dashboard-header {
            font-size: 0.9rem;
            padding: 3px;
            margin-bottom: 2px;
        }
        .expander-content {
            padding: 8px;
            font-size: 0.8rem;
        }
        .emotion-text {
            font-size: 1rem;
        }
        .search-container {
            flex-direction: column;
            align-items: flex-start;
        }
        .header-container {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
        .emotion-block {
            font-size: 1rem;
            padding: 6px 10px;
        }
    }
</style>""", unsafe_allow_html=True)

def get_emotion_emoji(emotion):
    emotion_emoji_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "fearful": "😨",
        "disgust": "🤢",
        "surprise": "😲",
        "neutral": "😐",
        "unknown": "❓",
        "excited": "🤩",
        "confused": "😕",
        "calm": "😌"
    }
    return emotion_emoji_map.get(emotion.lower(), "❓")

def get_interpretation(score):
    if score >= 70:
        return "High Interest Lead", "high-score"
    elif score >= 40:
        return "Moderate Interest Lead", "medium-score"
    else:
        return "Low Interest Lead", "low-score"

def get_intent_interpretation(score):
    if score >= 70:
        return "Strong Intent Score", "high-score"
    elif score >= 40:
        return "Moderate Intent Score", "medium-score"
    else:
        return "Low Intent Score", "low-score"

def display_lead_score(score):
    interpretation, css_class = get_interpretation(score)
    st.markdown(f"""
    <div class="score-card {css_class}">
        <div class="lead-score">{score}/100</div>
        <div>{interpretation}</div>
    </div>
    """, unsafe_allow_html=True)

def display_intent_score(score):
    interpretation, css_class = get_intent_interpretation(score)
    st.markdown(f"""
    <div class="score-card {css_class}">
        <div class="intent-score">{score}/100</div>
        <div>{interpretation}</div>
    </div>
    """, unsafe_allow_html=True)

def convert_to_whisper_format(input_path):
    supported_formats = ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.wma']
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext == '.wav':
        return input_path
    if file_ext not in supported_formats:
        logger.error(f"Unsupported audio format: {file_ext}")
        raise ValueError(f"Unsupported audio format: {file_ext}")
    temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    wav_path = os.path.join(temp_dir, f"temp_{timestamp}.wav")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        if not os.path.exists(wav_path):
            logger.error("AudioSegment exported successfully, but WAV file was not created.")
            raise RuntimeError("Failed to create WAV file during conversion.")
        return wav_path
    except Exception as e:
        logger.error(f"AudioSegment conversion failed: {str(e)}")
        try:
            import subprocess
            cmd = ['ffmpeg', '-i', input_path, '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', wav_path]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not os.path.exists(wav_path):
                logger.error("FFmpeg ran successfully, but output WAV file was not created.")
                raise RuntimeError("FFmpeg ran successfully, but output WAV file was not created.")
            return wav_path
        except FileNotFoundError:
            logger.error("FFmpeg not found in system PATH.")
            raise RuntimeError("FFmpeg is not installed or not found in system PATH. Please install FFmpeg: https://ffmpeg.org/download.html")
        except subprocess.CalledProcessError as cpe:
            logger.error(f"FFmpeg failed: {cpe.stderr}")
            raise RuntimeError(f"FFmpeg failed to convert audio: {cpe.stderr}. Ensure input file is valid.")
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            raise RuntimeError(f"Failed to convert audio: {str(e)}. Ensure FFmpeg is installed and input file is accessible.")

def clean_old_files():
    try:
        retention_map = {
            "1_month": 30,
            "3_months": 90,
            "6_months": 180,
            "1_year": 365
        }
        retention_period = st.session_state.get('retention_period', '1_month')
        retention_days = retention_map[retention_period]
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        metadata = st.session_state.storage_metadata
        updated_metadata = []
        for entry in metadata:
            entry_date = datetime.fromisoformat(entry['timestamp'])
            if entry_date >= cutoff_date:
                updated_metadata.append(entry)
            else:
                try:
                    for path in [entry.get('audio_path'), entry.get('report_path'),
                                 entry.get('raw_transcript_path'), entry.get('translated_text_path')]:
                        if path and os.path.exists(path):
                            os.remove(path)
                            logger.info(f"Deleted old file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old file {path}: {str(e)}")
        st.session_state.storage_metadata = updated_metadata
        metadata_path = os.path.join(st.session_state.storage_dir, 'storage_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(updated_metadata, f, indent=2)
        logger.info(f"Cleaned old files. Retained {len(updated_metadata)} entries.")
    except Exception as e:
        logger.error(f"Failed to clean old files: {str(e)}")
        st.error(f"Failed to clean old files: {str(e)}")

def initialize_session_state():
    defaults = {
        'results': None,
        'analysis_complete': False,
        'zip_created': False,
        'whisper_model': None,
        'temp_files': [],
        'zip_filename': None,
        'processing_error': None,
        'wav_path': None,
        'audio_path': None,
        'enable_drive': True,
        'drive_folder_id': None,
        'process_triggered': False,
        'custom_filename': None,
        'audio_buffer': None,
        'selected_audio': None,
        'storage_metadata': [],
        'storage_dir': 'temp_files',
        'dashboard_display_count': 10,
        'dashboard_search_results': None,
        'dashboard_search_active': False,
        'retention_period': '1_month',
        'gdrive_search_results': None,
        'gdrive_search_active': False,
        'gdrive_display_count': 10,
        'dashboard_filter': 'Filename',
        'dashboard_clear_search_trigger': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    os.makedirs(st.session_state.storage_dir, exist_ok=True)
    metadata_path = os.path.join(st.session_state.storage_dir, 'storage_metadata.json')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump([], f)
    elif not st.session_state.storage_metadata:
        try:
            with open(metadata_path, 'r') as f:
                st.session_state.storage_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            st.session_state.storage_metadata = []
    clean_old_files()

def store_audio_and_report(audio_file, audio_buffer, pdf_path=None, custom_filename=None, raw_transcription=None, ml_translation=None):
    try:
        storage_dir = st.session_state.storage_dir  # Fixed typo: storage_pinpoint -> storage_dir
        metadata_path = os.path.join(storage_dir, 'storage_metadata.json')
        metadata = st.session_state.storage_metadata
        original_filename = os.path.splitext(audio_file.name)[0]
        base_filename = custom_filename or original_filename
        base_filename = "".join(c for c in base_filename if c.isalnum() or c in ('-', '_')).rstrip() or original_filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_ext = os.path.splitext(audio_file.name)[1]
        audio_filename = f"{base_filename}_{timestamp}{audio_ext}"
        audio_path = os.path.join(storage_dir, audio_filename)
        with open(audio_path, 'wb') as f:
            f.write(audio_buffer.getvalue())
        report_path = None
        report_filename = None
        if pdf_path and os.path.exists(pdf_path):
            report_filename = os.path.basename(pdf_path)
            report_path = os.path.join(storage_dir, report_filename)
            shutil.copy(pdf_path, report_path)
        raw_transcript_path = os.path.join(storage_dir, f"{base_filename}_{timestamp}_raw.txt")
        translated_text_path = os.path.join(storage_dir, f"{base_filename}_{timestamp}_translated.txt")
        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
            f.write(raw_transcription or "Not Available")
        with open(translated_text_path, 'w', encoding='utf-8') as f:
            f.write(ml_translation or "Not Available")
        metadata_entry = {
            'audio_filename': audio_filename,
            'audio_path': audio_path,
            'report_filename': report_filename,
            'report_path': report_path,
            'raw_transcript_path': raw_transcript_path,
            'translated_text_path': translated_text_path,
            'timestamp': datetime.now().isoformat(),
            'base_filename': base_filename,
            'emotion': st.session_state.get('results', {}).get('audio_metadata', {}).get('emotion', 'unknown')
        }
        metadata.append(metadata_entry)
        st.session_state.storage_metadata = metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Stored metadata for {audio_filename}. Total entries: {len(metadata)}")
        clean_old_files()
        return audio_path, report_path
    except Exception as e:
        logger.error(f"Failed to store audio/report: {str(e)}")
        st.error(f"Failed to store audio/report: {str(e)}")
        return None, None

def cleanup_temp_files():
    try:
        if st.session_state.get('zip_created') and st.session_state.get('zip_filename') and os.path.exists(st.session_state.get('zip_filename', '')):
            for file_path in st.session_state.get('temp_files', []):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {str(e)}")
                    st.error(f"Failed to delete {file_path}: {str(e)}")
            st.session_state.temp_files = []
            temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp dir: {temp_dir}")
                except Exception as e:
                    logger.error(f"Failed to delete temp dir {temp_dir}: {str(e)}")
    except:
        pass

def reset_analysis():
    try:
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            if key not in ['storage_metadata', 'storage_dir']:
                del st.session_state[key]
        initialize_session_state()
        cleanup_temp_files()
        clean_old_files()
        st.success("Analysis reset successfully!")
        st.rerun()
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        st.error(f"Reset failed: {str(e)}")

def create_zip_archive(audio_path, raw_transcription, ml_translation, pdf_path, en_analysis, ml_analysis, comparison, base_filename):
    try:
        temp_dir = os.path.join(tempfile.gettempdir(), f"zip_staging_{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file {audio_path} not found. Attempting to use original audio from session state.")
            audio_path = st.session_state.get('audio_path', audio_path)
            if not os.path.exists(audio_path):
                logger.error(f"Original audio file {audio_path} also not found.")
                raise RuntimeError(f"Audio file not found: {audio_path}. Ensure the file was not deleted.")
        audio_ext = os.path.splitext(audio_path)[1]
        audio_filename = f"audio{audio_ext}"
        audio_dest = os.path.join(temp_dir, audio_filename)
        with open(audio_path, 'rb') as src, open(audio_dest, 'wb') as dst:
            dst.write(src.read())
        raw_transcription_path = os.path.join(temp_dir, "transcript_raw.txt")
        with open(raw_transcription_path, 'w', encoding='utf-8') as f:
            f.write(raw_transcription)
        ml_translation_path = os.path.join(temp_dir, "transcript_translated.txt")
        with open(ml_translation_path, 'w', encoding='utf-8') as f:
            f.write(ml_translation)
        en_csv_path = save_analysis_to_csv(en_analysis, os.path.join(temp_dir, f"{base_filename}_en"))
        ml_csv_path = save_analysis_to_csv(ml_analysis, os.path.join(temp_dir, f"{base_filename}_ml"))
        comparison_csv_path = save_analysis_to_csv(comparison, os.path.join(temp_dir, f"{base_filename}_comparison"))
        summary_pdf_path = os.path.join(temp_dir, "summary_report.pdf")
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file {pdf_path} not found.")
            raise RuntimeError(f"PDF file not found: {pdf_path}. Ensure the PDF was generated correctly.")
        with open(pdf_path, 'rb') as src, open(summary_pdf_path, 'wb') as dst:
            dst.write(src.read())
        en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
        ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
        combined_avg = (en_avg_score + ml_avg_score) / 2
        lead_score = int(combined_avg * 100)
        positive_intents = sum(1 for item in en_analysis if item["intent"] in ["Strong_interest", "Fee_query", "Moderate_interest", "Confirmation"])
        intent_score = int((positive_intents / len(en_analysis)) * 100) if en_analysis else 0
        intent_counts = pd.Series([item["intent"] for item in en_analysis]).value_counts()
        primary_intent = intent_counts.index[0] if not intent_counts.empty else "Unknown"
        summary_data = {
            "filename": base_filename,
            "timestamp": datetime.now().isoformat(),
            "lead_score": lead_score,
            "intent_score": intent_score,
            "primary_intent": primary_intent,
            "english_analysis": en_analysis,
            "malayalam_analysis": ml_analysis,
            "comparison_analysis": comparison,
            "audio_metadata": {
                "extension": audio_ext,
                "duration": len(AudioSegment.from_file(audio_path)) / 1000,
                "emotion": st.session_state.get('results', {}).get('audio_metadata', {}).get('emotion', 'unknown')
            }
        }
        summary_json_path = os.path.join(temp_dir, "summary_analyzed.json")
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(tempfile.gettempdir(), f"{base_filename}_L{lead_score}_I{intent_score}_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(audio_dest, audio_filename)
            zipf.write(raw_transcription_path, "transcript_raw.txt")
            zipf.write(ml_translation_path, "transcript_translated.txt")
            zipf.write(en_csv_path, f"{base_filename}_en.csv")
            zipf.write(ml_csv_path, f"{base_filename}_ml.csv")
            zipf.write(comparison_csv_path, f"{base_filename}_comparison.csv")
            zipf.write(summary_json_path, "summary_analyzed.json")
            zipf.write(summary_pdf_path, "summary_report.pdf")
        st.session_state.temp_files.extend([
            audio_dest, raw_transcription_path, ml_translation_path,
            en_csv_path, ml_csv_path, comparison_csv_path,
            summary_json_path, summary_pdf_path, zip_path
        ])
        return zip_path
    except Exception as e:
        logger.error(f"Failed to create ZIP file: {str(e)}")
        raise RuntimeError(f"Failed to create ZIP file: {str(e)}")

def process_audio(audio_file, model_size, progress_callback=None):
    try:
        st.session_state.analysis_complete = False
        st.session_state.results = None
        st.session_state.processing_error = None
        steps = 9
        step_increment = 100 / steps
        if progress_callback:
            progress_callback(0, "Saving uploaded file...")
        original_filename = os.path.splitext(audio_file.name)[0]
        custom_filename = st.session_state.custom_filename
        base_filename = custom_filename if custom_filename else original_filename
        base_filename = "".join(c for c in base_filename if c.isalnum() or c in ('-', '_')).rstrip() or original_filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_buffer = BytesIO(audio_file.read())
        st.session_state.audio_buffer = audio_buffer.getvalue()
        temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
        os.makedirs(temp_dir, exist_ok=True)
        audio_ext = os.path.splitext(audio_file.name)[1]
        audio_path = os.path.join(temp_dir, f"original_{timestamp}{audio_ext}")
        with open(audio_path, 'wb') as f:
            f.write(audio_buffer.getvalue())
        st.session_state.temp_files.append(audio_path)
        st.session_state.audio_path = audio_path
        if progress_callback:
            progress_callback(int(step_increment), "Saving uploaded file...")
        if progress_callback:
            progress_callback(int(step_increment), "Converting audio format...")
        try:
            wav_path = convert_to_whisper_format(audio_path)
            if wav_path != audio_path:
                st.session_state.temp_files.append(wav_path)
                st.session_state.wav_path = wav_path
            else:
                st.session_state.wav_path = audio_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            raise RuntimeError(f"Audio conversion failed: {str(e)}. Try a different audio format or ensure ffmpeg is installed.")
        if progress_callback:
            progress_callback(int(2 * step_increment), "Converting audio format...")
        if progress_callback:
            progress_callback(int(2 * step_increment), "Loading transcription model...")
        if st.session_state.whisper_model is None:
            import torch
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            st.session_state.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        if progress_callback:
            progress_callback(int(3 * step_increment), "Loading transcription model...")
        if progress_callback:
            progress_callback(int(3 * step_increment), "Transcribing audio...")
        segments, _ = st.session_state.whisper_model.transcribe(st.session_state.wav_path, beam_size=5, language="en")
        full_text = ""
        segment_list = []
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            confidence = seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0
            segment_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "confidence": round(confidence, 3),
                "overlap": i > 0 and seg.start < segment_list[i-1]["end"]
            })
            full_text += f" {text}"
        raw_transcription = full_text.strip()
        logger.info(f"Raw transcription: {raw_transcription}")
        if progress_callback:
            progress_callback(int(4 * step_increment), "Transcribing audio...")
        if progress_callback:
            progress_callback(int(4 * step_increment), "Translating to English...")
        try:
            if 'transcriber' not in st.session_state:
                st.session_state.transcriber = MalayalamTranscriptionPipeline(model_size=model_size)
            transcriber = st.session_state.transcriber
            input_data = {'raw_transcription': raw_transcription}
            logger.debug(f"Translation input: {input_data}")
            
            from tenacity import retry, stop_after_attempt, wait_fixed
            @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
            def translate_with_timeout(data):
                return transcriber.translate_to_malayalam(data)
            
            results = translate_with_timeout(input_data)
            ml_translation = results.get('translated_malayalam', '')
            
            if not ml_translation:
                logger.warning("Translation returned empty result")
            logger.info(f"Translated Malayalam: {ml_translation}")
            
            sentences = split_into_sentences(raw_transcription, "en")
            ml_translations = split_into_sentences(ml_translation, "ml") if ml_translation else ["" for _ in sentences]
            
            if len(ml_translations) < len(sentences):
                ml_translations.extend(["" for _ in range(len(sentences) - len(ml_translations))])
            elif len(ml_translations) > len(sentences):
                ml_translations = ml_translations[:len(sentences)]
            logger.debug(f"Aligned: English sentences={len(sentences)}, Malayalam sentences={len(ml_translations)}")
            
            transcriber.cleanup()
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            ml_translation = ""
            sentences = split_into_sentences(raw_transcription, "en")
            ml_translations = ["" for _ in sentences]
            st.session_state.processing_error = f"Translation failed: {str(e)}. Proceeding with transcription."
        if progress_callback:
            progress_callback(int(5 * step_increment), "Translating to English...")
        if progress_callback:
            progress_callback(int(5 * step_increment), "Analyzing content...")
        en_analysis = []
        for sentence in sentences:
            analysis = analyze_text(sentence, "en")
            if analysis:
                en_analysis.extend(analysis)
        ml_analysis = []
        for i in range(len(sentences)):
            translated = ml_translations[i] if i < len(ml_translations) else ""
            if translated:
                analysis = analyze_text(translated, "ml")
                if analysis:
                    ml_analysis.extend(analysis)
            else:
                ml_analysis.append({
                    "sentence_id": f"ml_{i+1}",
                    "text": "",
                    "language": "ml",
                    "intent": "Neutral_response",
                    "sentiment": "neutral",
                    "sentiment_score": 0.5,
                    "word_count": 0,
                    "char_count": 0
                })
        min_length = min(len(en_analysis), len(ml_analysis))
        en_analysis = en_analysis[:min_length]
        ml_analysis = ml_analysis[:min_length]
        logger.debug(f"Analysis lengths: English={len(en_analysis)}, Malayalam={len(ml_analysis)}")
        comparison = compare_analyses(en_analysis, ml_analysis)
        if progress_callback:
            progress_callback(int(6 * step_increment), "Analyzing content...")
        if progress_callback:
            progress_callback(int(6 * step_increment), "Calculating scores...")
        en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
        ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
        combined_avg = (en_avg_score + ml_avg_score) / 2 if ml_analysis else en_avg_score
        lead_score = int(combined_avg * 100)
        positive_intents = sum(1 for item in en_analysis if item["intent"] in ["Strong_interest", "Fee_query", "Moderate_interest", "Confirmation"])
        intent_score = int((positive_intents / len(en_analysis)) * 100) if en_analysis else 0
        if progress_callback:
            progress_callback(int(7 * step_increment), "Calculating scores...")
        if progress_callback:
            progress_callback(int(7 * step_increment), "Finalizing results...")
        results = {
            "raw_transcription": raw_transcription,
            "ml_translation": ml_translation,
            "en_analysis": en_analysis,
            "ml_analysis": ml_analysis,
            "comparison": comparison,
            "lead_score": lead_score,
            "intent_score": intent_score,
            "audio_path": audio_path,
            "original_filename": base_filename,
            "audio_metadata": {
                "emotion": "unknown"
            }
        }
        try:
            if 'transcriber' in st.session_state:
                emotion = st.session_state.transcriber.analyze_emotion(wav_path)
                results["audio_metadata"]["emotion"] = emotion
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {str(e)}. Using default 'unknown' emotion.")
        
        st.session_state.results = results
        if progress_callback:
            progress_callback(int(8 * step_increment), "Finalizing results...")
        if progress_callback:
            progress_callback(int(8 * step_increment), "Storing audio and report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"{base_filename}_L{lead_score}_I{intent_score}_{timestamp}"
        pdf_path = generate_analysis_pdf(
            en_analysis,
            ml_analysis,
            comparison,
            final_filename
        )
        st.session_state.temp_files.append(pdf_path)
        stored_audio_path, stored_pdf_path = store_audio_and_report(
            audio_file,
            BytesIO(st.session_state.audio_buffer),
            pdf_path=pdf_path,
            custom_filename=custom_filename,
            raw_transcription=raw_transcription,
            ml_translation=ml_translation
        )
        if stored_pdf_path:
            st.session_state.temp_files.append(stored_pdf_path)
        st.session_state.analysis_complete = True
        if progress_callback:
            progress_callback(100, "Analysis complete!")
        logger.info("Audio processing completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        st.session_state.processing_error = f"Processing failed: {str(e)}. Try a different audio file or check dependencies."
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")

def display_results():
    try:
        results = st.session_state.results
        if not results:
            st.error("No results available to display")
            return

        emotion = results.get('audio_metadata', {}).get('emotion', 'unknown')
        emoji = get_emotion_emoji(emotion)
        # Calculate primary sentiment from en_analysis
        sentiments = [item["sentiment"] for item in results.get("en_analysis", []) if item.get("sentiment")]
        sentiment = pd.Series(sentiments).mode()[0] if sentiments else "unknown"
        
        st.markdown(f"""
        <div class="header-container">
            <div class="dashboard-header">Analysis Results</div>
            <div class="emotion-block">Detected Emotion:   {emotion.capitalize()} {emoji}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.audio_buffer:
            st.markdown("Uploaded Audio")
            st.audio(st.session_state.audio_buffer, format=f"audio/{os.path.splitext(results['audio_path'])[1][1:]}")
        tab1, tab2, tab3 = st.tabs(["Results", "Analysis", "Visualizations"])
        with tab1:
            st.header("Transcription Results")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Malayalam Transcription")
                st.text_area("Raw Transcription", results["ml_translation"], height=200)
            with col2:
                st.subheader("English Transcription")
                st.text_area("Raw Transcription", results["raw_transcription"], height=200)
        with tab2:
            st.header("Detailed Analysis")
            col1, col2 = st.columns(2)
            with col1:
                display_lead_score(results['lead_score'])
            with col2:
                display_intent_score(results['intent_score'])
            st.subheader("Malayalam Analysis")
            st.dataframe(pd.DataFrame(results["ml_analysis"]), use_container_width=True)
            with st.expander("Export Malayalam Analysis"):
                csv_path = save_analysis_to_csv(results["ml_analysis"], results["original_filename"] + "_ml")
                if csv_path:
                    with open(csv_path, 'rb') as f:
                        st.download_button("Download Malayalam CSV", f, file_name=os.path.basename(csv_path))
                    st.session_state.temp_files.append(csv_path)
            st.subheader("English Analysis")
            st.dataframe(pd.DataFrame(results["en_analysis"]), use_container_width=True)
            with st.expander("Export English Analysis"):
                csv_path = save_analysis_to_csv(results["en_analysis"], results["original_filename"] + "_en")
                if csv_path:
                    with open(csv_path, "rb") as f:
                        st.download_button("Download English CSV", f, file_name=os.path.basename(csv_path))
                    st.session_state.temp_files.append(csv_path)
            st.subheader("Comparison Analysis")
            st.dataframe(pd.DataFrame(results["comparison"]), use_container_width=True)
            with st.expander("Export Comparison Analysis"):
                csv_path = save_analysis_to_csv(results["comparison"], results["original_filename"] + "_comparison")
                if csv_path:
                    with open(csv_path, 'rb') as f:
                        st.download_button("Download Comparison CSV", f.read(), file_name=os.path.basename(csv_path))
                    st.session_state.temp_files.append(csv_path)
        with tab3:
            st.header("Interactive Visualizations")
            st.subheader("Sentiment Distribution")
            en_sentiments = pd.Series([item["sentiment"] for item in results["en_analysis"]]).value_counts().reset_index()
            en_sentiments.columns = ['sentiment', 'count']
            ml_sentiments = pd.Series([item["sentiment"] for item in results["ml_analysis"]]).value_counts().reset_index()
            ml_sentiments.columns = ['sentiment', 'count']
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(en_sentiments, x='sentiment', y='count', title="English Sentiment", color='sentiment')
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.bar(ml_sentiments, x='sentiment', y='count', title="Malayalam Sentiment", color='sentiment')
                st.plotly_chart(fig2, use_container_width=True)
            st.subheader("Intent Distribution")
            en_intents = pd.Series([item["intent"] for item in results["en_analysis"]]).value_counts().reset_index()
            en_intents.columns = ['intent', 'count']
            fig3 = px.bar(en_intents, x='intent', y='count', title="Intent Distribution", color='intent')
            st.plotly_chart(fig3, use_container_width=True)
            st.subheader("Sentiment Trend")
            valid_en = [item for item in results["en_analysis"]
                        if "sentiment_score" in item and isinstance(item["sentiment_score"], (int, float))
                        and not pd.isna(item["sentiment_score"])]
            valid_ml = [item for item in results["ml_analysis"]
                        if "sentiment_score" in item and isinstance(item["sentiment_score"], (int, float))
                        and not pd.isna(item["sentiment_score"])]
            min_length = min(len(valid_en), len(valid_ml))
            if min_length > 0:
                df_trend = pd.DataFrame({
                    'Sentence': list(range(1, min_length + 1)),
                    'English': [item["sentiment_score"] for item in valid_en][:min_length],
                    'Malayalam': [item["sentiment_score"] for item in valid_ml][:min_length]
                })
                fig4 = px.line(
                    df_trend,
                    x='Sentence',
                    y=['English', 'Malayalam'],
                    title="Sentiment Trend Over Conversation"
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning(f"No valid data for trend analysis. English has {len(valid_en)} valid entries, "
                          f"Malayalam has {len(valid_ml)} valid entries. Check transcription and translation outputs.")
            st.subheader("Sentiment Differences")
            valid_en = [item for item in results["en_analysis"]
                        if "sentiment_score" in item and isinstance(item["sentiment_score"], (int, float))
                        and not pd.isna(item["sentiment_score"])]
            valid_ml = [item for item in results["ml_analysis"]
                        if "sentiment_score" in item and isinstance(item["sentiment_score"], (int, float))
                        and not pd.isna(item["sentiment_score"])]
            min_length = min(len(valid_en), len(valid_ml))
            if min_length > 0:
                sentiment_diffs = [
                    abs(en["sentiment_score"] - ml["sentiment_score"])
                    for en, ml in zip(valid_en[:min_length], valid_ml[:min_length])
                ]
                fig5 = px.histogram(
                    sentiment_diffs,
                    nbins=10,
                    title="English-Malayalam Sentiment Differences"
                )
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.warning(f"No valid data for sentiment differences. English has {len(valid_en)} valid entries, "
                          f"Malayalam has {len(valid_ml)} valid entries. Check transcription and translation outputs.")
        st.markdown("---")
        st.header("Export Full Report")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"{results['original_filename']}_L{results['lead_score']}_I{results['intent_score']}_{timestamp}"
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Report"):
                with st.spinner("Generating report..."):
                    try:
                        pdf_path = generate_analysis_pdf(
                            results["en_analysis"],
                            results["ml_analysis"],
                            results["comparison"],
                            final_filename
                        )
                        st.session_state.temp_files.append(pdf_path)
                        zip_filename = create_zip_archive(
                            results["audio_path"],
                            results["raw_transcription"],
                            results["ml_translation"],
                            pdf_path,
                            results["en_analysis"],
                            results["ml_analysis"],
                            results["comparison"],
                            results["original_filename"]
                        )
                        st.session_state.zip_created = True
                        st.session_state.zip_filename = zip_filename
                        st.success("Report generated successfully!")
                    except Exception as e:
                        logger.error(f"Failed to generate report: {str(e)}")
                        st.markdown(f"""
                        <div class="error-message">
                            <strong>Error:</strong> Failed to generate report: {str(e)}<br>
                            <strong>Suggestion:</strong> Ensure sufficient disk space, check dependencies, or try a different audio file.
                        </div>
                        """, unsafe_allow_html=True)
        with col2:
            if st.session_state.zip_created and st.session_state.zip_filename and os.path.exists(st.session_state.zip_filename):
                if st.button("Upload to Google Drive"):
                    with st.spinner("Uploading to Google Drive..."):
                        try:
                            if st.session_state.enable_drive:
                                audio_ext = os.path.splitext(results["audio_path"])[1]
                                audio_filename = f"{final_filename}{audio_ext}"
                                temp_audio_path = os.path.join(tempfile.gettempdir(), audio_filename)
                                with open(results["audio_path"], 'rb') as src, open(temp_audio_path, 'wb') as dst:
                                    dst.write(src.read())
                                st.session_state.temp_files.append(temp_audio_path)
                                uploaded_file = upload_to_gdrive(
                                    temp_audio_path,
                                    folder_id=st.session_state.drive_folder_id,
                                    custom_filename=audio_filename
                                )
                                if uploaded_file:
                                    st.success(f"Uploaded {audio_filename} to Google Drive!")
                                    if hasattr(uploaded_file, 'get') and callable(uploaded_file.get):
                                        st.markdown(f"[View file]({uploaded_file.get('webViewLink')})")
                                else:
                                    st.markdown(f"""
                                    <div class="error-message">
                                        <strong>Error:</strong> Google Drive upload failed<br>
                                        <strong>Suggestion:</strong> Verify Google Drive credentials and folder ID.
                                    </div>
                                    """, unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"Google Drive upload failed: {str(e)}")
                            st.markdown(f"""
                            <div class="error-message">
                                <strong>Error:</strong> Google Drive upload failed: {str(e)}<br>
                                <strong>Suggestion:</strong> Verify Google Drive credentials and folder ID.
                            </div>
                            """, unsafe_allow_html=True)
            if st.session_state.zip_created and st.session_state.zip_filename and os.path.exists(st.session_state.zip_filename):
                with open(st.session_state.zip_filename, "rb") as f:
                    st.download_button(
                        label="Download Analysis (ZIP)",
                        data=f.read(),
                        file_name=os.path.basename(st.session_state.zip_filename),
                        mime="application/zip",
                        key="zip_download"
                    )
    except Exception as e:
        logger.error(f"Failed to display results: {str(e)}")
        st.error(f"Failed to display results: {str(e)}")

def display_gdrive_search_results(files):
    try:
        if files:
            st.success(f"Found {len(files)} files")
            sorted_files = sorted(
                files,
                key=lambda x: x['createdTime'],
                reverse=True
            )
            display_count = st.session_state.gdrive_display_count
            for idx, file in enumerate(sorted_files[:display_count]):
                filename = file['name']
                lead_score = intent_score = "N/A"
                lead_match = re.search(r'_L(\d+)', filename)
                intent_match = re.search(r'_I(\d+)', filename)
                if lead_match:
                    lead_score = lead_match.group(1)
                if intent_match:
                    intent_score = intent_match.group(1)
                emotion = file.get('emotion', 'unknown')
                emoji = get_emotion_emoji(emotion)
                expander_label = f"{filename} (Stored: {datetime.fromisoformat(file['createdTime'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')})"
                with st.expander(expander_label):
                    st.markdown('<div class="expander-content">', unsafe_allow_html=True)
                    st.markdown(f"**Detected Emotion**: {emotion.capitalize()} {emoji}")
                    st.markdown(f"**Lead Score**: {lead_score}/100")
                    st.markdown(f"**Intent Score**: {intent_score}/100")
                    st.markdown(f"**Created**: {file['createdTime']}")
                    st.markdown(f"[View in Drive]({file['webViewLink']})")
                    download_key = f"download_{file['id']}"
                    if download_key not in st.session_state:
                        st.session_state[download_key] = False
                    if st.button("Download", key=f"btn_download_{file['id']}_{idx}"):
                        with st.spinner(f"Downloading {filename}..."):
                            try:
                                downloaded_file = download_from_gdrive(file['id'])
                                if downloaded_file:
                                    st.session_state[download_key] = {
                                        'content': downloaded_file['content'],
                                        'filename': downloaded_file['name'],
                                        'mimeType': downloaded_file['mimeType']
                                    }
                                else:
                                    st.error(f"Failed to download {filename}: No content returned")
                            except Exception as e:
                                logger.error(f"Download failed: {str(e)}")
                                st.error(f"Download failed: {str(e)}")
                    if st.session_state.get(download_key):
                        st.download_button(
                            label="Save File",
                            data=st.session_state[download_key]['content'],
                            file_name=st.session_state[download_key]['filename'],
                            mime=st.session_state[download_key]['mimeType'],
                            key=f"save_{file['id']}_{idx}"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
            if display_count < len(sorted_files):
                if st.button("Show More", key="gdrive_show_more"):
                    st.session_state.gdrive_display_count += 10
                    st.rerun()
        else:
            st.warning("No files found")
    except Exception as e:
        logger.error(f"Failed to display Google Drive search results: {str(e)}")
        st.error(f"Failed to display search results: {str(e)}")

def display_dashboard():
    # Check if clear search was triggered and reset search state
    if st.session_state.get('dashboard_clear_search_trigger', False):
        st.session_state.dashboard_search_results = None
        st.session_state.dashboard_search_active = False
        st.session_state.dashboard_display_count = 10
        st.session_state.dashboard_clear_search_trigger = False
        st.rerun()

    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.dashboard_filter == "Filename":
            st.text_input("Search by Filename", key="dashboard_search_query")
        elif st.session_state.dashboard_filter == "Lead Score Range":
            col_min, col_max = st.columns(2)
            min_lead = col_min.number_input("Min Lead Score", 0, 100, 0, key="dashboard_min_lead")
            max_lead = col_max.number_input("Max Lead Score", 0, 100, 100, key="dashboard_max_lead")
        elif st.session_state.dashboard_filter == "Intent Score Range":
            col_min, col_max = st.columns(2)
            min_intent = col_min.number_input("Min Intent Score", 0, 100, 0, key="dashboard_min_intent")
            max_intent = col_max.number_input("Max Intent Score", 0, 100, 100, key="dashboard_max_intent")
        elif st.session_state.dashboard_filter == "Date Range":
            col_min, col_max = st.columns(2)
            start_date = col_min.date_input("Start Date", key="dashboard_start_date")
            end_date = col_max.date_input("End Date", key="dashboard_end_date")
    with col2:
        filter_options = ["Filename", "Lead Score Range", "Intent Score Range", "Date Range"]
        st.selectbox("Filter by", filter_options, key="dashboard_filter_select", on_change=lambda: st.session_state.update({'dashboard_filter': st.session_state.dashboard_filter_select}))

    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Search", key="dashboard_search_button"):
            if st.session_state.dashboard_filter == "Filename":
                search_query = st.session_state.get('dashboard_search_query', '')
                if search_query:
                    st.session_state.dashboard_search_results = [
                        entry for entry in st.session_state.storage_metadata
                        if search_query.lower() in entry['audio_filename'].lower() or
                           search_query.lower() in entry['base_filename'].lower()
                    ]
                    st.session_state.dashboard_search_active = True
                else:
                    st.session_state.dashboard_search_results = None
                    st.session_state.dashboard_search_active = False
            elif st.session_state.dashboard_filter == "Lead Score Range":
                min_lead = st.session_state.get('dashboard_min_lead', 0)
                max_lead = st.session_state.get('dashboard_max_lead', 100)
                st.session_state.dashboard_search_results = [
                    entry for entry in st.session_state.storage_metadata
                    if entry.get('report_filename') and
                    re.search(r'_L(\d+)', entry['report_filename']) and
                    min_lead <= int(re.search(r'_L(\d+)', entry['report_filename']).group(1)) <= max_lead
                ]
                st.session_state.dashboard_search_active = True
            elif st.session_state.dashboard_filter == "Intent Score Range":
                min_intent = st.session_state.get('dashboard_min_intent', 0)
                max_intent = st.session_state.get('dashboard_max_intent', 100)
                st.session_state.dashboard_search_results = [
                    entry for entry in st.session_state.storage_metadata
                    if entry.get('report_filename') and
                    re.search(r'_I(\d+)', entry['report_filename']) and
                    min_intent <= int(re.search(r'_I(\d+)', entry['report_filename']).group(1)) <= max_intent
                ]
                st.session_state.dashboard_search_active = True
            elif st.session_state.dashboard_filter == "Date Range":
                start_date = st.session_state.get('dashboard_start_date')
                end_date = st.session_state.get('dashboard_end_date')
                if start_date and end_date and start_date <= end_date:
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.max.time())
                    st.session_state.dashboard_search_results = [
                        entry for entry in st.session_state.storage_metadata
                        if start_datetime <= datetime.fromisoformat(entry['timestamp']) <= end_datetime
                    ]
                    st.session_state.dashboard_search_active = True
                else:
                    st.error("End date must be after start date")
                    st.session_state.dashboard_search_results = None
                    st.session_state.dashboard_search_active = False
            st.session_state.dashboard_display_count = 10
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.get('dashboard_search_active') and st.session_state.get('dashboard_search_results'):
        st.markdown('<div class="clear-search-button">', unsafe_allow_html=True)
        if st.button("Clear Search", key="dashboard_clear_search"):
            st.session_state.dashboard_clear_search_trigger = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<h4 style="text-underline-offset: 4px; font-weight: 400;color:white;margin-bottom:3px;text-decoration: underline; margin-top:-15px;">Recent files</h4>', unsafe_allow_html=True)

    if st.session_state.storage_metadata:
        sort_options = ["Date (Newest First)", "Lead Score (High to Low)", "Intent Score (High to Low)"]
        sort_choice = st.selectbox("Sort by", sort_options, key="dashboard_sort_by")
        metadata_with_scores = []
        for entry in st.session_state.storage_metadata:
            lead_score = intent_score = 0
            if entry.get('report_filename'):
                lead_match = re.search(r'_L(\d+)', entry['report_filename'])
                intent_match = re.search(r'_I(\d+)', entry['report_filename'])
                if lead_match:
                    lead_score = int(lead_match.group(1))
                if intent_match:
                    intent_score = int(intent_match.group(1))
            metadata_with_scores.append({
                **entry,
                'lead_score': lead_score,
                'intent_score': intent_score,
                'emotion': entry.get('emotion', 'unknown')
            })

        if st.session_state.dashboard_search_active and st.session_state.search_results:
            filtered_metadata = [
                m for m in metadata_with_scores
                if any(
                    m['audio_filename'] == r['audio_filename'] and
                    m['timestamp'] == r['timestamp']
                    for r in st.session_state.search_results
                )
            ]
        else:
            filtered_metadata = metadata_with_scores

        if sort_choice == "Lead Score (High to Low)":
            sorted_metadata = sorted(
                filtered_metadata,
                key=lambda x: x['lead_score'],
                reverse=True
            )
        elif sort_choice == "Intent Score (High to Low)":
            sorted_metadata = sorted(
                filtered_metadata,
                key=lambda x: x['intent_score'],
                reverse=True
            )
        else:
            sorted_metadata = sorted(
                filtered_metadata,
                key=lambda x: x['timestamp'],
                reverse=True
            )

        display_count = st.session_state.dashboard_display_count
        total_files = len(sorted_metadata)
        for idx, entry in enumerate(sorted_metadata[:display_count]):
            expander_label = f"{entry['audio_filename']} (Stored: {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"
            with st.expander(expander_label):
                st.markdown('<div class="expander-content">', unsafe_allow_html=True)
                
                emotion = entry.get('emotion', 'unknown')
                emoji = get_emotion_emoji(emotion)
                st.markdown(f"**Detected Emotion**: {emotion.capitalize()} {emoji}")
                
                st.markdown(f"**Lead Score**: {entry['lead_score']}/100")
                st.markdown(f"**Intent Score**: {entry['intent_score']}/100")
                st.markdown("**Summary PDF**")
                if entry.get('report_path') and os.path.exists(entry['report_path']):
                    st.markdown('<div class="pdf-viewer-button">', unsafe_allow_html=True)
                    if st.button("View PDF", key=f"view_pdf_{idx}"):
                        with open(entry['report_path'], 'rb') as f:
                            pdf_data = f.read()
                        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600px" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    with open(entry['report_path'], 'rb') as f:
                        st.download_button(
                            label="Download PDF",
                            data=f.read(),
                            file_name=entry['report_filename'],
                            mime="application/pdf",
                            key=f"download_pdf_{idx}"
                        )
                else:
                    st.write("Not Available")
                st.markdown("**Audio File**")
                if os.path.exists(entry['audio_path']):
                    with open(entry['audio_path'], 'rb') as f:
                        st.audio(f.read(), format=f"audio/{os.path.splitext(entry['audio_filename'])[1][1:]}")
                st.markdown("**Raw Transcription**")
                raw_transcript = "Not Available"
                if entry.get('raw_transcript_path') and os.path.exists(entry['raw_transcript_path']):
                    try:
                        with open(entry['raw_transcript_path'], 'r', encoding='utf-8') as f:
                            raw_transcript = f.read()
                    except Exception as e:
                        logger.error(f"Failed to read raw transcript: {str(e)}")
                        st.warning(f"Failed to load raw transcript: {str(e)}")
                st.text_area("Raw Transcript", raw_transcript, height=100, disabled=True, key=f"raw_transcript_{idx}")
                st.markdown("**Translated Text (Malayalam)**")
                translated_text = "Not Available"
                if entry.get('translated_text_path') and os.path.exists(entry['translated_text_path']):
                    try:
                        with open(entry['translated_text_path'], 'r', encoding='utf-8') as f:
                            translated_text = f.read()
                    except Exception as e:
                        logger.error(f"Failed to read translated text: {str(e)}")
                        st.warning(f"Failed to load translated text: {str(e)}")
                st.text_area("Translated Text", translated_text, height=100, disabled=True, key=f"translated_text_{idx}")
                st.markdown("**Stored At**")
                st.write(datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S'))
                st.markdown("**Action**")
                st.markdown('<div class="dashboard-analysis-button">', unsafe_allow_html=True)
                if st.button("Start Analysis", key=f"analyze_{idx}"):
                    try:
                        st.session_state.selected_audio = {
                            'path': entry['audio_path'],
                            'name': entry['audio_filename'],
                            'custom_filename': entry['base_filename']
                        }
                        st.session_state.custom_filename = entry['base_filename']
                        st.session_state.process_triggered = True
                        logger.info(f"Triggered analysis for {entry['audio_filename']}")
                    except Exception as e:
                        logger.error(f"Failed to trigger analysis: {str(e)}")
                        st.error(f"Failed to trigger analysis: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if display_count < total_files:
            if st.button("Show More", key="show_more"):
                st.session_state.dashboard_display_count += 10
                st.rerun()
    else:
        st.info("No audio files stored yet. Upload an audio file to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    try:
        initialize_session_state()
        st.markdown("""
        <div class="logo-container">
            <div class="glow-logo">LSS</div>
            <div class="logo-text">Lead Scoring System</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.analysis_complete and st.session_state.results:
            display_results()
            display_dashboard()

        with st.sidebar:
            st.header("Upload an audio to analyse lead potential")
            st.markdown("Audio Upload")
            
            audio_file = st.file_uploader("Choose audio file (MP3, WAV, etc.)", type=['mp3', 'wav', 'aac', 'm4a', 'flac'])
            if audio_file:
                custom_filename = st.text_input("Custom filename (optional)")
                st.session_state.custom_filename = custom_filename.strip() if custom_filename else None
            if audio_file:
                if st.button("Start Analysis", key="sidebar_start_analysis"):
                    st.session_state.process_triggered = True
            if st.button("Reset & Cleanup", key="reset_cleanup"):
                reset_analysis()
            st.markdown("---")
            st.header("Settings")
            model_size = st.selectbox("Model Size", ["base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"], index=1)
            retention_period = st.selectbox(
                "Dashboard Retention Period",
                ["1 Month", "3 Months", "6 Months", "1 Year"],
                index=0,
                key="retention_period_select"
            )
            retention_map = {
                "1 Month": "1_month",
                "3 Months": "3_months",
                "6 Months": "6_months",
                "1 Year": "1_year"
            }
            st.session_state.retention_period = retention_map[retention_period]
            if st.session_state.retention_period:
                clean_old_files()
            st.markdown("---")
            st.header("Google Drive")
            drive_folder_id = st.text_input("Folder ID (optional)")
            enable_drive = st.checkbox("Enable Drive Upload", value=True)
            st.session_state.drive_folder_id = drive_folder_id
            st.session_state.enable_drive = enable_drive
            st.markdown("---")
            st.header("About")
            st.markdown("""
            Analyzes Malayalam audio to:
            - Transcribe to Malayalam
            - Translate to English
            - Detect sentiment and intent
            - Calculate lead scores
            - Export to Google Drive
            """)

        progress_bar = st.empty()
        status_text = st.empty()

        def update_progress(progress, status):
            try:
                progress_bar.progress(progress)
                status_text.text(status)
                if progress >= 100:
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
            except:
                pass

        if not st.session_state.analysis_complete or not st.session_state.results:
            display_dashboard()

        st.markdown("---")
        st.markdown('<h4 style="color:white;font-weight: 400;text-decoration: underline;text-underline-offset: 4px; margin-top:5px;">Search Audio From Drive</h4>', unsafe_allow_html=True)
        with st.expander("Search Google Drive Files"):
            col1, col2 = st.columns([3, 1])
            with col2:
                filter_options = ["Filename", "Lead Score Range", "Intent Score Range", "Date Range"]
                gdrive_filter = st.selectbox("Filter by", filter_options, key="gdrive_filter")
            with col1:
                if gdrive_filter == "Filename":
                    search_query = st.text_input("Search by Filename", key="gdrive_search_query")
                    if st.button("Search", key="gdrive_search_button_filename"):
                        if search_query:
                            with st.spinner("Searching..."):
                                files = search_gdrive_files(query=search_query)
                                st.session_state.gdrive_search_results = files
                                st.session_state.gdrive_search_active = True
                        else:
                            st.session_state.gdrive_search_results = None
                            st.session_state.gdrive_search_active = False
                        st.session_state.gdrive_display_count = 10
                        st.rerun()
                elif gdrive_filter == "Lead Score Range":
                    col_min, col_max = st.columns(2)
                    min_lead = col_min.number_input("Min Lead Score", 0, 100, 0, key="gdrive_min_lead")
                    max_lead = col_max.number_input("Max Lead Score", 0, 100, 100, key="gdrive_max_lead")
                    if st.button("Search", key="gdrive_search_lead"):
                        with st.spinner("Searching..."):
                            files = search_gdrive_files(min_lead=min_lead, max_lead=max_lead)
                            st.session_state.gdrive_search_results = files
                            st.session_state.gdrive_search_active = True
                            st.session_state.gdrive_display_count = 10
                            st.rerun()
                elif gdrive_filter == "Intent Score Range":
                    col_min, col_max = st.columns(2)
                    min_intent = col_min.number_input("Min Intent Score", 0, 100, 0, key="gdrive_min_intent")
                    max_intent = col_max.number_input("Max Intent Score", 0, 100, 100, key="gdrive_max_intent")
                    if st.button("Search", key="gdrive_search_intent"):
                        with st.spinner("Searching..."):
                            files = search_gdrive_files(min_intent=min_intent, max_intent=max_intent)
                            st.session_state.gdrive_search_results = files
                            st.session_state.gdrive_search_active = True
                            st.session_state.gdrive_display_count = 10
                            st.rerun()
                elif gdrive_filter == "Date Range":
                    col_min, col_max = st.columns(2)
                    start_date = col_min.date_input("Start Date", key="gdrive_start_date")
                    end_date = col_max.date_input("End Date", key="gdrive_end_date")
                    if st.button("Search", key="gdrive_search_date"):
                        if start_date <= end_date:
                            with st.spinner("Searching..."):
                                start_datetime = datetime.combine(start_date, datetime.min.time()).isoformat() + 'Z'
                                end_datetime = datetime.combine(end_date, datetime.max.time()).isoformat() + 'Z'
                                files = search_gdrive_files(start_date=start_datetime, end_date=end_datetime)
                                st.session_state.gdrive_search_results = files
                                st.session_state.gdrive_search_active = True
                        else:
                            st.error("End date must be after start date")
                            st.session_state.gdrive_search_results = None
                            st.session_state.gdrive_search_active = False
                        st.session_state.gdrive_display_count = 10
                        st.rerun()
                if st.button("Clear Search", key="gdrive_clear_search"):
                    st.session_state.gdrive_search_results = None
                    st.session_state.gdrive_search_active = False
                    st.session_state.gdrive_display_count = 10
                    st.rerun()

        sort_options = ["Date (Newest First)", "Lead Score (High to Low)", "Intent Score (High to Low)"]
        sort_choice = st.selectbox("Sort by", sort_options, key="gdrive_sort_by")
        files_with_scores = []
        files = st.session_state.gdrive_search_results if st.session_state.gdrive_search_active else search_gdrive_files()
        for file in files:
            lead_score = intent_score = 0
            lead_match = re.search(r'_L(\d+)', file['name'])
            intent_match = re.search(r'_I(\d+)', file['name'])
            if lead_match:
                lead_score = int(lead_match.group(1))
            if intent_match:
                intent_score = int(intent_match.group(1))
            files_with_scores.append({
                **file,
                'lead_score': lead_score,
                'intent_score': intent_score,
                'emotion': file.get('emotion', 'unknown')
            })

        if sort_choice == "Lead Score (High to Low)":
            sorted_files = sorted(
                files_with_scores,
                key=lambda x: x['lead_score'],
                reverse=True
            )
        elif sort_choice == "Intent Score (High to Low)":
            sorted_files = sorted(
                files_with_scores,
                key=lambda x: x['intent_score'],
                reverse=True
            )
        else:
            sorted_files = sorted(
                files_with_scores,
                key=lambda x: x['createdTime'],
                reverse=True
            )

        display_gdrive_search_results(sorted_files)

        if audio_file and st.session_state.get('process_triggered') and not st.session_state.get('analysis_complete'):
            with st.spinner("Processing audio..."):
                try:
                    process_audio(audio_file, model_size, update_progress)
                    if st.session_state.get('processing_error'):
                        st.error(f"Error: {st.session_state['processing_error']}")
                    else:
                        st.rerun()
                except Exception as e:
                    logger.error(f"Audio processing failed: {str(e)}")
                    st.error(f"Audio processing failed: {str(e)}")

        if st.session_state.get('selected_audio') and st.session_state.get('process_triggered') and not st.session_state.get('analysis_complete'):
            with st.spinner("Processing stored audio..."):
                try:
                    selected = st.session_state.selected_audio
                    with open(selected['path'], 'rb') as f:
                        audio_buffer = BytesIO(f.read())
                    audio_file = type('UploadedFile', (), {
                        'name': selected['name'],
                        'read': lambda self: audio_buffer.read(),
                        'seek': lambda self, pos: audio_buffer.seek(pos)
                    })()
                    st.session_state.custom_filename = selected['custom_filename']
                    process_audio(audio_file, model_size, update_progress)
                    if st.session_state.get('processing_error'):
                        st.error(f"Error: {st.session_state['processing_error']}")
                    else:
                        st.rerun()
                except Exception as e:
                    logger.error(f"Stored audio processing failed: {str(e)}")
                    st.error(f"Stored audio processing failed: {str(e)}")
    except Exception as e:
        logger.error(f"Main function failed: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()