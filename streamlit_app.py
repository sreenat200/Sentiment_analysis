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
    :root {
        --text-color: #ffffff;
        --background-color: #202427;
    }

    @media (prefers-color-scheme: light) {
        :root {
            --text-color: #000000;
            --background-color: #ffffff;
        }
        .stApp h1, .dashboard-header, .emotion-block, .expander-content, .score-card, .emotion-text {
            color: #E0E0E0 !important;

        }
        .high-score, .low-score {
            color: #ffffff !important;
        }
        .medium-score {
            color: #2c3e50 !important;
        }
    }

    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        margin-top: 0;
        padding-top: 0;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 22px;
        margin-top: -27px;
    }
    .logo-text {
        font-size: 24px;
        font-weight: bold;
        font-family: 'Playfair Display', serif;
        color: var(--text-color);
    }
    .glow-logo {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #d3d3d3, #ffffff);
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
        font-size: 2rem !important;
        animation: slideUpFade 1.5s ease-out;
        margin-top: 0;
    }
    @keyframes slideUpFade {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .stText, .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput, .stCheckbox {
        color: #E0E0E0 !important;
    }
    .stButton>button {
        background-color: #E81828;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        transition: transform 0.2s, background-color 0.2s;
        width: auto;
        max-width: 180px;
        display: inline-block;
        text-align: center;
        font-size: 10px;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: transparent;
        color: #ffffff;
    }
    .clear-search-button>button {
        background-color: transparent;
        color: var(--text-color, #D22B2B);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        padding: 8px 16px;
        transition: transform 0.2s, background-color 0.2s, border-color 0.2s;
        width: auto;
        max-width: 180px;
        display: inline-block;
        text-align: center;
        font-size: 14px;
    }
    .clear-search-button>button:hover {
        transform: scale(1.05);
        background-color: rgba(255, 255, 255, 0.1);
        border-color: #ffffff;
        color: var(--text-color, #B02424);
    }

    
    .use-selected-dir-button>button {
        background-color: transparent;
        color: var(--text-color, #ffffff);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 8px;
        padding: 8px 16px;
        transition: transform 0.2s, background-color 0.2s, border-color 0.2s;
        width: auto;
        max-width: 180px;
        display: inline-block;
        text-align: center;
        font-size: 14px;
    }
    .use-selected-dir-button>button:hover {
        transform: scale(1.05);
        background-color: rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.8);

        border-color: #ffffff;
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
    }
    .medium-score {
        background-color: #f1c40f;
    }
    .low-score {
        background-color: #e74c3c;
    }
    .lead-score, .intent-score {
        font-size: 2rem;
        font-weight: bold;
    }
    .st.spinner{
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.6), rgba(60, 60, 60, 0.4));
        margin-bottom: -30px;    

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
        margin-top: 20px;
    }
    .dashboard-header {
        background-color: transparent;
        padding: 6px;
        margin-bottom: 4px;
        text-color: #E0E0E0;
        text-align: left;
        font-size: 1.9rem;
        font-weight: bold;
        font-family: 'Merriweather', serif;
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
        border: 1px solid rgba(255, 255, 255, 0.8);
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
        border: 1px solid rgba(255, 255, 255, 0.8);
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
        .stButton>button, .clear-search-button>button, .use-selected-dir-button>button {
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

def select_folder():
    try:
        from tkinter import Tk
        from tkinter.filedialog import askdirectory
        root = Tk()
        root.withdraw()
        folder = askdirectory(title="Select Storage Directory")
        root.destroy()
        if folder and os.path.isdir(folder):
            return folder
        else:
            logger.warning("No directory selected or invalid directory.")
            return None
    except Exception as e:
        logger.error(f"Folder selection failed: {str(e)}")
        return None

def load_agent_config():
    config_path = os.path.join(os.getcwd(), 'app_storage', 'agent_config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f).get('agents', [])
        except Exception as e:
            logger.error(f"Failed to load agent config: {str(e)}")
    return []

def save_agent_config(agents):
    config_path = os.path.join(os.getcwd(), 'app_storage', 'agent_config.json')
    try:
        with open(config_path, 'w') as f:
            json.dump({'agents': agents}, f, indent=2)
        logger.info(f"Saved agent config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save agent config: {str(e)}")

def add_agent(agent_name):
    agent_name = "".join(c for c in agent_name if c.isalnum() or c in ('-', '_')).rstrip()
    if not agent_name:
        return False, "Invalid agent name. Use alphanumeric characters, hyphens, or underscores."
    agents = load_agent_config()
    if agent_name in agents:
        return False, "Agent name already exists."
    agents.append(agent_name)
    save_agent_config(agents)
    return True, "Agent added successfully."

def delete_agent(agent_name):
    agents = load_agent_config()
    if agent_name in agents:
        agents.remove(agent_name)
        save_agent_config(agents)
        return True, "Agent deleted successfully."
    return False, "Agent not found."

def load_storage_config():
    config_path = os.path.join(os.getcwd(), 'app_storage', 'storage_config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    default_storage_dir = os.path.join(os.getcwd(), 'app_storage', 'user_dirs', 'default')
    default_config = {'recent_dirs': [], 'last_selected_dir': default_storage_dir, 'retention_period': '1_month'}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                config['recent_dirs'] = [d for d in config.get('recent_dirs', []) if os.path.isdir(d)]
                if 'last_selected_dir' not in config or not os.path.isdir(config.get('last_selected_dir')):
                    config['last_selected_dir'] = default_storage_dir
                if 'retention_period' not in config:
                    config['retention_period'] = '1_month'
                return config
        except Exception as e:
            logger.error(f"Failed to load storage config: {str(e)}")
    return default_config

def save_storage_config(config):
    config_path = os.path.join(os.getcwd(), 'app_storage', 'storage_config.json')
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved storage config to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save storage config: {str(e)}")

def update_recent_dirs(new_dir):
    config = load_storage_config()
    recent_dirs = config.get('recent_dirs', [])
    if new_dir in recent_dirs:
        recent_dirs.remove(new_dir)
    recent_dirs.insert(0, new_dir)
    config['recent_dirs'] = recent_dirs[:5]
    config['last_selected_dir'] = new_dir
    save_storage_config(config)

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
        retention_period = st.session_state.get('retention_period', '1_month')
        if retention_period == 'disable':
            logger.info("Retention period set to 'disable'. Skipping cleanup of old files.")
            return
        retention_map = {
            "1_month": 30,
            "3_months": 90,
            "6_months": 180,
            "1_year": 365
        }
        retention_days = retention_map.get(retention_period, 30)
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
    base_storage_path = os.path.join(os.getcwd(), 'app_storage', 'user_dirs')
    os.makedirs(base_storage_path, exist_ok=True)
    
    config = load_storage_config()
    default_storage_dir = os.path.join(base_storage_path, 'default')
    storage_dir = config.get('last_selected_dir', default_storage_dir)
    if not storage_dir or not os.path.isdir(storage_dir):
        storage_dir = default_storage_dir
        config['last_selected_dir'] = storage_dir
        save_storage_config(config)
    
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
        'process_triggered': False,
        'custom_filename': None,
        'audio_buffer': None,
        'selected_audio': None,
        'storage_metadata': [],
        'storage_dir': storage_dir,
        'dashboard_display_count': 10,
        'dashboard_search_results': None,
        'dashboard_search_active': False,
        'retention_period': config.get('retention_period', '1_month'),
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
        storage_dir = st.session_state.storage_dir
        metadata_path = os.path.join(storage_dir, 'storage_metadata.json')
        metadata = st.session_state.storage_metadata
        original_filename = os.path.splitext(audio_file.name)[0]
        base_filename = custom_filename or original_filename
        base_filename = "".join(c for c in base_filename if c.isalnum() or c in ('-', '_')).rstrip() or original_filename
        agent_name = st.session_state.get('selected_agent', 'NoAgent')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_ext = os.path.splitext(audio_file.name)[1]
        audio_filename = f"{agent_name}_{base_filename}_{timestamp}{audio_ext}"
        audio_path = os.path.join(storage_dir, audio_filename)
        with open(audio_path, 'wb') as f:
            f.write(audio_buffer.getvalue())
        report_path = None
        report_filename = None
        if pdf_path and os.path.exists(pdf_path):
            report_filename = f"{agent_name}_{base_filename}_{timestamp}.pdf"
            report_path = os.path.join(storage_dir, report_filename)
            shutil.copy(pdf_path, report_path)
        raw_transcript_path = os.path.join(storage_dir, f"{agent_name}_{base_filename}_{timestamp}_raw.txt")
        translated_text_path = os.path.join(storage_dir, f"{agent_name}_{base_filename}_{timestamp}_translated.txt")
        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
            f.write(raw_transcription or "Not Available")
        with open(translated_text_path, 'w', encoding='utf-8') as f:
            f.write(ml_translation or "Not Available")
        metadata_entry = {
    'audio_filename': audio_filename,  # e.g., "AgentName_BaseFilename_20230607202230.mp3"
    'audio_path': audio_path,  # Full path to stored audio file
    'report_filename': report_filename,  # e.g., "AgentName_BaseFilename_20230607202230.pdf"
    'report_path': report_path,  # Full path to stored PDF report or None
    'raw_transcript_path': raw_transcript_path,  # Path to raw transcription text
    'translated_text_path': translated_text_path,  # Path to translated Malayalam text
    'timestamp': datetime.now().isoformat(),  # ISO timestamp, e.g., "2025-06-21T18:57:00.123456"
    'base_filename': base_filename,  # Base filename (custom or derived from audio file)
    'emotion': st.session_state.get('results', {}).get('audio_metadata', {}).get('emotion', 'unknown'),  # e.g., "happy", "neutral"
    'agent_name': agent_name,  # e.g., "Agent1" or "NoAgent"
    'lead_score': st.session_state.get('results', {}).get('lead_score', 0),  # Integer from 0 to 100
    'intent_score': st.session_state.get('results', {}).get('intent_score', 0)  # Integer from 0 to 100
}
        metadata.append(metadata_entry)
        st.session_state.storage_metadata = metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Stored metadata for {audio_filename}. Total entries: {len(metadata)}")
        update_recent_dirs(storage_dir)
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
        agent_name = st.session_state.get('selected_agent', 'NoAgent')
        temp_dir = os.path.join(tempfile.gettempdir(), f"zip_staging_{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file {audio_path} not found. Attempting to use original audio from session state.")
            audio_path = st.session_state.get('audio_path', audio_path)
            if not os.path.exists(audio_path):
                logger.error(f"Original audio file {audio_path} also not found.")
                raise RuntimeError(f"Audio file not found: {audio_path}. Ensure the file was not deleted.")
        audio_ext = os.path.splitext(audio_path)[1]
        audio_filename = f"{agent_name}_audio{audio_ext}"
        audio_dest = os.path.join(temp_dir, audio_filename)
        with open(audio_path, 'rb') as src, open(audio_dest, 'wb') as dst:
            dst.write(src.read())
        raw_transcription_path = os.path.join(temp_dir, f"{agent_name}_transcript_raw.txt")
        with open(raw_transcription_path, 'w', encoding='utf-8') as f:
            f.write(raw_transcription)
        ml_translation_path = os.path.join(temp_dir, f"{agent_name}_transcript_translated.txt")
        with open(ml_translation_path, 'w', encoding='utf-8') as f:
            f.write(ml_translation)
        en_csv_path = save_analysis_to_csv(en_analysis, os.path.join(temp_dir, f"{agent_name}_{base_filename}_en"))
        ml_csv_path = save_analysis_to_csv(ml_analysis, os.path.join(temp_dir, f"{agent_name}_{base_filename}_ml"))
        comparison_csv_path = save_analysis_to_csv(comparison, os.path.join(temp_dir, f"{agent_name}_{base_filename}_comparison"))
        summary_pdf_path = os.path.join(temp_dir, f"{agent_name}_summary_report.pdf")
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
            "agent_name": agent_name,
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
        summary_json_path = os.path.join(temp_dir, f"{agent_name}_summary_analyzed.json")
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(tempfile.gettempdir(), f"{agent_name}_{base_filename}_L{lead_score}_I{intent_score}_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(audio_dest, audio_filename)
            zipf.write(raw_transcription_path, f"{agent_name}_transcript_raw.txt")
            zipf.write(ml_translation_path, f"{agent_name}_transcript_translated.txt")
            zipf.write(en_csv_path, f"{agent_name}_{base_filename}_en.csv")
            zipf.write(ml_csv_path, f"{agent_name}_{base_filename}_ml.csv")
            zipf.write(comparison_csv_path, f"{agent_name}_{base_filename}_comparison.csv")
            zipf.write(summary_json_path, f"{agent_name}_summary_analyzed.json")
            zipf.write(summary_pdf_path, f"{agent_name}_summary_report.pdf")
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

        steps = 10  # Adjusted for clearer progress increments
        step_increment = 100 / steps
        current_step = 0

        def update_step(step, status):
            nonlocal current_step
            current_step = min(step, steps)  # Cap at max steps
            if progress_callback:
                progress_callback(int(current_step * step_increment), status)
                logger.info(f"Progress: {int(current_step * step_increment)}%, Status: {status}")
                # Small delay to ensure Streamlit renders the update
                import time
                time.sleep(0.1)

        update_step(0, "Saving uploaded file...")

        # Save uploaded file
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

        update_step(1, "Converting audio format...")

        # Convert audio
        try:
            wav_path = convert_to_whisper_format(audio_path)
            if wav_path != audio_path:
                st.session_state.temp_files.append(wav_path)
                st.session_state.wav_path = wav_path
            else:
                st.session_state.wav_path = audio_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            update_step(current_step, f"Error: Audio conversion failed: {str(e)}")
            raise RuntimeError(f"Audio conversion failed: {str(e)}. Try a different audio format or ensure ffmpeg is installed.")

        update_step(2, "Loading transcription model...")

        # Load transcription model
        if st.session_state.whisper_model is None:
            import torch
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            st.session_state.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)

        update_step(3, "Transcribing audio...")

        # Transcribe audio
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

        update_step(4, "Translating to English...")

        # Translate to Malayalam
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
                ml_translations.extend([""] for _ in range(len(sentences) - len(ml_translations)))
            elif len(ml_translations) > len(sentences):
                ml_translations = ml_translations[:len(sentences)]
            logger.debug(f"Aligned: English sentences={len(sentences)}, Malayalam={len(ml_translations)}")

            transcriber.cleanup()
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            update_step(current_step, f"Warning: Translation failed: {str(e)}")
            ml_translation = ""
            sentences = split_into_sentences(raw_transcription, "en")
            ml_translations = ["" for _ in sentences]
            st.session_state.processing_error = f"Translation failed: {str(e)}. Proceeding with transcription."

        update_step(5, "Analyzing English content...")

        # Analyze English content
        en_analysis = []
        for sentence in sentences:
            analysis = analyze_text(sentence, "en")
            if analysis:
                en_analysis.extend(analysis)

        update_step(6, "Analyzing Malayalam content...")

        # Analyze Malayalam content
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

        update_step(7, "Calculating lead scores...")

        # Calculate lead scores
        positive_keywords_en = ["share", "interested", "send whatsapp", "don't have any other", "got it",
                               "acknowledge", "noted", "please send", "sent details", "agreed"]
        positive_keywords_ml = ["പങ്കിടുക", "താൽപ്പര്യം", "ശരി", "താല്പര്യമുണ്ട്", "തിരയുന്നു", "ഇഷ്ടമുണ്ട്",
                               "വാട്സാപ്പിൽ അയക്കൂ", "വാട്സാപ്പ്", "വാട്ട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു",
                               "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", "ഓക്കെ", "യെസ്", "അക്ക്നലഡ്ജ്",
                               "ക്ലിയർ", "തയാറാണ്", "അറിയിപ്പ് ലഭിച്ചു", "വാട്ട്സാപ്പിലേ",
                               "ഞാൻ അതിനായി നോക്കിയിരുന്നു"]
        negative_keywords_en = ["not interested", "not looking", "can't", "don't have any other", "won't", "don't like",
                               "not now", "later", "not suitable", "decline"]
        negative_keywords_ml = ["താല്പര്യമില്ല", "നോക്കുന്നില്ല", "ഇല്ല", "വേണ്ട", "മറ്റ് ജോലികൾ ചെയ്യാനില്ലേ?", "സാധ്യമല്ല", "ഇഷ്ടമല്ല"]
        positive_extra_points = 8
        negative_extra_points = -8

        last_en_sentences = [item["text"].lower() for item in en_analysis[-5:]] if en_analysis else []
        last_ml_sentences = [item["text"] for item in ml_analysis[-5:]] if ml_analysis else []

        positive_matches = []
        negative_matches = []
        extra_points = 0

        for i, sentence in enumerate(last_en_sentences):
            for keyword in positive_keywords_en:
                if keyword in sentence:
                    positive_matches.append({
                        "language": "English",
                        "sentence_index": len(en_analysis) - 5 + i,
                        "keyword": keyword,
                        "points": positive_extra_points
                    })
                    extra_points += positive_extra_points
            for keyword in negative_keywords_en:
                if keyword in sentence:
                    negative_matches.append({
                        "language": "English",
                        "sentence_index": len(en_analysis) - 5 + i,
                        "keyword": keyword,
                        "points": negative_extra_points
                    })
                    extra_points += negative_extra_points

        for i, sentence in enumerate(last_ml_sentences):
            for keyword in positive_keywords_ml:
                if keyword in sentence:
                    positive_matches.append({
                        "language": "Malayalam",
                        "sentence_index": len(ml_analysis) - 5 + i,
                        "keyword": keyword,
                        "points": positive_extra_points
                    })
                    extra_points += positive_extra_points
            for keyword in negative_keywords_ml:
                if keyword in sentence:
                    negative_matches.append({
                        "language": "Malayalam",
                        "sentence_index": len(ml_analysis) - 5 + i,
                        "keyword": keyword,
                        "points": negative_extra_points
                    })
                    extra_points += negative_extra_points

        en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
        ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
        combined_avg = (en_avg_score + ml_avg_score) / 2 if ml_analysis else en_avg_score
        base_lead_score = int(combined_avg * 100)
        lead_score = max(0, min(base_lead_score + extra_points, 100))

        positive_intents = sum(1 for item in en_analysis if item["intent"] in ["Strong_interest", "Fee_query", "Moderate_interest", "Confirmation"])
        intent_score = int((positive_intents / len(en_analysis)) * 100) if en_analysis else 0

        update_step(8, "Finalizing results...")

        # Finalize results
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
            },
            "lead_score_breakdown": {
                "base_lead_score": base_lead_score,
                "positive_matches": positive_matches,
                "negative_matches": negative_matches,
                "total_extra_points": extra_points
            },
            "agent_name": st.session_state.get('selected_agent', 'NoAgent')
        }

        try:
            if 'transcriber' in st.session_state:
                emotion = st.session_state.transcriber.analyze_emotion(wav_path)
                results["audio_metadata"]["emotion"] = emotion
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {str(e)}. Using default 'unknown' emotion.")

        st.session_state.results = results

        update_step(9, "Storing audio and report...")

        # Store audio and report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = st.session_state.get('selected_agent', 'NoAgent')
        final_filename = f"{agent_name}_{base_filename}_L{lead_score}_I{intent_score}_{timestamp}"
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

        update_step(10, "Analysis complete!")

        st.session_state.analysis_complete = True
        logger.info("Audio processing completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        st.session_state.processing_error = f"Processing failed: {str(e)}. Try a different audio file or check dependencies."
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        raise

def display_results():
    try:
        results = st.session_state.results
        if not results:
            st.error("No results available to display")
            return

        emotion = results.get('audio_metadata', {}).get('emotion', 'unknown')
        emoji = get_emotion_emoji(emotion)
        sentiments = [item["sentiment"] for item in results.get("en_analysis", []) if item.get("sentiment")]
        sentiment = pd.Series(sentiments).mode()[0] if sentiments else "unknown"
        
        agent_name = results.get('agent_name', 'Unknown')
        st.markdown(f"""
        <div class="header-container">
            <div class="dashboard-header">Analysis Results - Agent: {agent_name}</div>
            <div class="emotion-block">Detected Emotion: {emotion.capitalize()} {emoji}</div>
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
                csv_path = save_analysis_to_csv(results["ml_analysis"], f"{results['agent_name']}_{results['original_filename']}_ml")
                if csv_path:
                    with open(csv_path, 'rb') as f:
                        st.download_button("Download Malayalam CSV", f, file_name=os.path.basename(csv_path))
                    st.session_state.temp_files.append(csv_path)
            st.subheader("English Analysis")
            st.dataframe(pd.DataFrame(results["en_analysis"]), use_container_width=True)
            with st.expander("Export English Analysis"):
                csv_path = save_analysis_to_csv(results["en_analysis"], f"{results['agent_name']}_{results['original_filename']}_en")
                if csv_path:
                    with open(csv_path, "rb") as f:
                        st.download_button("Download English CSV", f, file_name=os.path.basename(csv_path))
                    st.session_state.temp_files.append(csv_path)
            st.subheader("Comparison Analysis")
            st.dataframe(pd.DataFrame(results["comparison"]), use_container_width=True)
            with st.expander("Export Comparison Analysis"):
                csv_path = save_analysis_to_csv(results["comparison"], f"{results['agent_name']}_{results['original_filename']}_comparison")
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
        agent_name = results.get('agent_name', 'NoAgent')
        final_filename = f"{agent_name}_{results['original_filename']}_L{results['lead_score']}_I{results['intent_score']}_{timestamp}"
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

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def display_dashboard():
    try:
        logger.info("Starting display_dashboard function")

        # Initialize session state defaults
        if 'dashboard_display_count' not in st.session_state:
            st.session_state.dashboard_display_count = 10
        if 'dashboard_search_results' not in st.session_state:
            st.session_state.dashboard_search_results = None
        if 'dashboard_search_active' not in st.session_state:
            st.session_state.dashboard_search_active = False
        if 'dashboard_clear_search_trigger' not in st.session_state:
            st.session_state.dashboard_clear_search_trigger = False
        if 'dashboard_filter' not in st.session_state:
            st.session_state.dashboard_filter = 'Filename'

        # Check if clear search was triggered
        if st.session_state.get('dashboard_clear_search_trigger', False):
            logger.info("Clearing search results")
            st.session_state.dashboard_search_results = None
            st.session_state.dashboard_search_active = False
            st.session_state.dashboard_display_count = 10
            st.session_state.dashboard_clear_search_trigger = False

        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)

        # Search and Filter Section
        col1, col2 = st.columns([3, 1])
        with col1:
            try:
                if st.session_state.dashboard_filter == "Filename":
                    search_query = st.text_input("Search by Filename", key="dashboard_search_query")
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
                elif st.session_state.dashboard_filter == "Agent Name":
                    agents = load_agent_config()
                    if not agents:
                        logger.warning("No agents found in configuration")
                        st.warning("No agents available for filtering. Please create an agent in Agent Management.")
                    else:
                        selected_agent = st.selectbox("Select Agent", agents, key="dashboard_search_agent")
            except Exception as e:
                logger.error(f"Error in filter input section: {str(e)}", exc_info=True)
                st.error(f"Failed to render filter inputs: {str(e)}")
                return

        with col2:
            filter_options = ["Filename", "Lead Score Range", "Intent Score Range", "Date Range", "Agent Name"]
            st.selectbox("Filter by", filter_options, key="dashboard_filter_select",
                         on_change=lambda: st.session_state.update({'dashboard_filter': st.session_state.dashboard_filter_select}))

        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("Search", key="dashboard_search_button"):
                try:
                    logger.info(f"Executing search with filter: {st.session_state.dashboard_filter}")
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
                            if min_lead <= entry.get('lead_score', 0) <= max_lead
                        ]
                        st.session_state.dashboard_search_active = True
                    elif st.session_state.dashboard_filter == "Intent Score Range":
                        min_intent = st.session_state.get('dashboard_min_intent', 0)
                        max_intent = st.session_state.get('dashboard_max_intent', 100)
                        st.session_state.dashboard_search_results = [
                            entry for entry in st.session_state.storage_metadata
                            if min_intent <= entry.get('intent_score', 0) <= max_intent
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
                    elif st.session_state.dashboard_filter == "Agent Name":
                        selected_agent = st.session_state.get('dashboard_search_agent')
                        if selected_agent:
                            st.session_state.dashboard_search_results = [
                                entry for entry in st.session_state.storage_metadata
                                if entry.get('agent_name') == selected_agent
                            ]
                            st.session_state.dashboard_search_active = True
                        else:
                            st.session_state.dashboard_search_results = None
                            st.session_state.dashboard_search_active = False
                    st.session_state.dashboard_display_count = 10
                    logger.info(f"Search completed. Found {len(st.session_state.dashboard_search_results or [])} results")
                except Exception as e:
                    logger.error(f"Search failed: {str(e)}", exc_info=True)
                    st.error(f"Search failed: {str(e)}")
                    return

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get('dashboard_search_active') and st.session_state.get('dashboard_search_results'):
            st.markdown('<div class="clear-search-button">', unsafe_allow_html=True)
            if st.button("Clear Search", key="dashboard_clear_search"):
                logger.info("Clear search button clicked")
                st.session_state.dashboard_search_results = None
                st.session_state.dashboard_search_active = False
                st.session_state.dashboard_display_count = 10
                st.session_state.dashboard_clear_search_trigger = False
            st.markdown('</div>', unsafe_allow_html=True)

        # Create tabs for Recent Files and Agent Performance
        tab1, tab2 = st.tabs(["Recent Files", "Agent Performance"])

        with tab1:
            try:
                if not st.session_state.storage_metadata:
                    logger.info("No storage metadata available")
                    st.info("No audio files stored yet. Upload an audio file to begin.")
                else:
                    sort_options = ["Date (Newest First)", "Lead Score (High to Low)", "Intent Score (High to Low)", "Agent Name"]
                    sort_choice = st.selectbox("Sort by", sort_options, key="dashboard_sort_by")
                    metadata_with_scores = []
                    for entry in st.session_state.storage_metadata:
                        try:
                            lead_score = entry.get('lead_score', 0)
                            intent_score = entry.get('intent_score', 0)
                            if lead_score == 0 or intent_score == 0:
                                if entry.get('report_filename'):
                                    lead_match = re.search(r'_L(\d+)', entry['report_filename'])
                                    intent_match = re.search(r'_I(\d+)', entry['report_filename'])
                                    if lead_match:
                                        lead_score = int(lead_match.group(1))
                                    if intent_match:
                                        intent_score = int(intent_match.group(1))
                                    if lead_score == 0 or intent_score == 0:
                                        logger.warning(f"Failed to extract scores from filename: {entry['report_filename']}")
                            metadata_with_scores.append({
                                **entry,
                                'lead_score': lead_score,
                                'intent_score': intent_score,
                                'emotion': entry.get('emotion', 'unknown'),
                                'agent_name': entry.get('agent_name', 'Unknown')
                            })
                        except Exception as e:
                            logger.error(f"Error processing metadata entry {entry.get('audio_filename', 'unknown')}: {str(e)}")
                            continue

                    if st.session_state.dashboard_search_active and st.session_state.dashboard_search_results:
                        filtered_metadata = [
                            m for m in metadata_with_scores
                            if any(
                                m['audio_filename'] == r.get('audio_filename') and
                                m['timestamp'] == r.get('timestamp')
                                for r in st.session_state.dashboard_search_results
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
                    elif sort_choice == "Agent Name":
                        sorted_metadata = sorted(
                            filtered_metadata,
                            key=lambda x: x['agent_name']
                        )
                    else:
                        sorted_metadata = sorted(
                            filtered_metadata,
                            key=lambda x: x['timestamp'],
                            reverse=True
                        )

                    display_count = st.session_state.dashboard_display_count
                    total_files = len(sorted_metadata)
                    logger.info(f"Displaying {min(display_count, total_files)} of {total_files} metadata entries")
                    for idx, entry in enumerate(sorted_metadata[:display_count]):
                        try:
                            expander_label = f"{entry['audio_filename']} (Agent: {entry['agent_name']}, Stored: {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"
                            with st.expander(expander_label):
                                st.markdown('<div class="expander-content">', unsafe_allow_html=True)

                                emotion = entry.get('emotion', 'unknown')
                                emoji = get_emotion_emoji(emotion)
                                st.markdown(f"**Detected Emotion**: {emotion.capitalize()} {emoji}")

                                lead_score = entry['lead_score']
                                intent_score = entry['intent_score']
                                if lead_score == 0:
                                    st.markdown("**Lead Score**: Not Available (Analysis may be incomplete)")
                                    logger.warning(f"Lead score missing for {entry['audio_filename']}")
                                else:
                                    st.markdown(f"**Lead Score**: {lead_score}/100")
                                if intent_score == 0:
                                    st.markdown("**Intent Score**: Not Available (Analysis may be incomplete)")
                                    logger.warning(f"Intent score missing for {entry['audio_filename']}")
                                else:
                                    st.markdown(f"**Intent Score**: {intent_score}/100")

                                st.markdown(f"**Agent**: {entry['agent_name']}")
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
                                    logger.warning(f"PDF not found for {entry['audio_filename']}")
                                st.markdown("**Audio File**")
                                if os.path.exists(entry['audio_path']):
                                    with open(entry['audio_path'], 'rb') as f:
                                        st.audio(f.read(), format=f"audio/{os.path.splitext(entry['audio_filename'])[1][1:]}")
                                else:
                                    st.write("Not Available")
                                    logger.warning(f"Audio file not found: {entry['audio_path']}")
                                st.markdown("**Raw Transcription**")
                                raw_transcript = "Not Available"
                                if entry.get('raw_transcript_path') and os.path.exists(entry['raw_transcript_path']):
                                    try:
                                        with open(entry['raw_transcript_path'], 'r', encoding='utf-8') as f:
                                            raw_transcript = f.read()
                                    except Exception as e:
                                        logger.error(f"Failed to read raw transcript for {entry['audio_filename']}: {str(e)}")
                                        st.warning(f"Failed to load raw transcript: {str(e)}")
                                st.text_area("Raw Transcript", raw_transcript, height=100, disabled=True, key=f"raw_transcript_{idx}")
                                st.markdown("**Translated Text (Malayalam)**")
                                translated_text = "Not Available"
                                if entry.get('translated_text_path') and os.path.exists(entry['translated_text_path']):
                                    try:
                                        with open(entry['translated_text_path'], 'r', encoding='utf-8') as f:
                                            translated_text = f.read()
                                    except Exception as e:
                                        logger.error(f"Failed to read translated text for {entry['audio_filename']}: {str(e)}")
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
                                            'custom_filename': entry['base_filename'],
                                            'agent_name': entry['agent_name']
                                        }
                                        st.session_state.custom_filename = entry['base_filename']
                                        st.session_state.selected_agent = entry['agent_name']
                                        st.session_state.process_triggered = True
                                        logger.info(f"Triggered analysis for {entry['audio_filename']}")
                                    except Exception as e:
                                        logger.error(f"Failed to trigger analysis for {entry['audio_filename']}: {str(e)}")
                                        st.error(f"Failed to trigger analysis: {str(e)}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            logger.error(f"Error rendering metadata entry {idx} ({entry.get('audio_filename', 'unknown')}): {str(e)}", exc_info=True)
                            st.error(f"Failed to render entry {entry.get('audio_filename', 'unknown')}: {str(e)}")

                    if display_count < total_files:
                        if st.button("Show More", key="show_more"):
                            st.session_state.dashboard_display_count += 10
                            logger.info("Show More button clicked, increasing display count")
            except Exception as e:
                logger.error(f"Error in Recent Files tab: {str(e)}", exc_info=True)
                st.error(f"Failed to render Recent Files tab: {str(e)}")
                return

        with tab2:
            try:
                #st.markdown('<div style="text-underline-offset: 4px; font-weight: 400; color: var(--text-color); margin-bottom: 3px; text-decoration: underline; margin-top: -15px;">Agent Performance</div>', unsafe_allow_html=True)

                # Define categorize_lead function at tab scope
                def categorize_lead(score):
                    try:
                        score = float(score)
                        if score >= 70:
                            return 'High Interest'
                        elif score >= 40:
                            return 'Moderate Interest'
                        else:
                            return 'Low Interest'
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid lead score: {score}")
                        return 'Low Interest'

                # Agent selection for performance analysis
                agents = load_agent_config()
                if not agents:
                    logger.warning("No agents available for performance analysis")
                    st.warning("No agents available. Please create an agent in Agent Management.")
                else:
                    selected_agent = st.selectbox(
                        "Select Agent for Performance Analysis",
                        agents,
                        key="agent_performance_select"
                    )

                    # Weekly Performance Graph and Pie Chart for Selected Agent
                    if selected_agent:
                        # Filter metadata for the selected agent
                        agent_data = [
                            entry for entry in st.session_state.storage_metadata
                            if entry.get('agent_name') == selected_agent
                        ]

                        if not agent_data:
                            logger.info(f"No data available for agent {selected_agent}")
                            st.info(f"No data available for agent {selected_agent}.")
                        else:
                            # Process data for weekly aggregation
                            df = pd.DataFrame(agent_data)
                            try:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                            except Exception as e:
                                logger.error(f"Failed to convert timestamps for agent {selected_agent}: {str(e)}")
                                st.error(f"Invalid timestamp data for agent {selected_agent}: {str(e)}")
                                return
                            df['week'] = df['timestamp'].dt.to_period('W').apply(lambda x: x.start_time)

                            df['lead_category'] = df['lead_score'].apply(categorize_lead)

                            # Group by week and lead category
                            weekly_performance = df.groupby(['week', 'lead_category']).size().unstack(fill_value=0).reset_index()
                            weekly_performance['Week'] = weekly_performance['week'].dt.strftime('%Y-%m-%d')

                            # Ensure required columns exist
                            for col in ['High Interest', 'Moderate Interest', 'Low Interest']:
                                if col not in weekly_performance:
                                    weekly_performance[col] = 0
                            logger.debug(f"Weekly performance data: {weekly_performance.to_dict()}")

                            # Calculate total counts for each lead category
                            category_counts = df['lead_category'].value_counts().to_dict()
                            high_count = category_counts.get('High Interest', 0)
                            moderate_count = category_counts.get('Moderate Interest', 0)
                            low_count = category_counts.get('Low Interest', 0)

                            # Create line plot with colored markers
                            st.subheader("Weekly Lead Categories")
                            if weekly_performance.empty:
                                logger.warning(f"No weekly performance data for agent {selected_agent}")
                                st.warning(f"No weekly performance data available for agent {selected_agent}.")
                            else:
                                try:
                                    fig_line = px.line(
                                        weekly_performance,
                                        x='Week',
                                        y=['High Interest', 'Moderate Interest', 'Low Interest'],
                                        title=f"Weekly Lead Categories for Agent: {selected_agent}",
                                        markers=True
                                    )

                                    # Update legend labels with counts
                                    fig_line.update_traces(
                                        selector=dict(name='High Interest'),
                                        name=f'High Interest ({high_count})',
                                        line=dict(color='green'),
                                        marker=dict(symbol='circle', size=10, color='green')
                                    )
                                    fig_line.update_traces(
                                        selector=dict(name='Moderate Interest'),
                                        name=f'Moderate Interest ({moderate_count})',
                                        line=dict(color='yellow'),
                                        marker=dict(symbol='circle', size=10, color='yellow')
                                    )
                                    fig_line.update_traces(
                                        selector=dict(name='Low Interest'),
                                        name=f'Low Interest ({low_count})',
                                        line=dict(color='red'),
                                        marker=dict(symbol='circle', size=10, color='red')
                                    )

                                    fig_line.update_layout(
                                        xaxis_title="Week Starting",
                                        yaxis_title="Number of Leads",
                                        yaxis_range=[0, max(weekly_performance[['High Interest', 'Moderate Interest', 'Low Interest']].max().max() + 5, 10)],
                                        xaxis_tickangle=45,
                                        legend_title="Lead Category (Count)"
                                    )
                                    st.plotly_chart(fig_line, use_container_width=True)
                                except Exception as e:
                                    logger.error(f"Failed to render weekly lead categories plot: {str(e)}", exc_info=True)
                                    st.error(f"Failed to render weekly lead categories plot: {str(e)}")
                                    return

                            # Pie Chart for Lead Category Distribution
                            st.subheader("Lead Category Distribution")
                            lead_counts = df['lead_category'].value_counts().reset_index()
                            lead_counts.columns = ['Lead Category', 'Count']

                            try:
                                fig_pie = px.pie(
                                    lead_counts,
                                    names='Lead Category',
                                    values='Count',
                                    title=f"Lead Category Distribution for Agent: {selected_agent}",
                                    color='Lead Category',
                                    color_discrete_map={
                                        'High Interest': 'green',
                                        'Moderate Interest': 'yellow',
                                        'Low Interest': 'red'
                                    }
                                )
                                # Update labels to show percentage and count
                                fig_pie.update_traces(
                                    textinfo='percent+label',
                                    hovertemplate='%{label}: %{value} leads (%{percent})'
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            except Exception as e:
                                logger.error(f"Failed to render lead category pie chart: {str(e)}", exc_info=True)
                                st.error(f"Failed to render lead category pie chart: {str(e)}")
                                return

                # All-Agent Comparison Bar Plot
                st.subheader("All Agents High Interest Lead Comparison")
                if not st.session_state.storage_metadata:
                    logger.info("No storage metadata available for agent comparison")
                    st.info("No data available for agent comparison.")
                else:
                    # Aggregate data for all agents
                    df_all = pd.DataFrame(st.session_state.storage_metadata)
                    df_all['lead_score'] = df_all['lead_score'].fillna(0)
                    df_all['agent_name'] = df_all['agent_name'].fillna('Unknown')
                    df_all['lead_category'] = df_all['lead_score'].apply(categorize_lead)

                    # Count high interest leads per agent
                    agent_comparison = df_all[df_all['lead_category'] == 'High Interest'].groupby('agent_name').size().reset_index()
                    agent_comparison.columns = ['Agent', 'High Interest Leads']

                    # Bar plot for high interest leads
                    if agent_comparison.empty:
                        logger.warning("No high interest leads data available for any agent")
                        st.warning("No high interest leads data available for any agent.")
                    else:
                        try:
                            fig_bar = px.bar(
                                agent_comparison,
                                x='Agent',
                                y='High Interest Leads',
                                title="High Interest Leads by Agent",
                                color='Agent',
                                text='High Interest Leads'
                            )
                            fig_bar.update_layout(
                                xaxis_title="Agent",
                                yaxis_title="Number of High Interest Leads",
                                yaxis_range=[0, max(agent_comparison['High Interest Leads'].max() + 5, 10)],
                                xaxis_tickangle=45,
                                showlegend=False
                            )
                            fig_bar.update_traces(textposition='auto')
                            st.plotly_chart(fig_bar, use_container_width=True)
                        except Exception as e:
                            logger.error(f"Failed to render high interest leads bar plot: {str(e)}", exc_info=True)
                            st.error(f"Failed to render high interest leads bar plot: {str(e)}")
                            return

                # Export Report Option
                st.markdown("**Export Performance Reports**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Selected Agent Report", key="agent_performance_report_button"):
                        if selected_agent and agent_data:
                            try:
                                # Generate charts for PDF
                                # Pie Chart
                                fig_pie = px.pie(
                                    lead_counts,
                                    names='Lead Category',
                                    values='Count',
                                    title=f"Lead Category Distribution for Agent: {selected_agent}",
                                    color='Lead Category',
                                    color_discrete_map={
                                        'High Interest': 'green',
                                        'Moderate Interest': 'yellow',
                                        'Low Interest': 'red'
                                    }
                                )
                                fig_pie.update_traces(
                                    textinfo='percent+label',
                                    hovertemplate='%{label}: %{value} leads (%{percent})'
                                )

                                # Line Chart
                                fig_line = px.line(
                                    weekly_performance,
                                    x='Week',
                                    y=['High Interest', 'Moderate Interest', 'Low Interest'],
                                    title=f"Weekly Lead Categories for Agent: {selected_agent}",
                                    markers=True
                                )
                                fig_line.update_traces(
                                    selector=dict(name='High Interest'),
                                    name=f'High Interest ({high_count})',
                                    line=dict(color='green'),
                                    marker=dict(symbol='circle', size=10, color='green')
                                )
                                fig_line.update_traces(
                                    selector=dict(name='Moderate Interest'),
                                    name=f'Moderate Interest ({moderate_count})',
                                    line=dict(color='yellow'),
                                    marker=dict(symbol='circle', size=10, color='yellow')
                                )
                                fig_line.update_traces(
                                    selector=dict(name='Low Interest'),
                                    name=f'Low Interest ({low_count})',
                                    line=dict(color='red'),
                                    marker=dict(symbol='circle', size=10, color='red')
                                )
                                fig_line.update_layout(
                                    xaxis_title="Week Starting",
                                    yaxis_title="Number of Leads",
                                    yaxis_range=[0, max(weekly_performance[['High Interest', 'Moderate Interest', 'Low Interest']].max().max() + 5, 10)],
                                    xaxis_tickangle=45,
                                    legend_title="Lead Category (Count)"
                                )

                                # Save charts as images
                                pie_chart_path = os.path.join(tempfile.gettempdir(), f"{selected_agent}_pie_chart.png")
                                line_chart_path = os.path.join(tempfile.gettempdir(), f"{selected_agent}_line_chart.png")
                                fig_pie.write_image(file=pie_chart_path, format="png", width=600, height=400)
                                fig_line.write_image(file=line_chart_path, format="png", width=600, height=400)
                                st.session_state.temp_files.extend([pie_chart_path, line_chart_path])

                                # Collect lead filenames by category
                                high_leads = [entry['audio_filename'] for entry in agent_data if categorize_lead(entry['lead_score']) == 'High Interest']
                                moderate_leads = [entry['audio_filename'] for entry in agent_data if categorize_lead(entry['lead_score']) == 'Moderate Interest']
                                low_leads = [entry['audio_filename'] for entry in agent_data if categorize_lead(entry['lead_score']) == 'Low Interest']

                                # Create PDF
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                report_filename = f"{selected_agent}_performance_report_{timestamp}.pdf"
                                report_path = os.path.join(tempfile.gettempdir(), report_filename)
                                st.session_state.temp_files.append(report_path)

                                doc = SimpleDocTemplate(report_path, pagesize=letter)
                                elements = []
                                styles = getSampleStyleSheet()
                                title_style = styles['Title']
                                heading_style = styles['Heading2']
                                normal_style = styles['Normal']
                                custom_style = ParagraphStyle(name='Custom', parent=normal_style, fontSize=10, leading=12)

                                # Title
                                elements.append(Paragraph(f"Performance Report for Agent: {selected_agent}", title_style))
                                elements.append(Spacer(1, 0.2 * inch))

                                # Lead Category Counts
                                elements.append(Paragraph("Lead Category Counts", heading_style))
                                count_data = [
                                    ['Category', 'Count'],
                                    ['High Interest', str(high_count)],
                                    ['Moderate Interest', str(moderate_count)],
                                    ['Low Interest', str(low_count)]
                                ]
                                count_table = Table(count_data, colWidths=[3 * inch, 1.5 * inch])
                                count_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                ]))
                                elements.append(count_table)
                                elements.append(Spacer(1, 0.2 * inch))

                                # Pie Chart
                                elements.append(Paragraph("Lead Category Distribution", heading_style))
                                pie_image = Image(pie_chart_path, width=5 * inch, height=3.33 * inch)
                                elements.append(pie_image)
                                elements.append(Spacer(1, 0.2 * inch))

                                # Weekly Line Chart
                                elements.append(Paragraph("Weekly Lead Categories", heading_style))
                                line_image = Image(line_chart_path, width=5 * inch, height=3.33 * inch)
                                elements.append(line_image)
                                elements.append(Spacer(1, 0.2 * inch))

                                # Lead Filenames by Category
                                elements.append(Paragraph("Lead Filenames by Category", heading_style))
                                lead_data = [['Category', 'Audio Filenames']]
                                lead_data.append(['High Interest', ', '.join(high_leads) if high_leads else 'None'])
                                lead_data.append(['Moderate Interest', ', '.join(moderate_leads) if moderate_leads else 'None'])
                                lead_data.append(['Low Interest', ', '.join(low_leads) if low_leads else 'None'])
                                lead_table = Table(lead_data, colWidths=[2 * inch, 5 * inch])
                                lead_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                                ]))
                                elements.append(lead_table)
                                elements.append(Spacer(1, 0.2 * inch))

                                # Build PDF
                                doc.build(elements)

                                # Provide download button
                                with open(report_path, 'rb') as f:
                                    st.download_button(
                                        label="Download Agent Performance PDF",
                                        data=f.read(),
                                        file_name=report_filename,
                                        mime="application/pdf",
                                        key=f"download_pdf_report_{selected_agent}"
                                    )
                                st.success("Agent performance report generated successfully!")
                                logger.info(f"Generated PDF report: {report_filename}")
                            except Exception as e:
                                logger.error(f"Failed to generate PDF report: {str(e)}", exc_info=True)
                                st.error(f"Failed to generate PDF report: {str(e)}")
                        else:
                            st.error("No data available for the selected agent.")
                            logger.warning(f"No data for selected agent {selected_agent} for report generation")
                with col2:
                    if st.button("Generate All Agents Report", key="all_agents_report_button"):
                        if agent_comparison.empty:
                            st.error("No data available for agent comparison.")
                            logger.warning("No data available for all agents report")
                        else:
                            try:
                                # Save all-agent comparison to CSV
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                report_filename = f"all_agents_high_interest_leads_{timestamp}.csv"
                                report_path = os.path.join(tempfile.gettempdir(), report_filename)
                                agent_comparison.to_csv(report_path, index=False)
                                st.session_state.temp_files.append(report_path)

                                # Provide download button
                                with open(report_path, 'rb') as f:
                                    st.download_button(
                                        label="Download All Agents High Interest Leads CSV",
                                        data=f.read(),
                                        file_name=report_filename,
                                        mime="text/csv",
                                        key="download_all_agents_report"
                                    )
                                st.success("All agents high interest leads report generated successfully!")
                                logger.info(f"Generated all agents report: {report_filename}")
                            except Exception as e:
                                logger.error(f"Failed to generate all agents performance report: {str(e)}", exc_info=True)
                                st.error(f"Failed to generate report: {str(e)}")
            except Exception as e:
                logger.error(f"Error in Agent Performance tab: {str(e)}", exc_info=True)
                st.error(f"Failed to render Agent Performance tab: {str(e)}")
                return

        st.markdown('</div>', unsafe_allow_html=True)
        logger.info("Completed display_dashboard function successfully")
    except Exception as e:
        logger.error(f"Critical error in display_dashboard: {str(e)}", exc_info=True)
        st.error(f"Critical error rendering dashboard: {str(e)}. Check logs for details.")

def main():
    try:
        # Initialize session state
        initialize_session_state()
        logger.info("Session state initialized successfully")

        # Reset process_triggered to avoid stale state
        if 'process_triggered' not in st.session_state:
            st.session_state.process_triggered = False

        # Create main layout containers
        header_container = st.container()
        loading_placeholder = st.empty()
        results_container = st.container()
        dashboard_container = st.container()

        # Render logo and title
        with header_container:
            st.markdown("""
                <div class="logo-container">
                    <div class="glow-logo">LSS</div>
                    <div class="logo-text">Lead Scoring System</div>
                </div>
                """, unsafe_allow_html=True)

        # Display results
        if st.session_state.get('analysis_complete') and st.session_state.get('results'):
            with results_container:
                try:
                    display_results()
                    logger.info("Results rendered successfully")
                except Exception as e:
                    st.error("Failed to render results. Check logs for details.")
                    logger.error(f"Failed to render results: {str(e)}", exc_info=True)

        # Display dashboard
        with dashboard_container:
            try:
                display_dashboard()
                logger.info("Dashboard rendered successfully")
            except Exception as e:
                st.error("Failed to render dashboard. Check logs for details.")
                logger.error(f"Failed to render dashboard: {str(e)}", exc_info=True)

        # Sidebar
        with st.sidebar:
            st.header("Upload an Audio to Analyze Lead Potential")
            st.markdown("**Audio Upload**")
            
            audio_file = st.file_uploader(
                "Choose audio file (MP3, WAV, etc.)",
                type=['mp3', 'wav', 'aac', 'm4a', 'flac'],
                key="sidebar_audio_uploader"
            )
            
            if audio_file:
                custom_filename = st.text_input("Custom filename (optional)", key="sidebar_custom_filename_input")
                st.session_state.custom_filename = custom_filename.strip() if custom_filename else None
                
                agents = load_agent_config()
                agent_options = ["None"] + agents
                agent_name = st.selectbox(
                    "Select Agent",
                    agent_options,
                    index=0,
                    key="sidebar_upload_agent_select"
                )
                st.session_state.selected_agent = agent_name if agent_name != "None" else "NoAgent"
                
                if not agents and agent_name == "None":
                    st.warning("No agents created yet. Files will be stored without an agent association.")
            
            if audio_file and st.session_state.get('selected_agent'):
                if st.button("Start Analysis", key="sidebar_start_analysis_button"):
                    st.session_state.process_triggered = True
                    logger.info("Analysis triggered for uploaded audio")
            
            if st.button("Reset & Cleanup", key="sidebar_reset_cleanup_button"):
                reset_analysis()
                st.success("Reset completed. Please upload a new audio file.")
                st.rerun()
            
            st.markdown("---")
            st.header("Settings")
            
            model_size = st.selectbox(
                "Model Size",
                ["base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"],
                index=1,
                key="sidebar_model_size_select"
            )
            st.session_state.model_size = model_size
            
            retention_period = st.selectbox(
                "Dashboard Retention Period",
                ["Disable", "1 Month", "3 Months", "6 Months", "1 Year"],
                index=["Disable", "1 Month", "3 Months", "6 Months", "1 Year"].index(
                    {"disable": "Disable", "1_month": "1 Month", "3_months": "3 Months",
                     "6_months": "6 Months", "1_year": "1 Year"}.get(
                        st.session_state.retention_period, "1 Month")),
                key="sidebar_retention_period_select"
            )
            retention_map = {
                "Disable": "disable",
                "1 Month": "1_month",
                "3 Months": "3_months",
                "6 Months": "6_months",
                "1 Year": "1_year"
            }
            new_retention_period = retention_map[retention_period]
            if new_retention_period != st.session_state.retention_period:
                st.session_state.retention_period = new_retention_period
                config = load_storage_config()
                config['retention_period'] = new_retention_period
                save_storage_config(config)
                clean_old_files()
            
            st.markdown("---")
            st.header("Agent Management")
            
            new_agent_name = st.text_input("Create New Agent", key="sidebar_new_agent_name")
            if st.button("Add Agent", key="sidebar_add_agent_button"):
                success, message = add_agent(new_agent_name)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            
            agents = load_agent_config()
            if agents:
                delete_agent_name = st.selectbox(
                    "Select Agent to Delete",
                    agents,
                    key="sidebar_delete_agent_select"
                )
                pin_input = st.text_input("Enter PIN to delete agent", type="password", key="sidebar_delete_pin")
                if st.button("Delete Agent", key="sidebar_delete_agent_button"):
                    if pin_input == "8593":
                        success, message = delete_agent(delete_agent_name)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Failed deletion: Incorrect PIN")
            else:
                st.warning("No agents available to delete.")
            
            st.markdown("---")
            st.header("Storage Settings")
            
            st.write(f"Current Storage Directory: {st.session_state.storage_dir}")
            storage_config = load_storage_config()
            recent_dirs = storage_config.get('recent_dirs', [])
            storage_options = ["Default Storage", "Custom Directory", "Select Directory (Desktop)", "Select Directory (Mobile)"]
            if recent_dirs:
                storage_options.insert(0, "Recent Directories")
            
            storage_choice = st.selectbox(
                "Select Storage Location",
                storage_options,
                key="sidebar_storage_choice_select"
            )
            base_storage_path = os.path.join(os.getcwd(), 'app_storage', 'user_dirs')
            default_storage_dir = os.path.join(base_storage_path, 'default')
            
            if storage_choice == "Recent Directories" and recent_dirs:
                selected_recent = st.selectbox(
                    "Choose Recent Directory",
                    recent_dirs,
                    key="sidebar_recent_dir_select"
                )
                st.markdown('<div class="use-selected-dir-button">', unsafe_allow_html=True)
                if st.button("Selected Path", key="sidebar_use_recent_dir_button"):
                    if os.path.isdir(selected_recent):
                        st.session_state.storage_dir = selected_recent
                        os.makedirs(st.session_state.storage_dir, exist_ok=True)
                        initialize_session_state()
                        update_recent_dirs(selected_recent)
                        st.success(f"Storage directory set to: {st.session_state.storage_dir}")
                        st.rerun()
                    else:
                        st.error(f"Directory no longer exists: {selected_recent}")
                st.markdown('</div>', unsafe_allow_html=True)
            elif storage_choice == "Default Storage":
                if default_storage_dir != st.session_state.storage_dir:
                    st.session_state.storage_dir = default_storage_dir
                    os.makedirs(st.session_state.storage_dir, exist_ok=True)
                    initialize_session_state()
                    update_recent_dirs(st.session_state.storage_dir)
                    st.success(f"Storage directory set to: {st.session_state.storage_dir}")
                    st.rerun()
            elif storage_choice == "Custom Directory":
                custom_dir_name = st.text_input("Enter Directory Name (e.g., my_storage)", key="sidebar_custom_storage_dir")
                if custom_dir_name:
                    custom_dir_name = "".join(c for c in custom_dir_name if c.isalnum() or c in ('-', '_')).rstrip()
                    if custom_dir_name:
                        new_storage_dir = os.path.join(base_storage_path, custom_dir_name)
                        if new_storage_dir != st.session_state.storage_dir:
                            st.session_state.storage_dir = new_storage_dir
                            os.makedirs(st.session_state.storage_dir, exist_ok=True)
                            initialize_session_state()
                            update_recent_dirs(st.session_state.storage_dir)
                            st.success(f"Storage directory set to: {st.session_state.storage_dir}")
                            st.rerun()
                    else:
                        st.error("Invalid directory name. Use alphanumeric characters, hyphens, or underscores.")
            elif storage_choice == "Select Directory (Desktop)":
                if st.button("Select Directory", key="sidebar_tkinter_select_dir_button"):
                    folder = select_folder()
                    if folder:
                        st.session_state.storage_dir = folder
                        os.makedirs(st.session_state.storage_dir, exist_ok=True)
                        initialize_session_state()
                        update_recent_dirs(st.session_state.storage_dir)
                        st.success(f"Storage directory set to: {st.session_state.storage_dir}")
                        st.rerun()
                    else:
                        st.error("No directory selected. Please choose a valid directory or enter manually.")
                manual_dir = st.text_input("Or enter directory path manually", key="sidebar_manual_storage_dir")
                if manual_dir:
                    if os.path.isdir(manual_dir):
                        if manual_dir != st.session_state.storage_dir:
                            st.session_state.storage_dir = manual_dir
                            os.makedirs(st.session_state.storage_dir, exist_ok=True)
                            initialize_session_state()
                            update_recent_dirs(st.session_state.storage_dir)
                            st.success(f"Storage directory set to: {st.session_state.storage_dir}")
                            st.rerun()
                    else:
                        st.error("Invalid directory path. Please enter a valid path.")
            elif storage_choice == "Select Directory (Mobile)":
                st.warning("Directory selection is supported in Chrome/Edge on Android. iOS Safari and Firefox may not support this; use manual input instead.")
                directory_picker_html = """
                <input type="file" id="directoryPicker" webkitdirectory directory style="display: none;">
                <button id="selectDirButton">Select Directory</button>
                <script>
                    const picker = document.getElementById('directoryPicker');
                    const button = document.getElementById('selectDirButton');
                    button.onclick = () => picker.click();
                    picker.onchange = () => {
                        if (picker.files.length > 0) {
                            const file = picker.files[0];
                            const dirPath = file.path.substring(0, file.path.lastIndexOf('/')) || file.path;
                            sessionStorage.setItem('selectedDirectory', dirPath);
                            window.location.reload();
                        }
                    };
                </script>
                """
                st.components.v1.html(directory_picker_html, height=50)
                
                selected_dir = st.session_state.get('selected_directory', None)
                if selected_dir is None:
                    get_session_storage_js = """
                    <script>
                        const dirPath = sessionStorage.getItem('selectedDirectory');
                        if (dirPath) {
                            window.parent.postMessage({type: 'selectedDirectory', value: dirPath}, '*');
                        }
                    </script>
                    """
                    st.components.v1.html(get_session_storage_js, height=0)
                    
                    if 'selected_directory' in st.session_state:
                        selected_dir = st.session_state.selected_directory
                
                if selected_dir:
                    if os.path.isdir(selected_dir):
                        if selected_dir != st.session_state.storage_dir:
                            st.session_state.storage_dir = selected_dir
                            os.makedirs(st.session_state.storage_dir, exist_ok=True)
                            initialize_session_state()
                            update_recent_dirs(st.session_state.storage_dir)
                            st.success(f"Storage directory set to: {st.session_state.storage_dir}")
                            clear_session_storage_js = """
                            <script>
                                sessionStorage.removeItem('selectedDirectory');
                            </script>
                            """
                            st.components.v1.html(clear_session_storage_js, height=0)
                            st.rerun()
                    else:
                        st.error(f"Invalid directory: {selected_dir}")
                
                manual_dir = st.text_input("Or enter directory path manually", key="sidebar_manual_storage_dir_mobile")
                if manual_dir:
                    if os.path.isdir(manual_dir):
                        if manual_dir != st.session_state.storage_dir:
                            st.session_state.storage_dir = manual_dir
                            os.makedirs(st.session_state.storage_dir, exist_ok=True)
                            initialize_session_state()
                            update_recent_dirs(st.session_state.storage_dir)
                            st.success(f"Storage directory set to: {st.session_state.storage_dir}")
                            st.rerun()
                    else:
                        st.error("Invalid directory path. Please enter a valid path.")
            
            st.markdown("---")
            st.header("About")
            st.markdown("""
                Analyzes Malayalam audio to:
                - Transcribe to Malayalam
                - Translate to English
                - Detect sentiment and intent
                - Calculate lead scores
                """)

        # Process uploaded audio
        if audio_file and st.session_state.get('process_triggered') and not st.session_state.get('analysis_complete'):
            with loading_placeholder.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, status):
                    try:
                        progress_value = min(max(float(progress) / 100.0, 0.0), 1.0)
                        progress_bar.progress(progress_value, text=status)
                        status_text.markdown(f"**Status**: {status} ({int(progress)}%)")
                        logger.info(f"Progress: {progress}%, Status: {status}")
                        import time
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Failed to update progress: {str(e)}")
                
                with st.spinner(""):
                    try:
                        logger.info("Starting audio processing")
                        update_progress(0, "Initializing analysis...")
                        process_audio(audio_file, st.session_state.get('model_size', 'medium'), update_progress)
                        st.session_state.process_triggered = False
                        if st.session_state.get('processing_error'):
                            st.error(f"Error: {st.session_state['processing_error']}")
                            progress_bar.empty()
                            status_text.empty()
                        else:
                            update_progress(100, "Analysis completed")
                            st.success("Analysis completed successfully!")
                            progress_bar.empty()
                            status_text.empty()
                            st.rerun()
                    except Exception as e:
                        st.session_state.process_triggered = False
                        logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
                        st.error(f"Audio processing failed: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

        # Process stored audio
        if st.session_state.get('selected_audio') and st.session_state.get('process_triggered') and not st.session_state.get('analysis_complete'):
            with loading_placeholder.container():
                st.markdown("**Processing Stored Audio**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, status):
                    try:
                        progress_value = min(max(float(progress) / 100.0, 0.0), 1.0)
                        progress_bar.progress(progress_value, text=status)
                        status_text.markdown(f"**Status**: {status} ({int(progress)}%)")
                        logger.info(f"Progress: {progress}%, Status: {status}")
                        import time
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Failed to update progress: {str(e)}")
                
                with st.spinner(""):
                    try:
                        logger.info("Starting stored audio processing")
                        update_progress(0, "Initializing stored audio analysis...")
                        selected = st.session_state.selected_audio
                        with open(selected['path'], 'rb') as f:
                            audio_buffer = BytesIO(f.read())
                        audio_file = type('UploadedFile', (), {
                            'name': selected['name'],
                            'read': lambda self: audio_buffer.read(),
                            'seek': lambda self, pos: audio_buffer.seek(pos)
                        })()
                        st.session_state.custom_filename = selected['custom_filename']
                        st.session_state.selected_agent = selected.get('agent_name', 'NoAgent')
                        process_audio(audio_file, st.session_state.get('model_size', 'medium'), update_progress)
                        st.session_state.process_triggered = False
                        if st.session_state.get('processing_error'):
                            st.error(f"Error: {st.session_state['processing_error']}")
                            progress_bar.empty()
                            status_text.empty()
                        else:
                            update_progress(100, "Analysis completed")
                            st.success("Analysis completed successfully!")
                            progress_bar.empty()
                            status_text.empty()
                            st.rerun()
                    except Exception as e:
                        st.session_state.process_triggered = False
                        logger.error(f"Stored audio processing failed: {str(e)}", exc_info=True)
                        st.error(f"Stored audio processing failed: {str(e)}")
                        progress_bar.empty()
                        status_text.empty()

    except Exception as e:
        logger.error(f"Main function failed: {str(e)}", exc_info=True)
        st.markdown(f"""
            <div class="error-message">
                <strong>Error:</strong> Application failed to initialize: {str(e)}<br>
                <strong>Suggestion:</strong> Check logs, ensure dependencies are installed, and verify file permissions.
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting Streamlit application")
    main()