import streamlit as st
import os
from main_python import (
    analyze_text,
    compare_analyses,
    generate_analysis_pdf,
    create_zip_archive,
    upload_to_gdrive,
    search_gdrive_files,
    save_analysis_to_csv
)
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pydub import AudioSegment
import zipfile
from indicnlp.tokenize import sentence_tokenize
import nltk
import traceback
from nltk.tokenize import sent_tokenize
import shutil

# Download NLTK data (punkt tokenizer) if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Malayalam Audio Lead Scoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with modifications
st.markdown("""
<style>
    .main .block-container {
        background-color: #ffffff;
    }
    
    /* Main title styling */
    .stApp h1 {
        font-size: 1.8rem !important;  /* Reduced title size */
        font-family: Arial, sans-serif !important;  /* Changed font type */
        color: #d3d3d3 !important;  /* Lowered brightness from white to off-white */
    }
    
    /* General text color adjustment for white text */
    .stApp, .stText, .stMarkdown, .stTextInput, .stSelectbox, .stNumberInput, .stCheckbox, .stButton>button, .stDownloadButton>button {
        color: #d3d3d3 !important;  /* Lowered brightness for all white text */
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: #d3d3d3;  /* Adjusted button text color */
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    
    .stDownloadButton>button {
        background-color: #2196F3;
        color: #d3d3d3;  /* Adjusted download button text color */
    }
    
    .lead-score {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    
    .score-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
    }
    
    .high-score {
        background-color: #d4edda;
        color: #155724;
    }
    
    .medium-score {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .low-score {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .error-message {
        color: #721c24;
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    .search-header h2 {
        font-size: 1.5rem !important;  /* Reduced size of search header */
    }
    
    .search-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .cleanup-button {
        background-color: #f44336 !important;
    }
</style>
""", unsafe_allow_html=True)

def get_interpretation(score):
    if score >= 70:
        return "High Interest Lead", "high-score"
    elif score >= 40:
        return "Moderate Interest Lead", "medium-score"
    else:
        return "Low Interest Lead", "low-score"

def display_lead_score(score):
    interpretation, css_class = get_interpretation(score)
    
    st.markdown(f"""
    <div class="score-card {css_class}">
        <div class="lead-score">{score}/100</div>
        <div style="text-align: center; font-size: 1.2rem;">{interpretation}</div>
    </div>
    """, unsafe_allow_html=True)

def convert_to_whisper_format(input_path):
    """Convert audio file to WAV format if needed"""
    supported_formats = ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg', '.wma']
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext == '.wav':
        return input_path
        
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported audio format: {file_ext}")

    temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    wav_path = os.path.join(temp_dir, f"temp_{timestamp}.wav")

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        # Try using ffmpeg directly if pydub fails
        try:
            import subprocess
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ac', '1',
                '-ar', '16000',
                '-acodec', 'pcm_s16le',
                wav_path
            ]
            subprocess.run(cmd, check=True)
            return wav_path
        except Exception as e:
            raise RuntimeError(f"Failed to convert audio to WAV format: {str(e)}")

def split_into_sentences(text, language="en"):
    try:
        # First try NLTK for both English and Malayalam
        sentences = sent_tokenize(text)
        
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If NLTK returned just one sentence (might have failed), try Indic NLP
        if len(sentences) <= 1:
            raise Exception("NLTK returned only one sentence, trying Indic NLP")
            
        return sentences
        
    except Exception as nltk_error:
        try:
            st.warning(f"NLTK sentence splitting failed, trying Indic NLP...")
            if language == "ml":
                # Use Indic NLP for Malayalam
                sentences = sentence_tokenize.sentence_split(text, lang='mal')
            else:
                # Use Indic NLP for English as fallback
                sentences = sentence_tokenize.sentence_split(text, lang='en')
            
            return [s.strip() for s in sentences if s.strip()]
        except Exception as indic_error:
            st.error(f"Both NLTK and Indic NLP sentence splitting failed")
            return [text] if text.strip() else []

def initialize_session_state():
    """Initialize all required session state variables"""
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
        'process_triggered': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def cleanup_temp_files():
    """Clean up temporary files"""
    if 'temp_files' in st.session_state:
        for file_path in st.session_state.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.error(f"Error deleting temp file {file_path}: {str(e)}")
        st.session_state.temp_files = []
    
    # Clean up whisper_temp directory
    temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            st.error(f"Error cleaning up temp directory: {str(e)}")

def reset_analysis():
    """Reset the analysis state and clean up files"""
    st.session_state.analysis_complete = False
    st.session_state.results = None
    st.session_state.zip_created = False
    st.session_state.zip_filename = None
    st.session_state.process_triggered = False
    cleanup_temp_files()
    st.success("Analysis reset and temporary files cleaned up!")

def process_audio_file(audio_file, model_size):
    """Process the uploaded audio file through the full pipeline with step-by-step progress"""
    try:
        # Clear any previous results
        st.session_state.analysis_complete = False
        st.session_state.results = None
        st.session_state.processing_error = None
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Save uploaded file
        status_text.text("Step 1/7: Saving uploaded file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.read())
            audio_path = tmp_file.name
        st.session_state.temp_files.append(audio_path)
        st.session_state.audio_path = audio_path
        progress_bar.progress(15)
        
        # Step 2: Convert to WAV if needed
        status_text.text("Step 2/7: Converting audio format if needed...")
        try:
            wav_path = convert_to_whisper_format(audio_path)
            if wav_path != audio_path:
                st.session_state.temp_files.append(wav_path)
                st.session_state.wav_path = wav_path
        except Exception as e:
            st.error(f"Audio conversion failed: {str(e)}")
            wav_path = audio_path
        progress_bar.progress(30)
        
        # Step 3: Initialize Whisper model
        status_text.text("Step 3/7: Loading model (this may take a minute)...")
        if st.session_state.whisper_model is None:
            import torch
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            st.session_state.whisper_model = WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type
            )
        progress_bar.progress(45)
        
        # Step 4: Transcribe audio
        status_text.text("Step 4/7: Transcribing audio content...")
        segments, info = st.session_state.whisper_model.transcribe(
            wav_path,
            beam_size=5,
            language="en"
        )
        
        full_text = ""
        segment_list = []
        for i, seg in enumerate(segments):
            text = seg.text.strip()
            confidence = seg.avg_logprob if hasattr(seg, 'avg_logprob') else 1.0
            segment_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "confidence": round(confidence, 3),
                "overlap": i > 0 and seg.start < segment_list[i - 1]["end"]
            })
            full_text += f" {text}"
        
        raw_transcription = full_text.strip()
        progress_bar.progress(60)
        
        # Step 5: Translate to Malayalam
        status_text.text("Step 5/7: Translating to Malayalam...")
        from deep_translator import GoogleTranslator
        ml_translation = GoogleTranslator(source='en', target='ml').translate(raw_transcription)
        progress_bar.progress(75)
        
        # Step 6: Analyze texts
        status_text.text("Step 6/7: Analyzing content...")
        en_analysis = analyze_text(raw_transcription, "en")
        ml_analysis = analyze_text(ml_translation, "ml")
        comparison = compare_analyses(en_analysis, ml_analysis)
        
        # Calculate lead score
        en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
        ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
        combined_avg = (en_avg_score + ml_avg_score) / 2
        lead_score = int(combined_avg * 100)
        
        # Step 7: Store results
        status_text.text("Step 7/7: Finalizing results...")
        st.session_state.results = {
            "raw_transcription": raw_transcription,
            "ml_translation": ml_translation,
            "en_analysis": en_analysis,
            "ml_analysis": ml_analysis,
            "comparison": comparison,
            "lead_score": lead_score,
            "audio_path": wav_path if hasattr(st.session_state, 'wav_path') and st.session_state.wav_path else audio_path,
            "original_filename": os.path.splitext(audio_file.name)[0]
        }
        st.session_state.analysis_complete = True
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Small delay to show completion before hiding
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"An error occurred during processing: {str(e)}")
        st.error(traceback.format_exc())
    finally:
        # Don't cleanup temp files yet - we need them for export
        pass

def display_results():
    """Display the analysis results in tabs"""
    results = st.session_state.results
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Results", "Analysis", "Visualizations"])
    
    with tab1:
        st.header("Transcription Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("English Transcription")
            st.text_area("Raw Transcription", results["raw_transcription"], height=200, key="en_transcription")
        
        with col2:
            st.subheader("Malayalam Translation")
            st.text_area("Translated Text", results["ml_translation"], height=200, key="ml_translation")
    
    with tab2:
        st.header("Detailed Analysis")
        
        # Metrics cards
        col1, col2 = st.columns(2)
        with col1:
            display_lead_score(results['lead_score'])
        
        # Analysis dataframes
        st.subheader("English Analysis")
        st.dataframe(pd.DataFrame(results["en_analysis"]))
        
        st.subheader("Malayalam Analysis")
        st.dataframe(pd.DataFrame(results["ml_analysis"]))
        
        st.subheader("Comparison Analysis")
        st.dataframe(pd.DataFrame(results["comparison"]))
    
    with tab3:
        st.header("Analysis Visualizations")
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # English sentiment
        en_sentiments = [item["sentiment"] for item in results["en_analysis"]]
        pd.Series(en_sentiments).value_counts().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("English Sentiment")
        ax1.set_xlabel("Sentiment")
        ax1.set_ylabel("Count")
        
        # Malayalam sentiment
        ml_sentiments = [item["sentiment"] for item in results["ml_analysis"]]
        pd.Series(ml_sentiments).value_counts().plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title("Malayalam Sentiment")
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Count")
        
        st.pyplot(fig)
        
        # Sentiment trend
        st.subheader("Sentiment Trend")
        fig, ax = plt.subplots(figsize=(10, 5))
        en_scores = [item["sentiment_score"] for item in results["en_analysis"]]
        ml_scores = [item["sentiment_score"] for item in results["ml_analysis"]]
        
        # Ensure consistent lengths by taking the minimum
        min_length = min(len(en_scores), len(ml_scores))
        if min_length > 0:
            sentence_numbers = list(range(1, min_length + 1))
            ax.plot(sentence_numbers, en_scores[:min_length], marker='o', label='English', color='blue')
            ax.plot(sentence_numbers, ml_scores[:min_length], marker='s', label='Malayalam', color='green')
            ax.set_xlabel('Sentence Number')
            ax.set_ylabel('Sentiment Score')
            ax.set_title('Sentiment Trend Over Conversation')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.warning("No sentiment scores available to plot trend.")
    
    # Export Results section at the bottom
    st.markdown("---")
    st.header("Export Results")
    
    # Get base filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"{results['original_filename']}_{timestamp}_L{results['lead_score']}"
    
    # Generate PDF report
    if st.button("Generate Full Report"):
        with st.spinner("Creating report..."):
            try:
                pdf_path = generate_analysis_pdf(
                    results["en_analysis"],
                    results["ml_analysis"],
                    results["comparison"],
                    final_filename
                )
                
                # Create zip archive - use the WAV file if available
                audio_file_path = st.session_state.wav_path if hasattr(st.session_state, 'wav_path') and st.session_state.wav_path else results["audio_path"]
                
                # Handle both tuple and single return value from create_zip_archive
                zip_result = create_zip_archive(
                    audio_file_path,
                    results["raw_transcription"],
                    results["ml_translation"],
                    pdf_path,
                    results["en_analysis"],
                    results["ml_analysis"],
                    final_filename
                )
                
                # Ensure we have a single path string
                if isinstance(zip_result, tuple):
                    zip_filename = zip_result[0]  # Take first element if tuple
                else:
                    zip_filename = zip_result
                
                st.session_state.zip_created = True
                st.session_state.zip_filename = zip_filename
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate report: {str(e)}")
                st.error(traceback.format_exc())
    
    if st.session_state.get('zip_created', False) and st.session_state.zip_filename:
        # Download zip file
        try:
            if os.path.exists(st.session_state.zip_filename):
                with open(st.session_state.zip_filename, "rb") as f:
                    st.download_button(
                        label="Download Full Analysis (ZIP)",
                        data=f,
                        file_name=os.path.basename(st.session_state.zip_filename),
                        mime="application/zip"
                    )
            else:
                st.error("Zip file not found for download")
        except Exception as e:
            st.error(f"Failed to prepare download: {str(e)}")
            st.error(traceback.format_exc())
        
        # Upload to Google Drive
        if st.session_state.get('enable_drive', False):
            if st.button("Upload to Google Drive"):
                with st.spinner("Uploading to Google Drive..."):
                    try:
                        if not os.path.exists(st.session_state.zip_filename):
                            st.error("Zip file not found for upload")
                            return
                            
                        drive_folder = st.session_state.get('drive_folder_id', None)
                        uploaded_file = upload_to_gdrive(st.session_state.zip_filename, drive_folder)
                        if uploaded_file:
                            st.success(f"Successfully uploaded to Google Drive!")
                            if hasattr(uploaded_file, 'get') and callable(uploaded_file.get):
                                st.markdown(f"[View file]({uploaded_file.get('webViewLink')})")
                            else:
                                st.write("Upload complete. File ID:", uploaded_file)
                        else:
                            st.error("Google Drive upload failed - no file returned")
                    except Exception as e:
                        st.error(f"Google Drive upload failed: {str(e)}")
                        st.error(traceback.format_exc())

def display_search_results(files):
    """Display search results in a consistent format"""
    if files:
        st.success(f"Found {len(files)} files:")
        for file in files:
            # Extract lead score from filename if available
            filename = file['name']
            lead_score = "N/A"
            
            # Try to extract score from filename pattern
            import re
            match = re.search(r'_L(\d+)', filename)
            if match:
                lead_score = match.group(1)
            
            # Use columns to create a card-like layout
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown("📄")
            with col2:
                st.markdown(f"**{filename}**")
                st.markdown(f"Lead Score: {lead_score}")
                st.markdown(f"Created: {file['createdTime']}")
                st.markdown(f"[View file]({file['webViewLink']})")
            st.markdown("---")
    else:
        st.warning("No files found matching your criteria")

def main():
    # Initialize session state first
    initialize_session_state()
    
    st.title("Lead Scoring System 📊")
    
    # Search functionality at the top with filter on the right
    st.markdown('<div class="search-header"><h2>Search recent analysis files </h2></div>', unsafe_allow_html=True)
    
    # Create columns for search and filter
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Create a dropdown for filter options
        filter_options = ["Filename", "Lead Score Range"]
        selected_filter = st.selectbox("Filter by", filter_options, index=0, key="filter_select")
    
    with col1:
        if selected_filter == "Filename":
            search_query = st.text_input("Search by filename", key="filename_search")
            if st.button("Search", key="search_button"):
                with st.spinner("Searching Google Drive..."):
                    files = search_gdrive_files(query=search_query if search_query else None)
                    display_search_results(files)
        
        elif selected_filter == "Lead Score Range":
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                min_lead = st.number_input("Minimum lead score", min_value=0, max_value=100, value=0, key="min_lead")
            with score_col2:
                max_lead = st.number_input("Maximum lead score", min_value=0, max_value=100, value=100, key="max_lead")
            
            if st.button("Search by Lead Score", key="score_search_button"):
                with st.spinner("Searching Google Drive..."):
                    files = search_gdrive_files(
                        min_lead=min_lead if min_lead > 0 else None,
                        max_lead=max_lead if max_lead < 100 else None
                    )
                    display_search_results(files)
    
    st.markdown("""
    Upload an audio file to get transcriptions, sentiment analysis, 
    and lead scoring.
    """)

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload Audio File")
        audio_file = st.file_uploader(
            "Choose an audio file (MP3, WAV, etc.)",
            type=['mp3', 'wav', 'aac', 'm4a', 'flac', 'ogg']
        )
        
        # Add process button below file uploader
        if audio_file and not st.session_state.process_triggered:
            if st.button("Start Analysis", key="process_button"):
                st.session_state.process_triggered = True
                st.rerun()
        
        # Add cleanup button
        if st.button("Reset & Cleanup", key="cleanup_button", help="Reset analysis and clean up temporary files"):
            reset_analysis()
            st.rerun()
        
        st.markdown("---")
        st.header("Analysis Settings")
        model_size = st.selectbox(
            "Select Model Size",
            ["base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"],
            index=1  # Default to small
        )
        
        st.markdown("---")
        st.header("Google Drive Settings")
        drive_folder_id = st.text_input("Google Drive Folder ID (optional)")
        enable_drive = st.checkbox("Enable Google Drive Upload", value=True)
        
        # Store these in session state
        st.session_state.drive_folder_id = drive_folder_id
        st.session_state.enable_drive = enable_drive
        
        st.markdown("---")
        st.header("About")
        st.markdown("""
        This app analyzes Malayalam audio to:
        - Transcribe to English
        - Translate to Malayalam
        - Detect sentiment
        - Calculate lead score
        - Save results to Google Drive
        """)

    # Main processing
    if audio_file and st.session_state.process_triggered and not st.session_state.analysis_complete:
        process_audio_file(audio_file, model_size)

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.results:
        display_results()

    # Display any processing error
    if st.session_state.get('processing_error'):
        st.markdown(f"""
        <div class="error-message">
            <strong>Processing Error:</strong> {st.session_state.processing_error}
        </div>
        """, unsafe_allow_html=True)

    # Cleanup temp files when the session ends
    if not audio_file and st.session_state.get('temp_files'):
        cleanup_temp_files()

if __name__ == "__main__":
    main()