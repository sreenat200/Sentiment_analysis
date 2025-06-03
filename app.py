import streamlit as st
import os
from Lead_score_conversion import (
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
import base64
from datetime import datetime
from pydub import AudioSegment
import zipfile

# Set page config
st.set_page_config(
    page_title="Malayalam Audio Lead Scoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
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
        color: white;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #3498db;
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

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    
    return wav_path

def main():
    st.title("📊 Malayalam Audio Lead Scoring System")
    st.markdown("""
    Upload an audio file to get English transcription, Malayalam translation, sentiment analysis, 
    intent detection, and lead scoring with Google Drive integration.
    """)

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'zip_created' not in st.session_state:
        st.session_state.zip_created = False
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = None
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload Audio File")
        audio_file = st.file_uploader(
            "Choose an audio file (MP3, WAV, etc.)",
            type=['mp3', 'wav', 'aac', 'm4a', 'flac', 'ogg']
        )
        
        st.markdown("---")
        st.header("Analysis Settings")
        model_size = st.selectbox(
            "Select Whisper Model Size",
            ["tiny", "base", "small", "medium", "large-v1", "large-v2","large-v3"],
            index=5  # Default to large-v2
        )
        
        st.markdown("---")
        st.header("Google Drive Settings")
        drive_folder_id = st.text_input("Google Drive Folder ID (optional)")
        enable_drive = st.checkbox("Enable Google Drive Upload", value=True)
        
        st.markdown("---")
        st.header("About")
        st.markdown("""
        This app analyzes Malayalam audio to:
        - Transcribe to English
        - Translate to Malayalam
        - Detect sentiment and intent
        - Calculate lead score
        - Save results to Google Drive
        """)

    # Main processing
    if audio_file and not st.session_state.analysis_complete:
        with st.spinner("Processing audio..."):
            try:
                # Save uploaded file to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
                    tmp_file.write(audio_file.read())
                    audio_path = tmp_file.name
                
                # Convert to WAV if needed
                try:
                    wav_path = convert_to_whisper_format(audio_path)
                    if wav_path != audio_path:
                        st.session_state.temp_files.append(wav_path)
                except Exception as e:
                    st.error(f"Audio conversion failed: {str(e)}")
                    wav_path = audio_path
                
                st.session_state.temp_files.append(audio_path)
                
                # Initialize Whisper model if not already loaded
                if st.session_state.whisper_model is None:
                    with st.spinner("Loading Whisper model (this may take a minute)..."):
                        import torch
                        from faster_whisper import WhisperModel
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        compute_type = "float16" if device == "cuda" else "int8"
                        st.session_state.whisper_model = WhisperModel(
                            model_size, 
                            device=device, 
                            compute_type=compute_type
                        )
                
                # Transcribe audio
                with st.spinner("Transcribing audio..."):
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
                
                # Translate to Malayalam
                with st.spinner("Translating to Malayalam..."):
                    from deep_translator import GoogleTranslator
                    ml_translation = GoogleTranslator(source='en', target='ml').translate(raw_transcription)
                
                # Analyze texts
                with st.spinner("Analyzing content..."):
                    en_analysis = analyze_text(raw_transcription, "en")
                    ml_analysis = analyze_text(ml_translation, "ml")
                    comparison = compare_analyses(en_analysis, ml_analysis)
                    
                    # Calculate lead score
                    en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
                    ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
                    combined_avg = (en_avg_score + ml_avg_score) / 2
                    lead_score = int(combined_avg * 100)
                    
                    # Calculate intent score
                    positive_intents = sum(1 for item in en_analysis if item["intent"] in ["Strong_interest", "Moderate_interest"])
                    intent_score = int((positive_intents / len(en_analysis))) * 100 if en_analysis else 0
                    
                    # Store results in session state
                    st.session_state.results = {
                        "raw_transcription": raw_transcription,
                        "ml_translation": ml_translation,
                        "en_analysis": en_analysis,
                        "ml_analysis": ml_analysis,
                        "comparison": comparison,
                        "lead_score": lead_score,
                        "intent_score": intent_score,
                        "audio_path": audio_path,
                        "original_filename": audio_file.name
                    }
                    st.session_state.analysis_complete = True
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Cleanup temp files after processing
                for file_path in st.session_state.temp_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        st.error(f"Error deleting temp file {file_path}: {str(e)}")
                st.session_state.temp_files = []

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Results", "Analysis", "Visualizations", "Export"])
        
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
            
            # Intent distribution
            st.subheader("Intent Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            intents = [item["english_intent"] for item in results["comparison"]]
            pd.Series(intents).value_counts().plot(kind='bar', ax=ax, color='orange')
            ax.set_title("Intent Distribution")
            ax.set_xlabel("Intent")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Sentiment trend
            st.subheader("Sentiment Trend")
            fig, ax = plt.subplots(figsize=(10, 5))
            en_scores = [item["sentiment_score"] for item in results["en_analysis"]]
            ml_scores = [item["sentiment_score"] for item in results["ml_analysis"]]
            sentence_numbers = list(range(1, len(en_scores)+1))
            
            ax.plot(sentence_numbers, en_scores, marker='o', label='English', color='blue')
            ax.plot(sentence_numbers, ml_scores, marker='s', label='Malayalam', color='green')
            ax.set_xlabel('Sentence Number')
            ax.set_ylabel('Sentiment Score')
            ax.set_title('Sentiment Trend Over Conversation')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        with tab4:
            st.header("Export Results")
            
            # Get base filename
            base_filename = os.path.splitext(results["original_filename"])[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{base_filename}_{timestamp}_L{results['lead_score']}_I{results['intent_score']}"
            
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
                        
                        # Create zip archive
                        zip_filename, _ = create_zip_archive(
                            results["audio_path"],
                            results["raw_transcription"],
                            results["ml_translation"],
                            pdf_path,
                            results["en_analysis"],
                            results["ml_analysis"],
                            final_filename
                        )
                        
                        st.session_state.zip_created = True
                        st.session_state.zip_filename = zip_filename
                        st.success("Report generated successfully!")
                    except Exception as e:
                        st.error(f"Failed to generate report: {str(e)}")
            
            if st.session_state.get('zip_created', False):
                # Download zip file
                with open(st.session_state.zip_filename, "rb") as f:
                    st.download_button(
                        label="Download Full Analysis (ZIP)",
                        data=f,
                        file_name=os.path.basename(st.session_state.zip_filename),
                        mime="application/zip"
                    )
                
                # Upload to Google Drive
                if enable_drive:
                    if st.button("Upload to Google Drive"):
                        with st.spinner("Uploading to Google Drive..."):
                            drive_folder = drive_folder_id if drive_folder_id else None
                            uploaded_file = upload_to_gdrive(st.session_state.zip_filename, drive_folder)
                            if uploaded_file:
                                st.success(f"Successfully uploaded to Google Drive!")
                                st.markdown(f"[View file]({uploaded_file.get('webViewLink')})")
                            else:
                                st.error("Google Drive upload failed")
            
            # Search functionality
            st.markdown("---")
            st.header("Search Google Drive")
            
            search_query = st.text_input("Search term (filename)")
            min_lead = st.number_input("Minimum lead score", min_value=0, max_value=100, value=0)
            max_lead = st.number_input("Maximum lead score", min_value=0, max_value=100, value=100)
            min_intent = st.number_input("Minimum intent score", min_value=0, max_value=100, value=0)
            max_intent = st.number_input("Maximum intent score", min_value=0, max_value=100, value=100)
            
            if st.button("Search"):
                with st.spinner("Searching Google Drive..."):
                    files = search_gdrive_files(
                        query=search_query if search_query else None,
                        min_lead=min_lead if min_lead > 0 else None,
                        max_lead=max_lead if max_lead < 100 else None,
                        min_intent=min_intent if min_intent > 0 else None,
                        max_intent=max_intent if max_intent < 100 else None
                    )
                    
                    if files:
                        st.success(f"Found {len(files)} files:")
                        for file in files:
                            with st.expander(f"📄 {file['name']}"):
                                st.markdown(f"**Created:** {file['createdTime']}")
                                st.markdown(f"[View file]({file['webViewLink']})")
                    else:
                        st.warning("No files found matching your criteria")

if __name__ == "__main__":
    main()
