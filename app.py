import os
import tempfile
from datetime import datetime
import torch
from pydub import AudioSegment
from deep_translator import GoogleTranslator
from transformers import pipeline
import pandas as pd
from faster_whisper import WhisperModel
import nltk
import streamlit as st

# Check and install missing packages
try:
    import torch
except ImportError:
    st.warning("PyTorch not found. Installing...")
    os.system("pip install torch")
    import torch

try:
    from faster_whisper import WhisperModel
except ImportError:
    st.warning("faster-whisper not found. Installing...")
    os.system("pip install faster-whisper")
    from faster_whisper import WhisperModel

# Initialize NLTK data with error handling
try:
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    if not nltk.data.find('tokenizers/punkt'):
        nltk.download('punkt', download_dir=nltk_data_path)
except Exception as e:
    st.error(f"Error initializing NLTK: {str(e)}")

class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="large-v2"):  # Updated to v2 which is more stable
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Loading Faster-Whisper {model_size} model on {self.device}...")
        compute_type = "float16" if self.device == "cuda" else "int8"
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type)
        except Exception as e:
            st.error(f"Failed to load Whisper model: {str(e)}")
            raise
        self.temp_files = []

    def convert_to_whisper_format(self, input_path):
        supported_formats = ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg']
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        file_ext = os.path.splitext(input_path)[1].lower()
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
            self.temp_files.append(wav_path)
            return wav_path
        except Exception as e:
            st.error(f"Audio conversion failed: {str(e)}")
            return None

    def transcribe_audio(self, audio_path):
        if not audio_path.lower().endswith('.wav'):
            audio_path = self.convert_to_whisper_format(audio_path)
            if not audio_path:
                return None

        try:
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="ml"  # Changed to Malayalam for better transcription
            )

            full_text = " ".join(segment.text for segment in segments)
            return {
                "raw_transcription": full_text,
                "audio_metadata": {
                    "original_path": audio_path,
                    "sample_rate": 16000,
                    "duration": len(AudioSegment.from_wav(audio_path)) / 1000
                }
            }
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            return None

    def translate_to_english(self, text_or_dict):
        try:
            if isinstance(text_or_dict, dict):
                text = text_or_dict.get('raw_transcription', '')
            else:
                text = text_or_dict

            if not text.strip():
                return text_or_dict

            return GoogleTranslator(source='ml', target='en').translate(text)
        except Exception as e:
            st.warning(f"Translation error: {str(e)}")
            return text_or_dict

    def cleanup(self):
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        self.temp_files = []

# Initialize sentiment analysis pipeline
@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",  # Lighter model
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        st.error(f"Failed to load sentiment model: {str(e)}")
        return None

sentiment_pipeline = load_sentiment_pipeline()

def analyze_sentiment(text):
    if not sentiment_pipeline:
        return {"label": "neutral", "score": 0.5}
    
    try:
        result = sentiment_pipeline(text)[0]
        return {
            "label": result['label'],
            "score": result['score']
        }
    except:
        return {"label": "neutral", "score": 0.5}

def display_results():
    st.set_page_config(
        page_title="Malayalam Audio Analyzer",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("Malayalam Audio Analyzer")
    st.write("Upload a Malayalam audio file to analyze sentiment and intent")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "m4a", "ogg"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name

        transcriber = MalayalamTranscriptionPipeline()
        try:
            with st.spinner("Processing audio..."):
                results = transcriber.transcribe_audio(audio_path)
                if results:
                    st.subheader("Malayalam Transcription")
                    st.text_area("Transcription", results['raw_transcription'], height=150)

                    translation = transcriber.translate_to_english(results)
                    st.subheader("English Translation")
                    st.text_area("Translation", translation, height=150)

                    sentiment = analyze_sentiment(translation)
                    st.subheader("Sentiment Analysis")
                    st.write(f"Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.2f})")

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            transcriber.cleanup()
            try:
                os.remove(audio_path)
            except:
                pass

if __name__ == "__main__":
    display_results()
