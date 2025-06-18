import os
import tempfile
from datetime import datetime
import torch
from pydub import AudioSegment
import subprocess
import pandas as pd
from indicnlp.tokenize import sentence_tokenize
import re
import io
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
from gdrive_utils import upload_to_gdrive, create_gdrive_folder, update_metadata_csv, authenticate_gdrive
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
import matplotlib.font_manager as fm
from matplotlib import rcParams
import requests
import cssutils
import unicodedata
from indicnlp import common
from googleapiclient.http import MediaIoBaseDownload
import json
import shutil
import streamlit as st
from googleapiclient.http import MediaFileUpload
import zipfile
from transformers import AutoModelForAudioClassification
from huggingface_hub import login

# Hugging Face login
login(token="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")

# Configure NLTK data path
nltk_data_path = os.path.expanduser('~/.nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.download(['punkt', 'punkt_tab'], download_dir=nltk_data_path, quiet=False)
    punkt_path = nltk.data.find('tokenizers/punkt')
    punkt_tab_path = nltk.data.find('tokenizers/punkt_tab')
    print(f"NLTK data successfully configured: punkt at {punkt_path}, punkt_tab at {punkt_tab_path}")
except Exception as e:
    print(f"Error setting up NLTK data: {str(e)}")
    try:
        nltk.download(['punkt', 'punkt_tab'], quiet=False)
        print("Fell back to default NLTK data path for punkt and punkt_tab")
    except Exception as fallback_error:
        print(f"Failed to download NLTK data entirely: {str(fallback_error)}")
        raise SystemExit("Critical error: NLTK punkt and punkt_tab resources are required but could not be downloaded.")

# Set the path for Indic NLP resources
indic_nlp_resource_path = os.path.expanduser('~/.indic_nlp_resources')
os.makedirs(indic_nlp_resource_path, exist_ok=True)

try:
    common.set_resources_path(indic_nlp_resource_path)
    common.init()
    print(f"Indic NLP resources successfully initialized at {indic_nlp_resource_path}")
except Exception as e:
    print(f"Error initializing Indic NLP resources: {str(e)}")
    try:
        import urllib.request
        import zipfile
        resource_url = "https://github.com/anoopkunchukuttan/indic_nlp_resources/archive/master.zip"
        resource_zip_path = os.path.join(indic_nlp_resource_path, "indic_nlp_resources.zip")
        resource_extract_path = os.path.join(indic_nlp_resource_path, "indic_nlp_resources-master")

        print("Downloading Indic NLP resources...")
        urllib.request.urlretrieve(resource_url, resource_zip_path)

        with zipfile.ZipFile(resource_zip_path, 'r') as zip_ref:
            zip_ref.extractall(indic_nlp_resource_path)

        resource_final_path = os.path.join(indic_nlp_resource_path, "resources")
        if os.path.exists(resource_final_path):
            shutil.rmtree(resource_final_path)
        shutil.move(os.path.join(resource_extract_path, "resources"), resource_final_path)

        os.remove(resource_zip_path)
        shutil.rmtree(resource_extract_path)

        common.set_resources_path(indic_nlp_resource_path)
        common.init()
        print("Indic NLP resources downloaded and initialized successfully")
    except Exception as indic_error:
        print(f"Failed to download or initialize Indic NLP resources: {str(indic_error)}")
        raise SystemExit("Critical error: Indic NLP resources are required but could not be loaded.")

# Configure Matplotlib to embed fonts in PDF output
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Configure Matplotlib to use a Malayalam-supporting font
font_path = r'E:\vscode_files\NotoSerifMalayalam-Regular.ttf'
malayalam_font = None
default_font = fm.FontProperties(family='DejaVu Sans')

if os.path.exists(font_path):
    print("Found local Malayalam font at", font_path)
    malayalam_font = fm.FontProperties(fname=font_path)
else:
    print("Local Malayalam font not found. Attempting to download static font...")
    temp_dir = tempfile.gettempdir()
    downloaded_font_path = os.path.join(temp_dir, "NotoSerifMalayalam-Regular.ttf")

    try:
        fallback_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerifMalayalam/NotoSerifMalayalam-Regular.ttf"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(fallback_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(downloaded_font_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded Malayalam font to {downloaded_font_path}")
        if os.path.exists(downloaded_font_path):
            malayalam_font = fm.FontProperties(fname=downloaded_font_path)
        else:
            print("Failed to save the downloaded font.")
    except Exception as e:
        print(f"Failed to download font: {str(e)}")
        print("Falling back to DejaVu Sans.")
        malayalam_font = default_font

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Serif Malayalam']

def is_malayalam(text):
    """Check if text contains Malayalam characters (Unicode range U+0D00–U+0D7F)."""
    if not text:
        return False
    for char in text:
        if 0x0D00 <= ord(char) <= 0x0D7F:
            return True
    return False

class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Faster-Whisper {model_size} model on {self.device}...")
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type)
        # Initialize emotion classifier
        self.emotion_classifier = pipeline(
            "audio-classification",
            model="m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition",
            device=0 if torch.cuda.is_available() else -1
        )
        # Initialize translation pipeline
        self.translator = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang="eng_Latn",
            tgt_lang="mal_Mlym",
            device=0 if torch.cuda.is_available() else -1
        )
        self.temp_files = []

    def convert_to_whisper_format(self, input_path):
        supported_formats = ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg', '.wma']
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        file_ext = os.path.splitext(input_path)[1].lower()
        if file_ext not in supported_formats:
            raise ValueError(f"Unsupported audio format: {file_ext}")

        try:
            temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            wav_path = os.path.join(temp_dir, f"temp_{timestamp}.wav")

            try:
                audio = AudioSegment.from_file(input_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(wav_path, format="wav")
            except Exception as e:
                print(f"Pydub conversion failed, trying ffmpeg directly: {str(e)}")
                cmd = [
                    'ffmpeg',
                    '-i', input_path,
                    '-ac', '1',
                    '-ar', '16000',
                    '-acodec', 'pcm_s16le',
                    wav_path
                ]
                subprocess.run(cmd, check=True)

            if not os.path.exists(wav_path):
                raise RuntimeError(f"Failed to create temporary WAV file: {wav_path}")

            self.temp_files.append(wav_path)
            print(f"Converted to temporary WAV: {wav_path}")
            return wav_path
        except Exception as e:
            print(f"Error converting audio file: {str(e)}")
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
            raise
    
    def analyze_emotion(self, audio_path):
        """Analyze emotion from audio file"""
        try:
            result = self.emotion_classifier(audio_path)
            return max(result, key=lambda x: x['score'])['label']
        except Exception as e:
            print(f"Emotion analysis failed: {e}")
            return "unknown"

    def transcribe_audio(self, audio_path):
        try:
            if not audio_path.lower().endswith('.wav'):
                audio_path = self.convert_to_whisper_format(audio_path)
                if not audio_path:
                    return None

            print("Transcribing audio with Faster-Whisper...")
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="en"
            )

            # Analyze emotion
            emotion = self.analyze_emotion(audio_path)
            print(f"Detected Emotion: {emotion.upper()}")

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

            return {
                "raw_transcription": full_text.strip(),
                "segments": segment_list,
                "audio_metadata": {
                    "original_path": audio_path,
                    "sample_rate": 16000,
                    "duration": len(AudioSegment.from_wav(audio_path)) / 1000,
                    "emotion": emotion
                }
            }
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None

    def clean_transcription(self, text, remove_duplicate_sentences=True, remove_duplicate_words=False):
        """
        Clean transcription by removing repeated sentences or words.
        
        Args:
            text (str): The raw transcription text.
            remove_duplicate_sentences (bool): If True, remove duplicate sentences.
            remove_duplicate_words (bool): If True, remove consecutive duplicate words.
        
        Returns:
            str: Cleaned transcription text.
        """
        if not text.strip():
            print("Empty transcription, returning empty string.")
            return ""
        
        if remove_duplicate_sentences:
            sentences = split_into_sentences(text, language="en")
            seen_sentences = set()
            unique_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen_sentences:
                    seen_sentences.add(sentence)
                    unique_sentences.append(sentence)
                else:
                    print(f"Removed duplicate sentence: {sentence}")
            text = " ".join(unique_sentences)
        
        if remove_duplicate_words:
            words = text.split()
            cleaned_words = [words[0]] if words else []
            for i in range(1, len(words)):
                if words[i] != words[i-1]:
                    cleaned_words.append(words[i])
                else:
                    print(f"Removed duplicate word: {words[i]}")
            text = " ".join(cleaned_words)
        
        return text.strip()

    def translate_to_malayalam(self, text_or_dict):
        malayalam_reverse_map = {
            'ആശയവുമില്ല': 'ഐഡിയയുമില്ല',
            'എന്ന് ': 'നിന്നും ',
            'ചെയ്യാൻ': 'ചെയ്യാവോ',
            'പരിഹാരങ്ങളും': 'സൊല്യൂഷൻസ്',
            'പരിശീലനാർത്ഥി': 'ട്രെയിനീ',
            'തീരുമാനം': 'ഡിസിഷൻ',
            'ഉദ്യോഗം': 'ജോബ്',
            'പദ്ധതി': 'പ്രൊജക്ട്',
            'സമസ്യ': 'പ്രോബ്ലം',
            'സഹായം': 'ഹെൽപ്പ്',
            'ഉദാഹരണം': 'എക്സാംപിൾ',
            'വിവരണം': 'ഡിസ്‌ക്രിപ്ഷൻ',
            'വിലയിരുത്തൽ': 'ഇവാലുവേഷൻ',
            'പരീക്ഷണം': 'ടെസ്റ്റ്',
            'പരീക്ഷ': 'എക്സാം',
            'പഠനം': 'സ്റ്റഡി',
            'വിദ്യാർത്ഥി': 'സ്റ്റുഡന്റ്',
            'വ്യാപാരം': 'ബിസിനസ്',
            'തൊഴിലാളി': 'എംപ്ലോയി',
            'ഉദ്യോഗസ്ഥൻ': 'സ്റ്റാഫ്',
            'കൂടിയാ�ലോചന': 'മീറ്റിംഗ്',
            'ആരംഭം': 'സ്റ്റാർട്ട്',
            'അവസാനം': 'എൻഡ്',
            'സംശയം': 'ഡൗട്ട്',
            'പിന്തുണ': 'സപ്പോർട്ട്',
            'ആശ്വാസം': 'റിലീഫ്',
            'താല്പര്യം': 'ഇന്ററസ്റ്റ്',
            'പ്രതിഫലം': 'റിവാർഡ്',
            'പരിശോധന': 'ചെക്ക്',
            'ഉപകരണങ്ങൾ': 'ടൂൾസ്',
            'തെളിവ്': 'പ്രൂഫ്',
            'അനുമതി': 'പർമിഷൻ',
            'ലക്ഷ്യം': 'ഗോൾ',
            'വൈഭവം': 'ഗ്ലാമർ',
            'സംരക്ഷണം': 'സെക്യൂരിറ്റി',
            'നയം': 'പോളിസി',
            'പരിധി': 'ലിമിറ്റ്',
            'പരീക്ഷണഘട്ടം': 'പൈലറ്റ്',
            'സവിശേഷത': 'ഫീച്ചർ',
            'തികഞ്ഞത്': 'ഓപ്റ്റിമൽ',
            'മാറ്റം': 'ചേഞ്ച്',
            'ഉപഭോക്താവ്': 'യൂസർ',
            'പ്രകടനം': 'പെർഫോമൻസ്',
            'വിശ്വാസം': 'ട്രസ്റ്റ്',
            'വൈഭവം': 'ഗ്രാൻഡർ',
            'വിഫലം': 'ഫെയിൽഡ്',
            'വിജയം': 'സക്സസ്',
            'നിര്‍ണ്ണയം': 'ജഡ്ജ്മെന്റ്',
            'പ്രതീക്ഷ': 'എക്‌സ്‌പെക്ടേഷൻ',
            'അവസരം': 'ഓപ്പർച്യൂണിറ്റി',
            'തീരുവിൽ': 'ഫൈനലായി',
            'കണ്ണോട്ട്': 'ഫോക്കസ്',
            'തെളിവാക്കുക': 'വെരിഫൈ ചെയ്യുക',
            'മാറ്റിവെക്കുക': 'റീഷെഡ്യൂൾ ചെയ്യുക',
            'നൽകുക': 'പ്രൊവൈഡ് ചെയ്യുക',
            'പരിഷ്കരണം': 'അപ്പ്‌ഡേറ്റ്',
            'തിരഞ്ഞെടുത്തത്': 'സെലെക്ട് ചെയ്തത്',
            'തിരഞ്ഞെടുപ്പ്': 'ഓപ്ഷൻ',
            'സ്ഥിരീകരണം': 'കോൺഫർമേഷൻ',
            'ചിട്ട': 'പ്ലാൻ',
            'പരാമർശം': 'റഫറൻസ്',
            'മാറ്റം': 'മോഡിഫിക്കേഷൻ',
            'അഭിപ്രായം': 'ഫീഡ്‌ബാക്ക്',
            'കാഴ്ചപ്പാട്': 'വ്യൂ',
            'പിന്തുടരുക': 'ഫോളോ ചെയ്യുക',
            'ഭരണസംവിധാനം': 'മാനേജ്മെന്റ്',
            'അനുഭവം': 'എക്സ്പീരിയൻസ്',
            'ഉണ്ടോ': 'വേണോ',
            'തന്ത്രശാസ്ത്ര സ്ഥാപനം': 'ഐടി ഫിർം',
            'നയം': 'പോളിസി',
            'പരിഗണന': 'കൺസിഡറേഷൻ',
            'നിബന്ധനകൾ': 'ടേംസ്',
            'സമിതി': 'കമ്മിറ്റി',
            'നിക്ഷേപം': 'ഇൻവെസ്റ്റ്‌മെന്റ്',
            'സംരംഭം': 'സ്റ്റാർട്ടപ്പ്',
            'തൊഴിൽ': 'ജോബ്',
            'പരിശീലനം': 'ട്രെയിനിംഗ്',
            'വായ്പ': 'ലോൺ',
            'തരംഗം': 'ട്രെൻഡ്',
            'നിക്ഷേപകന്‍': 'ഇൻവെസ്റ്റർ',
            'പദ്ധതി': 'പ്രൊജക്ട്',
            'അവകാശം': 'റൈറ്റ്‌സ്',
            'സുരക്ഷ': 'സെക്യൂരിറ്റി',
            'അനുമതി': 'പരമിഷൻ',
            'നയതന്ത്രം': 'ഡിപ്ലോമസി',
            'വിനിമയം': 'ട്രാൻസാക്ഷൻ',
            'പരിഹാരങ്ങളും': 'സൊല്യൂഷൻസ്',
            'മീൻ/മെൻ ടെക്‌നോളജി പ്ലാറ്റ്ഫോം': 'മീൻ ആൻഡ് മെൻ സ്റ്റാക്ക്',
            'ആവശ്യ പ്രയോഗ നിർമ്മാണം': 'ആപ്ലിക്കേഷൻ ഡെവലപ്മെന്റ്',
            'അനുഭവ സർട്ടിഫിക്കറ്റ്': 'എക്സ്പീരിയൻസ് സർട്ടിഫിക്കറ്റ്',
            'തിരഞ്ഞെടുത്ത പ്രക്രിയ': 'സ്ക്രീനിംഗ് പ്രോസസ്സ്',
            'ഇന്റർനെറ്റ് സുരക്ഷാ': 'സൈബർ സെക്യൂരിറ്റി',
            'ഡാറ്റാ ശാസ്ത്രം': 'ഡേറ്റ സയൻസ്',
            'ജീവന്ത പദ്ധതികൾ': 'ലൈവ് പ്രൊജക്ട്സ്',
            'ജീവന്ത പരിശീലനം': 'ലൈവ് ട്രെയിനിംഗ്',
            'പണമേറ്റ് പരിശീലനം': 'പെയ്ഡ് ഇന്റേൺഷിപ്പ്',
            'സുരക്ഷാ പരിഹാരങ്ങൾ': 'സെക്യൂരിറ്റി സൊല്യൂഷൻസ്',
            'പൂർത്തീകരണ സർട്ടിഫിക്കറ്റ്': 'സർട്ടിഫിക്കറ്റ് ഓഫ് കമ്പ്ലീഷൻ',
            'കൂടുതൽ വിവരം': 'മോർ ഇൻഫർമേഷൻ',
            'അവശ്യമില്ല': 'നോ നീഡ്',
            'സാധ്യമല്ല': 'നോട്ട് പൊസിബിൾ',
            'അറിയിപ്പ് ലഭിച്ചു': 'നോട്ടിഫിക്കേഷൻ റിസീവ്ഡ്',
            'ആറായിരം': 'സിക്സ് തൗസൻഡ്',
            'സോഫ്റ്റ്വെയറിൽ': 'സോഫ്റ്റ്‌വെയർ',
            'പരിഹാരങ്ങൾ': 'സൊല്യൂഷൻസ്',
            'പരിഹാരം': 'സൊല്യൂഷൻ',
            'മൃദുസ്ഥിതി പ്രയോഗം': 'സോഫ്റ്റ്‌വെയർ',
            'പരിശീലനം': 'ഇന്റേൺഷിപ്പ്',
            'സാക്ഷ്യപത്രം': 'സർട്ടിഫിക്കറ്റ്',
            'അടിസ്ഥാനഘടന': 'ഫ്രെയിംവർക്ക്',
            'ചോദ്യോത്തരസമരം': 'ഇന്റർവ്യൂ',
            'തൊഴിൽനിയമനം': 'പ്ലേസ്മെന്റ്',
            'ഭരണസംവിധാനം': 'മാനേജ്മെന്റ്',
            'അനുഭവം': 'എക്സ്പീരിയൻസ്',
            'ഉണ്ടോ': 'വേണോ',
            'തന്ത്രശാസ്ത്ര സ്ഥാപനം': 'ഐടി ഫിർം',
            'പരിഹാരവുമായി': 'സൊലൂഷൻസ്',
            'വാട്ട്‌സാപ്പ്': 'വാട്ട്‌സാപ്പ്',
            'സുരക്ഷാ': 'സെക്യൂരിറ്റി',
            'സ്ഥാപനം': 'കമ്പനി',
            'പദ്ധതി': 'പ്രൊജക്ട്',
            'ബന്ധപ്പെടുന്നു': 'വിളിക്കുന്നു',
            'ഗൂഗിൾ': 'ഗൂഗിൾ',
            'പരിഹാരവും': 'സൊലൂഷൻസ്',
            'കൃത്രിമ ബുദ്ധിമത്താ': 'എഐ',
            'മെഷീൻ ലേണിംഗ്': 'എംഎൽ',
            'മാനവ വിഭവശേഷി': 'എച് ആർ',
            'യോഗ്യത': 'ക്വാളിഫിക്കേഷൻ',
            'നികുതി': 'ഫീ',
            'വിശദാംശങ്ങൾ': 'ഡീറ്റെയിൽസ്',
            'പ്രചാരണ പത്രിക': 'ബ്രോഷർ',
            'അവകാശമാണോ?': 'ആണോ',
            'വിവരങ്ങൾ': 'ഡീറ്റെയിൽസ്',
            'അന്വേഷണം': 'ഇൻക്വയറി',
            'തിരയുന്നു': 'നോക്കിയിരുന്നു',
            'പങ്കിടാൻ': 'ഷെയർ',
            'പരിഹാരങ്ങളിൽ': 'സൊല്യൂഷൻസ്', 
            'വിശദാംശങ്ങളും': 'ഡീറ്റെയിൽസ്',
            'അവസരങ്ങൾ': 'ഓപ്പർച്യൂണിറ്റീസ്',
            'സാങ്കേതികവിദ്യ': 'ടെക്‌നോളജി',
            'സാങ്കേതികവിദ്യകൾ': 'ടെക്‌നോളജീസ്',
            'സാങ്കേതികവിദ്യയുടെ': 'ടെക്‌നോളജിയുടെ',
            'സാങ്കേതികവിദ്യകൾക്ക്': 'ടെക്‌നോളജികൾക്ക്',
            'ശരി':'യെസ് ',
            'ഇന്റർവ്യൂ': 'ഇന്റർവ്യൂ',
            
        }

        try:
            # Extract text from input
            if isinstance(text_or_dict, dict):
                text = text_or_dict.get('raw_transcription', '')
            else:
                text = text_or_dict

            if not text.strip():
                raise ValueError("No text found for translation")

            print("Translating to Malayalam...")
            # Perform translation
            ml_text = self.translator(text, max_length=512)[0]['translation_text']

            # Apply post-processing with reverse map
            for original, replacement in malayalam_reverse_map.items():
                ml_text = ml_text.replace(original, replacement)

            # Return based on input type
            if isinstance(text_or_dict, dict):
                text_or_dict['translated_malayalam'] = ml_text
                return text_or_dict
            else:
                return ml_text

        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text_or_dict

    def cleanup(self):
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted temp file: {file_path}")
            except Exception as e:
                print(f"Error deleting temp file {file_path}: {str(e)}")
        self.temp_files = []

def split_into_sentences(text, language="en"):
    try:
        if not text or not text.strip():
            print(f"No text provided for sentence splitting (language: {language})")
            return []

        if language == "en":
            try:
                sentences = sent_tokenize(text)
                if len(sentences) > 1:
                    print(f"Successfully split {len(sentences)} sentences using NLTK (English)")
                    return [s.strip() for s in sentences if s.strip()]
                else:
                    print("NLTK returned single or no sentences, trying fallback")
            except Exception as nltk_error:
                print(f"NLTK English tokenizer failed: {str(nltk_error)}")

        if language == "ml":
            try:
                sentences = sentence_tokenize.sentence_split(text, lang='mal')
                if len(sentences) > 1:
                    print(f"Successfully split {len(sentences)} sentences using Indic NLP (Malayalam)")
                    return [s.strip() for s in sentences if s.strip()]
                else:
                    print("Indic NLP returned single or no sentences, trying fallback")
            except Exception as indic_error:
                print(f"Indic NLP Malayalam tokenizer failed: {str(indic_error)}")

        print(f"Using fallback regex sentence splitting for language: {language}")
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        print(f"Regex split resulted in {len(sentences)} sentences")
        return sentences

    except Exception as e:
        print(f"All sentence splitting methods failed: {str(e)}")
        return [text.strip()] if text.strip() else []

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

def analyze_sentiment_batch(texts):
    results = sentiment_pipeline(texts)
    outputs = []
    for result in results:
        label = result['label']
        
        if "1 star" in label or "2 stars" in label:
            sentiment = {"label": "negative", "score": 0.2}
        elif "3 stars" in label:
            sentiment = {"label": "neutral", "score": 0.5}
        elif "4 stars" in label:
            sentiment = {"label": "positive", "score": 0.7}
        elif "5 stars" in label:
            sentiment = {"label": "very positive", "score": 0.9}
        else:
            sentiment = {"label": "neutral", "score": 0.5}
            
        outputs.append(sentiment)
    return outputs

def detect_intent(text, language="en"):
    """Enhanced intent detection for internship interest analysis in English and Malayalam"""
    text_lower = text.lower().strip()
    
    intent_keywords = {
        "en": {
            "Strong_interest": [
                "yes", "definitely", "ready", "want to join", "interested", 
                "share details", "send brochure", "i'll join", "let's proceed",
                "where do i sign", "share", "when can i start", "accept",
                "looking forward", "excited", "happy to", "glad to", "eager",
                "share it", "whatsapp", "i'm in"
            ],
            "Moderate_interest": [
                "maybe", "consider", "think about", "let me think", "tell me more",
                "more details", "explain", "clarify", "not sure", "possibly",
                "might", "could be", "depends", "need to check", "will decide",
                "get back", "discuss", "consult", "review", "evaluate"
            ],
            "No_interest": [
                 "can't", "won't", "don't like",
                "not now", "later", "not suitable", "decline"
            ],
            "company_query": [
                "tino software and security solutions","Tino software IT company","Tino software", "Tino software","I am Tino Software.", 
                "i am calling you from tino software and security solutions",
                "tinos software"
            ],
            "Qualification_query": [
                "qualification", "education", "computer science", "degree", "studying", "course",
                "background", "academics", "university", "college", "bsc",
                "graduate", "year of study", "curriculum", "syllabus"
            ],
            "Internship_details": [
                "internship","placement", "program","is looking for an internship", "duration", "Data Science","months", "period",
                "schedule", "timing", "timeframe", "1 to 3", "three months",
                "structure", "plan", "placement", "framework", "looking for an internship in data science"
            ],
            "Location_query": [
                "online", "offline", "location", "place", "where",
                "address", "relocate", "relocating", "from", "coming",
                "kozhikode", "kochi", "palarivattam", "hybrid", "remote"
            ],
            "Certificate_query": [
                "certificate", "certification", "document", "proof",
                "experience certificate", "training certificate", "letter",
                "completion", "award", "recognition"
            ],
            "Fee_query": [
                "fee", "payment", "cost", "amount", "charge",
                "6000", "six thousand", "money", "stipend", "salary",
                "compensation", "paid", "free"
            ],
            "Project_details": [
                "live project", "work", "assignment", "task", "project",
                "trainee", "superiors", "team", "collaborate", "develop",
                "build", "create", "implement", "hands-on", "practical"
            ],
            "Confirmation": [
                "ok", "interested", "send whatsapp", "got it",
                "acknowledge", "noted", "please send", "sent details", "agreed"
            ]
        },
        "ml": {
            "Strong_interest": [
                "തയ്യാറാണ്", "ആവശ്യമുണ്ട്", "ചെയ്യാം", "ആഗ്രഹമുണ്ട്", 
                "ഇഷ്ടമാണ്", "അറിയിച്ചോളൂ", "താൽപ്പര്യമുണ്ട്.", "ബ്രോഷർ വേണം", "വിശദാംശങ്ങൾ വേണം",
                "ശെയർ ചെയ്യുക", "ഞാൻ വരാം", "താൽപ്പര്യപ്പെടുന്നു", "ഉത്സാഹം", "താത്പര്യം",
                "സമ്മതം", "അംഗീകരിക്കുന്നു", "ഹാപ്പിയാണ്", "ഞാൻ ചെയ്യാം",
                "വാട്സാപ്പിൽ അയക്കൂ", "ആവശ്യമാണ്"
            ],
            "Moderate_interest": [
                "ആലോചിക്കാം", "നോക്കാം", "താല്പര്യമുണ്ട്", "ഇന്റെറസ്റ്റഡ്",
                "പറയാം", "ക്ഷണിക്കുക", "ചിന്തിക്കാം", "കാണാം", "ഉത്തരമില്ല",
                "കൂടുതൽ വിവരങ്ങൾ", "വ്യാഖ്യാനിക്കുക", "അവലംബിക്കുക"
            ],
            "No_interest": [
                "ഇല്ല", "വേണ്ട", "സാധ്യമല്ല", "ഇഷ്ടമല്ല"
            ],
            "company_query": [
                "ടിനോ സോഫ്റ്റ്വെയറിൽ", "ടിനോ സോഫ്റ്റ്വെയർ", "ടിനോ"
            ],
            "Qualification_query": [
                "വിദ്യാഭ്യാസം", "ഡിഗ്രി", "ബിസി", "പഠിക്കുന്നു", 
                "പഠനം", "അധ്യയനം", "ക്ലാസ്", "വർഷം", 
                "കോഴ്‌സ്", "സിലബസ്", "വിദ്യാർഥി", "ഗണിതം", "സയൻസ്"
            ],
            "Internship_details": [
                "ഇന്റെണ്ഷിപ്", "പരിശീലനം","ഡാറ്റാ സയൻസിലെ", "പ്ലെയ്സ്മെന്റ്", 
                "മാസം", "സമയക്രമം", "ടൈമിംഗ്", "1 മുതൽ 3 വരെ", 
                "അവസാന വർഷം", "ലൈവ്", "ഫ്രെയിംവർക്ക്", "സ്ഥിരമായി", 
                "ഡാറ്റാ സയൻസിലെ", "ഇന്റേൺഷിപ്പ്","ഡാറ്റാ സയൻസിലെ ഇന്റേൺഷിപ്പ്"
            ],
            "Location_query": [
                "ഓൺലൈൻ", "ഓഫ്ലൈൻ", "സ്ഥലം", "വിലാസം", "കഴിഞ്ഞ്", 
                "എവിടെ", "കൊഴിക്കോട്", "പാലാരിവട്ടം", "മാറ്റം", 
                "റിലൊക്കേറ്റ്", "വരുന്നു", "എവിടെ നിന്നാണ്", "ഹൈബ്രിഡ്", "വിലാസം"
            ],
            "Certificate_query": [
                "സർട്ടിഫിക്കറ്റ്", "ഡോക്യുമെന്റ്", "പ്രമാണം", "സാക്ഷ്യപത്രം", "കമ്പ്ലീഷൻ"
            ],
            "Fee_query": [
                "ഫീസ്", "പണം", "6000", "ആറ് ആയിരം", "കാണിക്ക്", 
                "മാസതൊട്ടി", "ചാർജ്", "റുമണറേഷൻ", "ഫ്രീ", 
                "ശമ്പളം", "സ്റ്റൈപെൻഡ്"
            ],
            "Project_details": [
                "പ്രോജക്ട്", "ലൈവ് പ്രോജക്ട്", "പ്രവൃത്തി", "ടാസ്‌ക്", 
                "ടീം", "മേധാവി", "ട്രെയിനി", "സഹപ്രവർത്തനം", 
                "ഡവലപ്പുചെയ്യുക", "സൃഷ്ടിക്കുക", "ഇമ്പ്ലിമെന്റുചെയ്യുക", 
                "പ്രായോഗികം", "അഭ്യാസം"
            ],
            "Confirmation": [
                "ശരി", "താല്പര്യമുണ്ട്", "തിരയുന്നു", "ഇഷ്ടമുണ്ട്", "വാട്സാപ്പിൽ അയക്കൂ", 
                "വാട്സാപ്പ്", "വാട്ട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു", 
                "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", "ഓക്കെ","യെസ്  ",
                "അക്ക്നലഡ്ജ്", "ക്ലിയർ", 
                "തയാറാണ്", "അറിയിപ്പ് ലഭിച്ചു", "വാട്ട്സ്ആപ്പിലെ","ഞാൻ അതിനായി നോക്കിയിരുന്നു"
            ]
        }
    }

    if any(keyword in text_lower for keyword in intent_keywords[language]["Confirmation"]):
        return {"intent": "Confirmation", "sentiment": "very positive", "sentiment_score": 0.9}
    
    if any(keyword in text_lower for keyword in intent_keywords[language]["Strong_interest"]):
        return {"intent": "Strong_interest", "sentiment": "positive", "sentiment_score": 0.7}
    
    if any(keyword in text_lower for keyword in intent_keywords[language]["company_query"]):
        return {"intent": "company_query", "sentiment": "neutral", "sentiment_score": 0.5}
    
    if any(keyword in text_lower for keyword in intent_keywords[language]["No_interest"]):
        return {"intent": "No_interest", "sentiment": "negative", "sentiment_score": 0.2}
    
    if any(keyword in text_lower for keyword in intent_keywords[language]["Moderate_interest"]):
        return {"intent": "Moderate_interest", "sentiment": "neutral", "sentiment_score": 0.5}
    
    for intent, keywords in intent_keywords[language].items():
        if intent not in ["Confirmation", "company_query", "Strong_interest", "No_interest", "Moderate_interest"]:
            if any(keyword in text_lower for keyword in keywords):
                return {"intent": intent, "sentiment": "neutral", "sentiment_score": 0.5}
    
    return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

def analyze_text(text, language="en"):
    sentences = split_into_sentences(text, language)
    if not sentences:
        return []

    analysis = []
    for i, sentence in enumerate(sentences):
        intent_result = detect_intent(sentence, language)
        
        sentiment_result = analyze_sentiment_batch([sentence])[0] if sentence.strip() else {
            "label": "neutral",
            "score": 0.5
        }
        
        final_sentiment = intent_result.get("sentiment", sentiment_result["label"])
        final_score = intent_result.get("sentiment_score", sentiment_result["score"])
        
        analysis.append({
            "sentence_id": f"{language}_{i+1}",
            "text": sentence,
            "language": language,
            "intent": intent_result["intent"],
            "sentiment": final_sentiment,
            "sentiment_score": final_score,
            "word_count": len(sentence.split()),
            "char_count": len(sentence)
        })
    return analysis

def save_analysis_to_csv(analysis, filename_prefix):
    if not analysis:
        print("No analysis data to save")
        return None

    df = pd.DataFrame(analysis)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_analysis_{timestamp}.csv"
    os.makedirs("analysis_results", exist_ok=True)
    full_path = os.path.join("analysis_results", filename)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"✅ Analysis saved to {full_path}")
    return full_path

def csv_to_pdf(csv_path, title, output_pdf_path):
    """Convert a CSV file to a PDF with a table representation"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    rows_per_page = 20
    num_rows = len(df)
    num_pages = (num_rows // rows_per_page) + (1 if num_rows % rows_per_page else 0)

    with PdfPages(output_pdf_path) as pdf:
        for page in range(num_pages):
            start_row = page * rows_per_page
            end_row = min(start_row + rows_per_page, num_rows)
            page_df = df.iloc[start_row:end_row]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')

            if page == 0:
                fig.suptitle(title, fontsize=16, y=0.95, fontproperties=default_font)

            table = ax.table(
                cellText=page_df.values,
                colLabels=page_df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#f0f0f0'] * len(page_df.columns)
            )

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for key, cell in table.get_celld().items():
                cell.set_fontsize(8)
                text = str(cell.get_text().get_text())
                if is_malayalam(text) and malayalam_font:
                    cell.set_text_props(fontproperties=malayalam_font)
                else:
                    cell.set_text_props(fontproperties=default_font)
            table.scale(1, 1.5)

            table.auto_set_column_width(range(len(page_df.columns)))

            plt.figtext(0.95, 0.05, f'Page {page + 1} of {num_pages}', ha='right', fontsize=10, fontproperties=default_font)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"✅ CSV converted to PDF: {output_pdf_path}")
    return output_pdf_path

def combine_pdfs(pdf_paths, output_path):
    """Combine multiple PDF files into a single PDF with font embedding"""
    merger = PdfMerger()

    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            merger.append(pdf_path)
        else:
            print(f"Warning: PDF file not found for merging: {pdf_path}")

    merger.write(output_path)
    merger.close()
    print(f"✅ Combined PDFs into: {output_path}")
    return output_path

def generate_analysis_pdf(en_analysis, ml_analysis, comparison, filename_prefix):
    """Generate a PDF report with analysis metrics and visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"analysis_results/{filename_prefix}_visual_report_{timestamp}.pdf"
    os.makedirs("analysis_results", exist_ok=True)
    
    if malayalam_font:
        font_path = malayalam_font.get_file()
        font_name = malayalam_font.get_name()
        print(f"Using Malayalam font: {font_name} at {font_path}")
    else:
        print("No Malayalam font configured.")
    print(f"Using default font: DejaVu Sans")
    
    en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
    ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
    combined_avg = (en_avg_score + ml_avg_score) / 2
    lead_score = int(combined_avg * 100)
    
    en_sentiments = [item["sentiment"] for item in en_analysis]
    ml_sentiments = [item["sentiment"] for item in ml_analysis]
    en_scores = [item["sentiment_score"] for item in en_analysis]
    ml_scores = [item["sentiment_score"] for item in ml_analysis]
    sentence_numbers = list(range(1, len(en_analysis)+1)) if en_analysis else []
    
    with PdfPages(pdf_filename) as pdf:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.8, "Conversation Analysis Report", ha='center', va='center', size=20, fontproperties=default_font)
        plt.text(0.5, 0.7, f"Generated on {datetime.now().strftime('%Y-%m-d %H:%M:%S')}", 
                ha='center', va='center', size=12, fontproperties=default_font)
        plt.text(0.5, 0.6, f"Filename: {filename_prefix}", ha='center', va='center', size=12, fontproperties=default_font)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.text(0.1, 0.9, "Key Metrics", size=16, fontproperties=default_font)
        plt.text(0.1, 0.8, f"English Avg Sentiment: {en_avg_score:.2f}", size=12, fontproperties=default_font)
        plt.text(0.1, 0.7, f"Malayalam Avg Sentiment: {ml_avg_score:.2f}", size=12, fontproperties=default_font)
        plt.text(0.1, 0.6, f"Combined Avg Sentiment: {combined_avg:.2f}", size=12, fontproperties=default_font)
        plt.text(0.1, 0.5, f"Calculated Lead Score: {lead_score}/100", size=12, fontproperties=default_font)
        
        interpretation = ""
        if lead_score >= 70:
            interpretation = "High interest lead"
        elif lead_score >= 40:
            interpretation = "Moderate interest lead"
        else:
            interpretation = "Low interest lead"
        plt.text(0.1, 0.4, f"Interpretation: {interpretation}", size=12, fontproperties=default_font)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        if en_analysis and ml_analysis:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            pd.Series(en_sentiments).value_counts().plot(kind='bar', color='skyblue')
            plt.title('English Sentiment Distribution', fontproperties=default_font)
            plt.xticks(rotation=45, fontproperties=default_font)
            
            plt.subplot(1, 2, 2)
            pd.Series(ml_sentiments).value_counts().plot(kind='bar', color='lightgreen')
            plt.title('Malayalam Sentiment Distribution', fontproperties=malayalam_font if malayalam_font else default_font)
            plt.xticks(rotation=45, fontproperties=malayalam_font if malayalam_font else default_font)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(sentence_numbers, en_scores, marker='o', label='English', color='blue')
            plt.plot(sentence_numbers, ml_scores, marker='s', label='Malayalam', color='green')
            plt.xlabel('Sentence Number', fontproperties=default_font)
            plt.ylabel('Sentiment Score', fontproperties=default_font)
            plt.title('Sentiment Trend Over Conversation', fontproperties=default_font)
            plt.legend(prop=default_font)
            plt.grid(True)
            for label in plt.gca().get_xticklabels():
                label.set_fontproperties(default_font)
            for label in plt.gca().get_yticklabels():
                label.set_fontproperties(default_font)
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            en_intents = [item["intent"] for item in en_analysis]
            pd.Series(en_intents).value_counts().plot(kind='bar', color='orange')
            plt.title('Intent Distribution', fontproperties=default_font)
            plt.xticks(rotation=45, fontproperties=default_font)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sentiment_diffs = [abs(en - ml) for en, ml in zip(en_scores, ml_scores)]
            plt.hist(sentiment_diffs, bins=10, color='purple', alpha=0.7)
            plt.xlabel('Sentiment Score Difference', fontproperties=default_font)
            plt.ylabel('Frequency', fontproperties=default_font)
            plt.title('English-Malayalam Sentiment Differences', fontproperties=default_font)
            for label in plt.gca().get_xticklabels():
                label.set_fontproperties(default_font)
            for label in plt.gca().get_yticklabels():
                label.set_fontproperties(default_font)
            pdf.savefig()
            plt.close()
    
    print(f"✅ Visual PDF report generated: {pdf_filename}")
    return pdf_filename

def compare_analyses(en_analysis, ml_analysis):
    comparison = []
    for en, ml in zip(en_analysis, ml_analysis):
        comparison.append({
            "sentence_id": en["sentence_id"],
            "english_text": en["text"],
            "malayalam_text": ml["text"],
            "intent_match": en["intent"] == ml["intent"],
            "english_intent": en["intent"],
            "malayalam_intent": ml["intent"],
            "sentiment_diff": abs(en["sentiment_score"] - ml["sentiment_score"]),
            "english_sentiment": en["sentiment"],
            "malayalam_sentiment": ml["sentiment"]
        })
    return comparison

def print_analysis_summary(analysis, title):
    print(f"\n=== {title} Analysis Summary ===")
    print(f"Total Sentences: {len(analysis)}")
    if not analysis:
        return
    sentiment_counts = pd.Series([item["sentiment"] for item in analysis]).value_counts()
    print("\nSentiment Distribution:")
    print(sentiment_counts.to_string())

    intent_counts = pd.Series([item["intent"] for item in analysis]).value_counts()
    print("\nIntent Distribution:")
    print(intent_counts.to_string())

    avg_score = sum(item["sentiment_score"] for item in analysis) / len(analysis)
    print(f"\nAverage Sentiment Score: {avg_score:.2f}")

def create_zip_archive(audio_path, raw_transcription, ml_translation, pdf_report, 
                       en_analysis, ml_analysis, user_filename, drive_folder_id=None):
    """Create a zip archive with all analysis files, including a merged summary report, and upload to Google Drive."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
    ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
    combined_avg = (en_avg_score + ml_avg_score) / 2
    lead_score = int(combined_avg * 100)
    
    positive_intents = sum(1 for item in en_analysis if item["intent"] in ["Strong_interest","Fee_query", "Moderate_interest","Confirmation"])
    intent_score = int((positive_intents / len(en_analysis)) * 100) if en_analysis else 0
    
    base_filename = f"{user_filename}_L{lead_score}_I{intent_score}_{timestamp}"
    zip_filename = f"analysis_results/{base_filename}.zip"
    
    wav_path = None
    if audio_path.lower().endswith('.wav'):
        wav_path = audio_path
    else:
        temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                if file.endswith('.wav'):
                    wav_path = os.path.join(temp_dir, file)
                    break
    
    os.makedirs("analysis_results", exist_ok=True)
    temp_files = []
    
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            if wav_path and os.path.exists(wav_path):
                audio_ext = '.wav'
                new_audio_name = f"{base_filename}{audio_ext}"
                zipf.write(wav_path, arcname=new_audio_name)
            elif os.path.exists(audio_path):
                audio_ext = os.path.splitext(audio_path)[1]
                new_audio_name = f"{base_filename}{audio_ext}"
                zipf.write(audio_path, arcname=new_audio_name)
            else:
                print("Warning: No audio file found to include in archive")
            
            zipf.writestr(f"{base_filename}_transcription.txt", raw_transcription)
            zipf.writestr(f"{base_filename}_translation.txt", ml_translation)
            
            en_csv = save_analysis_to_csv(en_analysis, "english")
            ml_csv = save_analysis_to_csv(ml_analysis, "malayalam")
            comparison = compare_analyses(en_analysis, ml_analysis)
            comparison_csv = save_analysis_to_csv(comparison, "comparison")
            
            pdf_paths = []
            if en_csv and os.path.exists(en_csv):
                en_pdf = f"analysis_results/{base_filename}_english_analysis.pdf"
                csv_to_pdf(en_csv, "English Analysis", en_pdf)
                if os.path.exists(en_pdf):
                    pdf_paths.append(en_pdf)
                    temp_files.append(en_pdf)
                zipf.write(en_csv, arcname=f"{base_filename}_english_analysis.csv")
                temp_files.append(en_csv)
            
            if ml_csv and os.path.exists(ml_csv):
                ml_pdf = f"analysis_results/{base_filename}_malayalam_analysis.pdf"
                csv_to_pdf(ml_csv, "Malayalam Analysis", ml_pdf)
                if os.path.exists(ml_pdf):
                    pdf_paths.append(ml_pdf)
                    temp_files.append(ml_pdf)
                zipf.write(ml_csv, arcname=f"{base_filename}_malayalam_analysis.csv")
                temp_files.append(ml_csv)
            
            if comparison_csv and os.path.exists(comparison_csv):
                comparison_pdf = f"analysis_results/{base_filename}_comparison_analysis.pdf"
                csv_to_pdf(comparison_csv, "Comparison Analysis", comparison_pdf)
                if os.path.exists(comparison_pdf):
                    pdf_paths.append(comparison_pdf)
                    temp_files.append(comparison_pdf)
                zipf.write(comparison_csv, arcname=f"{base_filename}_comparison.csv")
                temp_files.append(comparison_csv)
            
            if os.path.exists(pdf_report):
                pdf_paths.append(pdf_report)
            else:
                print(f"Warning: PDF report not found at {pdf_report}")
            
            merged_pdf = f"analysis_results/{base_filename}_summary_report.pdf"
            if pdf_paths:
                combine_pdfs(pdf_paths, merged_pdf)
                if os.path.exists(merged_pdf):
                    zipf.write(merged_pdf, arcname=f"{base_filename}_summary_report.pdf")
                    temp_files.append(merged_pdf)
                else:
                    print(f"Warning: Merged PDF not created at {merged_pdf}")
            else:
                print("Warning: No PDFs available to merge")
    
        print(f"✅ Created zip archive: {zip_filename}")
        
        if drive_folder_id:
            service = authenticate_gdrive()
            if service:
                folder_id = create_gdrive_folder(f"Analysis_{base_filename}", parent_id=drive_folder_id, service=service)
                if folder_id:
                    uploaded_file = upload_to_gdrive(
                        zip_filename,
                        folder_id=folder_id,
                        service=service,
                        custom_filename=f"{base_filename}.zip"
                    )
                    if uploaded_file:
                        metadata_entry = {
                            'filename': base_filename,
                            'lead_score': lead_score,
                            'intent_score': intent_score,
                            'timestamp': timestamp,
                            'folder_id': folder_id
                        }
                        update_metadata_csv(folder_id, metadata_entry, service=service, root_folder_id=drive_folder_id)
        
        return zip_filename, base_filename, lead_score, intent_score
    
    finally:
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted temp file: {file_path}")
            except Exception as e:
                print(f"Error deleting temp file {file_path}: {str(e)}")

def main_analysis_workflow():
    """Main workflow for audio analysis"""
    transcriber = MalayalamTranscriptionPipeline()

    try:
        audio_path = input("Enter path to Malayalam audio file: ").strip()
        if not os.path.exists(audio_path):
            print("Error: File not found")
            exit(1)

        print("\n🔊 Transcribing audio...")
        results = transcriber.transcribe_audio(audio_path)
        if not results or not results.get("raw_transcription"):
            print("Transcription failed.")
            exit(1)

        raw_transcription = results["raw_transcription"]
        print("\n=== Raw English Transcription ===")
        print(raw_transcription)

        print("\n🌐 Translating to Malayalam...")
        results = transcriber.translate_to_malayalam(results)
        ml_translation = results.get("translated_malayalam", "")
        print("\n=== Malayalam Translation ===")
        print(ml_translation)

        print("\n🔍 Analyzing texts...")
        en_analysis = analyze_text(raw_transcription, "en")
        ml_analysis = analyze_text(ml_translation, "ml")

        user_filename = input("\nEnter a name for your analysis files (without extension): ").strip()
        if not user_filename:
            user_filename = "conversation_analysis"
        
        drive_folder_id = input("\nEnter Google Drive folder ID for upload (leave empty to skip): ").strip() or None
        
        comparison = compare_analyses(en_analysis, ml_analysis)
        pdf_report = generate_analysis_pdf(en_analysis, ml_analysis, comparison, user_filename)

        print("\n=== Analysis Complete ===")
        print_analysis_summary(en_analysis, "English")
        print_analysis_summary(ml_analysis, "Malayalam")

        zip_filename, base_filename, lead_score, intent_score = create_zip_archive(
            audio_path, raw_transcription, ml_translation, pdf_report,
            en_analysis, ml_analysis, user_filename, drive_folder_id
        )

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    finally:
        transcriber.cleanup()

def search_menu():
    print("Search functionality not implemented yet.")
    # Add your search logic here

if __name__ == "__main__":
    while True:
        print("\n=== Main Menu ===")
        print("1. Analyze new audio file")
        print("2. Search existing analyses")
        print("3. Exit")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            main_analysis_workflow()
        elif choice == "2":
            search_menu()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")