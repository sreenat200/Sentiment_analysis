import os
import tempfile
from datetime import datetime
import torch
from pydub import AudioSegment
from deep_translator import GoogleTranslator
import pandas as pd
from indicnlp.tokenize import sentence_tokenize
import re
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
from matplotlib import rcParams
import requests
from transformers import pipeline
from PyPDF2 import PdfMerger
import zipfile
import shutil
import urllib.request
from deep_translator import GoogleTranslator
from transformers import pipeline
import logging
logger = logging.getLogger(__name__)
import os
import streamlit as st
from huggingface_hub import login as hf_login

def load_token_from_secrets():
    return st.secrets.get("HF_TOKEN", "")

hf_token = load_token_from_secrets()
if hf_token:
    hf_login(token=hf_token)
else:
    st.error("Hugging Face token not found in Streamlit secrets.")



indic_nlp_resource_path = os.path.expanduser('~/.indic_nlp_resources')
os.makedirs(indic_nlp_resource_path, exist_ok=True)

try:
    from indicnlp import common
    common.set_resources_path(indic_nlp_resource_path)
    common.init()
    print(f"Indic NLP resources successfully initialized at {indic_nlp_resource_path}")
except Exception as e:
    print(f"Error initializing Indic NLP resources: {str(e)}")
    try:
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

# Check if font already exists in permanent storage
if os.path.exists(font_path):
    print("Found local Malayalam font at", font_path)
    malayalam_font = fm.FontProperties(fname=font_path)
else:
    # Only download if not already saved
    print("Local Malayalam font not found. Attempting to download static font...")
    try:
        fallback_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerifMalayalam/NotoSerifMalayalam-Regular.ttf"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(fallback_url, headers=headers, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(font_path), exist_ok=True)  # Ensure directory exists

        with open(font_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Downloaded Malayalam font to {font_path}")
        malayalam_font = fm.FontProperties(fname=font_path)

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

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
from pydub import AudioSegment


import os
import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Faster-Whisper {model_size} model on {self.device}...")

        if self.device == "cpu":
            num_threads = os.cpu_count() or 1
            torch.set_num_threads(num_threads)
            print(f"Using {num_threads} CPU threads")
            compute_type = "int8"
            cpu_threads = num_threads
        else:
            print("Using CUDA")
            compute_type = "float16"
            cpu_threads = 0

        try:
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type,
                cpu_threads=cpu_threads
            )
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Error loading WhisperModel: {e}")
            raise

        self.emotion_model = None
        self.emotion_tokenizer = None
        try:
            model_name = "j-hartmann/emotion-english-multilingual"
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.emotion_model.to(self.device)
            print("Emotion model loaded to device.")
        except Exception as e:
            print(f"Failed to load emotion model: {e}")

        self.temp_files = []
        self.translator = None

        self.lang_names = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'ks': 'Kashmiri',
    'ne': 'Nepali',
    'sd': 'Sindhi',
    'sa': 'Sanskrit',
    'bho': 'Bhojpuri',
    'doi': 'Dogri',
    'mai': 'Maithili',
    'mni': 'Manipuri',
    'kok': 'Konkani',
    'brx': 'Bodo',
    'sat': 'Santali',
    'ml': 'Malayalam',
    'raj': 'Rajasthani',
    'awa': 'Awadhi',
    'bh': 'Bihari',
    'chh': 'Chhattisgarhi',
    'mag': 'Magahi',
    'lep': 'Lepcha',
    'lmn': 'Lambadi',
    'mwr': 'Marwari',
    'noe': 'Nocte',
    'nnp': 'Nepali (India)',
    'gbm': 'Garhwali',
    'hne': 'Haryanvi',
}


    def convert_to_whisper_format(self, audio_path):
        """Convert any audio to 16kHz mono WAV for Whisper."""
        try:
            audio = AudioSegment.from_file(audio_path)
            print("Original Audio - Sample rate:", audio.frame_rate, "Channels:", audio.channels)
            audio = audio.set_frame_rate(16000).set_channels(1)
            temp_path = "temp_audio.wav"
            audio.export(temp_path, format="wav")
            self.temp_files.append(temp_path)
            print("Converted to:", temp_path)
            return temp_path
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            return None

    def analyze_emotion(self, text):
        if not self.emotion_model or not self.emotion_tokenizer:
            return "unknown"
        inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return "positive" if predicted_class == 1 else "negative"

    def detect_language_from_audio(self, audio_path):
        """
        Detect the language from an audio file using the Whisper model.
        Converts audio to WAV if needed and returns detected language and segments.
        """
        import time
        start_time = time.time()

        try:
            if not audio_path.lower().endswith('.wav'):
                # Convert to WAV for language detection
                audio_path = self.convert_to_whisper_format(audio_path)
                if not audio_path:
                    # Audio conversion failed
                    return None

            # Run language detection
            try:
                segments, info = self.model.transcribe(
                    audio_path,
                    language=None,
                    vad_filter=True,
                    beam_size=5,
                    task="transcribe"
                )
            except Exception as e:
                if "onnxruntime" in str(e).lower():
                    # Retry without VAD if onnxruntime is not installed
                    segments, info = self.model.transcribe(
                        audio_path,
                        language=None,
                        vad_filter=False,
                        beam_size=5,
                        task="transcribe"
                    )
                else:
                    # Transcription for language detection failed
                    raise

            detected_lang = getattr(info, 'language', None)
            if not detected_lang:
                # No language detected
                return None

            detected_lang_name = self.lang_names.get(detected_lang, detected_lang)
            # Optionally, print or log detection info here if needed
            # print(f"Detected Language: {detected_lang_name} ({detected_lang})")
            # print(f"Language detection took {time.time() - start_time:.2f} seconds")

            return {
                'language': detected_lang,
                'language_name': detected_lang_name,
                'segments': list(segments)
            }

        except Exception as e:
            # Optionally, print or log the error here if needed
            # print(f"Language detection failed: {str(e)}")
            return None
        finally:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Optionally, print or log cache clearing
                print("Cleared CUDA cache")
    
    

    def analyze_emotion(self, text):
        """Manual emotion prediction. Always returns string."""
        if self.emotion_model is None or self.emotion_tokenizer is None:
            return "unknown"

        if not isinstance(text, str) or not text.strip():
            return "unknown"

        try:
            inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                top_class = torch.argmax(probs, dim=1).item()

            return self.emotion_model.config.id2label[top_class]

        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return "unknown"

    def translate_to_language(self, text_or_dict, source_lang, target_lang):
        # Standard ISO 639-1 language codes for GoogleTranslator
        lang_codes = {
            'en': 'en',  # English
            'hi': 'hi',  # Hindi
            'bn': 'bn',  # Bengali
            'te': 'te',  # Telugu
            'mr': 'mr',  # Marathi
            'ta': 'ta',  # Tamil
            'ur': 'ur',  # Urdu
            'gu': 'gu',  # Gujarati
            'kn': 'kn',  # Kannada
            'ml': 'ml',  # Malayalam
            'pa': 'pa',  # Punjabi
            'or': 'or',  # Odia
            'as': 'as',  # Assamese
            'ne': 'ne'   # Nepali
        }

        # Malayalam-specific reverse map
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
            'കൂടിയാലോചന': 'മീറ്റിംഗ്',
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
            'അനുമതി': 'പർമിഷൻ',
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
            'നികുതി': 'ടാക്സ്'
        }

        try:
            # Extract input text
            if isinstance(text_or_dict, dict):
                text = text_or_dict.get('raw_transcription', '')
                if not text:
                    logger.warning("No raw_transcription found in input dictionary")
                    return {'translated_text': ''}
            else:
                text = text_or_dict
                if not isinstance(text, str):
                    logger.warning(f"Input text_or_dict is not a string or dict: {type(text_or_dict)}. Converting to string.")
                    text = str(text_or_dict)

            # Validate input text
            text = text.strip()
            if not text:
                logger.warning("Input text is empty after stripping")
                return {'translated_text': ''}

            # Clean text to avoid issues with special characters
            text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except basic punctuation

            # Validate language codes
            if source_lang not in lang_codes:
                logger.error(f"Unsupported source language: {source_lang}")
                return {'translated_text': '', 'error': f"Unsupported source language: {source_lang}"}
            if target_lang not in lang_codes:
                logger.error(f"Unsupported target language: {target_lang}")
                return {'translated_text': '', 'error': f"Unsupported target language: {target_lang}"}

            # Get language codes for GoogleTranslator
            source_code = lang_codes.get(source_lang, 'en')
            target_code = lang_codes.get(target_lang, 'ml')
            logger.info(f"Translating from {source_lang} ({source_code}) to {target_lang} ({target_code})")

            # Perform translation
            try:
                translator = GoogleTranslator(source=source_code, target=target_code)
                translated_text = translator.translate(text)
            except Exception as e:
                logger.error(f"GoogleTranslator failed for {source_lang} to {target_lang}: {str(e)}")
                return {'translated_text': '', 'error': f"Translation failed: {str(e)}"}

            if not translated_text:
                logger.warning("Translation returned empty result")
                return {'translated_text': ''}

            # Apply Malayalam-specific reverse map if target language is Malayalam
            if target_lang == 'ml':
                for incorrect, correct in malayalam_reverse_map.items():
                    translated_text = translated_text.replace(incorrect, correct)

            logger.debug(f"Translated text (truncated): {translated_text[:100]}...")
            return {'translated_text': translated_text}

        except Exception as e:
            logger.error(f"Translation failed for {source_lang} to {target_lang}: {str(e)}", exc_info=True)
            return {'translated_text': '', 'error': f"Translation failed: {str(e)}"}

    def cleanup(self):
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted temp file: {file_path}")
            except Exception as e:
                print(f"Error deleting temp file {file_path}: {str(e)}")
        self.temp_files = []


def split_into_sentences(text: str, language: str = "en") -> list[str]:
    
    try:
        if not text or not text.strip():
            print(f"No text provided for sentence splitting (language: {language})")
            return []

        print(f"Using regex sentence splitting as primary for language: {language}")
        sentence_endings = re.compile(
            r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\d\.\d)(?<=[.,?!।॥]|\n)(?=\s|$|[^\s.,?!])'
        )
        split_pattern = re.compile(r'\s*,\s*\.\s*\?\s*')
        common_abbreviations = {
            'en': r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Inc|Co|Ltd|U\.S|yes)\.',
            'ml': r'\b(?:ഡോ|ശ്രീ|ശ്രീമതി|പ്രൊ|കോ|യെസ്)\.'
        }
        abbr_pattern = common_abbreviations.get(language, r'\b(?:Dr|Mr|Mrs|Ms)\.')

        sentences = sentence_endings.split(text)
        cleaned_sentences = []
        current_sentence = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            split_sentences = split_pattern.split(sent)
            split_sentences = [s.strip() for s in split_sentences if s.strip()]

            for split_sent in split_sentences:
                if current_sentence:
                    if re.search(abbr_pattern + r'$', current_sentence):
                        current_sentence += " " + split_sent
                    else:
                        cleaned_sentences.append(current_sentence)
                        current_sentence = split_sent
                else:
                    current_sentence = split_sent
        if current_sentence:
            cleaned_sentences.append(current_sentence)

        
        final_sentences = []
        for sent in cleaned_sentences:
            subsentences = re.split(
                r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\d\.\d)(?<=[.,])(?=\s|$|[^\s.,?!])',
                sent
            )
            final_sentences.extend(subsent.strip() for subsent in subsentences if subsent.strip())

        if final_sentences:
            print(f"Regex split resulted in {len(final_sentences)} sentences")
            print(f"Sentences: {final_sentences}")
            return final_sentences
        else:
            print(f"Regex splitter returned no sentences for {language}, falling back to Indic NLP")

        lang_code = 'eng' if language == "en" else 'mal'
        try:
            sentences = sentence_tokenize.sentence_split(text, lang=lang_code)
            if sentences:
                print(f"Successfully split {len(sentences)} sentences using Indic NLP ({language})")
                cleaned_sentences = []
                current_sentence = ""

                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    split_sentences = split_pattern.split(sent)
                    split_sentences = [s.strip() for s in split_sentences if s.strip()]

                    for split_sent in split_sentences:
                        if current_sentence:
                            if re.search(abbr_pattern + r'$', current_sentence):
                                current_sentence += " " + split_sent
                            else:
                                cleaned_sentences.append(current_sentence)
                                current_sentence = split_sent
                        else:
                            current_sentence = split_sent
                if current_sentence:
                    cleaned_sentences.append(current_sentence)

                final_sentences = []
                for sent in cleaned_sentences:
                    subsentences = re.split(
                        r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\d\.\d)(?<=[.,])(?=\s|$|[^\s.,?!])',
                        sent
                    )
                    final_sentences.extend(subsent.strip() for subsent in subsentences if subsent.strip())

                print(f"Indic NLP post-processed into {len(final_sentences)} sentences")
                print(f"Sentences: {final_sentences}")
                return final_sentences
            else:
                print(f"Indic NLP returned no sentences for {language}")
        except Exception as indic_error:
            print(f"Indic NLP {language} tokenizer failed: {str(indic_error)}")

        print("All splitting methods failed, returning text as single sentence")
        return [text.strip()] if text.strip() else []

    except Exception as e:
        print(f"Sentence splitting error: {str(e)}")
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
    """Enhanced intent detection for internship interest analysis in English, Malayalam, and Tamil"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid input text for detect_intent: {text}. Returning Neutral_response.")
            return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

        text_lower = text.lower().strip()

        intent_keywords = {
            "en": {
                "Strong_interest": [
                    "definitely", "ready", "want to join", "interested",
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
                    "tino software and security solutions", "Tino software IT company", "Tino software",
                    "i am calling you from tino software and security solutions", "tinos software"
                ],
                "Qualification_query": [
                    "qualification", "education", "computer science", "degree", "studying", "course",
                    "background", "academics", "university", "college", "bsc",
                    "graduate", "year of study", "curriculum", "syllabus"
                ],
                "Internship_details": [
                    "internship", "placement", "program", "is looking for an internship", "duration",
                    "Data Science", "months", "period", "schedule", "timing", "timeframe",
                    "1 to 3", "three months", "structure", "plan", "framework",
                    "looking for an internship in data science"
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
                    "interested", "send whatsapp", "got it",
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
                    "ഇന്റെണ്ഷിപ്", "പരിശീലനം", "ഡാറ്റാ സയൻസിലെ", "ഇന്റേൺഷിപ്പിനൊപ്പം", "പ്ലെയ്സ്മെന്റ്",
                    "മാസം", "സമയക്രമം", "ടൈമിംഗ്", "1 മുതൽ 3 വരെ",
                    "അവസാന വർഷം", "ലൈവ്", "ഫ്രെയിംവർക്ക്", "സ്ഥിരമായി",
                    "ഡാറ്റാ സയൻസിലെ", "ഇന്റേൺഷിപ്പ്", "ഡാറ്റാ സയൻസിലെ ഇന്റേൺഷിപ്പ്"
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
                    "ടീം", "മേധാവി", "ട്രെയിനി", "സഹപ്രവർത്തനം", "പ്രോജക്റ്റുകൾ",
                    "ഡവലപ്പുചെയ്യുക", "സൃഷ്ടിക്കുക", "ഇമ്പ്ലിമെന്റുചെയ്യുക",
                    "പ്രായോഗികം", "അഭ്യാസം"
                ],
                "Confirmation": [
                    "ശരി", "താല്പര്യമുണ്ട്", "തിരയുന്നു", "ഇഷ്ടമുണ്ട്", "വാട്സാപ്പിൽ അയക്കൂ", "ഷെയർ ചെയ്യുക",
                    "വാട്സാപ്പ്", "വാട്ട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു",
                    "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", "ഓക്കെ", "യെസ്",
                    "അക്ക്നലഡ്ജ്", "ക്ലിയർ", "തയാറാണ്", "അറിയിപ്പ് ലഭിച്ചു",
                    "വാട്ട്സ്ആപ്പിലെ", "ഞാൻ അതിനായി നോക്കിയിരുന്നു"
                ]
            },
            "ta": {
                "Strong_interest": [
                    "நிச்சயமாக", "தயார்", "சேர விரும்புகிறேன்", "ஆர்வமாக உள்ளேன்",
                    "விவரங்களைப் பகிரவும்", "ப்ரோஷர் அனுப்பவும்", "நான் சேருவேன்", "தொடரலாம்",
                    "எங்கு கையெழுத்திடுவது", "பகிரவும்", "எப்போது தொடங்கலாம்", "ஏற்கிறேன்",
                    "எதிர்பார்க்கிறேன்", "உற்சாகமாக உள்ளேன்", "மகிழ்ச்சியாக உள்ளேன்",
                    "மகிழ்ச்சியாக இருக்கிறது", "ஆர்வமாக உள்ளேன்", "பகிரவும்",
                    "வாட்ஸ்அப்", "நான் உள்ளேன்"
                ],
                "Moderate_interest": [
                    "ஒருவேளை", "பரிசீலிக்கிறேன்", "யோசிக்கிறேன்", "என்னை யோசிக்க விடுங்கள்",
                    "மேலும் சொல்லுங்கள்", "மேலும் விவரங்கள்", "விளக்கவும்", "தெளிவுபடுத்தவும்",
                    "உறுதியாக இல்லை", "சாத்தியமாக", "இருக்கலாம்", "தங்கியிருக்கலாம்",
                    "சார்ந்தது", "சரிபார்க்க வேண்டும்", "முடிவு செய்வேன்",
                    "திரும்பி பேசுகிறேன்", "விவாதிக்கவும்", "ஆலோசிக்கவும்", "மதிப்பாய்வு செய்யவும்",
                    "மதிப்பிடவும்"
                ],
                "No_interest": [
                    "முடியாது", "செய்ய மாட்டேன்", "பிடிக்கவில்லை",
                    "இப்போது இல்லை", "பின்னர்", "பொருத்தமில்லை", "மறு"
                ],
                "company_query": [
                    "டினோ மென்பொருள் மற்றும் பாதுகா�ப்பு தீர்வுகள்", "டினோ மென்பொருள் ஐடி நிறுவனம்",
                    "டினோ மென்பொருள்", "நான் டினோ மென்பொருளில் இருந்து பேசுகிறேன்",
                    "டினோஸ் மென்பொருள்"
                ],
                "Qualification_query": [
                    "தகுதி", "கல்வி", "கணினி அறிவியல்", "பட்டம்", "படிக்கிறேன்",
                    "பாடநெறி", "பின்னணி", "கல்வித்துறை", "பல்கலைக்கழகம்",
                    "கல்லூரி", "பிஎஸ்சி", "பட்டதாரி", "படிப்பு ஆண்டு",
                    "பாடத்திட்டம்", "பாடநிரல்"
                ],
                "Internship_details": [
                    "இன்டர்ன்ஷிப்", "வேலைவாய்ப்பு", "திட்டம்", "இன்டர்ன்ஷிப் தேடுகிறேன்",
                    "கால அளவு", "டேட்டா சயின்ஸ்", "மாதங்கள்", "காலம்",
                    "அட்டவணை", "நேரம்", "காலக்கெடு", "1 முதல் 3 வரை",
                    "மூன்று மாதங்கள்", "அமைப்பு", "திட்டம்", "கட்டமைப்பு",
                    "டேட்டா சயின்ஸில் இன்டர்ன்ஷிப் தேடுகிறேன்"
                ],
                "Location_query": [
                    "ஆன்லைன்", "ஆஃப்லைன்", "இடம்", "முகவரி", "எங்கிருந்து",
                    "வருகிறேன்", "கோழிக்கோடு", "கொச்சி", "பாலரிவட்டம்",
                    "ஹைப்ரிட்", "தொலைவில்", "இடமாற்றம்", "மாற்றுதல்"
                ],
                "Certificate_query": [
                    "சான்றிதழ்", "சான்று", "ஆவணம்", "அனுபவ சான்றிதழ்",
                    "பயிற்சி சான்றிதழ்", "கடிதம்", "முடித்தல்", "விருது",
                    "அங்கீகாரம்"
                ],
                "Fee_query": [
                    "கட்டணம்", "கட்டணம் செலுத்துதல்", "விலை", "தொகை",
                    "கட்டணம்", "6000", "ஆறு ஆயிரம்", "பணம்", "ஊதியம்",
                    "சம்பளம்", "இழப்பீடு", "கட்டணமில்லை"
                ],
                "Project_details": [
                    "நேரடி திட்டம்", "வேலை", "பணி", "திட்டம்", "பயிற்சியாளர்",
                    "மேலாளர்கள்", "குழு", "ஒத்துழைப்பு", "உருவாக்குதல்",
                    "கட்டமைப்பு", "இருப்பாக்குதல்", "நடைமுறைப்படுத்துதல்",
                    "நேரடி அனுபவம்", "பயிற்சி"
                ],
                "Confirmation": [
                    "ஆர்வமாக உள்ளேன்", "வாட்ஸ்அப் அனுப்பவும்", "புரிந்தது",
                    "ஒப்புக்கொள்", "குறிப்பு எடுக்கப்பட்டது", "தயவுசெய்து அனுப்பவும்",
                    "விவரங்கள் அனுப்பப்பட்டன", "ஒப்பந்தம்", "சரி", "ஆமாம்",
                    "உறுதிப்படுத்து"
                ]
            },
            "hi": {
    "Strong_interest": [
        "ज़रूर", "तैयार हूँ", "शामिल होना चाहता हूँ", "दिलचस्पी है",
        "विवरण साझा करें", "ब्रॉशर भेजें", "मैं शामिल हो जाऊँगा", "आगे बढ़ें",
        "कहाँ साइन करूँ", "साझा करें", "मैं कब शुरू कर सकता हूँ", "स्वीकार करता हूँ",
        "प्रतीक्षा कर रहा हूँ", "उत्साहित हूँ", "खुशी से", "अच्छा लगा", "उत्सुक हूँ",
        "इसे साझा करें", "व्हाट्सएप करें", "मैं तैयार हूँ"
    ],
    "Moderate_interest": [
        "शायद", "विचार करूँगा", "सोच रहा हूँ", "सोचने दो", "और बताएं",
        "अधिक जानकारी", "समझाएं", "स्पष्ट करें", "पक्का नहीं", "संभवतः",
        "हो सकता है", "शायद हो", "निर्भर करता है", "जांचना पड़ेगा", "फैसला करूँगा",
        "बाद में बताऊंगा", "चर्चा करूँगा", "सलाह लूंगा", "समीक्षा", "मूल्यांकन"
    ],
    "No_interest": [
        "नहीं कर सकता", "नहीं करूँगा", "पसंद नहीं है",
        "अभी नहीं", "बाद में", "उपयुक्त नहीं है", "अस्वीकार करता हूँ"
    ],
    "company_query": [
        "टिनो सॉफ्टवेयर एंड सिक्योरिटी सॉल्यूशंस", "टिनो सॉफ्टवेयर आईटी कंपनी", "टिनो सॉफ्टवेयर",
        "मैं टिनो सॉफ्टवेयर एंड सिक्योरिटी सॉल्यूशंस से बोल रहा हूँ", "टिनोस सॉफ्टवेयर"
    ],
    "Qualification_query": [
        "योग्यता", "शिक्षा", "कंप्यूटर साइंस", "डिग्री", "पढ़ाई", "कोर्स",
        "पृष्ठभूमि", "अकादमिक", "यूनिवर्सिटी", "कॉलेज", "बीएससी",
        "स्नातक", "कौन से साल में", "पाठ्यक्रम", "सिलेबस"
    ],
    "Internship_details": [
        "इंटर्नशिप", "प्लेसमेंट", "प्रोग्राम", "इंटर्नशिप ढूंढ रहा है", "अवधि",
        "डेटा साइंस", "महीने", "समयावधि", "समय", "शेड्यूल", "समयसीमा",
        "1 से 3", "तीन महीने", "संरचना", "योजना", "ढांचा",
        "डेटा साइंस में इंटर्नशिप ढूंढ रहा हूँ"
    ],
    "Location_query": [
        "ऑनलाइन", "ऑफलाइन", "स्थान", "जगह", "कहाँ",
        "पता", "स्थानांतरण", "स्थानांतरित", "से", "आ रहा हूँ",
        "कोझिकोड", "कोच्चि", "पालरिवट्टम", "हाइब्रिड", "रिमोट"
    ],
    "Certificate_query": [
        "प्रमाणपत्र", "सर्टिफिकेट", "दस्तावेज़", "सबूत",
        "अनुभव प्रमाणपत्र", "प्रशिक्षण प्रमाणपत्र", "पत्र",
        "समाप्ति", "पुरस्कार", "मान्यता"
    ],
    "Fee_query": [
        "शुल्क", "भुगतान", "लागत", "राशि", "चार्ज",
        "6000", "छह हजार", "पैसे", "स्टाइपेंड", "वेतन",
        "मुआवज़ा", "पेड", "मुफ्त"
    ],
    "Project_details": [
        "लाइव प्रोजेक्ट", "काम", "असाइनमेंट", "कार्य", "प्रोजेक्ट",
        "ट्रेनी", "सीनियर्स", "टीम", "मिलकर काम करें", "विकसित करें",
        "बनाएं", "क्रिएट करें", "इम्प्लीमेंट करें", "प्रैक्टिकल", "हैंड्स-ऑन"
    ],
    "Confirmation": [
        "दिलचस्पी है", "व्हाट्सएप भेजें", "समझ गया",
        "स्वीकार करता हूँ", "नोट कर लिया", "कृपया भेजें", "विवरण भेजा", "सहमत हूँ"
    ]
},
"te": {
    "Strong_interest": [
        "ఖచ్చితంగా", "సిద్ధంగా ఉన్నాను", "చేరాలని ఉంది", "ఆసక్తిగా ఉన్నాను",
        "వివరాలు షేర్ చేయండి", "బ్రోచర్ పంపండి", "నేను చేరతాను", "ముందుకు పోదాం",
        "ఎక్కడ సైన్ చేయాలి?", "షేర్ చేయండి", "నేను ఎప్పటి నుంచి ప్రారంభించగలను?", "అంగీకరిస్తున్నాను",
        "ఎదురుచూస్తున్నాను", "ఆసక్తిగా ఉన్నాను", "సంతోషంగా", "ఆనందంగా ఉంది", "ఉత్సాహంగా ఉన్నాను",
        "దీనిని షేర్ చేయండి", "వాట్సాప్ చేయండి", "నేను రెడీ"
    ],
    "Moderate_interest": [
        "బహుశా", "ఆలోచిస్తాను", "చూస్తాను", "ఆలోచించనివ్వండి", "ఇంకా చెప్పండి",
        "మరిన్ని వివరాలు", "వివరించండి", "స్పష్టం చేయండి", "నాకు ఖచ్చితంగా తెలియదు", "కావచ్చు",
        "ఉండవచ్చు", "అలానే కావచ్చు", "అది ఆధారపడి ఉంటుంది", "చూసి చెబుతాను", "తర్వాత నిర్ణయం తీసుకుంటాను",
        "తరువాత చెబుతాను", "చర్చించాలి", "ఆలోచన అడగాలి", "సమీక్షించాలి", "విలువను అంచనా వేయాలి"
    ],
    "No_interest": [
        "చేయలేను", "చేయను", "ఇష్టం లేదు",
        "ఇప్పుడేమీ కాదు", "తర్వాత", "సరిపోదు", "నిరాకరిస్తున్నాను"
    ],
    "company_query": [
        "టినో సాఫ్ట్‌వేర్ అండ్ సెక్యూరిటీ సొల్యూషన్స్", "టినో సాఫ్ట్‌వేర్ ఐటీ కంపెనీ", "టినో సాఫ్ట్‌వేర్",
        "నేను టినో సాఫ్ట్‌వేర్ అండ్ సెక్యూరిటీ సొల్యూషన్స్ నుండి మాట్లాడుతున్నాను", "టినోస్ సాఫ్ట్‌వేర్"
    ],
    "Qualification_query": [
        "అర్హత", "విద్య", "కంప్యూటర్ సైన్స్", "డిగ్రీ", "చదువు", "కోర్సు",
        "నేపథ్యం", "అకాడెమిక్స్", "విశ్వవిద్యాలయం", "కాలేజ్", "బిఎస్సీ",
        "డిగ్రీ పూర్తి చేశాను", "ఏ యేళ్ళు చదువుతున్నారు?", "పాఠ్యప్రణాళిక", "సిలబస్"
    ],
    "Internship_details": [
        "ఇంటర్న్‌షిప్", "ప్లేస్‌మెంట్", "ప్రోగ్రామ్", "ఇంటర్న్‌షిప్ కావాలంటున్నాడు", "వ్యవధి",
        "డేటా సైన్స్", "నెలలు", "పీరియడ్", "సమయం", "షెడ్యూల్", "టైమ్‌ఫ్రేమ్",
        "1 నుండి 3", "మూడు నెలలు", "నిర్మాణం", "ప్లాన్", "ఫ్రేమ్‌వర్క్",
        "డేటా సైన్స్‌లో ఇంటర్న్‌షిప్ కావాలి అంటున్నాను"
    ],
    "Location_query": [
        "ఆన్లైన్", "ఆఫ్లైన్", "స్థానం", "ప్రదేశం", "ఎక్కడ",
        "చిరునామా", "ఇక్కడికి మారడం", "రీలోకేట్", "నుండి", "వస్తున్నాను",
        "కొళికోడు", "కొచ్చి", "పలరివట్టం", "హైబ్రిడ్", "రిమోట్"
    ],
    "Certificate_query": [
        "సర్టిఫికెట్", "ప్రామాణిక పత్రం", "డాక్యుమెంట్", "సాక్ష్యం",
        "అనుభవం సర్టిఫికెట్", "ట్రైనింగ్ సర్టిఫికేట్", "లేఖ",
        "పూర్తి అయిన", "అవార్డు", "గౌరవం"
    ],
    "Fee_query": [
        "ఫీజు", "చెల్లింపు", "ఖర్చు", "ధర", "చార్జ్",
        "6000", "ఆరు వేల", "డబ్బు", "స్టైఫండ్", "జీతం",
        "పగడభాగం", "పెయిడ్", "ఫ్రీ"
    ],
    "Project_details": [
        "లైవ్ ప్రాజెక్ట్", "పని", "అసైన్‌మెంట్", "టాస్క్", "ప్రాజెక్ట్",
        "ట్రైనీ", "సీనియర్స్", "టీమ్", "కలిసి పని చేయడం", "డెవలప్ చేయడం",
        "తయారు చేయడం", "సృష్టించడం", "అమలు చేయడం", "ప్రాక్టికల్", "హ్యాండ్స్-ఆన్"
    ],
    "Confirmation": [
        "ఆసక్తిగా ఉన్నాను", "వాట్సాప్ పంపండి", "అర్థమైంది",
        "అంగీకరిస్తున్నాను", "గమనించాను", "దయచేసి పంపండి", "వివరాలు పంపించారు", "ఒప్పుకుంటున్నాను"
    ]
},
"kn": {
    "Strong_interest": [
        "ಖಚಿತವಾಗಿ", "ನಾನು ಸಿದ್ಧ", "ಚೇರಲು ಇಚ್ಛೆ", "ಆಸಕ್ತಿಯಿದೆ",
        "ವಿವರಗಳನ್ನು ಹಂಚಿ", "ಬ್ರೋಷರ್ ಕಳುಹಿಸಿ", "ನಾನು ಸೇರುತ್ತೇನೆ", "ಮುಂದುವರೆಯೋಣ",
        "ಎಲ್ಲಿ ಸಹಿ ಹಾಕಬೇಕು?", "ಹಂಚಿ", "ನಾನು ಯಾವಾಗ ಆರಂಭಿಸಬಹುದು?", "ಸ್ವೀಕರಿಸುತ್ತೇನೆ",
        "ನೋಡಲಿರುವೆ", "ಉತ್ಸಾಹವಾಗಿದೆ", "ಸಂತೋಷವಾಗಿದೆ", "ಖುಷಿಯಾಗಿದೆ", "ಕಾತುರವಾಗಿದೆ",
        "ಇದನ್ನು ಹಂಚಿ", "ವಾಟ್ಸಪ್ ಮಾಡಿ", "ನಾನು ಸಿದ್ದ"
    ],
    "Moderate_interest": [
        "ಶಾಯದ", "ವಿಚಾರಿಸುತ್ತೇನೆ", "ಅಲೋಚಿಸುತ್ತೇನೆ", "ನನಗೆ ಯೋಚಿಸಲು ಬಿಡಿ", "ಹೆಚ್ಚು ಹೇಳಿ",
        "ಹೆಚ್ಚಿನ ವಿವರಗಳು", "ವಿವರಿಸಿ", "ಸ್ಪಷ್ಟಪಡಿಸಿ", "ಖಚಿತವಿಲ್ಲ", "ಸಂಭವವಿದೆ",
        "ಇರಬಹುದು", "ಆಗಬಹುದು", "ಅದು ಅವಲಂಬಿತವಾಗಿದೆ", "ಪರಿಶೀಲಿಸಬೇಕು", "ನಿರ್ಣಯಿಸುತ್ತೇನೆ",
        "ಮರುಕಳಿಸುತ್ತೇನೆ", "ಚರ್ಚಿಸಬೇಕು", "ಪರಾಮರ್ಶಿಸಬೇಕು", "ಪುನರ್ ವಿಮರ್ಶೆ", "ಮೌಲ್ಯಮಾಪನ"
    ],
    "No_interest": [
        "ಮಾಡಲು ಸಾಧ್ಯವಿಲ್ಲ", "ಮಾಡುವುದಿಲ್ಲ", "ಇಷ್ಟವಿಲ್ಲ",
        "ಈಗಲ್ಲ", "ನಂತರ", "ಯೋಗ್ಯವಲ್ಲ", "ನಿರಾಕರಿಸುತ್ತೇನೆ"
    ],
    "company_query": [
        "ಟಿನೋ ಸಾಫ್ಟ್‌ವೇರ್ ಅಂಡ್ ಸೆಕ್ಯುರಿಟಿ ಸೊಲ್ಯೂಶನ್ಸ್", "ಟಿನೋ ಸಾಫ್ಟ್‌ವೇರ್ ಐಟಿ ಕಂಪನಿ", "ಟಿನೋ ಸಾಫ್ಟ್‌ವೇರ್",
        "ನಾನು ಟಿನೋ ಸಾಫ್ಟ್‌ವೇರ್ ಅಂಡ್ ಸೆಕ್ಯುರಿಟಿ ಸೊಲ್ಯೂಶನ್ಸ್ ನಿಂದ ಕರೆ ಮಾಡುತ್ತಿದ್ದೇನೆ", "ಟಿನೋಸ್ ಸಾಫ್ಟ್‌ವೇರ್"
    ],
    "Qualification_query": [
        "ಅರ್ಹತೆ", "ಶಿಕ್ಷಣ", "ಕಂಪ್ಯೂಟರ್ ಸೈನ್ಸ್", "ಪದವಿ", "ಅಭ್ಯಾಸ", "ಕೋರ್ಸ್",
        "ಹಿನ್ನೆಲೆ", "ಶೈಕ್ಷಣಿಕ", "ವಿಶ್ವವಿದ್ಯಾಲಯ", "ಕಾಲೇಜು", "ಬಿಎಸ್ಸಿ",
        "ಪದವಿ", "ಏನೇ ವರ್ಷ ಓದುತ್ತಿದ್ದೀರಿ?", "ಪಠ್ಯಕ್ರಮ", "ಸಿಲೆಬಸ್"
    ],
    "Internship_details": [
        "ಇಂಟರ್ನ್‌ಶಿಪ್", "ಪ್ಲೇಸ್‌ಮೆಂಟ್", "ಕಾರ್ಯಕ್ರಮ", "ಇಂಟರ್ನ್‌ಶಿಪ್ ಬೇಕು ಎನ್ನುತ್ತಿದ್ದಾರೆ", "ಅವಧಿ",
        "ಡೇಟಾ ಸೈನ್ಸ್", "ತಿಂಗಳುಗಳು", "ಅವಧಿ", "ಸಮಯ", "ಪಟ್ಟಿ", "ಸಮಯಘಟ್ಟ",
        "1 ರಿಂದ 3", "ಮೂರು ತಿಂಗಳು", "ರಚನೆ", "ಯೋಜನೆ", "ಫ್ರೇಮ್‌ವರ್ಕ್",
        "ಡೇಟಾ ಸೈನ್ಸ್ ನಲ್ಲಿ ಇಂಟರ್ನ್‌ಶಿಪ್ ಬೇಕು"
    ],
    "Location_query": [
        "ಆನ್ಲೈನ್", "ಆಫ್ಲೈನ್", "ಸ್ಥಳ", "ಏನು ಸ್ಥಳ", "ಎಲ್ಲಿ",
        "ವಿಳಾಸ", "ಸ್ಥಳಾಂತರ", "ರೀಲೊಕೇಟ್", "ಇಂದ", "ಬರುತ್ತಿದ್ದೇನೆ",
        "ಕೊಝಿಕೋಡ್", "ಕೊಚ್ಚಿ", "ಪಲರಿವಟ್ಟಂ", "ಹೈಬ್ರಿಡ್", "ರಿಮೋಟ್"
    ],
    "Certificate_query": [
        "ಪ್ರಮಾಣಪತ್ರ", "ಸერტಿಫಿಕೇಟು", "ಡಾಕ್ಯುಮೆಂಟ್", "ಸಾಕ್ಷ್ಯ",
        "ಅನುಭವದ ಪ್ರಮಾಣಪತ್ರ", "ತರಬೇತಿ ಪ್ರಮಾಣಪತ್ರ", "ಪತ್ರ",
        "ಪೂರ್ಣಗೊಂಡ", "ಬಹುಮಾನ", "ಗುರುತಿನ"
    ],
    "Fee_query": [
        "ಫೀಸ್", "ಪಾವತಿ", "ಖರ್ಚು", "ಮೊತ್ತ", "ಚಾರ್ಜ್",
        "6000", "ಆರು ಸಾವಿರ", "ಹಣ", "ಸ್ಟೈಪೆಂಡ್", "ಸಂಬಳ",
        "ವಿಳೇವಾರು", "ಪೈಡ್", "ಉಚಿತ"
    ],
    "Project_details": [
        "ಲೈವ್ ಪ್ರಾಜೆಕ್ಟ್", "ಕೆಲಸ", "ಅಸೈನ್‌ಮೆಂಟ್", "ಕಾರ್ಯ", "ಪ್ರಾಜೆಕ್ಟ್",
        "ಟ್ರೈನಿ", "ಮುಂಬರುವವರು", "ಟೀಮ್", "ಸಹಕರಿಸಿ", "ಅಭಿವೃದ್ಧಿ ಮಾಡಿ",
        "ತಯಾರಿಸಿ", "ಸೃಷ್ಟಿಸಿ", "ಅಮಲುಮಾಡಿ", "ಪ್ರಾಯೋಗಿಕ", "ಹ್ಯಾಂಡ್ಸ್‌ಆನ್"
    ],
    "Confirmation": [
        "ಆಸಕ್ತಿ ಇದೆ", "ವಾಟ್ಸಪ್ ಕಳುಹಿಸಿ", "ಗೊತ್ತಾಯಿತು",
        "ಸ್ವೀಕರಿಸಿದೆ", "ಗಮನಿಸಿದೆ", "ದಯವಿಟ್ಟು ಕಳುಹಿಸಿ", "ವಿವರ ಕಳುಹಿಸಲಾಗಿದೆ", "ಒಪ್ಪಿಗೆ ಇದೆ"
    ]
},
"ne": {
    "Strong_interest": [
        "पक्कै", "म तयार छु", "जोडिन चाहन्छु", "चासो छ",
        "विवरण साझा गर्नुहोस्", "ब्रशर पठाउनुहोस्", "म सामेल हुनेछु", "अगाडि बढौं",
        "म कहाँ साइन गर्ने?", "शेयर गर्नुहोस्", "म कहिले सुरु गर्न सक्छु?", "स्वीकार गर्छु",
        "प्रतीक्षा गरिरहेको छु", "उत्साहित छु", "खुशी लाग्यो", "रमाइलो लाग्यो", "आतुर छु",
        "यसलाई शेयर गर्नुहोस्", "ह्वाट्सएप गर्नुहोस्", "म तयार छु"
    ],
    "Moderate_interest": [
        "शायद", "विचार गर्नेछु", "सोच्दै छु", "म सोच्न चाहन्छु", "थप जानकारी दिनुहोस्",
        "थप विवरण", "व्याख्या गर्नुहोस्", "स्पष्ट पार्नुहोस्", "पक्का छैन", "संभवतः",
        "हुन सक्छ", "शायद हो", "निर्भर गर्दछ", "जाँच गर्नुपर्छ", "निर्णय गर्नेछु",
        "फर्केर जवाफ दिन्छु", "छलफल गर्नु पर्छ", "सल्लाह लिनु पर्छ", "पुनरावलोकन गर्नुहोस्", "मूल्याङ्कन गर्नुहोस्"
    ],
    "No_interest": [
        "म गर्न सक्दिन", "म गर्दिन", "मन परेन",
        "अहिले होइन", "पछि", "उपयुक्त छैन", "अस्वीकार गर्छु"
    ],
    "company_query": [
        "टिनो सफ्टवेयर एन्ड सेक्युरिटी सोलुसन", "टिनो सफ्टवेयर आईटी कम्पनी", "टिनो सफ्टवेयर",
        "म टिनो सफ्टवेयर एन्ड सेक्युरिटी सोलुसनबाट बोल्दैछु", "टिनोस सफ्टवेयर"
    ],
    "Qualification_query": [
        "योग्यता", "शिक्षा", "कम्प्युटर साइन्स", "डिग्री", "पढाइ", "कोर्स",
        "पृष्ठभूमि", "शैक्षिक", "विश्वविद्यालय", "कलेज", "बिएस्सी",
        "स्नातक", "कुन वर्षमा पढ्दै?", "पाठ्यक्रम", "सिलेबस"
    ],
    "Internship_details": [
        "इन्टर्नशिप", "प्लेसमेन्ट", "प्रोग्राम", "इन्टर्नशिप खोज्दैछ", "अवधि",
        "डाटा साइन्स", "महिना", "समयावधि", "समय", "तालिका", "समयसीमा",
        "१ देखि ३", "तीन महिना", "संरचना", "योजना", "फ्रेमवर्क",
        "डाटा साइन्समा इन्टर्नशिप खोज्दैछु"
    ],
    "Location_query": [
        "अनलाइन", "अफलाइन", "स्थान", "ठाउँ", "कहाँ",
        "ठेगाना", "स्थानान्तरण", "रेलोकट", "बाट", "आउँदैछु",
        "कोझिकोड", "कोची", "पलारिवट्टम", "हाइब्रिड", "रिमोट"
    ],
    "Certificate_query": [
        "प्रमाणपत्र", "सर्टिफिकेट", "कागजात", "प्रमाण",
        "अनुभवको प्रमाणपत्र", "प्रशिक्षण प्रमाणपत्र", "पत्र",
        "समाप्ति", "पुरस्कार", "मान्यता"
    ],
    "Fee_query": [
        "शुल्क", "भुक्तानी", "लागत", "रकम", "चार्ज",
        "6000", "छ हजार", "पैसा", "स्टाइपेन्ड", "तलब",
        "क्षतिपूर्ति", "पेड", "निःशुल्क"
    ],
    "Project_details": [
        "लाइभ प्रोजेक्ट", "काम", "असाइनमेन्ट", "कार्य", "प्रोजेक्ट",
        "प्रशिक्षार्थी", "वरिष्ठ", "टोली", "सहकार्य", "विकास गर्नुहोस्",
        "बनाउनुहोस्", "सिर्जना गर्नुहोस्", "कार्यान्वयन गर्नुहोस्", "व्यावहारिक", "ह्याण्ड्स-अन"
    ],
    "Confirmation": [
        "चासो छ", "ह्वाट्सएप पठाउनुहोस्", "बुझियो",
        "स्वीकार गरें", "नोट गरें", "कृपया पठाउनुहोस्", "विवरण पठाइयो", "सहमति छ"
    ]
},
"pa": {
    "Strong_interest": [
        "ਜ਼ਰੂਰ", "ਮੈਂ ਤਿਆਰ ਹਾਂ", "ਸ਼ਾਮਲ ਹੋਣਾ ਚਾਹੁੰਦਾ ਹਾਂ", "ਦਿਲਚਸਪੀ ਹੈ",
        "ਵੇਰਵੇ ਸਾਂਝੇ ਕਰੋ", "ਬ੍ਰੋਸ਼ਰ ਭੇਜੋ", "ਮੈਂ ਜੁੜਾਂਗਾ", "ਚਲੋ ਅੱਗੇ ਵਧੀਏ",
        "ਕਿੱਥੇ ਸਾਇਨ ਕਰਨਾ ਹੈ?", "ਸਾਂਝਾ ਕਰੋ", "ਮੈਂ ਕਦੋਂ ਸ਼ੁਰੂ ਕਰ ਸਕਦਾ ਹਾਂ?", "ਸਵੀਕਾਰ ਹੈ",
        "ਉਡੀਕ ਕਰ ਰਿਹਾ ਹਾਂ", "ਉਤਸ਼ਾਹਤ ਹਾਂ", "ਖੁਸ਼ੀ ਹੋਈ", "ਚੰਗਾ ਲੱਗਿਆ", "ਉਤਸੁਕ ਹਾਂ",
        "ਇਹ ਸਾਂਝਾ ਕਰੋ", "ਵਾਟਸਐਪ ਕਰੋ", "ਮੈਂ ਤਿਆਰ ਹਾਂ"
    ],
    "Moderate_interest": [
        "ਸ਼ਾਇਦ", "ਸੋਚਾਂਗਾ", "ਸੋਚ ਰਿਹਾ ਹਾਂ", "ਮੈਨੂੰ ਸੋਚਣ ਦਿਓ", "ਹੋਰ ਦੱਸੋ",
        "ਹੋਰ ਵੇਰਵੇ", "ਵਿਆਖਿਆ ਕਰੋ", "ਸਪਸ਼ਟ ਕਰੋ", "ਪੱਕਾ ਨਹੀਂ", "ਸੰਭਵ ਹੈ",
        "ਹੋ ਸਕਦਾ ਹੈ", "ਸਕਦਾ ਹੈ", "ਇਹ ਨਿਰਭਰ ਕਰਦਾ ਹੈ", "ਜਾਂਚ ਕਰਨੀ ਪਏਗੀ", "ਫੈਸਲਾ ਕਰਾਂਗਾ",
        "ਮੁੜ ਦੱਸਾਂਗਾ", "ਚਰਚਾ ਕਰਾਂਗਾ", "ਸਲਾਹ ਲਵਾਂਗਾ", "ਸਮੀਖਿਆ ਕਰਨੀ", "ਮੁਲਾਂਕਣ"
    ],
    "No_interest": [
        "ਨਹੀਂ ਕਰ ਸਕਦਾ", "ਨਹੀਂ ਕਰਾਂਗਾ", "ਪਸੰਦ ਨਹੀਂ",
        "ਹੁਣ ਨਹੀਂ", "ਬਾਅਦ ਵਿਚ", "ਉਚਿਤ ਨਹੀਂ", "ਇਨਕਾਰ ਕਰਦਾ ਹਾਂ"
    ],
    "company_query": [
        "ਟਿਨੋ ਸਾਫਟਵੇਅਰ ਐਂਡ ਸਿਕ੍ਯੋਰਟੀ ਸੋਲੂਸ਼ਨਜ਼", "ਟਿਨੋ ਸਾਫਟਵੇਅਰ ਆਈਟੀ ਕੰਪਨੀ", "ਟਿਨੋ ਸਾਫਟਵੇਅਰ",
        "ਮੈਂ ਟਿਨੋ ਸਾਫਟਵੇਅਰ ਐਂਡ ਸਿਕ੍ਯੋਰਟੀ ਸੋਲੂਸ਼ਨਜ਼ ਤੋਂ ਗੱਲ ਕਰ ਰਿਹਾ ਹਾਂ", "ਟਿਨੋਜ਼ ਸਾਫਟਵੇਅਰ"
    ],
    "Qualification_query": [
        "ਯੋਗਤਾ", "ਸਿੱਖਿਆ", "ਕੰਪਿਊਟਰ ਸਾਇੰਸ", "ਡਿਗਰੀ", "ਪੜਾਈ", "ਕੋਰਸ",
        "ਪਿਛੋਕੜ", "ਅਕਾਦਮਿਕ", "ਯੂਨੀਵਰਸਿਟੀ", "ਕਾਲਜ", "ਬੀਐੱਸਸੀ",
        "ਗ੍ਰੈਜੂਏਟ", "ਕਿਹੜੇ ਸਾਲ ਵਿੱਚ ਹੋ?", "ਕਰੀਕੁਲਮ", "ਸਿਲੇਬਸ"
    ],
    "Internship_details": [
        "ਇੰਟਰਨਸ਼ਿਪ", "ਪਲੇਸਮੈਂਟ", "ਪ੍ਰੋਗਰਾਮ", "ਇੰਟਰਨਸ਼ਿਪ ਲੱਭ ਰਿਹਾ ਹੈ", "ਅवधि",
        "ਡਾਟਾ ਸਾਇੰਸ", "ਮਹੀਨੇ", "ਪੀਰੀਅਡ", "ਸਮਾਂ", "ਸ਼ੈਡੂਲ", "ਟਾਈਮਫ੍ਰੇਮ",
        "1 ਤੋਂ 3", "ਤਿੰਨ ਮਹੀਨੇ", "ਸੰਰਚਨਾ", "ਪਲਾਨ", "ਫਰੇਮਵਰਕ",
        "ਮੈਂ ਡਾਟਾ ਸਾਇੰਸ ਵਿੱਚ ਇੰਟਰਨਸ਼ਿਪ ਲੱਭ ਰਿਹਾ ਹਾਂ"
    ],
    "Location_query": [
        "ਆਨਲਾਈਨ", "ਆਫਲਾਈਨ", "ਥਾਂ", "ਸਥਾਨ", "ਕਿੱਥੇ",
        "ਐਡਰੈੱਸ", "ਰੀਲੋਕੇਟ", "ਟ੍ਰਾਂਸਫਰ", "ਤੋਂ", "ਆ ਰਿਹਾ ਹਾਂ",
        "ਕੋਝੀਕੋਡ", "ਕੋਚੀ", "ਪਲਾਰਿਵੱਟਮ", "ਹਾਈਬ੍ਰਿਡ", "ਰੀਮੋਟ"
    ],
    "Certificate_query": [
        "ਸਰਟੀਫਿਕੇਟ", "ਪ੍ਰਮਾਣ ਪੱਤਰ", "ਡੌਕੂਮੈਂਟ", "ਸਬੂਤ",
        "ਅਨੁਭਵ ਪੱਤਰ", "ਟ੍ਰੇਨਿੰਗ ਸਰਟੀਫਿਕੇਟ", "ਚਿੱਠੀ",
        "ਪੂਰਾ ਹੋਇਆ", "ਇਨਾਮ", "ਮਾਨਤਾ"
    ],
    "Fee_query": [
        "ਫੀਸ", "ਭੁਗਤਾਨ", "ਲਾਗਤ", "ਰਕਮ", "ਚਾਰਜ",
        "6000", "ਛੇ ਹਜ਼ਾਰ", "ਪੈਸਾ", "ਸਟਾਈਪੈਂਡ", "ਤਨਖਾਹ",
        "ਮੁਆਵਜ਼ਾ", "ਪੇਡ", "ਮੁਫ਼ਤ"
    ],
    "Project_details": [
        "ਲਾਈਵ ਪ੍ਰਾਜੈਕਟ", "ਕੰਮ", "ਅਸਾਈਨਮੈਂਟ", "ਟਾਸਕ", "ਪ੍ਰਾਜੈਕਟ",
        "ਟਰੇਨੀ", "ਸੀਨੀਅਰ", "ਟੀਮ", "ਮਿਲ ਕੇ ਕੰਮ ਕਰਨਾ", "ਡਿਵੈਲਪ ਕਰਨਾ",
        "ਤਿਆਰ ਕਰਨਾ", "ਬਣਾਉਣਾ", "ਲਾਗੂ ਕਰਨਾ", "ਪ੍ਰੈਕਟੀਕਲ", "ਹੈਂਡਜ਼-ਆਨ"
    ],
    "Confirmation": [
        "ਦਿਲਚਸਪੀ ਹੈ", "ਵਾਟਸਐਪ ਭੇਜੋ", "ਸਮਝ ਆ ਗਿਆ",
        "ਸਵੀਕਾਰ ਕੀਤਾ", "ਨੋਟ ਕਰ ਲਿਆ", "ਕਿਰਪਾ ਕਰਕੇ ਭੇਜੋ", "ਵੇਰਵੇ ਭੇਜੇ", "ਸਹਿਮਤ ਹਾਂ"
    ]
},
"gu": {
    "Strong_interest": [
        "ખરેખર", "હું તૈયાર છું", "જોડાવું છે", "રુચિ છે",
        "વિગતો શેર કરો", "બ્રોશર મોકલો", "હું જોડાઈ જઈશ", "ચાલો આગળ વધીએ",
        "હું સાઇન ક્યાં કરું?", "શેર કરો", "હું ક્યારે શરૂ કરી શકું?", "સ્વીકારું છું",
        "અપેક્ષા રાખી રહ્યો છું", "ઉત્સાહી છું", "ખુશીથી", "આનંદ થયો", "ઉત્સુક છું",
        "આને શેર કરો", "વોટ્સએપ કરો", "હું રેડી છું"
    ],
    "Moderate_interest": [
        "શાયદ", "વિચાર કરીશ", "વિચારમાં છું", "મને વિચારવા દો", "વધુ કહો",
        "વધુ વિગતો", "વિગતવાર કહો", "સ્પષ્ટ કરો", "ખાતરી નથી", "શક્ય છે",
        "હોઈ શકે", "શાયદ હોય", "આ નિર્ભર કરે છે", "તપાસવું પડશે", "નિર્ણય લઈશ",
        "પછી વાત કરીશ", "ચર્ચા કરીશું", "સલાહ લેશ", "સમીક્ષા", "મૂલ્યાંકન"
    ],
    "No_interest": [
        "નથી કરી શકતો", "નહી કરું", "ગમતું નથી",
        "હવે નહીં", "પછી", "ઉપયોગી નથી", "નકારું છું"
    ],
    "company_query": [
        "ટિનો સોફ્ટવેર એન્ડ સિક્યુરિટી સોલ્યુશન્સ", "ટિનો સોફ્ટવેર આઈટી કંપની", "ટિનો સોફ્ટવેર",
        "હું ટિનો સોફ્ટવેર એન્ડ સિક્યુરિટી સોલ્યુશન્સમાંથી બોલી રહ્યો છું", "ટિનોઝ સોફ્ટવેર"
    ],
    "Qualification_query": [
        "લાયકાત", "શિક્ષણ", "કમ્પ્યુટર વિજ્ઞાન", "ડિગ્રી", "અભ્યાસ", "કોર્સ",
        "પૃષ્ઠભૂમિ", "શૈક્ષણિક", "યૂનિવર્સિટી", "કોલેજ", "બીએસસી",
        "સ્નાતક", "કયા વર્ષમાં અભ્યાસ છે?", "પાઠ્યક્રમ", "સિલેબસ"
    ],
    "Internship_details": [
        "ઇન્ટરનશિપ", "પ્લેસમેન્ટ", "પ્રોગ્રામ", "ઇન્ટરનશિપ જોઈ રહ્યો છે", "અવધિ",
        "ડેટા સાયન્સ", "મહિનો", "સમયગાળો", "સમય", "શેડ્યૂલ", "ટાઈમફ્રેમ",
        "1 થી 3", "ત્રણ મહિનો", "બાંધકામ", "યોજના", "ફ્રેમવર્ક",
        "હું ડેટા સાયન્સમાં ઇન્ટરનશિપ શોધી રહ્યો છું"
    ],
    "Location_query": [
        "ઓનલાઈન", "ઓફલાઈન", "સ્થળ", "જગ્યા", "ક્યાં",
        "સરનામું", "સ્થળાંતર", "રીલોકેટ", "થી", "આવી રહ્યો છું",
        "કોઝિકોડ", "કોચી", "પલારિવટ્ટમ", "હાઈબ્રિડ", "રિમોટ"
    ],
    "Certificate_query": [
        "પ્રમાણપત્ર", "સર્ટિફિકેટ", "દસ્તાવેજ", "પુરાવો",
        "અનુભવ પ્રમાણપત્ર", "ટ્રેનિંગ સર્ટિફિકેટ", "પત્ર",
        "સમાપ્તી", "એવોર્ડ", "માન્યતા"
    ],
    "Fee_query": [
        "ફી", "ચૂકવણી", "ખર્ચ", "રકમ", "શુલ્ક",
        "6000", "છ હજાર", "પૈસા", "સ્ટાઈપેન્ડ", "પગાર",
        "વિતરણ", "પેડ", "મફત"
    ],
    "Project_details": [
        "લાઈવ પ્રોજેક્ટ", "કામ", "અસાઇનમેન્ટ", "કાર્ય", "પ્રોજેક્ટ",
        "ટ્રેની", "વરિષ્ઠ", "ટીમ", "સહયોગ", "વિક્સાવવું",
        "બનાવવું", "સર્જન", "અમલમાં મૂકવું", "પ્રેક્ટિકલ", "હેન્ડ્સ-ઓન"
    ],
    "Confirmation": [
        "રુચિ છે", "વોટ્સએપ મોકલો", "સમજી ગયો",
        "સ્વીકાર્યું", "નોંધ્યું", "મહેરબાની કરીને મોકલો", "વિગતો મોકલવામાં આવી", "સહમતી છે"
    ]
},
"mr": {
    "Strong_interest": [
        "नक्कीच", "मी तयार आहे", "जोडायचं आहे", "रुची आहे",
        "तपशील शेअर करा", "ब्रॉशर पाठवा", "मी सामील होईन", "चला पुढे जाऊया",
        "साइन कुठे करायचं?", "शेअर करा", "मी कधी सुरू करू शकतो?", "स्वीकारतो",
        "प्रतीक्षा करत आहे", "उत्सुक आहे", "आनंद झाला", "छान वाटलं", "आवड आहे",
        "हे शेअर करा", "व्हॉट्सअ‍ॅप करा", "मी तयार आहे"
    ],
    "Moderate_interest": [
        "कदाचित", "विचार करेन", "विचार करत आहे", "मला विचारू द्या", "थोडं अधिक सांगा",
        "अधिक तपशील", "समजावून सांगा", "स्पष्ट करा", "निश्चित नाही", "शक्यता आहे",
        "होऊ शकतं", "बहुधा", "ते अवलंबून आहे", "पाहावं लागेल", "नंतर ठरवेन",
        "नंतर कळवीन", "चर्चा करावी लागेल", "सल्ला घ्यावा लागेल", "पुनरावलोकन", "मूल्यांकन"
    ],
    "No_interest": [
        "झालं नाही", "करणार नाही", "आवडत नाही",
        "आत्ता नाही", "नंतर", "योग्य नाही", "नकार देतो"
    ],
    "company_query": [
        "टिनो सॉफ्टवेअर अँड सिक्युरिटी सोल्युशन्स", "टिनो सॉफ्टवेअर आयटी कंपनी", "टिनो सॉफ्टवेअर",
        "मी टिनो सॉफ्टवेअर अँड सिक्युरिटी सोल्युशन्समधून बोलतोय", "टिनोज सॉफ्टवेअर"
    ],
    "Qualification_query": [
        "पात्रता", "शिक्षण", "कॉम्प्युटर सायन्स", "डिग्री", "अभ्यास", "कोर्स",
        "पार्श्वभूमी", "शैक्षणिक", "विद्यापीठ", "कॉलेज", "बीएससी",
        "पदवीधर", "कोणत्या वर्षात?", "अभ्यासक्रम", "अभ्यासविषय"
    ],
    "Internship_details": [
        "इंटर्नशिप", "प्लेसमेंट", "प्रोग्रॅम", "इंटर्नशिप शोधत आहे", "कालावधी",
        "डेटा सायन्स", "महिने", "कालमर्यादा", "वेळ", "वेळापत्रक", "टाईमफ्रेम",
        "१ ते ३", "तीन महिने", "रचना", "योजना", "फ्रेमवर्क",
        "डेटा सायन्समध्ये इंटर्नशिप शोधत आहे"
    ],
    "Location_query": [
        "ऑनलाइन", "ऑफलाइन", "स्थान", "जागा", "कोठे",
        "पत्ता", "स्थानांतर", "रीलोकेट", "पासून", "येत आहे",
        "कोझिकोड", "कोची", "पलारीवत्तम", "हायब्रिड", "रिमोट"
    ],
    "Certificate_query": [
        "प्रमाणपत्र", "सर्टिफिकेट", "कागदपत्र", "पुरावा",
        "अनुभवाचे प्रमाणपत्र", "प्रशिक्षण प्रमाणपत्र", "पत्र",
        "पूर्ण झाले", "पुरस्कार", "ओळख"
    ],
    "Fee_query": [
        "शुल्क", "पेमेंट", "खर्च", "रक्कम", "चार्ज",
        "6000", "सहा हजार", "पैसे", "स्टायपेंड", "पगार",
        "वेतन", "पेड", "मोफत"
    ],
    "Project_details": [
        "थेट प्रोजेक्ट", "काम", "असाइनमेंट", "कार्य", "प्रोजेक्ट",
        "प्रशिक्षणार्थी", "वरिष्ठ", "संघ", "सहकार्य", "विकसित करा",
        "बनवा", "सर्जन करा", "अमलात आणा", "प्रॅक्टिकल", "हँड्स-ऑन"
    ],
    "Confirmation": [
        "रुची आहे", "व्हॉट्सअ‍ॅप पाठवा", "समजलं",
        "स्वीकारलं", "नोंद घेतली", "कृपया पाठवा", "तपशील पाठवला", "मान्य आहे"
    ]
},
"bn": {
    "Strong_interest": [
        "অবশ্যই", "আমি প্রস্তুত", "যোগ দিতে চাই", "আগ্রহী",
        "বিস্তারিত শেয়ার করুন", "ব্রোশার পাঠান", "আমি যোগ দেব", "চলুন এগিয়ে যাই",
        "আমি কোথায় সাইন করব?", "শেয়ার করুন", "আমি কখন শুরু করতে পারি?", "গ্রহণ করছি",
        "অপেক্ষায় আছি", "উৎসাহী", "খুশি", "ভালো লাগছে", "উত্সুক",
        "এটি শেয়ার করুন", "হোয়াটসঅ্যাপ করুন", "আমি রেডি"
    ],
    "Moderate_interest": [
        "হয়তো", "ভাবছি", "বিবেচনা করব", "আমাকে ভাবতে দিন", "আরও বলুন",
        "আরও বিস্তারিত", "ব্যাখ্যা করুন", "স্পষ্ট করুন", "নিশ্চিত না", "সম্ভবত",
        "হতে পারে", "হয়তো হতে পারে", "এটা নির্ভর করে", "চেক করতে হবে", "পরবর্তীতে সিদ্ধান্ত নেব",
        "ফিরে জানাবো", "আলোচনা করতে হবে", "পরামর্শ নিতে হবে", "পর্যালোচনা", "মূল্যায়ন"
    ],
    "No_interest": [
        "পারছি না", "করব না", "ভাল লাগছে না",
        "এখন নয়", "পরে", "উপযুক্ত নয়", "প্রত্যাখ্যান করছি"
    ],
    "company_query": [
        "টিনো সফটওয়্যার অ্যান্ড সিকিউরিটি সলিউশনস", "টিনো সফটওয়্যার আইটি কোম্পানি", "টিনো সফটওয়্যার",
        "আমি টিনো সফটওয়্যার অ্যান্ড সিকিউরিটি সলিউশনস থেকে বলছি", "টিনোস সফটওয়্যার"
    ],
    "Qualification_query": [
        "যোগ্যতা", "শিক্ষা", "কম্পিউটার সায়েন্স", "ডিগ্রি", "পড়াশোনা", "কোর্স",
        "পটভূমি", "একাডেমিক", "বিশ্ববিদ্যালয়", "কলেজ", "বিএসসি",
        "স্নাতক", "কোন বর্ষে পড়ছেন?", "পাঠ্যক্রম", "সিলেবাস"
    ],
    "Internship_details": [
        "ইন্টার্নশিপ", "প্লেসমেন্ট", "প্রোগ্রাম", "ইন্টার্নশিপ খুঁজছে", "সময়কাল",
        "ডেটা সায়েন্স", "মাস", "সময়সীমা", "সময়", "শিডিউল", "টাইমফ্রেম",
        "১ থেকে ৩", "তিন মাস", "গঠন", "পরিকল্পনা", "ফ্রেমওয়ার্ক",
        "আমি ডেটা সায়েন্সে ইন্টার্নশিপ খুঁজছি"
    ],
    "Location_query": [
        "অনলাইন", "অফলাইন", "অবস্থান", "স্থান", "কোথায়",
        "ঠিকানা", "স্থানান্তর", "রিলোকেট", "থেকে", "আসছি",
        "কোজিকোড", "কোচি", "পলারিভাট্টাম", "হাইব্রিড", "রিমোট"
    ],
    "Certificate_query": [
        "সার্টিফিকেট", "সনদপত্র", "নথি", "প্রমাণ",
        "অভিজ্ঞতার সার্টিফিকেট", "প্রশিক্ষণ সার্টিফিকেট", "চিঠি",
        "সম্পন্ন", "পুরস্কার", "স্বীকৃতি"
    ],
    "Fee_query": [
        "ফি", "পেমেন্ট", "খরচ", "পরিমাণ", "চার্জ",
        "৬০০০", "ছয় হাজার", "টাকা", "স্টাইপেন্ড", "বেতন",
        "পারিশ্রমিক", "পেইড", "ফ্রি"
    ],
    "Project_details": [
        "লাইভ প্রজেক্ট", "কাজ", "অ্যাসাইনমেন্ট", "টাস্ক", "প্রজেক্ট",
        "ইন্টার্ন", "সিনিয়র", "টিম", "সহযোগিতা", "উন্নয়ন করা",
        "তৈরি করা", "সৃষ্টি করা", "কার্যকর করা", "প্র্যাকটিক্যাল", "হ্যান্ডস-অন"
    ],
    "Confirmation": [
        "আগ্রহী", "হোয়াটসঅ্যাপে পাঠান", "বুঝেছি",
        "গ্রহণ করেছি", "নোট করে নিয়েছি", "দয়া করে পাঠান", "বিস্তারিত পাঠানো হয়েছে", "আমি সম্মত"
    ]
},
"or": {
    "Strong_interest": [
        "ନିଶ୍ଚିତଭାବେ", "ମୁଁ ପ୍ରସ୍ତୁତ", "ଯୋଗଦେବାକୁ ଚାହେଁ", "ଆଗ୍ରହ ଅଛି",
        "ବିବରଣୀ ଅଂଶୀଦାର କରନ୍ତୁ", "ବ୍ରୋଶର୍ ପଠାନ୍ତୁ", "ମୁଁ ଯୋଗଦେବି", "ଆଗକୁ ବଢ଼ିବା",
        "ମୁଁ କେଉଁଠି ସାଇନ କରିବି?", "ଅଂଶୀଦାର କରନ୍ତୁ", "ମୁଁ କେବେ ଆରମ୍ଭ କରିପାରିବି?", "ଗ୍ରହଣ କରିଛି",
        "ଅପେକ୍ଷା କରୁଛି", "ଉତ୍ସାହିତ", "ଖୁସି ଲାଗିଲା", "ଭଲ ଲାଗିଲା", "ଆତୁର ଅଛି",
        "ଏହାକୁ ଅଂଶୀଦାର କରନ୍ତୁ", "ଓଟସାପ୍ କରନ୍ତୁ", "ମୁଁ ରେଡି"
    ],
    "Moderate_interest": [
        "ସମ୍ଭବତଃ", "ଭାବିବି", "ଚିନ୍ତା କରୁଛି", "ମୁଁ ଭାବିବାକୁ ଦିଅ", "ଅଧିକ କୁହନ୍ତୁ",
        "ଅଧିକ ବିବରଣୀ", "ବ୍ୟାଖ୍ୟା କରନ୍ତୁ", "ସ୍ପଷ୍ଟ କରନ୍ତୁ", "ନିଶ୍ଚିତ ନୁହେଁ", "ଶାୟଦ",
        "ହେବା ସମ୍ଭବ", "ହେଇପାରେ", "ଏହା ଉପରେ ନିର୍ଭର କରେ", "ଯାଞ୍ଚ କରିବାକୁ ପଡ଼ିବ", "ସିଦ୍ଧାନ୍ତ ନେବି",
        "ପଛରୁ କୁହିବି", "ଚର୍ଚ୍ଚା କରିବାକୁ ପଡ଼ିବ", "ପରାମର୍ଶ ନେବାକୁ ପଡ଼ିବ", "ପୁନଃବିଚାର", "ମୂଲ୍ୟାଙ୍କନ"
    ],
    "No_interest": [
        "ମୁଁ ପାରିବିନାହିଁ", "ମୁଁ କରିବିନାହିଁ", "ଭଲ ଲାଗୁନାହିଁ",
        "ଏବେ ନୁହେଁ", "ପରେ", "ଉପଯୁକ୍ତ ନୁହେଁ", "ମୁଁ ଅସ୍ୱୀକାର କରେ"
    ],
    "company_query": [
        "ଟିନୋ ସଫ୍ଟୱେୟାର ଏବଂ ସିକ୍ୟୁରିଟି ସୋଲୁସନ୍ସ", "ଟିନୋ ସଫ୍ଟୱେୟାର ଆଇଟି କମ୍ପାନୀ", "ଟିନୋ ସଫ୍ଟୱେୟାର",
        "ମୁଁ ଟିନୋ ସଫ୍ଟୱେୟାର ଏବଂ ସିକ୍ୟୁରିଟି ସୋଲୁସନ୍ସରୁ କହୁଛି", "ଟିନୋଜ୍ ସଫ୍ଟୱେୟାର"
    ],
    "Qualification_query": [
        "ଯୋଗ୍ୟତା", "ଶିକ୍ଷା", "କମ୍ପ୍ୟୁଟର ସାଇନ୍ସ", "ଡିଗ୍ରୀ", "ପଢ଼ାଶୁଣା", "କୋର୍ସ",
        "ପୃଷ୍ଠଭୂମି", "ଶିକ୍ଷାତ୍ମକ", "ବିଶ୍ୱବିଦ୍ୟାଳୟ", "କଲେଜ୍", "ବିଏସ୍ସି",
        "ସ୍ନାତକ", "କେଉଁ ବର୍ଷରେ ପଢ଼ୁଛ?", "ପାଠ୍ୟକ୍ରମ", "ସିଲେବସ୍"
    ],
    "Internship_details": [
        "ଇଣ୍ଟର୍ନସିପ୍", "ପ୍ଲେସମେଣ୍ଟ", "ପ୍ରୋଗ୍ରାମ୍", "ଇଣ୍ଟର୍ନସିପ୍ ଖୋଜୁଛି", "ଅବଧି",
        "ଡେଟା ସାଇନ୍ସ", "ମାସ", "ପିରିଅଡ୍", "ସମୟ", "ସୂଚୀ", "ସମୟସୀମା",
        "୧ରୁ ୩", "ତିନି ମାସ", "ଢଂଗ", "ଯୋଜନା", "ଫ୍ରେମ୍ୱାର୍କ୍",
        "ମୁଁ ଡେଟା ସାଇନ୍ସରେ ଇଣ୍ଟର୍ନସିପ୍ ଚାହୁଁଛି"
    ],
    "Location_query": [
        "ଅନଲାଇନ୍", "ଅଫଲାଇନ୍", "ଅବସ୍ଥାନ", "ସ୍ଥାନ", "କେଉଁଠି",
        "ଠିକଣା", "ସ୍ଥାନାନ୍ତର", "ରିଲୋକେଟ୍", "ରୁ", "ଆସୁଛି",
        "କୋଝିକୋଡ୍", "କୋଚି", "ପଲାରିଭଟ୍ଟମ୍", "ହାଇବ୍ରିଡ୍", "ରିମୋଟ୍"
    ],
    "Certificate_query": [
        "ପ୍ରମାଣପତ୍ର", "ସର୍ଟିଫିକେଟ୍", "ଦଲିଲ", "ପ୍ରମାଣ",
        "ଅନୁଭବ ସର୍ଟିଫିକେଟ୍", "ପ୍ରଶିକ୍ଷଣ ପତ୍ର", "ଚିଠି",
        "ସମ୍ପୂର୍ଣ୍ଣ", "ପୁରସ୍କାର", "ସ୍ୱୀକୃତି"
    ],
    "Fee_query": [
        "ଶୁଳ୍କ", "ପେମେଣ୍ଟ", "ଖର୍ଚ୍ଚ", "ରାଶି", "ଚାର୍ଜ୍",
        "6000", "ଛଅ ହଜାର", "ଟଙ୍କା", "ସ୍ଟାଇପେଣ୍ଡ୍", "ବେତନ",
        "ପ୍ରତିଫଳ", "ପେଡ୍", "ମାଗଣା"
    ],
    "Project_details": [
        "ଲାଇଭ୍ ପ୍ରୋଜେକ୍ଟ୍", "କାମ", "ଅସାଇନ୍‌ମେଣ୍ଟ", "କାର୍ଯ୍ୟ", "ପ୍ରକଳ୍ପ",
        "ଇଣ୍ଟର୍ନ୍", "ସିନିଅର୍", "ଦଳ", "ସହଯୋଗ", "ବିକାଶ କରନ୍ତୁ",
        "ତିଆରି କରନ୍ତୁ", "ସୃଷ୍ଟି କରନ୍ତୁ", "କାର୍ଯ୍ୟାନ୍ୱୟନ କରନ୍ତୁ", "ପ୍ରାୟୋଗିକ", "ହ୍ୟାଣ୍ଡସଅନ୍"
    ],
    "Confirmation": [
        "ଆଗ୍ରହ ଅଛି", "ଓଟସାପ୍ ପଠାନ୍ତୁ", "ବୁଝିଗଲି",
        "ଗ୍ରହଣ କରିଛି", "ନୋଟ କରିଛି", "ଦୟାକରି ପଠାନ୍ତୁ", "ବିବରଣୀ ପଠାଇଦିଆଯାଇଛି", "ସମ୍ମତି ଅଛି"
    ]
},
"as": {
    "Strong_interest": [
        "নিশ্চয়", "মই প্রস্তুত", "যোগ দিব খুজিছো", "আগ্ৰহী",
        "বিৱৰণ শ্বেয়াৰ কৰক", "ব্ৰোছাৰ পঠিয়াওক", "মই যোগ দিম", "আগবাঢ়ো আহক",
        "মই ক’ত চাইন কৰিম?", "শ্বেয়াৰ কৰক", "মই কেতিয়া আৰম্ভ কৰিব পাৰিম?", "গ্ৰহণ কৰিছো",
        "অপেক্ষা কৰি আছো", "উৎসাহী", "আনন্দিত", "ভাল লাগিল", "উত্সুক",
        "এইটো শ্বেয়াৰ কৰক", "ৱাটছএপ কৰক", "মই ৰেডি"
    ],
    "Moderate_interest": [
        "সম্ভৱত", "চিন্তা কৰিম", "চিন্তা কৰি আছো", "চিন্তা কৰিব দিয়া", "আৰু ক’বা",
        "অধিক বিৱৰণ", "ব্যাখ্যা কৰক", "পৰিষ্কাৰ কৰক", "নিশ্চিত নহয়", "সম্ভৱ",
        "হ’ব পাৰে", "হয়তো", "ই নির্ভৰ কৰে", "পৰীক্ষা কৰিব লাগিব", "পিছত সিদ্ধান্ত ল’ম",
        "পাছত জনাম", "আলোচনা কৰিব লাগিব", "পৰামৰ্শ ল’ব লাগিব", "পুনঃসমীক্ষা", "মূল্যাংকন"
    ],
    "No_interest": [
        "নোৱাৰো", "কৰিম নোৱাৰো", "ভাল লগা নাই",
        "এতিয়া নহয়", "পিছত", "উপযুক্ত নহয়", "প্ৰত্যাখ্যান কৰিছো"
    ],
    "company_query": [
        "টিনো ছফ্টৱেৰ আৰু ছিকিউৰিটি ছলিউচনছ", "টিনো ছফ্টৱেৰ আইটি কোম্পানী", "টিনো ছফ্টৱেৰ",
        "মই টিনো ছফ্টৱেৰ আৰু ছিকিউৰিটি ছলিউচনছৰ পৰা কৈ আছো", "টিনোজ ছফ্টৱেৰ"
    ],
    "Qualification_query": [
        "যোগ্যতা", "শিক্ষা", "কম্পিউটাৰ চায়েন্স", "ডিগ্ৰী", "পঢ়া", "কোৰ্ছ",
        "পটভূমি", "শৈক্ষিক", "বিশ্ববিদ্যালয়", "কলেজ", "বিএছচি",
        "স্নাতক", "কোন বছৰত?", "পাঠ্যক্রম", "ছিলেবাছ"
    ],
    "Internship_details": [
        "ইন্টাৰ্ণশ্বিপ", "প্লেছমেন্ট", "প্ৰগ্ৰাম", "ইন্টাৰ্ণশ্বিপ বিচাৰি আছে", "সময়সীমা",
        "ডেটা চায়েন্স", "মাহ", "সময়কাল", "সময়", "তালিকা", "সময়ৰেখা",
        "১ ৰ পৰা ৩", "তিনি মাহ", "গঠন", "পরিকল্পনা", "ফ্ৰেমৱৰ্ক",
        "মই ডেটা চায়েন্সত ইন্টাৰ্ণশ্বিপ বিচাৰি আছো"
    ],
    "Location_query": [
        "অনলাইন", "অফলাইন", "অৱস্থান", "স্থান", "ক’ত",
        "ঠিকনা", "স্থানান্তৰ", "ৰিল’কেট", "ৰ পৰা", "আহি আছো",
        "কোজিকোড", "কোচি", "পলাৰিভাট্টম", "হাইব্ৰিড", "ৰিম’ট"
    ],
    "Certificate_query": [
        "প্ৰমাণপত্ৰ", "চাৰ্টিফিকেট", "দস্তাবেজ", "প্ৰমাণ",
        "অভিজ্ঞতাৰ চাৰ্টিফিকেট", "প্ৰশিক্ষণ চাৰ্টিফিকেট", "চিঠি",
        "সম্পূৰ্ণ", "পুৰস্কাৰ", "স্বীকৃতি"
    ],
    "Fee_query": [
        "ফি", "পেমেন্ট", "খৰচ", "পৰিমাণ", "চাৰ্জ",
        "৬০০০", "ছয় হাজাৰ", "টকা", "ষ্টাইপেণ্ড", "দৰমহা",
        "পৰিশ্ৰমিক", "পেইড", "বিনামূলীয়া"
    ],
    "Project_details": [
        "লাইভ প্ৰজেক্ট", "কাজ", "অ্যাসাইনমেণ্ট", "টাস্ক", "প্ৰকল্প",
        "ইণ্টাৰ্ণ", "সিনিয়ৰ", "দল", "সহযোগিতা", "উন্নয়ন কৰক",
        "তৈয়াৰ কৰক", "সৃষ্টি কৰক", "কাৰ্য্যন্বয় কৰক", "প্ৰাকটিকেল", "হেণ্ডছ-অন"
    ],
    "Confirmation": [
        "আগ্ৰহী", "ৱাটছএপত পঠিয়াওক", "বুজিছো",
        "গ্ৰহণ কৰিছো", "নোট কৰিছো", "অনুগ্ৰহ কৰি পঠিয়াওক", "বিৱৰণ পঠিওৱা হৈছে", "মই একমত"
    ]
}




        }

        # Check if language is supported
        if language not in intent_keywords:
            logger.warning(f"Unsupported language {language} in intent_keywords. Returning Neutral_response.")
            return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

        # Check for each intent type in order of priority
        if any(keyword in text_lower for keyword in intent_keywords[language]["Confirmation"]):
            logger.debug(f"Detected intent: Confirmation for text: {text_lower} in language: {language}")
            return {"intent": "Confirmation", "sentiment": "very positive", "sentiment_score": 0.9}

        if any(keyword in text_lower for keyword in intent_keywords[language]["Strong_interest"]):
            logger.debug(f"Detected intent: Strong_interest for text: {text_lower} in language: {language}")
            return {"intent": "Strong_interest", "sentiment": "positive", "sentiment_score": 0.7}

        if any(keyword in text_lower for keyword in intent_keywords[language]["company_query"]):
            logger.debug(f"Detected intent: company_query for text: {text_lower} in language: {language}")
            return {"intent": "company_query", "sentiment": "neutral", "sentiment_score": 0.5}

        if any(keyword in text_lower for keyword in intent_keywords[language]["No_interest"]):
            logger.debug(f"Detected intent: No_interest for text: {text_lower} in language: {language}")
            return {"intent": "No_interest", "sentiment": "negative", "sentiment_score": 0.2}

        if any(keyword in text_lower for keyword in intent_keywords[language]["Moderate_interest"]):
            logger.debug(f"Detected intent: Moderate_interest for text: {text_lower} in language: {language}")
            return {"intent": "Moderate_interest", "sentiment": "neutral", "sentiment_score": 0.5}

        for intent, keywords in intent_keywords[language].items():
            if intent not in ["Confirmation", "company_query", "Strong_interest", "No_interest", "Moderate_interest"]:
                if any(keyword in text_lower for keyword in keywords):
                    logger.debug(f"Detected intent: {intent} for text: {text_lower} in language: {language}")
                    return {"intent": intent, "sentiment": "neutral", "sentiment_score": 0.5}

        logger.debug(f"No specific intent detected for text: {text_lower} in language: {language}. Returning Neutral_response.")
        return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

    except Exception as e:
        logger.error(f"Error in detect_intent for language {language}: {str(e)}", exc_info=True)
        return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

def analyze_text(text, language="en"):
    import logging
    logger = logging.getLogger(__name__)

    try:
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid input text for analyze_text: {text}. Returning empty analysis.")
            return []

        # Split text into sentences
        sentences = split_into_sentences(text, language)
        if not sentences:
            logger.warning(f"No sentences found for text: {text} in language: {language}")
            return []

        analysis = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                logger.debug(f"Skipping empty sentence at index {i} for language {language}")
                analysis.append({
                    "sentence_id": f"{language}_{i+1}",
                    "text": sentence,
                    "language": language,
                    "intent": "Neutral_response",
                    "sentiment": "neutral",
                    "sentiment_score": 0.5,
                    "word_count": 0,
                    "char_count": 0
                })
                continue

            # Detect intent
            try:
                intent_result = detect_intent(sentence, language)
                if not isinstance(intent_result, dict) or "intent" not in intent_result:
                    logger.warning(f"Invalid intent_result for sentence: {sentence}. Using Neutral_response.")
                    intent_result = {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}
            except Exception as e:
                logger.error(f"detect_intent failed for sentence: {sentence} in language: {language}. Error: {str(e)}")
                intent_result = {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

            # Analyze sentiment
            try:
                sentiment_result = analyze_sentiment_batch([sentence])[0] if sentence.strip() else {
                    "label": "neutral",
                    "score": 0.5
                }
                if not isinstance(sentiment_result, dict) or "label" not in sentiment_result or "score" not in sentiment_result:
                    logger.warning(f"Invalid sentiment_result for sentence: {sentence}. Using default neutral.")
                    sentiment_result = {"label": "neutral", "score": 0.5}
            except Exception as e:
                logger.error(f"analyze_sentiment_batch failed for sentence: {sentence} in language: {language}. Error: {str(e)}")
                sentiment_result = {"label": "neutral", "score": 0.5}

            # Combine intent and sentiment results
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

        if not analysis:
            logger.warning(f"No analysis results generated for text: {text} in language: {language}")
            analysis.append({
                "sentence_id": f"{language}_1",
                "text": text,
                "language": language,
                "intent": "Neutral_response",
                "sentiment": "neutral",
                "sentiment_score": 0.5,
                "word_count": len(text.split()) if isinstance(text, str) else 0,
                "char_count": len(text) if isinstance(text, str) else 0
            })

        logger.debug(f"Analysis results for {language}: {len(analysis)} items")
        return analysis

    except Exception as e:
        logger.error(f"Error in analyze_text for language {language}: {str(e)}", exc_info=True)
        return [{
            "sentence_id": f"{language}_1",
            "text": text if isinstance(text, str) else "",
            "language": language,
            "intent": "Neutral_response",
            "sentiment": "neutral",
            "sentiment_score": 0.5,
            "word_count": len(text.split()) if isinstance(text, str) else 0,
            "char_count": len(text) if isinstance(text, str) else 0
        }]

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



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import matplotlib.patches as mpatches
import matplotlib as mpl

def generate_analysis_pdf(en_analysis, ml_analysis, comparison, filename_prefix):
    """Generate a visually appealing PDF report with analysis metrics and visualizations, ensuring equal page sizes"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"analysis_results/{filename_prefix}_visual_report_{timestamp}.pdf"
    os.makedirs("analysis_results", exist_ok=True)
    
    default_font = mpl.font_manager.FontProperties(family='DejaVu Sans', weight='normal', size=12)
    title_font = mpl.font_manager.FontProperties(family='DejaVu Sans', weight='bold', size=16)
    subtitle_font = mpl.font_manager.FontProperties(family='DejaVu Sans', weight='normal', size=14)
    
    if 'malayalam_font' in globals() and malayalam_font:
        malayalam_font_prop = malayalam_font
        font_path = malayalam_font.get_file()
        font_name = malayalam_font.get_name()
        print(f"Using Malayalam font: {font_name} at {font_path}")
    else:
        malayalam_font_prop = default_font
        print("No Malayalam font configured. Using default font: DejaVu Sans")
    
    en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
    ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
    combined_avg = (en_avg_score + ml_avg_score) / 2 if ml_analysis else en_avg_score
    base_lead_score = int(combined_avg * 100)
    
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
    positive_extra_points = 10
    negative_extra_points = -10

    last_en_sentences = [item["text"].lower() for item in en_analysis[-5:]] if en_analysis else []
    last_ml_sentences = [item["text"] for item in ml_analysis[-5:]] if ml_analysis else []

    extra_points = 0
    for i, sentence in enumerate(last_en_sentences):
        for keyword in positive_keywords_en:
            if keyword in sentence:
                extra_points += positive_extra_points
        for keyword in negative_keywords_en:
            if keyword in sentence:
                extra_points += negative_extra_points
    for i, sentence in enumerate(last_ml_sentences):
        for keyword in positive_keywords_ml:
            if keyword in sentence:
                extra_points += positive_extra_points
        for keyword in negative_keywords_ml:
            if keyword in sentence:
                extra_points += negative_extra_points
    lead_score = max(0, min(base_lead_score + extra_points, 100))

    positive_intents = sum(1 for item in en_analysis if item["intent"] in ["Strong_interest", "Fee_query", "Moderate_interest", "Confirmation"])
    intent_score = int((positive_intents / len(en_analysis)) * 100) if en_analysis else 0

    def get_score_color(score):
        if score >= 70:
            return '#4CBB17'  
        elif score >= 40:
            return '#F1C40F'  
        else:
            return '#E74C3C'  

    lead_score_color = get_score_color(lead_score)
    intent_score_color = get_score_color(intent_score)

    en_sentiments = [item["sentiment"] for item in en_analysis]
    ml_sentiments = [item["sentiment"] for item in ml_analysis]
    en_scores = [item["sentiment_score"] for item in en_analysis]
    ml_scores = [item["sentiment_score"] for item in ml_analysis]
    sentence_numbers = list(range(1, len(en_analysis)+1)) if en_analysis else []
    
    with PdfPages(pdf_filename) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#F5F6F5')
        fig.patch.set_facecolor('#F5F6F5')
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  
        rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='#2C3E50', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.85, "Lead Scoring System", ha='center', va='center', fontproperties=title_font, color='#2C3E50')
        ax.text(0.5, 0.75, "Conversation Analysis Report", ha='center', va='center', fontproperties=subtitle_font, color='#34495E')
        ax.text(0.5, 0.65, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ha='center', va='center', fontproperties=default_font, color='#34495E')
        ax.text(0.5, 0.55, f"Filename: {filename_prefix}", ha='center', va='center', fontproperties=default_font, color='#34495E')
        ax.text(0.5, 0.10, "Powered by LSS", ha='center', va='center', fontproperties=default_font, color='#7F8C8D')
        ax.text(0.5, 0.05, "Page 1", ha='center', va='center', fontproperties=default_font, color='#7F8C8D')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches=None)  
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#F5F6F5')
        fig.patch.set_facecolor('#F5F6F5')
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='#2C3E50', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.92, "Key Metrics", ha='center', va='center', fontproperties=title_font, color='#2C3E50')
        metrics = [
            ("English Avg Sentiment", f"{en_avg_score:.2f}", '#3498DB'),
            ("Malayalam Avg Sentiment", f"{ml_avg_score:.2f}", '#2ECC71'),
            ("Combined Avg Sentiment", f"{combined_avg:.2f}", '#9B59B6'),
            ("Lead Score", f"{lead_score:.2f}/100", lead_score_color),
            ("Intent Score", f"{intent_score:.2f}/100", intent_score_color),
        ]
        for i, (label, value, color) in enumerate(metrics):
            y_pos = 0.80 - i * 0.10
            ax.text(0.15, y_pos, label, ha='left', va='center', fontproperties=subtitle_font, color='#34495E')
            ax.text(0.85, y_pos, value, ha='right', va='center', fontproperties=subtitle_font, color=color, weight='bold')
        
        interpretation = "High interest lead" if lead_score >= 70 else "Moderate interest lead" if lead_score >= 40 else "Low interest lead"
        ax.text(0.15, 0.30, "Interpretation:", ha='left', va='center', fontproperties=subtitle_font, color='#34495E')
        ax.text(0.85, 0.30, interpretation, ha='right', va='center', fontproperties=subtitle_font, color=lead_score_color, weight='bold')
        ax.text(0.5, 0.05, "Page 2", ha='center', va='center', fontproperties=default_font, color='#7F8C8D')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches=None)
        plt.close(fig)
        
        
        if en_analysis and ml_analysis:
           
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#F5F6F5')
            fig.patch.set_facecolor('#F5F6F5')
            fig.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
            rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='#2C3E50', linewidth=2)
            ax.add_patch(rect)
            
            plt.subplot(1, 2, 1)
            en_counts = pd.Series(en_sentiments).value_counts()
            en_counts.plot(kind='bar', color='#3498DB', edgecolor='#2C3E50')
            plt.title('English Sentiment Distribution', fontproperties=title_font, color='#2C3E50')
            plt.xlabel('Sentiment', fontproperties=default_font)
            plt.ylabel('Count', fontproperties=default_font)
            plt.xticks(rotation=45, fontproperties=default_font)
            for i, v in enumerate(en_counts):
                plt.text(i, v + 0.2, str(v), ha='center', fontproperties=default_font, color='#2C3E50')
            
            plt.subplot(1, 2, 2)
            ml_counts = pd.Series(ml_sentiments).value_counts()
            ml_counts.plot(kind='bar', color='#2ECC71', edgecolor='#2C3E50')
            plt.title('Malayalam Sentiment Distribution', fontproperties=malayalam_font_prop, color='#2C3E50')
            plt.xlabel('Sentiment', fontproperties=malayalam_font_prop)
            plt.ylabel('Count', fontproperties=default_font)
            plt.xticks(rotation=45, fontproperties=malayalam_font_prop)
            for i, v in enumerate(ml_counts):
                plt.text(i, v + 0.2, str(v), ha='center', fontproperties=malayalam_font_prop, color='#2C3E50')
            
            plt.suptitle("Sentiment Distribution", fontproperties=title_font, y=0.95, color='#2C3E50')
            plt.tight_layout()
            plt.text(0.5, 0.05, "Page 3", ha='center', va='center', transform=fig.transFigure, fontproperties=default_font, color='#7F8C8D')
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
            
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#F5F6F5')
            fig.patch.set_facecolor('#F5F6F5')
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
            rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='#2C3E50', linewidth=2)
            ax.add_patch(rect)
            
            plt.plot(sentence_numbers, en_scores, marker='o', label='English', color='#3498DB', linewidth=2)
            plt.plot(sentence_numbers, ml_scores, marker='s', label='Malayalam', color='#2ECC71', linewidth=2)
            plt.xlabel('Sentence Number', fontproperties=default_font, color='#2C3E50')
            plt.ylabel('Sentiment Score', fontproperties=default_font, color='#2C3E50')
            plt.title('Sentiment Trend Over Conversation', fontproperties=title_font, color='#2C3E50')
            plt.legend(prop=default_font, loc='upper right', frameon=True, facecolor='#ECF0F1')
            plt.grid(True, linestyle='--', alpha=0.7)
            for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                label.set_fontproperties(default_font)
                label.set_color('#2C3E50')
            plt.text(0.5, 0.05, "Page 4", ha='center', va='center', transform=fig.transFigure, fontproperties=default_font, color='#7F8C8D')
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
            
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#F5F6F5')
            fig.patch.set_facecolor('#F5F6F5')
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
            rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='#2C3E50', linewidth=2)
            ax.add_patch(rect)
            
            en_intents = [item["intent"] for item in en_analysis]
            intent_counts = pd.Series(en_intents).value_counts()
            intent_counts.plot(kind='bar', color='#E67E22', edgecolor='#2C3E50')
            plt.title('Intent Distribution', fontproperties=title_font, color='#2C3E50')
            plt.xlabel('Intent', fontproperties=default_font, color='#2C3E50')
            plt.ylabel('Count', fontproperties=default_font, color='#2C3E50')
            plt.xticks(rotation=45, fontproperties=default_font)
            for i, v in enumerate(intent_counts):
                plt.text(i, v + 0.2, str(v), ha='center', fontproperties=default_font, color='#2C3E50')
            plt.tight_layout()
            plt.text(0.5, 0.05, "Page 5", ha='center', va='center', transform=fig.transFigure, fontproperties=default_font, color='#7F8C8D')
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_facecolor('#F5F6F5')
            fig.patch.set_facecolor('#F5F6F5')
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
            rect = mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='#2C3E50', linewidth=2)
            ax.add_patch(rect)
            
            sentiment_diffs = [abs(en - ml) for en, ml in zip(en_scores, ml_scores)]
            plt.hist(sentiment_diffs, bins=10, color='#9B59B6', alpha=0.7, edgecolor='#2C3E50')
            plt.xlabel('Sentiment Score Difference', fontproperties=default_font, color='#2C3E50')
            plt.ylabel('Frequency', fontproperties=default_font, color='#2C3E50')
            plt.title('English-Malayalam Sentiment Differences', fontproperties=title_font, color='#2C3E50')
            for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                label.set_fontproperties(default_font)
                label.set_color('#2C3E50')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.text(0.5, 0.05, "Page 6", ha='center', va='center', transform=fig.transFigure, fontproperties=default_font, color='#7F8C8D')
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
    
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




if __name__ == "__main__":
    while True:
        print("\n=== Main Menu ===")
        print("1. Analyze new audio file")
        print("2. Search existing analyses")
        print("3. Exit")
        
        
