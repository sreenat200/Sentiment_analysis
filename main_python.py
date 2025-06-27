import os
import tempfile
from datetime import datetime
import torch
from pydub import AudioSegment
import subprocess
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
import multiprocessing
import shutil
import urllib.request
import os
import os
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

def load_token_from_env():
    load_dotenv()
    return os.getenv("HF_TOKEN", "")


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

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
from pydub import AudioSegment


class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="base"):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Faster-Whisper {model_size} model on {self.device}...")

        # Set threading & compute type
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

        # Load Whisper model
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

        # Load emotion classifier manually (no pipeline)
        self.emotion_model = None
        self.emotion_tokenizer = None
        try:
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.emotion_model.to(self.device)
            print("Emotion model loaded to real device.")
        except Exception as e:
            print(f"Failed to load emotion model: {e}")

        # Other components
        self.temp_files = []
        self.translator = None

    def transcribe_audio(self, audio_path):
        try:
            if not audio_path.lower().endswith('.wav'):
                audio_path = self.convert_to_whisper_format(audio_path)
                if not audio_path:
                    return None

            print("Transcribing audio...")
            num_workers = os.cpu_count() or 1
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                num_workers=num_workers
            )

            full_text = " ".join(seg.text.strip() for seg in segments)

            # Emotion analysis (safe fallback)
            try:
                emotion = self.analyze_emotion(full_text)
                print(f"Detected Emotion: {emotion.upper()}")
            except Exception as e:
                print(f"Emotion analysis skipped: {e}")
                emotion = "unknown"

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
            print(f"Transcription failed: {e}")
            return None

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
            'ശരി': 'യെസ് ',
            'ഇന്റർവ്യൂ': 'ഇന്റർവ്യൂ',
            'പരിശീലനം': 'ട്രെയിനിംഗ്',
            'പരിശീലനത്തിനായി': 'ട്രെയിനിംഗിനായി',
            'സുപ്രഭാതം': 'ഗുഡ്മോറ്നിംഗ്',
            'തിരയുകയാണോ': 'നോക്കുകയാണോ ',
            'തിരയുന്നത്': 'നോക്കുന്നത്',
            'പങ്കിടുക': 'ഷെയർ ചെയ്യുക',
            'പങ്കിടുന്നു': 'ഷെയർ ചെയ്യുന്നു',
            'പങ്കിടാൻ': 'ഷെയർ ചെയ്യാൻ',
            'കിഴിവ് ': 'ഡിസ്‌കൗണ്ട്',
            'മൊബൈൽ വികസനത്തിനായി': 'മൊബൈൽ ഡെവലപ്മെന്റിനായി',
            'പ്രവർത്തിക്കുന്നു': 'വർക്കിംഗ്',
            'പ്രവർത്തിക്കാൻ': 'വർക്കിംഗ്',
            'പങ്കിടട്ടെ': 'ഷെയർ ചെയ്യട്ടെ',
            'പരിശീലനത്തെയും': 'ട്രെയിനിംഗിനെയും',
            'തിരയുകയാണ്.    ': 'നോക്കുകയാണ്.',
            'വികസനത്തിനായി':'ഡെവലൊപ്മെന്റിനായി ',
            'ഭൗതികശാസ്ത്രം': 'ഫിസിക്സ്',
            'പരിശീലന സർട്ടിഫിക്കേഷനാണ്': 'ട്രെയിനിംഗ് സർട്ടിഫിക്കേഷൻ'
        }
    
        try:
            # Extract text from input
            if isinstance(text_or_dict, dict):
                text = text_or_dict.get('raw_transcription', '')
            else:
                text = text_or_dict

            if not text.strip():
                raise ValueError("No text found for translation")

            print("Translating to Malayalam using Google Translator...")
            try:
                # Attempt translation with GoogleTranslator
                ml_text = GoogleTranslator(source='en', target='ml').translate(text)
            except Exception as google_error:
                print(f"Google Translator failed: {str(google_error)}. Falling back to NLLB model...")
                # Load NLLB model only if Google Translator fails
                if self.translator is None:
                    try:
                        self.translator = pipeline(
                            "translation",
                            model="facebook/nllb-200-distilled-600M",
                            src_lang="eng_Latn",
                            tgt_lang="mal_Mlym",
                            device=0 if self.device == "cuda" else -1  # GPU if available, else CPU
                        )
                        print(f"Translation pipeline loaded on {self.device}")
                    except Exception as e:
                        print(f"Error loading NLLB translation pipeline: {str(e)}")
                        raise
                # Perform translation with NLLB
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


def split_into_sentences(text: str, language: str = "en") -> list[str]:
    """
    Split text into sentences using a regex splitter as primary, with Indic NLP as fallback.
    Splits on full stop (.), comma (,), question mark (?), exclamation mark (!), Malayalam-specific
    punctuation (।,॥), newline, and sequences like ', . ?'. Handles abbreviations and decimals.
    """
    try:
        if not text or not text.strip():
            print(f"No text provided for sentence splitting (language: {language})")
            return []

        print(f"Using regex sentence splitting as primary for language: {language}")
        # Regex to split on sentence-ending punctuation: ., ,, ?, !, ।, ॥, or newline
        # Protects abbreviations (Dr., U.S.) and decimals (3.14), but allows splits after numbers (7000.)
        sentence_endings = re.compile(
            r'(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z]\.)(?<!\d\.\d)(?<=[.,?!।॥]|\n)(?=\s|$|[^\s.,?!])'
        )
        # Pattern for splitting on ', . ?'
        split_pattern = re.compile(r'\s*,\s*\.\s*\?\s*')
        common_abbreviations = {
            'en': r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|Inc|Co|Ltd|U\.S|yes)\.',
            'ml': r'\b(?:ഡോ|ശ്രീ|ശ്രീമതി|പ്രൊ|കോ|യെസ്)\.'
        }
        abbr_pattern = common_abbreviations.get(language, r'\b(?:Dr|Mr|Mrs|Ms)\.')

        # Step 1: Split using regex
        sentences = sentence_endings.split(text)
        cleaned_sentences = []
        current_sentence = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Split on ", . ?" if present
            split_sentences = split_pattern.split(sent)
            split_sentences = [s.strip() for s in split_sentences if s.strip()]

            for split_sent in split_sentences:
                if current_sentence:
                    # Check if the previous sentence ended with an abbreviation
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
            # Split on standalone commas or full stops, ensuring no abbreviation or decimal splits
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

        # Step 3: Fallback to Indic NLP
        lang_code = 'eng' if language == "en" else 'mal'
        try:
            sentences = sentence_tokenize.sentence_split(text, lang=lang_code)
            if sentences:
                print(f"Successfully split {len(sentences)} sentences using Indic NLP ({language})")
                # Post-process Indic NLP results to handle ", . ?" and additional . or , splitting
                cleaned_sentences = []
                current_sentence = ""

                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    # Split on ", . ?" if present
                    split_sentences = split_pattern.split(sent)
                    split_sentences = [s.strip() for s in split_sentences if s.strip()]

                    for split_sent in split_sentences:
                        if current_sentence:
                            # Check if the previous sentence ended with an abbreviation
                            if re.search(abbr_pattern + r'$', current_sentence):
                                current_sentence += " " + split_sent
                            else:
                                cleaned_sentences.append(current_sentence)
                                current_sentence = split_sent
                        else:
                            current_sentence = split_sent
                if current_sentence:
                    cleaned_sentences.append(current_sentence)

                # Additional splitting on standalone commas or full stops
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

        # Step 4: Ultimate fallback
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
    """Enhanced intent detection for internship interest analysis in English and Malayalam"""
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
                "ഇന്റെണ്ഷിപ്", "പരിശീലനം","ഡാറ്റാ സയൻസിലെ", "ഇന്റേൺഷിപ്പിനൊപ്പം","പ്ലെയ്സ്മെന്റ്", 
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
                "ടീം", "മേധാവി", "ട്രെയിനി", "സഹപ്രവർത്തനം","പ്രോജക്റ്റുകൾ ", 
                "ഡവലപ്പുചെയ്യുക", "സൃഷ്ടിക്കുക", "ഇമ്പ്ലിമെന്റുചെയ്യുക", 
                "പ്രായോഗികം", "അഭ്യാസം"
            ],
            "Confirmation": [
                "ശരി", "താല്പര്യമുണ്ട്", "തിരയുന്നു", "ഇഷ്ടമുണ്ട്", "വാട്സാപ്പിൽ അയക്കൂ", "ഷെയർ ചെയ്യുക",
                "വാട്സാപ്പ്", "വാട്ട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു", 
                "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", "ഓക്കെ","യെസ്  ",
                "അക്ക്നലഡ്ജ്", "ക്ലിയർ", "യെസ് .",
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
    
    # Font setup
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
    
    # Calculate lead_score and intent_score matching process_audio
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

    # Color scheme for scores
    def get_score_color(score):
        if score >= 70:
            return '#4CBB17'  # Green for high
        elif score >= 40:
            return '#F1C40F'  # Yellow for medium
        else:
            return '#E74C3C'  # Red for low

    lead_score_color = get_score_color(lead_score)
    intent_score_color = get_score_color(intent_score)

    # Prepare data for visualizations
    en_sentiments = [item["sentiment"] for item in en_analysis]
    ml_sentiments = [item["sentiment"] for item in ml_analysis]
    en_scores = [item["sentiment_score"] for item in en_analysis]
    ml_scores = [item["sentiment_score"] for item in ml_analysis]
    sentence_numbers = list(range(1, len(en_analysis)+1)) if en_analysis else []
    
    with PdfPages(pdf_filename) as pdf:
        # Cover Page
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#F5F6F5')
        fig.patch.set_facecolor('#F5F6F5')
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Consistent margins
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
        pdf.savefig(fig, bbox_inches=None)  # No tight bounding box
        plt.close(fig)
        
        # Key Metrics Page
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
        
        # Visualizations
        if en_analysis and ml_analysis:
            # Sentiment Distribution
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
            
            # Sentiment Trend
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
            
            # Intent Distribution
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
            
            # Sentiment Differences
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
        
        