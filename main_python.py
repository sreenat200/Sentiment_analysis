import os
import zipfile
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
import seaborn as sns
import numpy as np
from io import BytesIO
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
from gdrive_utils import upload_to_gdrive, search_menu

# Ensure NLTK data is available
nltk_data_path = os.path.expanduser('~/.nltk_data')
nltk.download('punkt', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

class MalayalamTranscriptionPipeline:
    def __init__(self, model_size=""):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Faster-Whisper {model_size} model on {self.device}...")
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=self.device, compute_type=compute_type)
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
                    "duration": len(AudioSegment.from_wav(audio_path)) / 1000
                }
            }
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None

    def translate_to_malayalam(self, text_or_dict):
        try:
            if isinstance(text_or_dict, dict):
                text = text_or_dict.get('raw_transcription', '')
            else:
                text = text_or_dict

            if not text.strip():
                raise ValueError("No text found for translation")

            print("Translating to Malayalam...")
            ml_text = GoogleTranslator(source='en', target='ml').translate(text)

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
        if language == "en":
            sentences = sent_tokenize(text)
        else:
            sentences = sent_tokenize(text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            raise Exception("NLTK returned only one sentence, trying Indic NLP")
            
        return sentences
        
    except Exception as nltk_error:
        try:
            print(f"NLTK sentence splitting failed ({str(nltk_error)}), trying Indic NLP...")
            if language == "ml":
                sentences = sentence_tokenize.sentence_split(text, lang='mal')
            else:
                sentences = sentence_tokenize.sentence_split(text, lang='en')
            
            return [s.strip() for s in sentences if s.strip()]
        except Exception as indic_error:
            print(f"Both NLTK and Indic NLP sentence splitting failed: {indic_error}")
            return [text] if text.strip() else []

# Sentiment analysis pipeline
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
                "no", "can't", "won't", "don't like",
                "not now", "later", "not suitable", "decline"
            ],
            "company_query": [
                "tino software and security solutions", "i am calling you from tino software and security solutions",
                "tinos software"
            ],
            "Qualification_query": [
                "qualification", "education", "computer science", "degree", "studying", "course",
                "background", "academics", "university", "college", "bsc",
                "graduate", "year of study", "curriculum", "syllabus"
            ],
            "Internship_details": [
                "internship", "program", "duration", "months", "period",
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
                "ok", "looking for", "interested", "send whatsapp", "got it",
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
                "ഇന്റെണ്ഷിപ്", "പരിശീലനം", "പ്ലെയ്സ്മെന്റ്", 
                "മാസം", "സമയക്രമം", "ടൈമിംഗ്", "1 മുതൽ 3 വരെ", 
                "അവസാന വർഷം", "ലൈവ്", "ഫ്രെയിംവർക്ക്", "സ്ഥിരമായി", "ഡാറ്റാ സയൻസിലെ", "ഇന്റേൺഷിപ്പ്"
            ],
            "Location_query": [
                "ഓൺലൈൻ", "ഓഫ്ലൈൻ", "സ്ഥലം", "വിലാസം", "കഴിഞ്ഞ്", 
                "എവിടെ", "കൊഴിക്കോട്", "പാലാരിവട്ടം", "മാറ്റം", 
                "റിലൊക്കേറ്റ്", "വരുന്നു", "എവിടെ നിന്നാണ്", "ഹൈബ്രിഡ്", "വിലാസം"
            ],
            "Certificate_query": [
                "സർട്ടിഫിക്കറ്റ്", "ഡോക്യുമെന്റ്", "അനുഭവ സർട്ടിഫിക്കറ്റ്", 
                "പരിശീലന സർട്ടിഫിക്കറ്റ്", "അവാർഡ്", "രജിസ്ട്രേഷൻ",
                "പ്രമാണം", "സാക്ഷ്യപത്രം", "കമ്പ്ലീഷൻ"
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
                "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", 
                "അക്ക്നലഡ്ജ്", "ക്ലിയർ", 
                "തയാറാണ്", "അറിയിപ്പ് ലഭിച്ചു", "വാട്ട്സ്ആപ്പിലെ"
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

def generate_analysis_pdf(en_analysis, ml_analysis, comparison, filename_prefix):
    """Generate a PDF report with analysis metrics and visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"analysis_results/{filename_prefix}_report_{timestamp}.pdf"
    os.makedirs("analysis_results", exist_ok=True)
    
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
        
        plt.text(0.5, 0.8, "Conversation Analysis Report", ha='center', va='center', size=20)
        plt.text(0.5, 0.7, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ha='center', va='center', size=12)
        plt.text(0.5, 0.6, f"Filename: {filename_prefix}", ha='center', va='center', size=12)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.text(0.1, 0.9, "Key Metrics", size=16)
        plt.text(0.1, 0.8, f"English Avg Sentiment: {en_avg_score:.2f}", size=12)
        plt.text(0.1, 0.7, f"Malayalam Avg Sentiment: {ml_avg_score:.2f}", size=12)
        plt.text(0.1, 0.6, f"Combined Avg Sentiment: {combined_avg:.2f}", size=12)
        plt.text(0.1, 0.5, f"Calculated Lead Score: {lead_score}/100", size=12)
        
        interpretation = ""
        if lead_score >= 70:
            interpretation = "High interest lead"
        elif lead_score >= 40:
            interpretation = "Moderate interest lead"
        else:
            interpretation = "Low interest lead"
        plt.text(0.1, 0.4, f"Interpretation: {interpretation}", size=12)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        if en_analysis and ml_analysis:
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            pd.Series(en_sentiments).value_counts().plot(kind='bar', color='skyblue')
            plt.title('English Sentiment Distribution')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            pd.Series(ml_sentiments).value_counts().plot(kind='bar', color='lightgreen')
            plt.title('Malayalam Sentiment Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            plt.plot(sentence_numbers, en_scores, marker='o', label='English', color='blue')
            plt.plot(sentence_numbers, ml_scores, marker='s', label='Malayalam', color='green')
            plt.xlabel('Sentence Number')
            plt.ylabel('Sentiment Score')
            plt.title('Sentiment Trend Over Conversation')
            plt.legend()
            plt.grid(True)
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            en_intents = [item["intent"] for item in en_analysis]
            pd.Series(en_intents).value_counts().plot(kind='bar', color='orange')
            plt.title('Intent Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sentiment_diffs = [abs(en - ml) for en, ml in zip(en_scores, ml_scores)]
            plt.hist(sentiment_diffs, bins=10, color='purple', alpha=0.7)
            plt.xlabel('Sentiment Score Difference')
            plt.ylabel('Frequency')
            plt.title('English-Malayalam Sentiment Differences')
            pdf.savefig()
            plt.close()
    
    print(f"✅ PDF report generated: {pdf_filename}")
    return pdf_filename

def create_zip_archive(audio_path, raw_transcription, ml_translation, pdf_report, 
                      en_analysis, ml_analysis, user_filename):
    """Create a zip archive with all analysis files"""
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
        
        if os.path.exists(pdf_report):
            zipf.write(pdf_report, arcname=f"{base_filename}_report.pdf")
        else:
            print(f"Warning: PDF report not found at {pdf_report}")
        
        en_csv = save_analysis_to_csv(en_analysis, "english")
        ml_csv = save_analysis_to_csv(ml_analysis, "malayalam")
        comparison = compare_analyses(en_analysis, ml_analysis)
        comparison_csv = save_analysis_to_csv(comparison, "comparison")
        
        if en_csv and os.path.exists(en_csv):
            zipf.write(en_csv, arcname=f"{base_filename}_english_analysis.csv")
        if ml_csv and os.path.exists(ml_csv):
            zipf.write(ml_csv, arcname=f"{base_filename}_malayalam_analysis.csv")
        if comparison_csv and os.path.exists(comparison_csv):
            zipf.write(comparison_csv, arcname=f"{base_filename}_comparison.csv")
    
    print(f"✅ Created zip archive: {zip_filename}")
    return zip_filename, base_filename, lead_score, intent_score

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
        
        comparison = compare_analyses(en_analysis, ml_analysis)
        pdf_report = generate_analysis_pdf(en_analysis, ml_analysis, comparison, user_filename)

        zip_filename, base_filename, lead_score, intent_score = create_zip_archive(
            audio_path, raw_transcription, ml_translation, pdf_report,
            en_analysis, ml_analysis, user_filename
        )

        drive_folder_id = input("Enter Google Drive folder ID (leave empty for root): ").strip()
        if drive_folder_id:
            upload_to_gdrive(zip_filename, drive_folder_id)
        else:
            upload_to_gdrive(zip_filename)

        print("\n=== Analysis Complete ===")
        print_analysis_summary(en_analysis, "English")
        print_analysis_summary(ml_analysis, "Malayalam")

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    finally:
        transcriber.cleanup()

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