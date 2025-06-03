import os
import zipfile
import tempfile
from datetime import datetime
import torch
from pydub import AudioSegment
from deep_translator import GoogleTranslator
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pandas as pd
import nltk
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
from io import BytesIO
from transformers import pipeline

nltk.download('punkt')

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Replace with your service account file

class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="large-v2"):
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

        temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        wav_path = os.path.join(temp_dir, f"temp_{timestamp}.wav")

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")

        self.temp_files.append(wav_path)
        print(f"Converted to temporary WAV: {wav_path}")
        return wav_path

    def transcribe_audio(self, audio_path):
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
            except Exception as e:
                print(f"Error deleting temp file {file_path}: {str(e)}")
        self.temp_files = []

# Sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

def authenticate_gdrive():
    """Authenticate with Google Drive using service account"""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def create_zip_archive(audio_path, raw_transcription, ml_translation, pdf_report, 
                      en_analysis, ml_analysis, user_filename):
    """Create a zip archive with all analysis files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate scores
    en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
    ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
    combined_avg = (en_avg_score + ml_avg_score) / 2
    lead_score = int(combined_avg * 100)
    
    # Calculate intent score (percentage of positive intents)
    positive_intents = sum(1 for item in en_analysis if item["intent"] in ["Strong_interest", "Moderate_interest"])
    intent_score = int((positive_intents / len(en_analysis)))* 100 if en_analysis else 0
    
    # Create base filename
    base_filename = f"{user_filename}_{timestamp}_L{lead_score}_I{intent_score}"
    zip_filename = f"analysis_results/{base_filename}.zip"
    
    # Rename audio file with scores
    audio_ext = os.path.splitext(audio_path)[1]
    new_audio_name = f"{base_filename}{audio_ext}"
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add original audio file (with new name)
        zipf.write(audio_path, arcname=new_audio_name)
        
        # Add text files
        zipf.writestr(f"{base_filename}_transcription.txt", raw_transcription)
        zipf.writestr(f"{base_filename}_translation.txt", ml_translation)
        
        # Add PDF report
        zipf.write(pdf_report, arcname=f"{base_filename}_report.pdf")
        
        # Add CSV analysis files
        en_csv = save_analysis_to_csv(en_analysis, "english")
        ml_csv = save_analysis_to_csv(ml_analysis, "malayalam")
        comparison = compare_analyses(en_analysis, ml_analysis)
        comparison_csv = save_analysis_to_csv(comparison, "comparison")
        
        if en_csv:
            zipf.write(en_csv, arcname=f"{base_filename}_english_analysis.csv")
        if ml_csv:
            zipf.write(ml_csv, arcname=f"{base_filename}_malayalam_analysis.csv")
        if comparison_csv:
            zipf.write(comparison_csv, arcname=f"{base_filename}_comparison.csv")
    
    print(f"✅ Created zip archive: {zip_filename}")
    return zip_filename, base_filename

def upload_to_gdrive(file_path, folder_id=None):
    """Upload file to Google Drive"""
    try:
        service = authenticate_gdrive()
        
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id] if folder_id else []
        }
        
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        
        print(f"✅ Uploaded to Google Drive: {file.get('name')}")
        print(f"🔗 View file at: {file.get('webViewLink')}")
        return file
    except Exception as e:
        print(f"❌ Google Drive upload failed: {str(e)}")
        return None

def search_gdrive_files(query=None, min_lead=None, max_lead=None, 
                       min_intent=None, max_intent=None):
    """Search files in Google Drive with filters"""
    try:
        service = authenticate_gdrive()
        query_parts = []
        
        if query:
            query_parts.append(f"name contains '{query}'")
        
        # Add score filters if provided
        if min_lead is not None:
            query_parts.append(f"name contains '_L{min_lead}' or name contains '_L{min_lead+1}'")
        if max_lead is not None:
            query_parts.append(f"name contains '_L{max_lead}' or name contains '_L{max_lead-1}'")
        if min_intent is not None:
            query_parts.append(f"name contains '_I{min_intent}' or name contains '_I{min_intent+1}'")
        if max_intent is not None:
            query_parts.append(f"name contains '_I{max_intent}' or name contains '_I{max_intent-1}'")
        
        full_query = " and ".join(query_parts) if query_parts else ""
        
        results = service.files().list(
            q=full_query,
            pageSize=10,
            fields="nextPageToken, files(id, name, webViewLink, createdTime)"
        ).execute()
        
        items = results.get('files', [])
        
        if not items:
            print("No files found matching your criteria.")
            return []
        
        print(f"Found {len(items)} files:")
        for item in items:
            print(f"\n📄 {item['name']}")
            print(f"🕒 Created: {item['createdTime']}")
            print(f"🔗 View: {item['webViewLink']}")
        
        return items
    except Exception as e:
        print(f"❌ Google Drive search failed: {str(e)}")
        return []

def split_into_sentences(text):
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception as e:
        print(f"Sentence splitting failed: {e}")
        return [text] if text.strip() else []

def analyze_sentiment_batch(texts):
    results = sentiment_pipeline(texts)
    outputs = []
    for result in results:
        label = result['label']
        
        # Set default scores based on sentiment labels
        if "1 star" in label or "2 stars" in label:
            sentiment = {"label": "negative", "score": 0.2}  # Default for negative
        elif "3 stars" in label:
            sentiment = {"label": "neutral", "score": 0.5}   # Default for neutral
        elif "4 stars" in label:
            sentiment = {"label": "positive", "score": 0.7}  # Default for positive
        elif "5 stars" in label:
            sentiment = {"label": "very positive", "score": 0.9}  # Default for very positive
        else:
            sentiment = {"label": "neutral", "score": 0.5}   # Default fallback
            
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
            "Qualification_query": [
                "qualification", "education", "degree", "studying", "course",
                "background", "academics", "university", "college", "bsc",
                "graduate", "year of study", "curriculum", "syllabus"
            ],
            "Internship_details": [
                "internship", "program", "duration", "months", "period",
                "schedule", "timing", "timeframe", "1 to 3", "three months",
                "structure", "plan", "framework"
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
                "ഇഷ്ടമാണ്", "അറിയിച്ചോളൂ", "ബ്രോഷർ വേണം", "വിശദാംശങ്ങൾ വേണം",
                "ശെയർ ചെയ്യുക", "ഞാൻ വരാം", "ഉത്സാഹം", "താത്പ�്യം",
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
            "Qualification_query": [
                "വിദ്യാഭ്യാസം", "ഡിഗ്രി", "ബിസി", "പഠിക്കുന്നു", 
                "പഠനം", "അധ്യയനം", "ക്ലാസ്", "വർഷം", 
                "കോഴ്‌സ്", "സിലബസ്", "വിദ്യാർഥി", "ഗണിതം", "സയൻസ്"
            ],
            "Internship_details": [
                "ഇന്റെണ്ഷിപ്", "പരിശീലനം", "പ്രോഗ്രാം", 
                "മാസം", "സമയക്രമം", "ടൈമിംഗ്", "1 മുതൽ 3 വരെ", 
                "അവസാന വർഷം", "ലൈവ്", "ഫ്രെയിംവർക്ക്", "സ്ഥിരമായി"
            ],
            "Location_query": [
                "ഓൺലൈൻ", "ഓഫ്ലൈൻ", "സ്ഥലം", "വിലാസം", "കഴിഞ്ഞ്", 
                "എവിടെ", "കൊഴിക്കോട്", "പാലാരിവട്ടം", "മാറ്റം", 
                "റിലൊക്കേറ്റ്", "വരുന്നു", "എവിടെ നിന്നാണ്", "ഹൈബ്രിഡ്"
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
                 "ശരി", "താല്പര്യമുണ്ട്", "ഇഷ്ടമുണ്ട്", "വാട്സാപ്പിൽ അയക്കൂ", 
                 "വാട്സാപ്പ്", "വാട്ട്സാപ്പ്", "കിട്ടി", "അറിയിച്ചു", 
                 "നോട്ടു ചെയ്തു", "സമ്മതം", "അംഗീകരിച്ചു", 
                 "അക്ക്നലഡ്ജ്", "ക്ലിയർ", 
                 "തയാറാണ്", "അറിയിപ്പ് �ലഭിച്ചു"
            ]
        }
    }

    if any(keyword in text_lower for keyword in intent_keywords[language]["Confirmation"]):
        return {"intent": "Confirmation", "sentiment": "very positive", "sentiment_score": 0.9}
    
    # Then check for strong interest
    if any(keyword in text_lower for keyword in intent_keywords[language]["Strong_interest"]):
        return {"intent": "Strong_interest", "sentiment": "positive", "sentiment_score": 0.7}
    
    # Then check for no interest
    if any(keyword in text_lower for keyword in intent_keywords[language]["No_interest"]):
        return {"intent": "No_interest", "sentiment": "negative", "sentiment_score": 0.2}
    
    # Then check for moderate interest
    if any(keyword in text_lower for keyword in intent_keywords[language]["Moderate_interest"]):
        return {"intent": "Moderate_interest", "sentiment": "neutral", "sentiment_score": 0.5}
    
    # Check other intents
    for intent, keywords in intent_keywords[language].items():
        if intent not in ["Confirmation", "Strong_interest", "No_interest", "Moderate_interest"]:
            if any(keyword in text_lower for keyword in keywords):
                return {"intent": intent, "sentiment": "neutral", "sentiment_score": 0.5}
    
    # Default response
    return {"intent": "Neutral_response", "sentiment": "neutral", "sentiment_score": 0.5}

def analyze_text(text, language="en"):
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Get sentiment analysis results with default scores
    sentiment_results = analyze_sentiment_batch(sentences)
    
    # Get intent analysis which provides both intent and sentiment
    intent_results = [detect_intent(sentence, language) for sentence in sentences]

    analysis = []
    for i, sentence in enumerate(sentences):
        # Use the intent-based sentiment label and score
        analysis.append({
            "sentence_id": f"{language}_{i+1}",
            "text": sentence,
            "language": language,
            "intent": intent_results[i]["intent"],
            "sentiment": intent_results[i]["sentiment"],
            "sentiment_score": intent_results[i]["sentiment_score"],  # Use intent-based score
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
    
    # Calculate metrics
    en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
    ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
    combined_avg = (en_avg_score + ml_avg_score) / 2
    lead_score = int(combined_avg * 100)
    
    # Prepare data for visualizations
    en_sentiments = [item["sentiment"] for item in en_analysis]
    ml_sentiments = [item["sentiment"] for item in ml_analysis]
    en_scores = [item["sentiment_score"] for item in en_analysis]
    ml_scores = [item["sentiment_score"] for item in ml_analysis]
    sentence_numbers = list(range(1, len(en_analysis)+1))
    
    with PdfPages(pdf_filename) as pdf:
        plt.figure(figsize=(10, 6))
        
        # Title page
        plt.text(0.5, 0.8, "Conversation Analysis Report", ha='center', va='center', size=20)
        plt.text(0.5, 0.7, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ha='center', va='center', size=12)
        plt.text(0.5, 0.6, f"Filename: {filename_prefix}", ha='center', va='center', size=12)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Key Metrics
        plt.figure(figsize=(10, 6))
        plt.text(0.1, 0.9, "Key Metrics", size=16)
        plt.text(0.1, 0.8, f"English Avg Sentiment: {en_avg_score:.2f}", size=12)
        plt.text(0.1, 0.7, f"Malayalam Avg Sentiment: {ml_avg_score:.2f}", size=12)
        plt.text(0.1, 0.6, f"Combined Avg Sentiment: {combined_avg:.2f}", size=12)
        plt.text(0.1, 0.5, f"Calculated Lead Score: {lead_score}/100", size=12)
        
        # Interpretation
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
        
        # Sentiment distribution comparison
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
        
        # Sentiment trend over time
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
        
        # Intent distribution
        plt.figure(figsize=(10, 6))
        en_intents = [item["intent"] for item in en_analysis]
        pd.Series(en_intents).value_counts().plot(kind='bar', color='orange')
        plt.title('Intent Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Sentiment difference histogram
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
    """Main workflow that integrates with the original analysis code"""
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

        # Generate PDF report
        user_filename = input("\nEnter a base name for your files (without extension): ").strip()
        if not user_filename:
            user_filename = "conversation_analysis"
        
        pdf_report = generate_analysis_pdf(en_analysis, ml_analysis, 
                                         compare_analyses(en_analysis, ml_analysis), 
                                         user_filename)

        # Create zip archive
        zip_filename, base_filename = create_zip_archive(
            audio_path, raw_transcription, ml_translation, pdf_report,
            en_analysis, ml_analysis, user_filename
        )

        # Upload to Google Drive
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

def search_menu():
    """Interactive search menu"""
    while True:
        print("\n=== Google Drive Search ===")
        print("1. Search by filename")
        print("2. Filter by lead score range")
        print("3. Filter by intent score range")
        print("4. Combined search")
        print("5. Back to main menu")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            query = input("Enter search term: ").strip()
            search_gdrive_files(query=query)
        elif choice == "2":
            min_lead = int(input("Minimum lead score (0-100): "))
            max_lead = int(input("Maximum lead score (0-100): "))
            search_gdrive_files(min_lead=min_lead, max_lead=max_lead)
        elif choice == "3":
            min_intent = int(input("Minimum intent score (0-100): "))
            max_intent = int(input("Maximum intent score (0-100): "))
            search_gdrive_files(min_intent=min_intent, max_intent=max_intent)
        elif choice == "4":
            query = input("Enter search term (leave empty to skip): ").strip() or None
            min_lead = int(input("Minimum lead score (0-100, leave empty to skip): ") or 0)
            max_lead = int(input("Maximum lead score (0-100, leave empty to skip): ") or 100)
            min_intent = int(input("Minimum intent score (0-100, leave empty to skip): ") or 0)
            max_intent = int(input("Maximum intent score (0-100, leave empty to skip): ") or 100)
            search_gdrive_files(
                query=query,
                min_lead=min_lead,
                max_lead=max_lead,
                min_intent=min_intent,
                max_intent=max_intent
            )
        elif choice == "5":
            break
        else:
            print("Invalid choice, please try again.")

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
