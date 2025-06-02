import os
import tempfile
from datetime import datetime
import torch
from pydub import AudioSegment
from deep_translator import GoogleTranslator
from transformers import pipeline
import pandas as pd
import nltk
from faster_whisper import WhisperModel

nltk.download('punkt')

class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="large-v1"):
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
        if "1 star" in label:
            sentiment = {"label": "very negative", "score": 0.1}
        elif "2 stars" in label:
            sentiment = {"label": "negative", "score": 0.3}
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
            # Interest Levels
            "Strong_interest": [
                "yes", "definitely", "ready", "want to join", "interested", 
                "share details", "send brochure", "i'll join", "let's proceed",
                "where do i sign", "how to apply", "when can i start", "accept",
                "looking forward", "excited", "happy to", "glad to", "eager",
                "share it", "i will come", "i'm in"
            ],
            "Moderate_interest": [
                "maybe", "consider", "think about", "let me think", "tell me more",
                "more details", "explain", "clarify", "not sure", "possibly",
                "might", "could be", "depends", "need to check", "will decide",
                "get back", "discuss", "consult", "review", "evaluate"
            ],
            "No_interest": [
                "no", "not interested", "can't", "won't", "don't like",
                "not now", "later", "not suitable", "inconvenient", "decline",
                "pass", "refuse", "reject", "not for me", "not my field"
            ],

            # Conversation Categories
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
            # Interest Levels
            "Strong_interest": [
                "തയ്യാറാണ്", "ആവശ്യമുണ്ട്", "ചെയ്യാം", "ആഗ്രഹമുണ്ട്", 
                "ഇഷ്ടമാണ്", "അറിയിച്ചോളൂ", "ബ്രോഷർ വേണം", "വിശദാംശങ്ങൾ വേണം",
                "ശെയർ ചെയ്യുക", "ഞാൻ വരാം", "ഉത്സാഹം", "താത്പര്യം",
                "സമ്മതം", "അംഗീകരിക്കുന്നു", "ഹാപ്പിയാണ്", "ഞാൻ ചെയ്യാം",
                "നിശ്ചിതമായി", "ആവശ്യമാണ്"
            ],
            "Moderate_interest": [
                "ആലോചിക്കാം", "നോക്കാം", "താല്പര്യമുണ്ട്", "ഇന്റെറസ്റ്റഡ്",
                "പറയാം", "ക്ഷണിക്കുക", "ചിന്തിക്കാം", "കാണാം", "ഉത്തരമില്ല",
                "കൂടുതൽ വിവരങ്ങൾ", "വ്യാഖ്യാനിക്കുക", "അവലംബിക്കുക"
            ],
            "No_interest": [
                "ഇല്ല", "വേണ്ട", "സാധ്യമല്ല", "ഇഷ്ടമല്ല", "ഇങ്ങനെയല്ല",
                "നിരസിക്കുക", "അനാവശ്യമാണ്", "പിന്തിരിയുക", "ഇതല്ല", "നിഷേധം"
            ],

            # Conversation Categories
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
                 "നോട്ടു ചെയ്തു", "സമ്മതം", "ബോധിച്ചിട്ടുണ്ട്", 
                 "അംഗീകരിച്ചു", "അക്ക്നലഡ്ജ്", "ക്ലിയർ", 
                 "തയാറാണ്", "അറിയിപ്പ് ലഭിച്ചു"
            ]

        }
    }

    # Step 1: Detect interest level
    if any(keyword in text_lower for keyword in intent_keywords[language]["Strong_interest"]):
        return "Strong_interest"
    if any(keyword in text_lower for keyword in intent_keywords[language]["Moderate_interest"]):
        return "Moderate_interest"
    if any(keyword in text_lower for keyword in intent_keywords[language]["No_interest"]):
        return "No_interest"

    # Step 2: Detect conversation category
    for intent, keywords in intent_keywords[language].items():
        if intent not in ["Strong_interest", "Moderate_interest", "No_interest"]:
            if any(keyword in text_lower for keyword in keywords):
                return intent

    return "Neutral_response"



def analyze_text(text, language="en"):
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    sentiment_results = analyze_sentiment_batch(sentences)

    analysis = []
    for i, sentence in enumerate(sentences):
        sentiment = sentiment_results[i]
        intent = detect_intent(sentence, language)
        analysis.append({
            "sentence_id": f"{language}_{i+1}",
            "text": sentence,
            "language": language,
            "intent": intent,
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"],
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

        en_csv = save_analysis_to_csv(en_analysis, "english")
        ml_csv = save_analysis_to_csv(ml_analysis, "malayalam")

        comparison = compare_analyses(en_analysis, ml_analysis)
        comparison_csv = save_analysis_to_csv(comparison, "comparison")

        print_analysis_summary(en_analysis, "English")
        print_analysis_summary(ml_analysis, "Malayalam")

        print("\n=== Translation Accuracy Insights ===")
        intent_matches = sum(1 for item in comparison if item["intent_match"])
        print(f"Intent Match Rate: {intent_matches / len(comparison):.1%}")
        avg_sentiment_diff = sum(item["sentiment_diff"] for item in comparison) / len(comparison)
        print(f"Average Sentiment Difference: {avg_sentiment_diff:.2f}")

        # Calculate Lead Score from average sentiment scores
        en_avg_score = sum(item["sentiment_score"] for item in en_analysis) / len(en_analysis) if en_analysis else 0
        ml_avg_score = sum(item["sentiment_score"] for item in ml_analysis) / len(ml_analysis) if ml_analysis else 0
        combined_avg = (en_avg_score + ml_avg_score) / 2
        
        # Convert to lead score (0-100 scale)
        lead_score = int(combined_avg * 100)
        print(f"\n=== Lead Score ===")
        print(f"Calculated Lead Score: {lead_score}/100")
        if lead_score >= 70:
            print("Interpretation: High interest lead")
        elif lead_score >= 40:
            print("Interpretation: Moderate interest lead")
        else:
            print("Interpretation: Low interest lead")

    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    finally:
        transcriber.cleanup()
