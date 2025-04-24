from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
from gtts import gTTS
import os
import csv
import datetime

app = Flask(__name__)

# Cache to avoid reloading models
model_cache = {}

# List of supported source languages
supported_langs = ['en', 'fr', 'de', 'es', 'hi']

def get_model(src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    if model_name not in model_cache:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model_cache[model_name] = (tokenizer, model)
    return model_cache[model_name]

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    audio_file = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        target_lang = request.form["target_lang"]
        tone = request.form["tone"]

        try:
            # Detect language and validate
            detected_lang = detect(input_text)
            if detected_lang not in supported_langs:
                raise ValueError(f"Detected language '{detected_lang}' is not supported.")

            # Tone adjustment
            if tone == "formal":
                input_text = "Please translate this formally: " + input_text
            elif tone == "casual":
                input_text = "Translate casually: " + input_text

            # Get model and translate
            tokenizer, model = get_model(detected_lang, target_lang)
            tokens = tokenizer([input_text], return_tensors="pt", padding=True)
            translation = model.generate(**tokens)
            result = tokenizer.decode(translation[0], skip_special_tokens=True)

            # Generate speech
            tts = gTTS(text=result, lang=target_lang)
            audio_file = "static/output.mp3"
            tts.save(audio_file)

            # Log translation
            with open("translation_history.csv", mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    input_text,
                    result,
                    detected_lang,
                    target_lang,
                    tone
                ])

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, audio_file=audio_file)

# Render-compatible server startup
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
