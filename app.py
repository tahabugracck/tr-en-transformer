from flask import Flask, request, render_template
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli yükleyelim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "translation_model_tr2en.zip"  # Modelin kaydedildiği dizin
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Çeviri fonksiyonu
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ana sayfa (form içeren HTML)
@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        text = request.form["text"]
        translation = translate(text)
    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)
