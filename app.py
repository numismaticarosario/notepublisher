import os
import base64
import json
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# --- Cargar variables de entorno (.env) ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- OpenAI SDK ---
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Home: sirve tu index.html ---
@app.route("/")
def index():
    return render_template("index.html")


def _to_data_url(file_bytes: bytes, filename: str) -> str:
    """
    Convierte los bytes de la imagen a un data URL base64 para enviar a OpenAI Vision.
    """
    ext = "png"
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ["png", "jpg", "jpeg", "webp"]:
            ext = "png"
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def _extract_json(text: str) -> dict:
    """
    Intenta extraer un JSON válido desde la respuesta del modelo.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {
            "pais": "",
            "valor": "",
            "referencia": "",
            "metal": "",
            "diametro": ""
        }

    raw = match.group(0)
    try:
        return json.loads(raw)
    except Exception:
        cleaned = raw.replace("\n", " ").replace("\r", " ")
        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "pais": "",
                "valor": "",
                "referencia": "",
                "metal": "",
                "diametro": ""
            }


@app.route("/extract", methods=["POST"])
def extract():
    """
    Recibe una imagen y llama a OpenAI Vision para extraer los campos principales.
    """
    if "image" not in request.files:
        return jsonify({"error": "No se recibió la imagen"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    data_url = _to_data_url(image_bytes, file.filename or "captura.png")

    # --- Prompt de extracción ---
    system_msg = (
        "Sos un asistente que extrae datos de listings numismáticos desde una imagen. "
        "Debes devolver SOLO un JSON válido con las claves solicitadas, sin texto adicional."
    )

    user_instructions = """A partir de la imagen (un printscreen simple de una ficha), hacé OCR y extraé estos campos:

1) 'pais': Tomar el nombre del país desde la leyenda 'ISSUER'.

2) 'valor': Tomar únicamente el número y la unidad monetaria desde 'VALUE'. 
   Ignorar cualquier otra palabra, símbolo o texto que aparezca en esa línea.
   Devolver solo en formato: "numero unidad", por ejemplo "1 Dollar" o "50 Centavos".

3) 'referencia': Revisar la leyenda 'REFERENCES'.
   - Si contiene 'P #<n>', devolver exactamente 'P #<n>'.
   - Si no hay P, buscar 'Bot. #<n>' y devolver 'Bot. #<n>'.
   - Si no hay P ni Bot., usar 'N #<n>' desde 'NUMBER' (si existe) y devolver 'N #<n>'.
   IMPORTANTE: Siempre devolver en formato: sigla (P/Bot./N) + espacio + '#' + número, sin nada más.
   Si no hay ningún valor, dejar cadena vacía.

4) 'metal': Tomar desde 'COMPOSITION' y traducirlo al español.

5) 'diametro': Tomar desde 'SIZE'. Devolver como '<numero> x <numero> mm' (agregar 'mm' si faltara).

Si falta algún dato, devolver cadena vacía para ese campo.

Formato de salida OBLIGATORIO (JSON plano, sin comentarios, sin markdown):
{
  "pais": "",
  "valor": "",
  "referencia": "",
  "metal": "",
  "diametro": ""
}"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_instructions},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
        )

        content = completion.choices[0].message.content or "{}"
        data = _extract_json(content)

        result = {
            "pais": data.get("pais", ""),
            "valor": data.get("valor", ""),
            "referencia": data.get("referencia", ""),
            "metal": data.get("metal", ""),
            "diametro": data.get("diametro", "")
        }

        return jsonify(result)

    except Exception as e:
        print("ERROR OpenAI:", e)
        return jsonify({
            "pais": "",
            "valor": "",
            "referencia": "",
            "metal": "",
            "diametro": "",
            "error": "Hubo un problema procesando la imagen"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)