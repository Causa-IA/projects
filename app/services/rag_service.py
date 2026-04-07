import json
import numpy as np
import os
import re
import psycopg2

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from deep_translator import GoogleTranslator


load_dotenv()

embedding_model = None
conn = None

print("Connecting to Groq...")
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def load_rag():
    global embedding_model, conn
    try:
        if embedding_model is None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        if conn is None or conn.closed:
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))

    except Exception as e:
        print("Error loading RAG:", e)


print("RAG service initialized (lazy loading enabled)")


def medical_rag_query(query: str):

    load_rag()

    if embedding_model is None or conn is None:
        return {
            "disease": None,
            "explanation": "El sistema no está disponible en este momento.",
            "recommendation": "Intente más tarde."
        }

    # Traducir síntomas a inglés
    query_en = GoogleTranslator(source="auto", target="en").translate(query)

    # Embedding normalizado
    query_embedding = embedding_model.encode([query_en], normalize_embeddings=True)
    query_embedding = query_embedding[0].tolist()

    # Búsqueda vectorial en pgvector
    cursor = conn.cursor()
    cursor.execute("""
        SELECT text, disease, section,
               1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        ORDER BY embedding <=> %s::vector
        LIMIT 10
    """, (query_embedding, query_embedding))

    results = cursor.fetchall()
    cursor.close()

    print("Query:", query_en)
    for row in results[:3]:
        print(f"  Score: {row[3]:.4f} | Disease: {row[1]} | Section: {row[2]}")

    # Enfermedades posibles (top 3 únicas)
    possible_diseases = []
    for row in results:
        disease = row[1]
        if disease not in possible_diseases:
            possible_diseases.append(disease)
        if len(possible_diseases) >= 3:
            break

    print("Possible diseases:", possible_diseases)

    # Contexto con las 3 secciones
    SECTIONS_TO_INCLUDE = {"symptoms", "description", "treatment"}
    context = ""

    cursor = conn.cursor()
    cursor.execute("""
        SELECT text, disease, section
        FROM medical_documents
        WHERE disease = ANY(%s) AND section = ANY(%s)
    """, (possible_diseases, list(SECTIONS_TO_INCLUDE)))

    context_rows = cursor.fetchall()
    cursor.close()

    for row in context_rows:
        context += row[0] + "\n"

    print("Context:", context)

    # Validar que el contexto no esté vacío
    if not context:
        return {
            "disease": None,
            "explanation": "No se encontró información suficiente.",
            "recommendation": "Consulte a un profesional de salud."
        }

    # Confianza basada en el mejor score
    best_score = results[0][3]

    if best_score < 0.35:
        return {
            "disease": None,
            "explanation": "Los síntomas no coinciden claramente con enfermedades del sistema.",
            "recommendation": "Consulte a un profesional de salud para una evaluación adecuada."
        }

    if best_score >= 0.55:
        confidence = "alta"
    elif best_score >= 0.45:
        confidence = "media"
    else:
        confidence = "baja"

    possible_diseases_json = json.dumps(possible_diseases)

    prompt = f"""
You are a strict medical extraction system.

Your task is to extract information ONLY from the provided medical context.

Medical context:
{context}

Possible diseases (from similarity search):
{possible_diseases}

User symptoms:
{query}

STRICT RULES:
- You MUST ONLY use information that appears explicitly in the medical context.
- DO NOT add any external medical knowledge.
- DO NOT infer or guess.
- Use the "description" sections to build the explanation of the disease.
- Use the "treatment" sections to build the recommendation.
- If treatment is not present in the context, write: "No disponible en la información."

OUTPUT RULES:
- Answer ONLY in Spanish.
- Return ONLY a valid JSON.
- Do NOT write text before or after the JSON.
- Do NOT explain anything outside the JSON.

FORMAT:

{{
"disease": "one disease from the list",
"possible_diseases": {possible_diseases_json},
"explanation": "description of the disease based ONLY on the description section of the context",
"recommendation": "treatment based ONLY on the treatment section of the context",
"confidence": "{confidence}"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.choices[0].message.content
    print(result)

    match = re.search(r"\{.*?\}", result, re.DOTALL)

    if match:
        json_text = match.group()
        data = json.loads(json_text)
        return data

    return {
        "disease": None,
        "explanation": "No se pudo interpretar la respuesta del modelo.",
        "recommendation": "Intente nuevamente.",
    }