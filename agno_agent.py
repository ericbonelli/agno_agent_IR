import os
import io
import tempfile
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from openai import OpenAI

# ------------------------------------------------------------------------------
# ⚙️ Configurações básicas
# ------------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ Variável de ambiente OPENAI_API_KEY não configurada.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="XP B3 Parser",
    description="API para leitura de Notas de Negociação B3 (XP Investimentos)",
    version="1.0.0"
)

# ------------------------------------------------------------------------------
# 🌍 Middleware CORS (para n8n ou dashboard)
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou especifique ['https://app.agno.com', 'https://n8n.seudominio.com']
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# 🔐 Proteção simples via API Key
# ------------------------------------------------------------------------------
@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if request.url.path.startswith("/predict"):
        key = request.headers.get("X-API-Key")
        if not API_KEY or key != API_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)

# ------------------------------------------------------------------------------
# 🧠 Função auxiliar: extrai texto do PDF
# ------------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        reader = PdfReader(tmp.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

# ------------------------------------------------------------------------------
# 🚀 Endpoint principal
# ------------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recebe um PDF de nota B3, extrai o texto e pede ao modelo GPT para retornar
    os campos principais da nota de negociação.
    """
    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)

        if not text:
            return {"error": "Nenhum texto encontrado no PDF."}

        # Prompt que orienta o modelo a extrair os campos relevantes
        prompt = f"""
        Extraia do texto abaixo os principais campos de uma nota de negociação B3 (XP):
        - Data do pregão
        - Número da nota
        - Valor dos negócios
        - Total de custos operacionais
        - IRRF (projeção)
        - Total líquido da nota
        Retorne em JSON estruturado com os nomes das chaves exatamente assim:
        {{ "data_pregao": "", "nota_numero": "", "valor_negocios": "", 
           "total_custos": "", "irrf_proj": "", "total_liquido_nota": "" }}
        Texto extraído:
        {text[:6000]}  # limita para evitar estouro de tokens
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Você é um extrator preciso de informações de PDFs da XP/B3."},
                {"role": "user", "content": prompt}
            ]
        )

        resposta = completion.choices[0].message.content
        return {"resultado": resposta}

    except Exception as e:
        return {"error": str(e)}

# ------------------------------------------------------------------------------
# 🧪 Teste rápido local
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "XP B3 Parser ativo", "docs": "/docs"}
