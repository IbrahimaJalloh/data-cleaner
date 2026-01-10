"""FastAPI Data Cleaner Diamant 2026."""
import os
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from cleaner import clean_data

app = FastAPI(title="🧹 Data Cleaner IA 95%", docs_url="/docs")  # ← AJOUTÉ

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← "*" prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)  # ← Root page
async def root():
    return """
    <h1 style="font-size:3em">🧹 Data Cleaner IA LIVE</h1>
    <p><a href="/docs" style="font-size:1.5em">→ Test Swagger /docs</a></p>
    <p>Upload CSV → JSON 95% clean (doublons/emails/tél)</p>
    <p><b>GRATUIT 50lignes | 500€ 5000lignes</b></p>
    """

@app.post("/clean")
async def clean_csv(file: UploadFile = File(...), x_ratelimit: str = Header(None)):
    """Nettoie CSV upload (size/type/tmp-delete)."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "CSV only")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Trop volumineux")
    
    tmp_path = f"/tmp/{uuid.uuid4()}.csv"
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        result: Dict = clean_data(tmp_path)
        print(f"Rate-limit header: {x_ratelimit}")
        return JSONResponse(result, headers={"X-RateLimit": "50"})
    except Exception as exc:
        raise HTTPException(500, "Erreur") from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 10000)), reload=False)  # ← PORT Render
