"""FastAPI Data Cleaner Diamant 2026."""
import os
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from cleaner import clean_data

app = FastAPI(title="🧹 Data Cleaner IA", docs_url="/docs", redoc_url="/redoc")  # Swagger auto !

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)  # ← FIX 404
async def root():
    return """
<!DOCTYPE html>
<html>
<head><title>🧹 Data Cleaner IA LIVE</title><meta charset="utf-8"></head>
<body style="font-family:sans-serif;max-width:800px;margin:50px auto;">
<h1 style="color:#3b82f6;font-size:3em">🧹 Data Cleaner IA 95%</h1>
<p><b>LIVE Render.com</b> | Nettoie CRM/doublons/emails/tél 24h</p>
<a href="/docs" style="background:#3b82f6;color:white;padding:15px 30px;font-size:1.5em;text-decoration:none;border-radius:10px;display:inline-block;">🚀 TEST GRATUIT /docs</a>
<hr>
<p><b>500€ / 5000 lignes</b> | <b>GRATUIT essai 50 lignes</b></p>
<p>Contact: LinkedIn Ibrahima Jalloh</p>
</body>
</html>
    """

@app.post("/clean")
async def clean_csv(file: UploadFile = File(...), x_ratelimit: str = Header(None)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "CSV only")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Max 10MB")
    
    tmp_path = f"/tmp/{uuid.uuid4()}.csv"
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        result: Dict = clean_data(tmp_path)
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
def health():
    return {"status": "OK", "live": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
