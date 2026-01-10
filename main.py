"""FastAPI Data Cleaner Diamant 2026."""
import os
import uuid
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from cleaner import clean_data

app = FastAPI(title="Data Cleaner Sécurisé")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

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
        print(f"Rate-limit header: {x_ratelimit}")  # Used log
        return JSONResponse(result, headers={"X-RateLimit": "50"})
    except Exception as exc:
        raise HTTPException(500, "Erreur") from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
def health() -> Dict[str, str]:
    """API saine."""
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
