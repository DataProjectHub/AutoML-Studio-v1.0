from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="AutoML Studio")
templates = Jinja2Templates(directory=str(ROOT / "streamlit_app/templates"))
@lru_cache(maxsize=1)
def get_model():
    path = ROOT / "models/best_model.pkl"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model unavailable. Run python main.py first.")
    return joblib.load(path)
@app.get("/health")
def health(): return {"status": "ok", "model_ready": (ROOT / "models/best_model.pkl").exists()}
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request): return templates.TemplateResponse(request, "form.html")
@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, Age: float = Form(...), BMI: float = Form(...), Exercise_Frequency: int = Form(...), Diet_Quality: float = Form(...), Sleep_Hours: float = Form(...), Smoking_Status: int = Form(...), Alcohol_Consumption: float = Form(...)):
    df = pd.DataFrame([{"Age": Age, "BMI": BMI, "Exercise_Frequency": Exercise_Frequency, "Diet_Quality": Diet_Quality, "Sleep_Hours": Sleep_Hours, "Smoking_Status": Smoking_Status, "Alcohol_Consumption": Alcohol_Consumption}])
    value = float(get_model().predict(df)[0])
    return templates.TemplateResponse(request, "form.html", {"result": round(value, 2)})
