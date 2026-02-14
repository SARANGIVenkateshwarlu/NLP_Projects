from fastapi import FastAPI, Request
import uvicorn
import os
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from pydantic import BaseModel
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class TextRequest(BaseModel):
    text: str


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(request: TextRequest):
    try:
        obj = PredictionPipeline()
        result = obj.predict(request.text) 
        #result = result.replace("<n>", "\n")
        return {"summary": result}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)