import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.utils.ml_utils.model.estimator import NetworkModel


client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        # Map predictions to user-friendly labels
        df['prediction_label'] = df['predicted_column'].map({1: 'Valid Network', 0: 'Not a Safe Network'})
        # Prepare data for table display
        table_data = df.to_dict(orient='records')
        # Determine overall result: if all are valid, safe; else, not safe
        if (df['prediction_label'] == 'Valid Network').all():
            result = 'Valid Network'
        else:
            result = 'Not a Safe Network'
        return templates.TemplateResponse("result.html", {"request": request, "result": result, "table_data": table_data})
    except Exception as e:
        raise NetworkSecurityException(e,sys)

@app.post("/predict_url")
async def predict_url_route(request: Request):
    try:
        form = await request.form()
        network_url = form.get("network_url")
        # Define all required columns in the correct order
        columns = [
            "Having Ip Address", "Url Length", "Shortining Service", "Having At Symbol", "Double Slash Redirecting",
            "Prefix Suffix", "Having Sub Domain", "Sslfinal State", "Domain Registeration Length", "Favicon", "Port",
            "Https Token", "Request Url", "Url Of Anchor", "Links In Tags", "Sfh", "Submitting To Email", "Abnormal Url",
            "Redirect", "On Mouseover", "Rightclick", "Popupwidnow", "Iframe", "Age Of Domain", "Dnsrecord", "Web Traffic",
            "Page Rank", "Google Index", "Links Pointing To Page", "Statistical Report"
        ]
        # Feature extraction logic, fill with 1 or -1 as appropriate
        feature_dict = {
            "Having Ip Address": 1 if any(char.isdigit() for char in network_url) else -1,
            "Url Length": len(network_url),
            "Shortining Service": 1 if ("bit.ly" in network_url or "tinyurl" in network_url) else -1,
            "Having At Symbol": 1 if "@" in network_url else -1,
            "Double Slash Redirecting": 1 if network_url.count("//") > 1 else -1,
            "Prefix Suffix": 1 if "-" in network_url else -1,
            "Having Sub Domain": 1 if network_url.count(".") > 2 else -1,
            "Sslfinal State": 1 if network_url.startswith("https") else -1,
            "Domain Registeration Length": 1,
            "Favicon": 1,
            "Port": 1 if ":" in network_url else -1,
            "Https Token": 1 if "https" in network_url else -1,
            "Request Url": 1,
            "Url Of Anchor": 1,
            "Links In Tags": 1,
            "Sfh": 1,
            "Submitting To Email": 1 if "mailto:" in network_url else -1,
            "Abnormal Url": -1,
            "Redirect": 1 if "//" in network_url[8:] else -1,
            "On Mouseover": -1,
            "Rightclick": -1,
            "Popupwidnow": -1,
            "Iframe": -1,
            "Age Of Domain": 1,
            "Dnsrecord": 1,
            "Web Traffic": 1,
            "Page Rank": 1,
            "Google Index": 1,
            "Links Pointing To Page": 1,
            "Statistical Report": -1
        }
        # Ensure correct order and types
        row = [int(feature_dict[col]) for col in columns]
        import pandas as pd
        df = pd.DataFrame([row], columns=columns)
        preprocesor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor, model=final_model)
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred
        df['prediction_label'] = df['predicted_column'].map({1: 'Valid Network', 0: 'Not a Safe Network'})
        table_data = df.to_dict(orient='records')
        if (df['prediction_label'] == 'Valid Network').all():
            result = 'Valid Network'
        else:
            result = 'Not a Safe Network'
        return templates.TemplateResponse("result.html", {"request": request, "table_data": table_data, "result": result})
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
    
if __name__=="__main__":
    app_run(app,host="127.0.0.1",port=8000)
