import pandas as pd
import uvicorn
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles


def add_url_features(X):
    df_temp = pd.DataFrame({"url": X})
    df_temp["https"] = df_temp["url"].str.contains("https").astype(int)
    df_temp["length"] = df_temp["url"].str.len()
    df_temp["num_digits"] = df_temp["url"].str.count(r'\d')
    df_temp["dots"] = df_temp["url"].str.count(r'\.')
    df_temp["at_symbol"] = df_temp["url"].str.contains("@").astype(int)
    df_temp["hyphen_count"] = df_temp["url"].str.count("-")
    return df_temp[["https","length","num_digits","dots","at_symbol","hyphen_count"]]


# Request schema
class UrlRequest(BaseModel):
    url: str

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load("static/Hurl_model.pkl")

@app.post("/predict")
def predict(data: UrlRequest):
    df = pd.DataFrame({"url": [data.url]})
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}

if __name__ == "__main__":
    uvicorn.run(app)
    #port = int(os.environ.get("PORT", 10000))
    #uvicorn.run("app:app", host="0.0.0.0", port=port)