import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictRequest, PredictResponse
from api.model   import predict_url

app      = FastAPI(title="Phishing Detection API")
executor = ThreadPoolExecutor(max_workers=4)


app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            predict_url,       # ← calls extract_features internally
            request.url
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug")
async def debug(request: PredictRequest):
    """Returns raw extracted features — useful for diagnosing wrong predictions."""
    from api.features import extract_static_features, extract_dynamic_features

    static  = extract_static_features(request.url)
    dynamic = extract_dynamic_features(request.url)

    return {
        "url":              request.url,
        "static_features":  static,
        "dynamic_features": dynamic,
        "failed_dynamic":   {k: v for k, v in dynamic.items() if v == -1},
        "total_features":   len(static) + len(dynamic),
    }