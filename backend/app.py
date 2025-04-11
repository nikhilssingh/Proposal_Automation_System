# app.py
from fastapi import FastAPI
from routes import api_router  # ✅ Import the central router from `routes/__init__.py`
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


app = FastAPI(title="RFP Automation API", version="1.0")

# ✅ Include all API routes
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Welcome to the RFP Automation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
