from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model_gemini import GeminiModel
import os
from dotenv import load_dotenv

# Load environment variables from .env locally
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def health_check():
    return {"status": "Ms. Potts backend is live!"}

# Initialize GeminiModel (OK for local use where env is already loaded)
model = GeminiModel()

@app.post("/query")
async def query_endpoint(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        user_context = data.get("context", {}).get("user_profile", {})

        print(f"✅ Received Query: {query}")
        print(f"✅ Received Context: {user_context}")

        response = model.get_response(query, user_context)

        print(f"✅ GeminiModel Response: {response}")

        return JSONResponse({
            "final_answer": response.get("final_answer", ""),
            "detected_intent": response.get("detected_intent", ""),
            "reasoning": response.get("reasoning", ""),
        })
    except Exception as e:
        print(f"❌ Exception inside /query: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Uncomment if you want to run it locally via `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
