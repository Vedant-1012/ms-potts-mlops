# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from model_gemini import GeminiModel
# import os
# from dotenv import load_dotenv
# from utils.monitoring import ModelMonitor


# # Initialize monitoring
# monitor = ModelMonitor(metrics_dir="./metrics")
# monitor.start_monitoring(interval=5)  # Check system metrics every 5 seconds


# # Load environment variables from .env locally
# load_dotenv()

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Health check
# @app.get("/")
# def health_check():
#     return {"status": "Ms. Potts backend is live!"}

# # Initialize GeminiModel (OK for local use where env is already loaded)
# model = GeminiModel()

# @app.post("/query")
# async def query_endpoint(request: Request):
#     try:
#         data = await request.json()
#         query = data.get("query", "")
#         user_context = data.get("context", {}).get("user_profile", {})

#         print(f"✅ Received Query: {query}")
#         print(f"✅ Received Context: {user_context}")

#         response = model.get_response(query, user_context)

#         print(f"✅ GeminiModel Response: {response}")

#         return JSONResponse({
#             "final_answer": response.get("final_answer", ""),
#             "detected_intent": response.get("detected_intent", ""),
#             "reasoning": response.get("reasoning", ""),
#         })
#     except Exception as e:
#         print(f"❌ Exception inside /query: {e}")
#         return JSONResponse({"error": str(e)}, status_code=500)

# # Uncomment if you want to run it locally via `python main.py`
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model_gemini import GeminiModel
import os
from dotenv import load_dotenv
from utils.monitoring import ModelMonitor
from utils.enhanced_logging import EnhancedLogger, log_with_context
import uuid
import time

# Load environment variables from .env
load_dotenv()

# Initialize enhanced logger
enhanced_logger = EnhancedLogger(
    name="ms_potts",
    level="info",
    log_dir="./logs",
    console_output=True,
    file_output=True,
    json_output=True,
    rich_formatting=True
)
logger = enhanced_logger.get_logger()

# Initialize monitoring
monitor = ModelMonitor(metrics_dir="./metrics")
monitor.start_monitoring(interval=5)

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
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
    logger.info("Health check endpoint called")
    return {"status": "Ms. Potts backend is live!"}

# Initialize GeminiModel
model = GeminiModel()

@app.post("/query")
async def query_endpoint(request: Request):
    request_id = f"req-{int(time.time())}-{str(uuid.uuid4())[:8]}"
    try:
        start_time = time.time()
        data = await request.json()
        query = data.get("query", "")
        user_context = data.get("context", {}).get("user_profile", {})

        log_with_context(
            logger,
            "info",
            "Received query",
            request_id=request_id,
            query=query,
            user_context_keys=list(user_context.keys()) if user_context else []
        )

        response = model.get_response(query, user_context)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        log_with_context(
            logger,
            "info",
            "Generated response",
            request_id=request_id,
            response_time_ms=duration_ms,
            intent=response.get("detected_intent"),
            response_length=len(response.get("final_answer", ""))
        )

        return JSONResponse({
            "final_answer": response.get("final_answer", ""),
            "detected_intent": response.get("detected_intent", ""),
            "reasoning": response.get("reasoning", ""),
        })

    except Exception as e:
        logger.exception("Exception inside /query")
        return JSONResponse({"error": str(e)}, status_code=500)

# Uncomment to run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)