# import json
# import time
# import logging
# import google.generativeai as genai
# from retriever_who import WHOBookRetriever
# from potts import IntentClassifier
# from tools import meal_logging, meal_planning
# from dotenv import load_dotenv
# import os

# from utils.experiment_tracking import ExperimentTracker
# from utils.monitoring import ModelMonitor  # ✅ import monitor
# from utils.debugging import DebugTracer, debug_value, exception_handler

# tracer = DebugTracer(output_dir="./debug_traces")
# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class GeminiModel:
#     def __init__(self):
#         try:
#             genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

#             # ✅ Initialize monitoring
#             self.monitor = ModelMonitor(metrics_dir="./metrics")

#             if hasattr(genai, 'GenerativeModel'):
#                 self.model = genai.GenerativeModel('gemini-1.5-flash')
#                 self.use_new_api = True
#             else:
#                 self.model = None
#                 self.use_new_api = False

#             self.intent_classifier = IntentClassifier()
#             self.retriever = WHOBookRetriever()

#         except Exception as e:
#             logger.error(f"Initialization error: {e}")
#             raise
    
#     @tracer.trace_function
#     def get_response(self, query: str, user_context: dict = None) -> dict:
#         if not query or not query.strip():
#             return {
#                 "reasoning": "No valid query provided.",
#                 "final_answer": "Please provide a valid query.",
#                 "detected_intent": None,
#                 "context_used": ""
#             }

#         try:
#             query_embedding = self.retriever.embed_query(query)
#             debug_value(query_embedding.shape, "Query embedding shape")

#             intent_result = self.intent_classifier.classify_from_embedding(query_embedding)
    
#             top_intent = intent_result['top_intent']
#             debug_value(top_intent, "Detected intent")

#             context = self.retriever.retrieve(query)
#             debug_value(len(context), "Context length")

#             if "OUT_OF_SCOPE" in context:
#                 return {
#                     "reasoning": context.replace("OUT_OF_SCOPE:", "Query out of scope."),
#                     "final_answer": "This question is outside my nutrition expertise.",
#                     "detected_intent": top_intent,
#                     "context_used": context
#                 }

#             if top_intent == "Meal-Logging":
#                 return meal_logging(query, user_context)
#             elif top_intent == "Meal-Planning-Recipes":
#                 return meal_planning(user_context)
#             else:
#                 user_profile_str = f"User Profile: {json.dumps(user_context)}" if user_context else ""
#                 full_prompt = f"""
#                 You are a helpful and expert nutrition assistant.
#                 Use the following context to answer the user's question:

#                 Context:
#                 {context}

#                 {user_profile_str}

#                 Question:
#                 {query}

#                 Provide a helpful, accurate, and empathetic response:
#                 """

#                 # ✅ Start inference timer
#                 start_time = time.time()

#                 # Generate content
#                 if self.use_new_api:
#                     response = self.model.generate_content(full_prompt)
#                     response_text = response.text.strip()
#                 else:
#                     response = genai.generate_text(
#                         model="gemini-1.5-flash",
#                         prompt=full_prompt,
#                         temperature=0.2,
#                         max_output_tokens=1024
#                     )
#                     response_text = response.result.strip()

#                 # ✅ End timer
#                 latency_ms = (time.time() - start_time) * 1000
#                 debug_value(latency_ms, "Inference latency in ms")

#                 tokens_generated = len(response_text.split())
#                 debug_value(tokens_generated, "Generated token count")

#                 # ✅ Log model metrics
#                 self.monitor.log_model_metrics({
#                     "latency_ms": latency_ms,
#                     "tokens_generated": tokens_generated
#                 })


#                 # ✅ Save debug trace
#                 tracer.save_trace(f"query_trace_{int(time.time())}.json")

#                 return {
#                     "reasoning": f"Used Gemini with intent: {top_intent}",
#                     "final_answer": response_text,
#                     "detected_intent": top_intent,
#                     "context_used": context
#                 }

#         except Exception as e:
#             logger.error(f"GeminiModel error: {e}")
#             tracer.save_trace(f"error_trace_{int(time.time())}.json")  # SAVE TRACE HERE TOO
#             return {
#                 "reasoning": f"Error occurred: {str(e)}",
#                 "final_answer": "Sorry, something went wrong.",
#                 "detected_intent": None,
#                 "context_used": ""
#             }

import json
import time
import logging
import google.generativeai as genai
from retriever_who import WHOBookRetriever
from potts import IntentClassifier
from tools import meal_logging, meal_planning
from dotenv import load_dotenv
import os

from utils.experiment_tracking import ExperimentTracker
from utils.monitoring import ModelMonitor
from utils.debugging import DebugTracer, debug_value, exception_handler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize debugging
tracer = DebugTracer(output_dir="./debug_traces")

class GeminiModel:
    def __init__(self):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            self.monitor = ModelMonitor(metrics_dir="./metrics")
            self.tracker = ExperimentTracker(
                experiment_name="ms_potts_gemini_responses",
                artifacts_dir="./mlruns"
            )

            if hasattr(genai, 'GenerativeModel'):
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_new_api = True
            else:
                self.model = None
                self.use_new_api = False

            self.intent_classifier = IntentClassifier()
            self.retriever = WHOBookRetriever()

        except Exception as e:
            logger.exception("Initialization failed in GeminiModel.")
            raise

    @tracer.trace_function
    def get_response(self, query: str, user_context: dict = None) -> dict:
        if not query or not query.strip():
            return {
                "reasoning": "No valid query provided.",
                "final_answer": "Please provide a valid query.",
                "detected_intent": None,
                "context_used": ""
            }

        try:
            with self.tracker.start_run(
                run_name="gemini_query",
                description="Query inference via GeminiModel"
            ):
                query_embedding = self.retriever.embed_query(query)
                debug_value(query_embedding.shape, "Query embedding shape")

                intent_result = self.intent_classifier.classify_from_embedding(query_embedding)
                top_intent = intent_result['top_intent']
                debug_value(top_intent, "Detected intent")

                context = self.retriever.retrieve(query)
                debug_value(len(context), "Context length")

                self.tracker.log_params({
                    "query": query,
                    "top_intent": top_intent,
                    "has_context": bool(context),
                    "user_context": json.dumps(user_context or {})
                })

                if "OUT_OF_SCOPE" in context:
                    self.tracker.log_metrics({"out_of_scope": 1})
                    return {
                        "reasoning": context.replace("OUT_OF_SCOPE:", "Query out of scope."),
                        "final_answer": "This question is outside my nutrition expertise.",
                        "detected_intent": top_intent,
                        "context_used": context
                    }

                if top_intent == "Meal-Logging":
                    return meal_logging(query, user_context)
                elif top_intent == "Meal-Planning-Recipes":
                    return meal_planning(user_context)

                # Prepare prompt
                user_profile_str = f"User Profile: {json.dumps(user_context)}" if user_context else ""
                full_prompt = f"""
                You are a helpful and expert nutrition assistant.
                Use the following context to answer the user's question:

                Context:
                {context}

                {user_profile_str}

                Question:
                {query}

                Provide a helpful, accurate, and empathetic response:
                """

                start_time = time.time()
                if self.use_new_api:
                    response = self.model.generate_content(full_prompt)
                    response_text = response.text.strip()
                else:
                    response = genai.generate_text(
                        model="gemini-1.5-flash",
                        prompt=full_prompt,
                        temperature=0.2,
                        max_output_tokens=1024
                    )
                    response_text = response.result.strip()

                latency_ms = (time.time() - start_time) * 1000
                tokens_generated = len(response_text.split())

                self.monitor.log_model_metrics({
                    "latency_ms": latency_ms,
                    "tokens_generated": tokens_generated
                })
                self.tracker.log_metrics({
                    "latency_ms": latency_ms,
                    "tokens_generated": tokens_generated
                })

                debug_value(latency_ms, "Inference latency in ms")
                debug_value(tokens_generated, "Generated token count")

                # Save result as artifact
                timestamp = int(time.time())
                result_data = {
                    "query": query,
                    "response": response_text,
                    "context_used": context,
                    "user_context": user_context
                }
                result_file = f"gemini_response_{timestamp}.json"
                with open(result_file, "w") as f:
                    json.dump(result_data, f, indent=2)
                self.tracker.log_artifact(result_file)

                tracer.save_trace(f"query_trace_{timestamp}.json")

                return {
                    "reasoning": f"Used Gemini with intent: {top_intent}",
                    "final_answer": response_text,
                    "detected_intent": top_intent,
                    "context_used": context
                }

        except Exception as e:
            logger.exception("Error during get_response execution.")
            self.tracker.log_metrics({"inference_error": 1})
            tracer.save_trace(f"error_trace_{int(time.time())}.json")
            return {
                "reasoning": f"Error occurred: {str(e)}",
                "final_answer": "Sorry, something went wrong.",
                "detected_intent": None,
                "context_used": ""
            }