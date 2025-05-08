import json
import logging
import google.generativeai as genai
# from retriever import Retriever
from retriever_who import WHOBookRetriever

from potts import IntentClassifier
from tools import meal_logging, meal_planning
from dotenv import load_dotenv
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiModel:
    def __init__(self):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.intent_classifier = IntentClassifier()
            # self.retriever = Retriever()
            self.retriever = WHOBookRetriever()

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def get_response(self, query: str, user_context: dict = None) -> dict:
        if not query or not query.strip():
            return {
                "reasoning": "No valid query provided.",
                "final_answer": "Please provide a valid query.",
                "detected_intent": None,
                "context_used": ""
            }

        try:
            query_embedding = self.retriever.embed_query(query)
            intent_result = self.intent_classifier.classify_from_embedding(query_embedding)
            top_intent = intent_result['top_intent']
            context = self.retriever.retrieve(query)

            if "OUT_OF_SCOPE" in context:
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
            else:
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

                response = self.model.generate_content(full_prompt)

                return {
                    "reasoning": f"Used Gemini with intent: {top_intent}",
                    "final_answer": response.text.strip(),
                    "detected_intent": top_intent,
                    "context_used": context
                }

        except Exception as e:
            logger.error(f"GeminiModel error: {e}")
            return {
                "reasoning": f"Error occurred: {str(e)}",
                "final_answer": f"Sorry, something went wrong.",
                "detected_intent": None,
                "context_used": ""
            }
