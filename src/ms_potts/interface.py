import gradio as gr
import requests
import os
import google.generativeai as genai

# configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# read backend URL from environment (set this in Cloud Run)
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8080")

# Global profile storage
user_profile = {}

# ------------------- Gradio App -------------------


def profile_page(profile, chat):
    gr.Markdown(
        "## 🧑‍⚕️ Welcome to Ms. Potts — Your AI Nutrition Assistant\n"
        "Please complete your profile to get started."
    )

    with gr.Row():
        name = gr.Textbox(label="Name")
        age = gr.Number(label="Age")
        sex = gr.Dropdown(choices=["male", "female", "other"], label="Sex")

    with gr.Row():
        height = gr.Number(label="Height (cm)")
        weight = gr.Number(label="Weight (kg)")

    activity_level = gr.Dropdown(
        choices=["sedentary", "moderate", "active"], label="Activity Level"
    )
    allergies = gr.Textbox(label="Allergies (comma-separated)")

    save_btn = gr.Button("Save Profile & Start Chatting")
    status = gr.Markdown("")

    def save_profile(name, age, sex, height, weight, activity_level, allergies):
        global user_profile
        user_profile = {
            "name": name,
            "age": int(age),
            "sex": sex,
            "height": int(height),
            "weight": int(weight),
            "activity_level": activity_level,
            "allergies": allergies,
        }
        status_text = f"✅ Welcome {name}! Profile saved. You can start chatting now."
        # hide profile form, show chat box
        return gr.update(visible=False), gr.update(visible=True), status_text

    save_btn.click(
        save_profile,
        inputs=[name, age, sex, height, weight, activity_level, allergies],
        outputs=[profile, chat, status],
    )


def chat_page():
    gr.Markdown("## 💬 Chat with Ms. Potts — Personalized Nutrition Guidance")

    chatbot = gr.Chatbot(type="messages")
    query_input = gr.Textbox(placeholder="Ask about food, diet, meal plans...")
    send_btn = gr.Button("Send")

    def ask_potts(query, history):
        payload = {"query": query, "context": {"user_profile": user_profile}}
        try:
            resp = requests.post(f"{BACKEND_URL}/query", json=payload)
            data = resp.json()

            final_answer = data.get("final_answer", "No answer received.")
            intent = data.get("detected_intent", "Unknown Intent")
            reasoning = data.get("reasoning", "")

            user_name = user_profile.get("name", "")
            if final_answer.strip().lower().startswith(("hi", "hello")):
                personalized = final_answer
            else:
                personalized = f"Hi {user_name}, {final_answer}"

            history.append(
                (
                    query,
                    personalized + f"\n\n📌 Intent: {intent}\n🧠 Reasoning: {reasoning}",
                )
            )
            return history, ""
        except Exception as e:
            history.append((query, f"❌ Error: {e}"))
            return history, ""

    send_btn.click(
        ask_potts, inputs=[query_input, chatbot], outputs=[chatbot, query_input]
    )
    query_input.submit(
        ask_potts, inputs=[query_input, chatbot], outputs=[chatbot, query_input]
    )


# Build Gradio App
with gr.Blocks() as gradio_app:
    with gr.Column(visible=True) as profile:
        pass

    with gr.Column(visible=False) as chat:
        chat_page()

    profile_page(profile, chat)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    gradio_app.launch(server_name="0.0.0.0", server_port=port)
