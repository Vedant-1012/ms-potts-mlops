import os
import gradio as gr
import requests
import google.generativeai as genai

# configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# read backend URL from environment (set this in Cloud Run)
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8080").rstrip("/")

# Global profile storage
user_profile = {}


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
        return gr.update(visible=False), gr.update(visible=True), status_text

    save_btn.click(
        save_profile,
        inputs=[name, age, sex, height, weight, activity_level, allergies],
        outputs=[profile, chat, status],
    )


def chat_page():
    gr.Markdown("## 💬 Chat with Ms. Potts — Personalized Nutrition Guidance")

    # switched to the correct plural form
    chatbot = gr.Chatbot(type="tuples")
    query_input = gr.Textbox(placeholder="Ask about food, diet, meal plans...")
    send_btn = gr.Button("Send")

    def ask_potts(query, history):
        payload = {"query": query, "context": {"user_profile": user_profile}}
        try:
            resp = requests.post(f"{BACKEND_URL}/query", json=payload)
            resp.raise_for_status()
            data = resp.json()

            final_answer = data.get("final_answer", "No answer received.")
            intent = data.get("detected_intent", "Unknown Intent")
            reasoning = data.get("reasoning", "")

            name = user_profile.get("name", "")
            if not final_answer.lower().startswith(("hi", "hello")) and name:
                final_answer = f"Hi {name}, {final_answer}"

            history.append(
                (
                    query,
                    final_answer + f"\n\n📌 Intent: {intent}\n🧠 Reasoning: {reasoning}",
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


# Build the app
with gr.Blocks() as gradio_app:
    with gr.Column(visible=True) as profile:
        pass
    with gr.Column(visible=False) as chat:
        chat_page()
    profile_page(profile, chat)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    gradio_app.launch(server_name="0.0.0.0", server_port=port)
