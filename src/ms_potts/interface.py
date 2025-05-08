# interface.py

import gradio as gr
import requests

# Global profile storage
user_profile = {}

# ------------------- Gradio App -------------------

def profile_page(profile, chat):
    gr.Markdown("## 🧑‍⚕️ Welcome to Ms. Potts — Your AI Nutrition Assistant\nPlease complete your profile to get started.")

    with gr.Row():
        name = gr.Textbox(label="Name")
        age = gr.Number(label="Age")
        sex = gr.Dropdown(choices=["male", "female", "other"], label="Sex")

    with gr.Row():
        height = gr.Number(label="Height (cm)")
        weight = gr.Number(label="Weight (kg)")

    activity_level = gr.Dropdown(choices=["sedentary", "moderate", "active"], label="Activity Level")
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
        outputs=[profile, chat, status]
    )

def chat_page():
    gr.Markdown(f"## 💬 Chat with Ms. Potts — Personalized Nutrition Guidance")

    chatbot = gr.Chatbot()
    query_input = gr.Textbox(placeholder="Ask about food, diet, meal plans...")
    send_btn = gr.Button("Send")

    def ask_potts(query, history):
        payload = {
            "query": query,
            "context": {
                "user_profile": user_profile
            }
        }
        try:
            # Send POST to localhost:8001/query
            response = requests.post("http://localhost:8080/query", json=payload)
            response_json = response.json()

            final_answer = response_json.get("final_answer", "No answer received.")
            intent = response_json.get("detected_intent", "Unknown Intent")
            reasoning = response_json.get("reasoning", "")

            user_name = user_profile.get("name", "")
            if final_answer.strip().lower().startswith(("hi", "hello")):
                personalized_answer = final_answer
            else:
                personalized_answer = f"Hi {user_name}, {final_answer}"

            history.append((query, personalized_answer + f"\n\n📌 Intent: {intent}\n🧠 Reasoning: {reasoning}"))
            return history, ""
        except Exception as e:
            history.append((query, f"❌ Error: {str(e)}"))
            return history, ""

    send_btn.click(
        ask_potts,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input]
    )

    query_input.submit(
        ask_potts,
        inputs=[query_input, chatbot],
        outputs=[chatbot, query_input]
    )

# Build Gradio App
with gr.Blocks() as gradio_app:
    with gr.Column(visible=True) as profile:
        pass  # Placeholder

    with gr.Column(visible=False) as chat:
        chat_page()

    profile_page(profile, chat)

# Launch Gradio App
if __name__ == "__main__":
    gradio_app.launch(server_name="0.0.0.0", server_port=7860)
