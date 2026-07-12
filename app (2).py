import os
import json
import time
import html
import gradio as gr
from fireworks import Fireworks


api_key = os.getenv("FIREWORKS_API_KEY")

client = Fireworks(
    api_key=api_key,
    account_id="fireworks",
) if api_key else None


def get_available_models():
    if not client:
        return [], "⚠️ FIREWORKS_API_KEY not found in Colab Secrets."

    try:
        models = list(
            client.models.list(filter="supports_serverless=true")
        )

        names = sorted(model.name for model in models)

        if not names:
            return [], "⚠️ No Fireworks serverless models are available."

        return names, f"✅ Loaded {len(names)} Fireworks models."

    except Exception as e:
        return [], f"⚠️ Could not load models: {html.escape(str(e))}"


def refresh_models():
    models, message = get_available_models()
    return gr.update(choices=models, value=models[0] if models else None), message


def clean_json(text):
    text = (text or "").strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    return json.loads(text)


def generate_quiz(topic, count, difficulty, model):
    if not (topic := (topic or "").strip()):
        return [], {}, "⚠️ Please enter a quiz topic."

    if not client:
        return [], {}, "⚠️ FIREWORKS_API_KEY not found in Colab Secrets."

    if not model:
        return [], {}, "⚠️ Click Refresh Models and select a model first."

    prompt = f"""Generate exactly {int(count)} unique multiple-choice questions about "{topic}" at {difficulty} difficulty.

Return ONLY valid JSON:
{{
  "quiz": [
    {{
      "question": "...",
      "options": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      }},
      "answer": "A"
    }}
  ]
}}"""

    try:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You create accurate educational quizzes. Return JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        data = clean_json(response.choices[0].message.content)
        quiz = data["quiz"]
        valid_answers = {"A", "B", "C", "D"}

        if not isinstance(quiz, list) or len(quiz) != int(count):
            raise ValueError("Incorrect number of questions returned.")

        for question in quiz:
            if (
                not isinstance(question.get("question"), str)
                or set(question.get("options", {}).keys()) != valid_answers
                or question.get("answer") not in valid_answers
            ):
                raise ValueError("Invalid quiz format returned.")

        return quiz, {}, "✅ Quiz generated successfully. Answer all questions."

    except json.JSONDecodeError:
        return [], {}, "⚠️ Invalid JSON received. Please generate again."
    except Exception as e:
        return [], {}, f"⚠️ Error: {html.escape(str(e))}"


def save_answer(choice, answers, index):
    answers = dict(answers or {})

    if choice:
        answers[index] = choice.split(".", 1)[0]

    return answers


def submit_quiz(quiz, answers):
    if not quiz:
        return "", "⚠️ Generate a quiz first."

    answers = answers or {}
    missing = [str(i + 1) for i in range(len(quiz)) if i not in answers]

    if missing:
        return "", f"⚠️ Answer every question first. Missing: {', '.join(missing)}"

    score = sum(
        answers[i] == question["answer"]
        for i, question in enumerate(quiz)
    )

    percentage = score / len(quiz) * 100
    passed = percentage >= 60
    status = "PASS 🎉" if passed else "FAIL ❌"
    color = "#16a34a" if passed else "#dc2626"

    review = []

    for i, question in enumerate(quiz):
        selected = answers[i]
        correct = question["answer"]
        icon = "✅" if selected == correct else "❌"

        review.append(f"""
        <div class="review-card">
            <b>Question {i + 1}: {html.escape(question["question"])}</b><br>
            {icon} Your answer:
            <b>{selected}. {html.escape(question["options"][selected])}</b><br>
            Correct answer:
            <b>{correct}. {html.escape(question["options"][correct])}</b>
        </div>
        """)

    result = f"""
    <div class="score-card">
        <h2>Score: {score} / {len(quiz)}</h2>
        <p>Percentage: <b>{percentage:.1f}%</b></p>
        <p style="color:{color}; font-size:18px"><b>{status}</b></p>
    </div>
    <h2 class="review-title">Answer Review</h2>
    {''.join(review)}
    """

    return result, "✅ Quiz submitted successfully."


def reset_quiz():
    return "", 5, "Moderate", [], {}, "Enter a topic and generate a new quiz."


models, startup_message = get_available_models()
selected_model = models[0] if models else None

css = """
body {
    background: linear-gradient(135deg, #dbeafe, #f3e8ff) !important;
}

.gradio-container {
    max-width: 900px !important;
    min-height: 100vh;
    background: linear-gradient(135deg, #dbeafe, #f3e8ff) !important;
}

.quiz-card,
.review-card,
.score-card {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #c7d2fe;
    border-radius: 16px;
    padding: 18px;
    margin: 14px 0;
    box-shadow: 0 5px 14px rgba(79, 70, 229, 0.12);
}

.quiz-card {
    border-left: 6px solid #6366f1;
}

.review-card {
    border-left: 6px solid #8b5cf6;
}

.score-card {
    border-left: 6px solid #4f46e5;
    background: #eef2ff !important;
}

.quiz-card h3,
.quiz-card p,
.review-card,
.review-card b,
.score-card,
.score-card h2,
.score-card p,
.review-title {
    color: #111827 !important;
}

h1, h2, h3 {
    color: #312e81 !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    css=css,
    title="Fireworks AI Quiz Generator",
) as demo:

    gr.HTML("""
    <div style="text-align:center; color:#312e81; padding:10px">
        <h1>🧠 Fireworks AI Quiz Generator</h1>
        <p>Create and solve AI-generated quizzes.</p>
    </div>
    """)

    quiz_state = gr.State([])
    answers_state = gr.State({})

    with gr.Row():
        topic = gr.Textbox(
            label="Quiz Topic",
            placeholder="Example: Python, Java, Machine Learning",
            scale=3,
        )

        count = gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Questions",
            scale=1,
        )

    difficulty = gr.Radio(
        choices=["Easy", "Moderate", "Difficult"],
        value="Moderate",
        label="Difficulty",
    )

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=models,
            value=selected_model,
            label="Fireworks Model",
            scale=4,
        )

        refresh_btn = gr.Button("Refresh Models", scale=1)

    with gr.Row():
        generate_btn = gr.Button("🚀 Generate Quiz", variant="primary")
        new_btn = gr.Button("New Quiz")

    message = gr.Markdown(startup_message)

    @gr.render(inputs=[quiz_state])
    def show_questions(quiz):
        if not quiz:
            return

        gr.Markdown("## 📝 Your Quiz")

        for i, question in enumerate(quiz):
            with gr.Group():
                gr.HTML(f"""
                <div class="quiz-card">
                    <h3>Question {i + 1}</h3>
                    <p>{html.escape(question["question"])}</p>
                </div>
                """)

                radio = gr.Radio(
                    choices=[
                        f"{key}. {question['options'][key]}"
                        for key in "ABCD"
                    ],
                    label="Select your answer",
                )

                radio.change(
                    lambda choice, answers, index=i: save_answer(
                        choice, answers, index
                    ),
                    inputs=[radio, answers_state],
                    outputs=answers_state,
                )

        submit_btn = gr.Button("Submit Quiz", variant="primary")
        result = gr.HTML()

        submit_btn.click(
            submit_quiz,
            inputs=[quiz_state, answers_state],
            outputs=[result, message],
        )

    refresh_btn.click(
        refresh_models,
        outputs=[model_dropdown, message],
    )

    generate_btn.click(
        generate_quiz,
        inputs=[topic, count, difficulty, model_dropdown],
        outputs=[quiz_state, answers_state, message],
        show_progress="full",
    )

    new_btn.click(
        reset_quiz,
        outputs=[
            topic,
            count,
            difficulty,
            quiz_state,
            answers_state,
            message,
        ],
    )

demo.launch(share=True, debug=True)
