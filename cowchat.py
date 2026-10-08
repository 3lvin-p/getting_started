
import cowsay
from transformers import pipeline
from transformers.utils import logging

# Remove unnecessary warnings from transformers
logging.set_verbosity_error()

gen = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-135M-Instruct"
)

while True:
    input_question = input("Chat, or type 'quit': ")

    if input_question.lower() == "quit":
        cowsay.cow("Goodbye! Moo! Moo!")
        break

    messages = [
        {
            "role": "system",
            "content": (
                "Pretend you are a friendly cow. "
                "Answer briefly, keeping the conversation "
                "in a context that matches the life of a cow. "
                "End each response with Moo!"
            )
        },
        {
            "role": "user",
            "content": input_question
        }
    ]

    response = gen(
        messages,
        max_new_tokens=100,
        clean_up_tokenization_spaces=False
    )

    response_text = response[0]["generated_text"][-1]["content"]

    cowsay.cow(response_text)
