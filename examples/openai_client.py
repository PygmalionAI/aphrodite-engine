import argparse
import openai

openai.api_key = "sasuga"
openai.api_base = "http://localhost:8000/v1"
model = "PygmalionAI/pygmalion-350m"

models = openai.Model.list()
print("Models:", models)

def get_completions(prompt, use_chat_completions):
    if use_chat_completions:
        completions = openai.Completion.create(
            model=model,
            message=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
        )
        return completions.choices[-1].message["content"]
    else:
        completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            echo=False,
            n=2,
            best_of=3,
            logprobs=3,
        )
        return completion.choices[0].text.strip()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-chat-completions", action="store_true")
    parser.add_argument("--prompt", type=str, default="A robot may injure a human being")
    args = parser.parse_args()

    completions = get_completions(args.prompt, args.use_chat_completions)

    print("Completion result:")
    print(completions)