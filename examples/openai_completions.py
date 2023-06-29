import openai

openai.api_key = "sasuga"
openai.api_base = "http://localhost:8000/v1"
model = "PygmalionAI/pygmalion-350m"

models = openai.Model.list()
print("Models:", models)

stream = True
completion = openai.Completion.create(
    model=model, prompt="What is a man? A", echo=False, n=2, best_of=3, stream=stream, logprobs=3)

if stream:
    for c in completion:
        print(c)
else:
    print("Completion result:", completion)