from aphrodite import LLM

# Sample prompts.
prompts = [
    "Once upon a time,",
    "In a galaxy far, far away,",
    "The quick brown fox jumps over the lazy dog.",
    "The meaning of life is",
]

# Create an LLM.
model = LLM(model="intfloat/e5-mistral-7b-instruct", enforce_eager=True)
# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = model.encode(prompts)
# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 4096 floats
