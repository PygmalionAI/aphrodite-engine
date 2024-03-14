from aphrodite import LLM, SamplingParams

# Sample prompts.
prompts = [
    "<|system|>Enter chat mode.<|user|>Hello!<|model|>",
    "<|system|>Enter RP mode.<|model|>Hello!<|user|>What are you doing?",
    "<|system|>Enter chat mode.<|user|>What is the meaning of life?<|model|>",
    "<|system|>Enter QA mode.<|user|>What is a man?<|model|>A miserable",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="PygmalionAI/pygmalion-2-7b"
          )  # pass additional arguments here, such as `quantization`
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
