from aphrodite import LLM, SamplingParams

prompts = [
  "What is a man? A",
  "The sun is a wondrous body, like a magnificent",
  "All flesh is grass and all the comeliness thereof",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/home/alpindale/AI-Stuff/models/Pythia-70M")
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
  prompt = output.prompt
  generated_text = output.outputs[0].text
  print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
