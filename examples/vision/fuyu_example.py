import os
from PIL import Image

from aphrodite import LLM, SamplingParams


def run_fuyu():
    llm = LLM(model="adept/fuyu-8b", max_model_len=4096)

    # single-image prompt
    prompt = "What is the content of this image?\n"
    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "burg.jpg")
    image = Image.open(image_path)

    sampling_params = SamplingParams(temperature=1.1,
                                     min_p=0.06,
                                     max_tokens=512)

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_fuyu()