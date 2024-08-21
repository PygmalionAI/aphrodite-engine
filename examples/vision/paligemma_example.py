import os

from PIL import Image

from aphrodite import LLM, SamplingParams


def run_paligemma():
    llm = LLM(model="google/paligemma-3b-mix-224")

    prompt = "caption es"

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


def main():
    run_paligemma()


if __name__ == "__main__":
    main()