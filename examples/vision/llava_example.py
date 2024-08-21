import os

from PIL import Image

from aphrodite import LLM, SamplingParams

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
# You can use `.buildkite/download-images.sh` to download them


def run_llava():
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
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
            }
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main():
    run_llava()


if __name__ == "__main__":
    main()
