import os
from PIL import Image

from aphrodite import LLM, SamplingParams


def run_llava_next():
    llm = LLM(
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        image_token_id=32000,
        image_input_shape="1,3,336,336",
        # Use the maximum possible value for memory profiling
        image_feature_size=2928,
    )

    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
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
    generated_text = ""
    for o in outputs:
        generated_text += o.outputs[0].text
    print(f"LLM output:{generated_text}")


if __name__ == "__main__":
    run_llava_next()
