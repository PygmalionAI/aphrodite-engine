import os

from PIL import Image

from aphrodite import LLM, SamplingParams


def run_phi3v():
    model_path = "microsoft/Phi-3-vision-128k-instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    # You may lower either max_num_seqs or max_model_len
    # to run this example on a machine with limited memory.
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_num_seqs=5,
    )

    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "burg.jpg")
    image = Image.open(image_path)

    # single-image prompt
    prompt = "<|user|>\n<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"  # noqa: E501

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
    run_phi3v()
