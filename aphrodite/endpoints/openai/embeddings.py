"""
This file is derived from
[text-generation-webui openai extension embeddings](https://github.com/oobabooga/text-generation-webui/blob/1a7c027386f43b84f3ca3b0ff04ca48d861c2d7a/extensions/openai/embeddings.py)
and modified.
The changes introduced are: Suppression of progress bar,
typing/pydantic classes moved into this file,
embeddings function declared async.
"""

import os
import base64
import numpy as np
from transformers import AutoModel

embeddings_params_initialized = False


def initialize_embedding_params():
    '''
    using 'lazy loading' to avoid circular import
    so this function will be executed only once
    '''
    global embeddings_params_initialized
    if not embeddings_params_initialized:

        global st_model, embeddings_model, embeddings_device

        st_model = os.environ.get("OPENAI_EMBEDDING_MODEL",
                                  'all-mpnet-base-v2')
        embeddings_model = None
        # OPENAI_EMBEDDING_DEVICE: auto (best or cpu),
        # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep,
        # hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta,
        # hpu, mtia, privateuseone
        embeddings_device = os.environ.get("OPENAI_EMBEDDING_DEVICE", 'cpu')
        if embeddings_device.lower() == 'auto':
            embeddings_device = None

        embeddings_params_initialized = True


def load_embedding_model(model: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        print("The sentence_transformers module has not been found. " +
              "Please install it manually with " +
              "pip install -U sentence-transformers.")
        raise ModuleNotFoundError from None

    initialize_embedding_params()
    global embeddings_device, embeddings_model
    try:
        print(f"Try embedding model: {model} on {embeddings_device}")
        if 'jina-embeddings' in model:
            # trust_remote_code is needed to use the encode method
            embeddings_model = AutoModel.from_pretrained(
                model, trust_remote_code=True)
            embeddings_model = embeddings_model.to(embeddings_device)
        else:
            embeddings_model = SentenceTransformer(
                model,
                device=embeddings_device,
            )

        print(f"Loaded embedding model: {model}")
    except Exception as e:
        embeddings_model = None
        raise Exception(f"Error: Failed to load embedding model: {model}",
                        internal_message=repr(e)) from None


def get_embeddings_model():
    initialize_embedding_params()
    global embeddings_model, st_model
    if st_model and not embeddings_model:
        load_embedding_model(st_model)  # lazy load the model

    return embeddings_model


def get_embeddings_model_name() -> str:
    initialize_embedding_params()
    global st_model
    return st_model


def get_embeddings(input: list) -> np.ndarray:
    model = get_embeddings_model()
    embedding = model.encode(input,
                             convert_to_numpy=True,
                             normalize_embeddings=True,
                             convert_to_tensor=False,
                             show_progress_bar=False)
    return embedding


async def embeddings(input: list,
                     encoding_format: str,
                     model: str = None) -> dict:
    if model is None:
        model = st_model
    else:
        load_embedding_model(model)

    embeddings = get_embeddings(input)
    if encoding_format == "base64":
        data = [{
            "object": "embedding",
            "embedding": float_list_to_base64(emb),
            "index": n
        } for n, emb in enumerate(embeddings)]
    else:
        data = [{
            "object": "embedding",
            "embedding": emb.tolist(),
            "index": n
        } for n, emb in enumerate(embeddings)]

    response = {
        "object": "list",
        "data": data,
        "model": st_model if model is None else model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
    }
    return response


def float_list_to_base64(float_array: np.ndarray) -> str:
    # Convert the list to a float32 array that the OpenAPI client expects
    # float_array = np.array(float_list, dtype="float32")

    # Get raw bytes
    bytes_array = float_array.tobytes()

    # Encode bytes into base64
    encoded_bytes = base64.b64encode(bytes_array)

    # Turn raw base64 encoded bytes into ASCII
    ascii_string = encoded_bytes.decode('ascii')
    return ascii_string
