import os
import base64
import numpy as np
from transformers import AutoModel
from pydantic import BaseModel, Field
from typing import List

embeddings_params_initialized = False

def initialize_embedding_params():
    '''
    using 'lazy loading' to avoid circular import
    so this function will be executed only once
    '''
    global embeddings_params_initialized
    if not embeddings_params_initialized:

        global st_model, embeddings_model, embeddings_device

        st_model = os.environ.get("OPENEDAI_EMBEDDING_MODEL", 'all-mpnet-base-v2')
        embeddings_model = None
        # OPENEDAI_EMBEDDING_DEVICE: auto (best or cpu), cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone
        embeddings_device = os.environ.get("OPENEDAI_EMBEDDING_DEVICE", 'cpu')
        if embeddings_device.lower() == 'auto':
            embeddings_device = None

        embeddings_params_initialized = True


def load_embedding_model(model: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        print("The sentence_transformers module has not been found. Please install it manually with pip install -U sentence-transformers.")
        raise ModuleNotFoundError

    initialize_embedding_params()
    global embeddings_device, embeddings_model
    try:
        print(f"Try embedding model: {model} on {embeddings_device}")
        if 'jina-embeddings' in model:
            embeddings_model = AutoModel.from_pretrained(model, trust_remote_code=True)  # trust_remote_code is needed to use the encode method
            embeddings_model = embeddings_model.to(embeddings_device)
        else:
            embeddings_model = SentenceTransformer(model, device=embeddings_device)

        print(f"Loaded embedding model: {model}")
    except Exception as e:
        embeddings_model = None
        raise Exception(f"Error: Failed to load embedding model: {model}", internal_message=repr(e))


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
    embedding = model.encode(input, convert_to_numpy=True, normalize_embeddings=True, convert_to_tensor=False, show_progress_bar=False)
    return embedding


async def embeddings(input: list, encoding_format: str) -> dict:
    embeddings = get_embeddings(input)
    if encoding_format == "base64":
        data = [{"object": "embedding", "embedding": float_list_to_base64(emb), "index": n} for n, emb in enumerate(embeddings)]
    else:
        data = [{"object": "embedding", "embedding": emb.tolist(), "index": n} for n, emb in enumerate(embeddings)]

    response = {
        "object": "list",
        "data": data,
        "model": st_model,  # return the real model
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



class EmbeddingsRequest(BaseModel):
    input: str | List[str] | List[int] | List[List[int]]
    model: str | None = Field(default=None, description="Unused parameter. To change the model, set the OPENEDAI_EMBEDDING_MODEL and OPENEDAI_EMBEDDING_DEVICE environment variables before starting the server.")
    encoding_format: str = Field(default="float", description="Can be float or base64.")
    user: str | None = Field(default=None, description="Unused parameter.")


class EmbeddingsResponse(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EncodeRequest(BaseModel):
    text: str


class EncodeResponse(BaseModel):
    tokens: List[int]
    length: int
