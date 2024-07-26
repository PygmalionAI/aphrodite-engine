import time
from typing import AsyncIterator, List, Tuple, Optional

from fastapi import Request

from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.endpoints.openai.protocol import (EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData, UsageInfo)
from aphrodite.endpoints.openai.serving_completions import parse_prompt_format
from aphrodite.endpoints.openai.serving_engine import OpenAIServing, LoRA
from aphrodite.common.outputs import EmbeddingRequestOutput
from aphrodite.common.utils import merge_async_iterators, random_uuid

TypeTokenIDs = List[int]


def request_output_to_embedding_response(
    final_res_batch: List[EmbeddingRequestOutput],
    request_id: str,
    created_time: int,
    model_name: str,
) -> EmbeddingResponse:
    data = []
    num_prompt_tokens = 0
    for idx, final_res in enumerate(final_res_batch):
        assert final_res is not None
        prompt_token_ids = final_res.prompt_token_ids

        embedding_data = EmbeddingResponseData(
            index=idx, embedding=final_res.outputs.embedding)
        data.append(embedding_data)

        num_prompt_tokens += len(prompt_token_ids)

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens,
    )

    return EmbeddingResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        data=data,
        usage=usage,
    )


class OpenAIServingEmbedding(OpenAIServing):

    def __init__(self, engine: AsyncAphrodite,
                 served_model_names: List[str],
                 lora_modules: Optional[List[LoRA]] = None):
        super().__init__(engine=engine,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules)

    async def create_embedding(self, request: EmbeddingRequest,
                               raw_request: Request):
        """Completion API similar to OpenAI's API.
        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Return error for unsupported features.
        if request.encoding_format == "base64":
            return self.create_error_response(
                "base64 encoding is not currently supported")
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.monotonic())

        # Schedule the request and get the result generator.
        generators = []
        try:
            prompt_is_tokens, prompts = parse_prompt_format(request.input)
            pooling_params = request.to_pooling_params()

            for i, prompt in enumerate(prompts):
                if prompt_is_tokens:
                    prompt_formats = self._validate_prompt_and_tokenize(
                        request, prompt_ids=prompt)
                else:
                    prompt_formats = self._validate_prompt_and_tokenize(
                        request, prompt=prompt)

                prompt_ids, prompt_text = prompt_formats

                generators.append(
                    self.engine.generate(prompt_text,
                                         pooling_params,
                                         f"{request_id}-{i}",
                                         prompt_token_ids=prompt_ids))
        except ValueError as e:
            # TODO: Use a aphrodite-specific Validation Error
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, EmbeddingRequestOutput]] = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: EmbeddingRequestOutput = [None] * len(prompts)
        async for i, res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(f"{request_id}-{i}")
                # TODO: Use a aphrodite-specific Validation Error
                return self.create_error_response("Client disconnected")
            final_res_batch[i] = res
        response = request_output_to_embedding_response(
            final_res_batch, request_id, created_time, model_name)

        return response
