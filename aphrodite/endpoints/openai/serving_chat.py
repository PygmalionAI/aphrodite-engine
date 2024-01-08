import time
from typing import AsyncGenerator, AsyncIterator, Union

from fastapi import Request


from aphrodite.common.logger import init_logger
from aphrodite.common.utils import random_uuid
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.endpoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo)
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.endpoints.openai.serving_engine import OpenAIServing
from aphrodite.common.logits_processor import BiasLogitsProcessor

logger = init_logger(__name__)


class OpenAIServingChat(OpenAIServing):

    def __init__(self,
                 engine: AsyncAphrodite,
                 served_model: str,
                 response_role: str,
                 chat_template=None):
        super().__init__(engine=engine,
                         served_model=served_model,
                         response_role=response_role,
                         chat_template=chat_template)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.
        See  https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI ChatCompletion API.
        NOTE: Currently we do not support the following features:
            - function_call (Users should implement this by themselves)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if not request.logit_bias:
            logit_processors = []
        else:
            biases = dict(
                map(lambda bias: (int(bias[0]), bias[1]),
                    request.logit_bias.items()))
            logit_processors = [BiasLogitsProcessor(biases)]

        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt)
        except Exception as e:
            logger.error(
                f"Error in applying chat template from request: {str(e)}")
            return self.create_error_response(str(e))

        token_ids, error_check_ret = await self._check_length(request,
                                                              prompt=prompt)
        if error_check_ret is not None:
            return error_check_ret

        request_id = f"cmpl-{random_uuid()}"
        # We disable top_k at -1, add this conversion for
        # compatibility
        if request.top_k == 0:
            request.top_k = -1
        try:
            spaces_between_special_tokens = request.spaces_between_special_tokens
            sampling_params = SamplingParams(
                n=request.n,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                top_a=request.top_a,
                min_p=request.min_p,
                tfs=request.tfs,
                eta_cutoff=request.eta_cutoff,
                epsilon_cutoff=request.epsilon_cutoff,
                typical_p=request.typical_p,
                mirostat_mode=request.mirostat_mode,
                mirostat_tau=request.mirostat_tau,
                mirostat_eta=request.mirostat_eta,
                stop=request.stop,
                stop_token_ids=request.stop_token_ids,
                include_stop_str_in_output=request.include_stop_str_in_output,
                max_tokens=request.max_tokens,
                best_of=request.best_of,
                ignore_eos=request.ignore_eos,
                use_beam_search=request.use_beam_search,
                skip_special_tokens=request.skip_special_tokens,
                spaces_between_special_tokens=request.
                spaces_between_special_tokens,  # pylint: disable=line-too-long
                custom_token_bans=request.custom_token_bans,
                logprobs=request.logprobs,
                prompt_logprobs=request.prompt_logprobs,
                logits_processors=logit_processors,
            )
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator = self.engine.generate(prompt, sampling_params,
                                                request_id, token_ids)
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id)
        else:
            return await self.chat_completion_full_generator(
                request, raw_request, result_generator, request_id)

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1].role

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str
    ) -> Union[ErrorResponse, AsyncGenerator[str, None]]:

        model_name = request.model
        created_time = int(time.monotonic())
        chunk_object_type = "chat.completion.chunk"

        # Send first response for each request.n (index) with the role
        role = self.get_chat_request_role(request)
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i, delta=DeltaMessage(role=role), finish_reason=None)
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 object=chunk_object_type,
                                                 created=created_time,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        # Send response to echo the input portion of the last message
        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]
            if last_msg_content:
                for i in range(request.n):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=last_msg_content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"

        # Send response for each token for each request.n (index)
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index

                if finish_reason_sent[i]:
                    continue

                if output.finish_reason is None:
                    # Send token-by-token response for each request.n
                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=delta_text),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    data = chunk.json(exclude_unset=True, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                else:
                    # Send the finish response for each request.n only once
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=previous_num_tokens[i],
                        total_tokens=prompt_tokens + previous_num_tokens[i],
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i, delta=[], finish_reason=output.finish_reason)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)
                    if final_usage is not None:
                        chunk.usage = final_usage
                    data = chunk.json(exclude_unset=True,
                                      exclude_none=True,
                                      ensure_ascii=False)
                    yield f"data: {data}\n\n"
                    finish_reason_sent[i] = True
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
            self, request: ChatCompletionRequest, raw_request: Request,
            result_generator: AsyncIterator[RequestOutput],
            request_id: str) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = request.model
        created_time = int(time.monotonic())
        final_res: RequestOutput = None

        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []
        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if request.messages and isinstance(
                    request.messages, list) and request.messages[-1].get(
                        "content") and request.messages[-1].get(
                            "role") == role:
                last_msg_content = request.messages[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response
    