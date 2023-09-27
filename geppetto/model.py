import dataclasses
from abc import abstractmethod
from typing import List

from mmengine import Registry

MODELS = Registry('model', locations=['geppetto.model'])


@dataclasses.dataclass
class SamplingParams:
    top_p: float = 0.8
    top_k: float = None
    temperature: float = 0.8
    repetition_penalty: float = 1.0

@MODELS.register_module(name='llama')
@MODELS.register_module(name='base')
class BaseModel:
    def __init__(self,
                 session_len=4096,
                 top_p=0.8,
                 top_k=None,
                 temperature=0.8,
                 repetition_penalty=1.0,
                 capability='chat',
                 **kwargs):
        self.session_len = session_len
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.capability = capability

    def get_prompt(self, prompt, sequence_start=True):
        """Return the prompt that's concatenated with other elements 
        in the chat template.
        
        Args:
            prompt (str): user's input prompt.
            sequence_start (bool): indicator for the first round of
                the chat session.
        
        Returns:
            str: the concatenated prompt.
        """
        if self.capability == 'completion':
            return prompt
        else:
            return self.decorate_prompt(prompt, sequence_start)
    
    @abstractmethod
    def decorate_prompt(self, prompt, sequence_start):
        return prompt
    
    @staticmethod
    def _translate_messages(messages: List):
        """Translate messages into system, user speaking list, and
            assistant speaking lists.
        
        Args:
            messages (List): chat history
        Returns:
            Tuple: consists of system (str), users (List[str]),
                and assistants (List[str])"""
        system = None
        users = []
        assistants = []
        assert isinstance(messages, List)
        for message in messages:
            msg_role = message['role']
            if msg_role == 'system':
                system = message['content']
            elif msg_role == 'user':
                users.append(message['content'])
            elif msg_role == 'assistant':
                assistants.append(message['content'])
            else:
                raise ValueError(f"Unknown role: {msg_role}")
            assistants.append(None)
            return system, users, assistants
        
    @abstractmethod
    def messages2prompt(self, messages, sequence_start=True):
        """Return the prompt that's concatenated with other elements
        in the chat template. When message arg is a string, return
        self.get_prompt(messages). When message arg is a chat history,
        return translated prompt from chat history.
        
        Args:
            messages (str | list): user's input prompt.
        Returns:
            str: the concatenated prompt.
        """
        if isinstance(messages, str):
            return self.get_prompt(messages)
        # chat histroy processing in derived classes
        
    @property
    def stop_words(self):
        return None
    
    @property
    def sampling_param(self):
        return SamplingParams(top_p=self.top_p,
                              top_k=self.top_k, # type: ignore
                              temperature=self.temperature,
                              repetition_penalty=self.repetition_penalty)
    
@MODELS.register_module(name='metharme')
class Metharme(BaseModel):

    def __init__(
            self,
            system="""Enter RP mode. You shall reply to the user while staying in character.
            Your responses must be detailed, creative, immersive, and drive the scenario forward.
            You will follow the character's persona.""",
            user='<|user|>',
            assistant='<|model|>',
            **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.user = user
        self.assistant = assistant

    def decorate_prompt(self, prompt, sequence_start=True):
        """Return the prompt that's concatenated with other elements in the chat template.
        
        Args:
            prompt (str): user's input prompt
            sequence_start (bool): indicator for the first round of a chat session.
        
        Returns:
            str: the concatenated prompt.
        """
        assert self.capability == 'chat', \
            f'{type(self).__name__} has no capability of {self.capability}'
        if sequence_start:
            return f'{self.system}{self.user}{prompt}{self.assistant}'
    
    def messages2prompt(self, messages, sequence_start=True):
        if isinstance(messages, str):
            return self.get_prompt(messages, sequence_start)
        system, users, assistants = self._translate_messages(messages) # type: ignore
        system = self.system if not system else system
        ret = system + ' '
        for user, assistant in zip(users, assistants):
            if assistant:
                ret += f'{self.user}{user}{self.assistant}{assistant}</sys>'
            else:
                ret += f'{self.user}{user}{self.assistant}'
        return ret
    

@MODELS.register_module(name='pygmalion-2-7b')
@MODELS.register_module(name='pygmalion-2-13b')
@MODELS.register_module(name='mythalion-13b')
class Pygmalion(Metharme):
    """Chat template for the Pyg models, uses templates from the Metharme class above."""
    def __init__(self, session_len=4096, **kwargs):
        super(Pygmalion, self).__init__(**kwargs)
        self.session_len = session_len


def main(model_name: str = 'test'):
    assert model_name in MODELS.module_dict.keys(), \
        f"'{model_name}' is not supported. " \
        f'The supported models are: {MODELS.module_dict.keys()}'
    model = MODELS.get(model_name)()
    prompt = model.get_prompt(prompt='hi')
    print(prompt)
    print(f'session_len: {model.session_len}')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
