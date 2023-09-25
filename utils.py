import logging

from transformers.generation.streamers import BaseStreamer

from .dist import get_rank, master_only, master_only_and_broadcast_general

try:
    import readline
except ImportError:
    pass

logger = logging.getLogger(__name__)

class TerminalIO:

    end_of_output = '\n'

    @master_only_and_broadcast_general
    def input(self):
        print('\ndouble enter to send input >>> ', end='')
        sentinel = ''
        try:
            return '\n'.join(iter(input, sentinel))
        except EOFError:
            print('Detect EOF, exit')
            exit()

    @master_only
    def output(self, string):
        print(string, end='', flush=True)

class BasicStreamer(BaseStreamer):

    def __init__(self,
                 decode_func,
                 output_func,
                 end_of_output='\n',
                 skip_prompt=True):
        self.decode = decode_func
        self.output = output_func
        self.end_of_output = end_of_output
        self.skip_prompt = skip_prompt
        self.gen_len = 0

    def put(self, value):
        if self.gen_len == 0 and self.skip_prompt:
            pass
        else:
            token = self.decode(value)
            self.output(token)
        
        self.gen_len += 1

    def end(self):
        self.output(self.end_of_output)
        self.gen_len = 0


def control(prompt, gen_config, sm):

    if prompt == 'exit':
        exit(0)
    
    if prompt == 'clear':
        sm.new_session()
        logger.info('Session cleared.')
        return True
    
    if prompt.startswith('config set'):
        try:
            keqv = prompt.split()[-1]
            k, v = keqv.split('=')
            v = eval(v)
            gen_config.__setattr__(k, v)
            logger.info(f"Worker {get_rank()} set {k} to {repr(v)}")
            logger.info(f"Generator config changed to: {gen_config}")

            return True
        except:
            logger.info(
                'illegal instruction, treated as normal conversation. ')
    return False