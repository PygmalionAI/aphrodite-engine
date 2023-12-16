import collections
import functools

class TokenIndex:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # map id -> token str (including whitespaces)
        self.norm_vocab = {}
        for token_id in tokenizer.vocab.values():
            norm_token = tokenizer.decode([tokenizer.bos_token_id, token_id])[
                len(tokenizer.bos_token):]
        
        # get index allowing efficient retrieval of valid tokens,
        # given a sequence

        # given tokens ["art", "artist", "argument", "alice"]
        # map "a" -> ["ar", "al"]
        # map "ar" -> ["art", "artist"]
        # map "art" -> [None, "artist"] (None indicates no match)
        self.char_map = collections.defaultdict(set)
        for word in self.norm_vocab:
            for i in range(1, len(word) + 1):
                prefix = word[:i]
                if i < len(word):
                    self.char_map[prefix].add(word[i])
                else:
                    # Add None for complete matches
                    self.char_map[prefix].add(None)
    
    def get_valid_next_charset(self, seq, legal_chars):
        results = set(self.char_map[seq]) & legal_chars
        return results
    
    def is_token(self, tok):
        return tok in self.norm_vocab

