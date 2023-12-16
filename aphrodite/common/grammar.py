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


class EpsilonNFA:
    """Traverses a Character-Based Epsilon-NFA.
    Used to find the next valid token given a sequence of tokens.
    
    self.nfa (dict): A dictionary representing the NFA. It includes:
        - 'states' (list): A list of states (UUID) in the NFA.
        - 'initial_state' (UUID or any hashable ID): The initial state.
        - 'final_states' (list): A list of final or accepting states (UUID).
        - 'alphabets' (list): The set of input symbols (characters).
        - 'transition_function' (dict): A dictionary representing the state
            transitions. Each key is a state (UUID), and its value is another
            dictionary mapping input symbols to list of next states (UUIDs).
    
    self.nfa should never be mutated.
    """

    def __init__(self, nfa):
        self.nfa = nfa

        # set of states you may be in
        self.current_states = set([nfa['initial_state']])
        self.current_str = ""

        self.legal_chars = set(
            [char for char in self.nfa['alphabets'] if char != "$"])
        self._resolved_epsilon_cache = {}

    def step_seq(self, seq):
        for char in seq:
            self.step(char)

    def step(self, char):
        """Updates the canonical state."""
        next_states = self.next_char_state_map()[char]
        if not next_states:
            raise ValueError(
                f"Illegal transition from '{self.current_str}', "
                f"no next state for '{char}'")
        self.current_states = next_states
        self.current_str += char

    def simulate_step(self, chars, state_step=None):
        """Return map of chars and their resulting next state-set
        given a current state-set and new chars."""
        if state_step is None:
            state_step = self.current_states
        state_map = self.next_char_state_map(state_set)
        return {tok: state_map[tok] for tok in chars if tok in state_map}

    def copy(self):
        new_nfa = EpsilonNFA(self.nfa)
        new_nfa.current_states = self.current_states
        new_nfa.current_str = self.current_str
        new_nfa._resolved_epsilon_cache = self._resolved_epsilon_cache
        return new_nfa

    @property
    def allow_stop_next(self):
        return None in self.next_char_state_map()

    def next_char_state_map(self, current_states=None):
        """Creates a mapping of possible next characters to a set of
        valid states for each character.
        """
        if current_states is None:
            current_states = self.current_states
        
        char_to_states = collections.defaultdict(set)

        if bool(current_states & set(self.nfa["final_states"])):
            char_to_states[None] = None
        
        for state in self._resolve_epsilon_closure(current_states):
            for char, next_states in self.nfa["transition_function"][state].items():
                if next_states and char != "$":
                    char_to_states[char].update(next_states)
        
        return char_to_states

    def _resolve_epsilon_closure(self, states):
        closure = set()
        for state in states:
            if state in self._resolved_epsilon_cache:
                new_closures = self._resolved_epsilon_cache[state]
            else:
                new_closures = self._get_epsilon_closure(state)
                self._resolved_epsilon_cache[state] = new_closures
            closure.update(self._get_epsilon_closure(state))
        return closure

    def _get_epsilon_closure(self, state, visited=None):
        if visited is None:
            visited = set()

        stack = [state]
        while stack:
            current_state = stack.pop()
            if current_state not in visited:
                visited.add(current_state)
                stack.extend(self.nfa["transition_function"][current_state].get("$", []))
        return visited

