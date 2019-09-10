class Vocabulary(object):
    def __init__(self, corpus):
        self._corpus = corpus

        self.vocab = set()
        for doc in corpus:
            self.vocab.update(doc)

        self.W = len(self.vocab)

        vocab_list = sorted(list(self.vocab))

        self.id2token = {}
        self.token2id = {}

        for idx, token in enumerate(vocab_list):
            self.id2token[idx] = token
            self.token2id[token] = idx


    @property
    def corpus(self):
        return self._corpus


    def form_id2token(self):
        self.id2token = {
            value: key
            for key, value in self.token2id.items()
        }


    def update_from_token2id(self):
        self.form_id2token()
        self.vocab = set(list(self.token2id.keys()))
        self.W = len(self.vocab)


    def remove_by_token(self, token_rmv):
        id_rmv = self.token2id.pop(token_rmv)

        # decrement value of token2id's values appropriately
        for key, value in self.token2id.items():
            if value > id_rmv:
                self.token2id[key] -= 1

        self.form_id2token()

        self.vocab = set(list(self.token2id.keys()))
        self.W = len(self.vocab)


    def remove_by_id(self, id_rmv):
        token_rmv = self.id2token[id_rmv]
        self.remove_by_token(token_rmv=token_rmv)
