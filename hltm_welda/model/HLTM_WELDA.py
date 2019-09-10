import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
import hltm_welda.model.hltm_welda.hltm_welda.model._lda as _lda
import json
from hltm_welda.model.Vocabulary import Vocabulary
from hltm_welda.model.Database import Database
from scipy.spatial import cKDTree


class HLTM_WELDA(object):
    """
    This object implements human-in-the-loop topic modeling with the Word
    Vector Latent Dirichlet Allocation (WELDA) model. WELDA is a topic model
    based on LDA that uses a pre-trained word vector embedding to enhance
    the topic model. The human-in-the-loop aspect of the model allows for
    a person to interject feedback in real-time to the model in a way that
    modifies either the underlying data or the prior distribution of the model.

	params
	--
		corpus_raw:: list ::
			list holding raw text of the corpus, each entry is a document
		corpus:: list ::
			list holding parsed text of the corpus, each entry a list of strings
		K:: int ::
			initial number of topics of the model
		alpha_init:: int ::
			initial value to use in matrix of document-topic priors
		eta_init:: int ::
			initial value to use in matrix of topic-word priors
		w2v_dim_red_dict:: dict ::
			dictionary whose keys are tokens in the (parsed) corpus, and value
            is the corresponding vector obtained from applying a dimensionality
            reduction algorithm like PCA or UMAP to the word vector
            corresponding to the token
		w2v_id2token:: dict ::
			dictionary whose keys are integers that are the id's of the words
            with respect to the word2vec mapping, and values are the word token
		w2v_token2id:: dict ::
			dictionary whose keys are word tokens from the word2vec corpus, and
            values are the id's of that token
        kdtree:: scipy.spatial.cKDTree ::
            nearest neighbors kd-tree used for nearest neighbors lookup. points
            here are vectors from word2vec. lookup returns index of word2vec token
        iterations_init:: int ::
            how many times to iterate over the corpus when initially training model
        random_state:: int ::
            integer seed used for producing a random state
    """
    def __init__(
        self,
        corpus_raw: np.ndarray,
        corpus: np.ndarray,
        K: int,
        alpha_init: float,
        eta_init: float,
        w2v_dim_red_dict: dict,
        w2v_id2token: dict,
        w2v_token2id: dict,
        kdtree: cKDTree,
        iterations_init: int,
        random_state: int,
    ):
        self._SAVES_DIR = 'hltm_welda/model/data/saves'
        self._corpus_raw_orig = corpus_raw.copy()
        self._corpus_orig = corpus.copy()
        self._w2v_dim_red_dict = w2v_dim_red_dict
        self._w2v_id2token = w2v_id2token
        self._w2v_token2id = w2v_token2id
        self._kdtree = kdtree
        self.corpus_raw = corpus_raw
        self.corpus = corpus
        self.K = K # number of topics
        self.alpha_init = alpha_init
        self.eta_init = eta_init
        self.iterations_init = iterations_init
        self.random_state = random_state
        self.vocabulary = Vocabulary(corpus=corpus)
        self.D = len(self.corpus)
        self.topic_name_ddict = defaultdict(lambda:None)
        self.topic_name_map = lambda x: self.topic_name_ddict[x] or x
        self.db = Database()

        # create kdtree based on intersection of corpus vocabulary and w2v vocabulary
        # creates self.intersection_vocab
        #         self.intersection_id2token
        #         self.intersection_token2id
        #         self.intersection_kdtree
        self.create_intersection_kdtree()

        # create priors
        self.alpha = np.full(
            shape=(self.D, self.K),
            fill_value=self.alpha_init,
            dtype='float',
        ) # document topic prior

        self.eta = np.full(
            shape=(self.K, self.vocabulary.W),
            fill_value=self.eta_init,
            dtype='float',
        ) # topic word prior

        self.token_count = 0
        for doc in self.corpus:
            self.token_count += len(doc)

        self.Nt = np.zeros(shape=self.token_count, dtype='int')
        self.Nw = np.zeros(shape=self.token_count, dtype='int')
        self.Nw_copy = np.zeros(shape=self.token_count, dtype='int')
        self.Nd = np.zeros(shape=self.token_count, dtype='int')

        # fill Nw, Nw_copy and Nd
        token_index = 0
        for doc_idx, doc in enumerate(self.corpus):
            for word_idx, word_id in enumerate(doc):
                self.Nw[token_index] = self.vocabulary.token2id[word_id]
                self.Nw_copy[token_index] = self.vocabulary.token2id[word_id]
                self.Nd[token_index] = doc_idx
                token_index += 1


    def __repr__(self):
        return (
            f"hltm_welda("
            f"corpus_raw={self.corpus_raw[:2]},"
            f"corpus={self.corpus[:2]},"
            f"K={self.K},"
            f"alpha_init={self.alpha_init},"
            f"eta_init={self.eta_init},"
            f"w2v_dim_red_dict={self.w2v_dim_red_dict},"
            f"w2v_id2token={self.w2v_id2token},"
            f"w2v_token2id={self.w2v_token2id},"
            f"kdtree={self.kdtree},"
            f"iterations_init={self.iterations_init},"
            f"random_state={self.random_state})"
        )

    @property
    def SAVES_DIR(self):
        return self._SAVES_DIR

    @property
    def corpus_raw_orig(self):
        return self._corpus_raw_orig

    @property
    def corpus_orig(self):
        return self._corpus_orig

    @property
    def w2v_dim_red_dict(self):
        return self._w2v_dim_red_dict

    @property
    def w2v_id2token(self):
        return self._w2v_id2token

    @property
    def w2v_token2id(self):
        return self._w2v_token2id

    @property
    def kdtree(self):
        return self._kdtree


    def create_intersection_vocab(self):
        """
        creates set containing vocabulary words that are in both the corpus
        and word2vec dictionary
        """
        self.intersection_vocab = self.vocabulary.vocab.intersection(set(self.w2v_dim_red_dict.keys()))


    def create_intersection_dicts(self):
        """
        creates dictionaries mapping
            id -> token
            token -> id
        for words in the set intersection_vocab
        """
        self.create_intersection_vocab()

        self.intersection_id2token = {
            id_: token
            for id_, token in
            enumerate(sorted(list(self.intersection_vocab)))
        }

        self.intersection_token2id = {
            token: id_
            for id_, token in
            self.intersection_id2token.items()
        }


    def create_intersection_kdtree(self):
        """
        creates kdtree based off of words in set intersection_vocab
        """
        self.create_intersection_dicts()

        intersection_vocab_sorted_vects = np.zeros(
            shape=(
                len(self.intersection_token2id),
                2
            ),
            dtype=np.float64,
        )

        for index, word in self.intersection_id2token.items():
            intersection_vocab_sorted_vects[index] = self.w2v_dim_red_dict[word]

        self.intersection_kdtree = cKDTree(
            data=intersection_vocab_sorted_vects,
            leafsize=16,
            compact_nodes=True,
            balanced_tree=True,
        )


    def save_token_maps(
        self,
        save_id: int,
    ):
        """
        saves contents of Nw, Nw_copy, Nt, and Nd to database

        params
        --
            save_id:: int :: integer identifying the entry in the save table
        """
        save_id_col = np.full(
            shape=self.Nd.shape,
            fill_value=save_id,
            dtype='int'
        )
        token_maps = list(
            zip(
                save_id_col.tolist(),
                self.Nd.tolist(),
                self.Nw.tolist(),
                self.Nw_copy.tolist(),
                self.Nt.tolist()
            )
        )
        self.db.persist_token_maps_by_save_id(
            save_id=save_id,
            token_maps=token_maps
        )


    def save_vocab(
        self,
        save_id: int,
    ):
        """
        saves the contents of vocabulary.token2id

        params
        --
            save_id:: int :: integer identifying the entry in the save table
        """
        words = [(save_id, int(key), value) for key, value in self.vocabulary.id2token.items()]
        self.db.persist_vocab_by_save_id(save_id=save_id, words=words)


    def save_alpha(
        self,
        save_id: int,
    ):
        """
        saves the contents of alpha

        params
        --
            save_id:: int :: integer identifying the entry in the save table
        """
        alpha_col = self.alpha.reshape(-1)
        save_id_col = np.full(shape=alpha_col.shape, fill_value=save_id, dtype='int')
        alpha_save = list(zip(save_id_col.tolist(), alpha_col.tolist()))
        self.db.persist_alpha_by_save_id(save_id=save_id, alpha=alpha_save)


    def save_eta(
        self,
        save_id: int,
    ):
        """
        saves the contents of eta

        params
        --
            save_id:: int :: integer identifying the entry in the save table
        """
        eta_col = self.eta.reshape(-1)
        save_id_col = np.full(shape=eta_col.shape, fill_value=save_id, dtype='int')
        eta_save = list(zip(save_id_col.tolist(), eta_col.tolist()))
        self.db.persist_eta_by_save_id(save_id=save_id, eta=eta_save)


    def save_topic_names(
        self,
        save_id: int,
    ):
        """
        saves the topic names

        params
        --
            save_id:: int :: integer identifying the entry in the save table
        """
        rows = [
            (save_id, topic_id, self.topic_name_map(topic_id))
            for topic_id in range(self.K)
        ]
        self.db.persist_topic_names_by_save_id(save_id=save_id, topic_names=rows)


    def save_model(
        self,
        save_name: str,
    ):
        """
        saves the model state

        params
        --
            save_name:: str :: the save name that will appear in the menu
                later for loading
        """
        save_id = self.db.create_model_record(save_name=save_name, model=self)
        self.save_token_maps(save_id=save_id)
        self.save_vocab(save_id=save_id)
        self.save_alpha(save_id=save_id)
        self.save_eta(save_id=save_id)
        self.save_topic_names(save_id=save_id)

        corpus_list = [list(doc) for doc in self.corpus]
        with open(self.SAVES_DIR + f'/{save_id}_corpus.json', 'w') as f:
            json.dump(obj=corpus_list, fp=f)

        corpus_raw_list = [list(doc) for doc in self.corpus_raw]
        with open(self.SAVES_DIR + f'/{save_id}_corpus_raw.json', 'w') as f:
            json.dump(obj=corpus_raw_list, fp=f)


    def get_save_names(self) -> list:
        """
        retrieves all save names and returns them as a list of strings
        params
        --
        """
        names = self.db.get_save_names()
        return names


    def load_token_maps_df(
        self,
        save_id: int,
    ) -> pd.DataFrame:
        """
        loads data for specified model for Nw, Nt, Nd. Returns pandas
        DataFrame with one column for each.
        params
        --
            save_id:: int :: integer specifying saved model to load
        """
        token_maps_df = self.db.get_token_maps_by_save_id(save_id=save_id)
        return token_maps_df


    def load_vocab_dict(
        self,
        save_id: int,
    ) -> dict:
        """
        loads dictionary of vocabulary for specified model
        params
        --
            save_id:: int :: integer specifying saved model to load
        """
        vocab_dict = self.db.get_vocabulary_by_save_id(save_id=save_id)
        return vocab_dict


    def load_alpha_stacked(
        self,
        save_id: int,
    ) -> np.ndarray:
        """
        load stacked version of alpha (document-topic priors). Return numpy array
        params
        --
            save_id:: int :: integer specifying saved model to load
        """
        alpha_stacked = self.db.get_alpha_by_save_id(save_id=save_id)
        return alpha_stacked


    def load_eta_stacked(
        self,
        save_id: int,
    ) -> np.ndarray:
        """
        load stacked version of eta (topic-word priors). Return numpy array
        params
        --
            save_id:: int :: integer specifying saved model to load
        """
        eta_stacked = self.db.get_eta_by_save_id(save_id=save_id)
        return eta_stacked


    def load_topic_names_dict(
        self,
        save_id: int,
    ) -> dict:
        """
        load dictionary of topic names. Return dictionary
        params
        --
            save_id:: int :: integer specifying saved model to load
        """
        topic_names_dict = self.db.get_topic_names_by_save_id(save_id=save_id)
        return topic_names_dict


    def load_corpus(
        self,
        save_id: int,
    ):
        """
        load corpus from save file. Returns numpy array.
        params
        --
            save_id:: int :: integer specifying saved model to load
        """
        with open(self.SAVES_DIR + f'/{save_id}_corpus.json', 'r') as f:
            corpus = json.load(fp=f)

        corpus = np.array([np.array(doc) for doc in corpus])
        return corpus


    def load_corpus_raw(
        self,
        save_id: int,
    ):
        """
        loads raw corpus from save file. Return list
        params
        --
            save_id:: int :: integer specifying saved model to load
        """
        with open(self.SAVES_DIR + f'/{save_id}_corpus_raw.json', 'r') as f:
            corpus_raw = json.load(fp=f)

        return corpus_raw


    def load_model(
        self,
        save_name: str,
    ):
        """
        loads all data related to a model from a specified save name.
        params
        --
            save_name:: str :: name of saved model to load
        """
        save_dict = self.db.get_save_record_by_name(save_name=save_name)
        save_id = save_dict['id']

        token_maps_df = self.load_token_maps_df(save_id=save_id)
        vocab_dict = self.load_vocab_dict(save_id=save_id)
        alpha_stacked = self.load_alpha_stacked(save_id=save_id)
        eta_stacked = self.load_eta_stacked(save_id=save_id)
        topic_names_dict = self.load_topic_names_dict(save_id=save_id)
        corpus = self.load_corpus(save_id=save_id)
        corpus_raw = self.load_corpus_raw(save_id=save_id)

        save_name = save_dict['name']
        K = save_dict['K']
        alpha_init = save_dict['alpha_init']
        eta_init = save_dict['eta_init']
        random_state = save_dict['random_state']
        Nd = token_maps_df['d'].values
        Nw = token_maps_df['w'].values
        Nw_copy = token_maps_df['w_cp'].values
        Nt = token_maps_df['t'].values
        D = token_maps_df['d'].nunique()
        W = token_maps_df['w'].nunique()
        alpha = alpha_stacked.reshape(D, K)
        eta = eta_stacked.reshape(K, W)

        # reset values of model
        self.corpus_raw = corpus_raw
        self.corpus = corpus
        self.K = K
        self.alpha_init = alpha_init
        self.eta_init = eta_init
        self.random_state = random_state
        self.Nd = Nd
        self.Nw = Nw
        self.Nw_copy = Nw_copy
        self.Nt = Nt
        self.alpha = alpha
        self.eta = eta

        self.vocabulary.token2id = vocab_dict
        self.vocabulary.update_from_token2id()
        self.D = D

        for key, value in topic_names_dict.items():
            self.topic_name_ddict[key] = value

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def name_topic(
        self,
        topic: int,
        name: str,
    ):
        """
        assign a user specified name to a chosen topic
        params
        --
            topic:: int :: topic id chosen by user
            name:: str :: name to be assigned to topic
        """
        self.topic_name_ddict[topic] = name


    def re_initialize_corpus(self):
        """
        loads original corpus and raw corpus
        params
        --
        """
        self.corpus_raw = self.corpus_raw_orig
        self.corpus = self.corpus_orig


    def re_initialize_priors(self):
        """
        re-initializes priors alpha and eta to the internal initial values
        params
        --
        """
        self.alpha = np.full(
            shape=(self.D, self.K),
            fill_value=self.alpha_init,
            dtype='float',
        ) # document topic prior

        self.eta = np.full(
            shape=(self.K, self.vocabulary.W),
            fill_value=self.eta_init,
            dtype='float',
        ) # topic word prior


    def init_random_topic_assign(self):
        """
        initializes Nt with random topic assignments
        params
        --
        """
        K = self.K
        Nt_len = len(self.Nt)
        self.Nt = np.random.randint(low=0, high=K, size=Nt_len, dtype='int')

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def form_wt(self):
        """
        forms the word-topic count matrix
        params
        --
        """
        # place these here (as opposed to as arguments with defaults) because
        # "Default argument values are evaluated at function define-time, but
        # self is an argument only available at function call time."
        # https://stackoverflow.com/questions/1802971/nameerror-name-self-is-not-defined
        Nw = self.Nw
        Nt = self.Nt
        K = self.K
        W = self.vocabulary.W

        nzw = _lda.form_wt(
            Nt_in=Nt,
            Nw_in=Nw,
            K=K,
            W=W,
        )

        # make sure the cython function did not give back error message (string)
        assert type(nzw) == np.ndarray, f'type(nzw) != numpy array. nzw: {nzw}'

        self.wt = nzw

    def form_wt_copy(self):
        """
        forms copy of word-topic count matrix based on Nw_copy; the array of
        word tokens with some words replaced based on WELDA algorithm.
        params
        --
        """
        Nw = self.Nw_copy
        Nt = self.Nt
        K = self.K
        W = self.vocabulary.W

        nzw = _lda.form_wt(
            Nt_in=Nt,
            Nw_in=Nw,
            K=K,
            W=W,
        )

        # make sure the cython function did not give back error message (string)
        assert type(nzw) == np.ndarray, f'type(nzw) != numpy array. nzw: {nzw}'

        self.wt_copy = nzw


    def form_dt(self):
        """
        forms the topic-document count matrix.
        params
        --
        """
        K = self.K

        ndz = _lda.form_dt(
            Nt_in=self.Nt,
            Nd_in=self.Nd,
            K=K,
        )

        assert type(ndz) == np.ndarray, f'type(ndz) != numpy array. ndz: {ndz}'

        self.dt = ndz


    def create_topic_wv_dict(
        self,
        top_n_words: int=50,
    ) -> dict:
        """
        create dictionary of word vectors for a specified topic based in
        specified number of top words for that topic.
        params
        --
            top_n_words:: int :: number of top words to use for topic
        """
        self.form_wt()
        # self.form_wt_copy()
        self.calc_theta()

        K = self.K
        id2token = self.vocabulary.id2token
        w2v_dim_red_dict = self.w2v_dim_red_dict
        # nzw = self.wt
        phi = self.phi

        topic_wv_dict = {
            topic: {}
            for topic in range(K)
        }

        for topic in topic_wv_dict.keys():
            top_n_umap_wv = np.array(
                [
                    w2v_dim_red_dict[word]
                    for word in
                    [
                        id2token[id_] for id_ in
                        phi[topic].argsort()[:-top_n_words:-1]
                    ]
                    if word in w2v_dim_red_dict.keys()
                ]
            )

            topic_wv_dict[topic]['mean'] = top_n_umap_wv.mean(axis=0)
            topic_wv_dict[topic]['cov'] = np.cov(top_n_umap_wv.T)

        return topic_wv_dict


    def replace_words(
        self,
        topic_wv_dict: dict,
        Nw: np.ndarray,
        Nt: np.ndarray,
        K: int,
        kdtree: cKDTree,
        token2id: dict,
        w2v_id2token: dict,
        lambda_: float=0.2,
    ) -> np.ndarray:
        """
        randomly replaces word tokens in Nw with words chosen based on topic
        from the WELDA algorithm.
        params
        --
            topic_wv_dict:: dict :: dictionary with keys topic ids and values
                vectors of top n words for that topic
            Nw:: np.ndarray :: numpy array of word tokens
            Nt:: np.ndarray :: numpy array of topic assignments
            K:: int :: number of topics in current model
            kdtree:: scipy.spatial.cKDTree :: kdtree based on word vectors
            token2id:: dict :: dictionary with keys tokens in the vocabulary and
                values their ids
            w2v_id2token:: dict :: dictionary with keys ids of tokens for the
                word2vec model, and values the token
            lambda_:: float :: probability that a word will be chosen to be replaced
        """
        replace_flags = np.random.binomial(n=1, p=lambda_, size=len(Nw))
        replace_flags_bool = replace_flags.astype(bool)

        for topic in range(K):
            index_to_replace = np.where((replace_flags_bool) & (Nt == topic))[0]

            samples = np.random.multivariate_normal(
                mean=topic_wv_dict[topic]['mean'],
                cov=topic_wv_dict[topic]['cov'],
                size=len(index_to_replace),
            )

            new_word_indexes_w2v = kdtree.query(samples)[1]
            new_words = [w2v_id2token[index] for index in new_word_indexes_w2v]
            new_word_indexes = np.array([token2id[word] for word in new_words])
            Nw[index_to_replace] = new_word_indexes

        return Nw


    def fit_cython(
        self,
        Nw: np.ndarray,
        wt: np.ndarray,
        iterations: int,
    ):
        """
        Fits the LDA model given the internal state of Nt, Nw, Nd and the
        priors alpha and eta. Calls cython function to fit.

        params
        --
            Nw:: np.array :: 1D numpy array holding word token ids
            wt:: np.array :: 2D numpy array holding word-topic counts
            iterations:: int :: how many iterations the model should go through
        """
        self.form_dt()
        self.form_wt()
        self.form_wt_copy()

        Nt_arr = _lda.fit(
            iterations=iterations,
            Nt_in=self.Nt,
            Nw_in=Nw,
            Nd_in=self.Nd,
            dt=self.dt,
            wt=wt,
            alpha=self.alpha,
            eta=self.eta,
            K=self.K,
        )

        if type(Nt_arr) != np.ndarray:
            print(Nt_arr)

        self.Nt = Nt_arr

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_phi_copy()
        self.calc_theta()


    def calc_theta(self):
        """
        calculates topic-document posterior distribution matrix

        params
        --
        """
        self.form_dt()
        self.theta = (self.dt+self.alpha) / (self.dt+self.alpha).sum(
            axis=1
        ).astype(float).reshape(-1, 1)


    def calc_phi(self):
        """
        calculates word-topic posterior distribution matrix

        params
        --
        """
        self.form_wt()
        self.phi = (self.wt + self.eta) / (self.wt + self.eta).sum(
            axis=1
        ).astype(float).reshape(-1, 1)


    def calc_phi_copy(self):
        """
        calculates copy of word-topic posterior distribution matrix

        params
        --
        """
        self.form_wt_copy()
        self.phi_copy = (self.wt_copy + self.eta) / (self.wt_copy + self.eta).sum(
            axis=1
        ).astype(float).reshape(-1, 1)


    def add_to_stopwords(
        self,
        sw: str,
    ):
        """
        removes a word from the corpus

        params
        --
            sw:: string :: token of the word to be removed from the corpus
        """
        sw_id = self.vocabulary.token2id[sw]

        # drop columns from eta prior
        self.eta = np.delete(arr=self.eta, obj=sw_id, axis=1)

        self.Nt, self.Nw, self.Nw_copy, self.Nd, doc_lengths = _lda.remove_word(
            Nt_in=self.Nt,
            Nw_in=self.Nw,
            Nw_copy_in=self.Nw_copy,
            Nd_in=self.Nd,
            sw_idx=sw_id,
        )

        self.vocabulary.remove_by_token(sw)
        effaced_doc_indexes = np.where(doc_lengths == 0)[0]

        if len(effaced_doc_indexes) > 0:
            self.alpha = np.delete(arr=self.alpha, obj=effaced_doc_indexes, axis=0)

        # remove documents from corpus
        self.corpus = np.delete(arr=self.corpus, obj=effaced_doc_indexes, axis=0)
        self.corpus_raw = np.delete(arr=self.corpus_raw, obj=effaced_doc_indexes, axis=0)
        self.D = len(np.unique(self.Nd))

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def remove_doc_from_corpus(
        self,
        doc_rmv_idx: int,
    ):
        """
        removes a document from the corpus

        params
        --
            doc_rmv_idx:: int :: index of document to remove from corpus
        """
        # remove document from corpus
        self.corpus = np.delete(arr=self.corpus, obj=doc_rmv_idx, axis=0)
        self.corpus_raw = np.delete(arr=self.corpus_raw, obj=doc_rmv_idx, axis=0)

        # remove document from alpha
        self.alpha = np.delete(arr=self.alpha, obj=doc_rmv_idx, axis=0)

        self.Nt, self.Nw, self.Nw_copy, self.Nd, word_counts = _lda.remove_doc(
            Nt_in=self.Nt,
            Nw_in=self.Nw,
            Nw_copy_in=self.Nw_copy,
            Nd_in=self.Nd,
            doc_idx=doc_rmv_idx,
        )

        effaced_word_indexes = np.where(word_counts == 0)[0]

        if len(effaced_word_indexes) > 0:
            self.eta = np.delete(
                arr=self.eta,
                obj=effaced_word_indexes,
                axis=1
            )

        effaced_word_indexes_set = set(effaced_word_indexes.tolist())

        num_popped_words = 0
        for word, word_idx in list(self.vocabulary.token2id.items()):
            self.vocabulary.token2id[word] -= num_popped_words
            if word_idx in effaced_word_indexes_set:
                self.vocabulary.token2id.pop(word)
                num_popped_words += 1

        self.vocabulary.update_from_token2id()
        self.D = len(np.unique(self.Nd))

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def merge_topics(
        self,
        topic1: int,
        topic2: int,
    ):
        """
        merges two specified topics. assigns all occurrences one topic to the other

        params
        --
            topic1:: int :: index specifying one topic to merge
            topic1:: int :: index specifying one topic to merge
        """
        topics = np.array([topic1, topic2])
        parent_topic = min(topics)
        child_topic = max(topics)

        if self.topic_name_ddict[parent_topic] is None and self.topic_name_ddict[child_topic] is not None:
            self.topic_name_ddict[parent_topic] = self.topic_name_ddict[child_topic]

        for topic in range(child_topic+1, self.K):
            self.topic_name_ddict[topic-1] = self.topic_name_ddict[topic]

        self.topic_name_ddict.pop(self.K, None)

        self.Nt[np.where(self.Nt == child_topic)] = parent_topic
        self.Nt[np.where(self.Nt > child_topic)] -= 1

        # decrement K
        self.K = self.K - 1

        # average child and parent rows and put into parent row and delete child topic row from eta
        self.eta[parent_topic] = (self.eta[parent_topic] + self.eta[child_topic])/2.0
        self.eta = np.delete(arr=self.eta, obj=child_topic, axis=0)

        # average child and parent columns and put into parent column, and
        # delete child topic column from alpha
        self.alpha[:,parent_topic] = (self.alpha[:,parent_topic] + self.alpha[:,child_topic])/2.0
        self.alpha = np.delete(arr=self.alpha, obj=child_topic, axis=1)

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def split_topic(
        self,
        topic_to_split: int,
        new_topic_seed_words: list,
    ):
        """
        creates new topic using specified words (seed words) from a specified topic

        params
        --
            topic_to_split:: int :: topic id to split into two topics
            new_topic_seed_words:: list :: list of strings containing tokens of
                words to seed new topic
        """
        # increment K
        self.K += 1

        # find ids of seed words
        new_topic_seed_words_id = np.array([
            self.vocabulary.token2id[w]
            for w in new_topic_seed_words
        ])

        idx_replace = (self.Nt == new_topic_seed_words_id.reshape(-1, 1)).any(axis=0)
        self.Nt[idx_replace] = np.random.randint(
            low=0,
            high=self.K,
            size=idx_replace.sum(),
            dtype='int',
        )

        # add column to alpha (document-topic prior): one new column for new topic
        self.alpha = np.concatenate(
            (
                self.alpha,
                np.full(
                    shape=self.alpha[:,0].reshape(-1, 1).shape,
                    fill_value=self.alpha_init
                )
            ),
            axis=1
        )

        # row to add to eta for new topic
        new_topic_word_row = np.full(
            shape=self.eta[0,:].reshape(1, -1).shape,
            fill_value=self.eta_init
        )

        # set large prior for index of seed words in new topic word row
        new_topic_word_row[0][new_topic_seed_words_id] = 1.0

        # add row to eta with large priors for seed words in new topic
        self.eta = np.concatenate(
            (
                self.eta,
                new_topic_word_row
            ),
            axis=0
        )

        self.fit_cython(
            iterations=1,
            Nw=self.Nw_copy,
            wt=self.wt_copy,
        )

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def add_word_to_topic(
        self,
        topic: int,
        word: str,
    ):
        """
        forgets topic assignment for all occurrences of a specified word and
        changes prior for that word to strongly prefer specified topic. then
        resamples topics for that word from a multinomial.

        params
        --
            topic:: int :: index specifying topic
            word:: string :: token for which the prior distribution will change
        """
        word_id = self.vocabulary.token2id[word]

        # increase the chosen words prior in the chosen topic
        self.eta[topic, word_id] \
        = np.abs(self.wt[topic].max() - self.wt[topic, word_id]) # should be positive, but
                                                                 # use np.abs() just in case

        idx_replace = np.where(self.Nw == word_id)[0]
        pvals = self.eta[:,word_id]
        pvals = pvals/pvals.sum()
        self.Nt[idx_replace] = np.random.multinomial(
            n=1,
            pvals=pvals,
            size=len(idx_replace),
        ).argmax(axis=1)

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def remove_word_from_topic(
        self,
        topic: int,
        word: str,
    ):
        """
        forgets topic assignment for all occurrences of a specified word and
        changes prior for that word to strongly avoid specified topic. then
        resamples topics for that word from a multinomial.

        params
        --
            topic:: int :: index specifying topic
            word:: string :: token for which the prior distribution will change
        """
        word_id = self.vocabulary.token2id[word]

        # set prior for word in topic to very small value
        self.eta[topic, word_id] = 1e-8

        idx_replace = np.where(self.Nw == word_id)[0]
        pvals = self.eta[:,word_id]
        pvals = pvals/pvals.sum()
        self.Nt[idx_replace] = np.random.multinomial(
            n=1,
            pvals=pvals,
            size=len(idx_replace),
        ).argmax(axis=1)

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def remove_doc_from_topic(
        self,
        topic: int,
        doc_idx: int,
    ):
        """
        forgets topic assignment for all occurrences of a specified document and
        changes prior for that document to strongly avoid specified topic. then
        resamples topics for that document from a multinomial.

        params
        --
            topic:: int :: index specifying topic
            doc_idx:: int :: document id for which the prior distribution will change
        """
        # set prior for document in topic to very small value
        self.alpha[doc_idx, topic] = 1e-8

        idx_replace = np.where(self.Nd == doc_idx)[0]
        pvals = self.alpha[doc_idx]
        pvals = pvals/pvals.sum()
        self.Nt[idx_replace] = np.random.multinomial(
            n=1,
            pvals=pvals,
            size=len(idx_replace),
        ).argmax(axis=1)

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def add_doc_to_topic(
        self,
        topic: int,
        doc_idx: int,
    ):
        """
        forgets topic assignment for all occurrences of a specified document and
        changes prior for that document to strongly prefer specified topic. then
        resamples topics for that document from a multinomial.

        params
        --
            topic:: int :: index specifying topic
            doc_idx:: int :: document id for which the prior distribution will change
        """
        # set prior for document in topic
        # similar to adding word to topic
        self.alpha[doc_idx, topic] = np.abs(self.dt[:,topic].max() - self.dt[doc_idx, topic])

        idx_replace = np.where(self.Nd == doc_idx)[0]
        pvals = self.alpha[doc_idx]
        pvals = pvals/pvals.sum()
        self.Nt[idx_replace] = np.random.multinomial(
            n=1,
            pvals=pvals,
            size=len(idx_replace),
        ).argmax(axis=1)

        self.form_wt()
        self.form_wt_copy()
        self.form_dt()
        self.calc_phi()
        self.calc_theta()


    def get_top_n_words_for_topic(
        self,
        topic: int,
        n: int,
    ) -> list:
        """
        gets the top n words for specified topic. Returns a list of dictionaries
        with word information.
        params
        --
            topic:: int :: integer specifying topic
            n:: int :: integer specifying how many words to return
        """
        self.form_wt()
        self.form_wt_copy()
        self.calc_phi()
        word_ids = self.phi[topic].argsort()[:-n:-1]
        top_n_words = [
            {
                't': topic,
                'wid': wid.item(),
                'w': self.vocabulary.id2token[wid]
            }
            for wid in word_ids
        ]
        return top_n_words


    def get_top_n_docs_for_topic(
        self,
        topic: int,
        n: int,
    ) -> list:
        """
        gets the top n documents for specified topic. Returns a list of dictionaries
        with document information.
        params
        --
            topic:: int :: integer specifying topic
            n:: int :: integer specifying how many documents to return
        """
        self.calc_theta()
        doc_ids = self.theta[:,topic].argsort()[:-n:-1]
        if type(self.corpus_raw[0][0]) == dict:
            # print('get_top_n_docs_for_topic case of dictionary')
            top_n_docs = [
                {
                    't': topic,
                    'docid': doc_id.item(),
                    'd': list(
                        filter(
                            lambda utterance: utterance['speaker'] != 'sys',
                            self.corpus_raw[doc_id]
                        )
                    )
                }
                for doc_id in doc_ids
            ]
        else:
            top_n_docs = [
                {
                    't': topic,
                    'docid': doc_id.item(),
                    'd': self.corpus_raw[doc_id]
                }
                for doc_id in doc_ids
            ]
        return top_n_docs


    def get_num_docs_per_topic(self) -> dict:
        """
        calculates and returns the number of documents per topic.
        params
        --
        """
        from collections import Counter
        self.form_dt()
        self.calc_theta()
        topics_num_docs = Counter(self.theta.argsort(axis=1)[:,-1])

        docs_per_topic = {
            key.item(): value
            for key, value in sorted(topics_num_docs.items(), key=lambda x: x[1])
        }
        return docs_per_topic
