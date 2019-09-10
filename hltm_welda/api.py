from hltm_welda import app
from hltm_welda.model.HLTM_WELDA import HLTM_WELDA
import pickle
from flask.json import jsonify
from flask import request
import numpy as np


with open('hltm_welda/model/data/newsgroups_preprocessed_data.pickle', 'rb') as f:
    corpus = pickle.load(f)

with open('hltm_welda/model/data/newsgroups_raw_data.pickle', 'rb') as f:
    corpus_raw = pickle.load(f)

with open('hltm_welda/model/data/w2v_wikipedia_pca_dict.pickle', 'rb') as f:
    w2v_dim_red_dict = pickle.load(f)

with open('hltm_welda/model/data/w2v_wikipedia_id2token.pickle', 'rb') as f:
    w2v_id2token = pickle.load(f)

with open('hltm_welda/model/data/w2v_wikipedia_token2id.pickle', 'rb') as f:
    w2v_token2id = pickle.load(f)

with open('hltm_welda/model/data/cKDTree_w2v_wikipedia_pca.pickle', 'rb') as f:
    kdtree = pickle.load(f)


lda = HLTM_WELDA(
    corpus_raw=corpus_raw,
    corpus=corpus,
    K=8,
    alpha_init=10.,
    eta_init=0.01,
    w2v_dim_red_dict=w2v_dim_red_dict,
    w2v_id2token=w2v_id2token,
    w2v_token2id=w2v_token2id,
    kdtree=kdtree,
    iterations_init=100,
    random_state=42
)


@app.route('/initialize_model')
def initialize():
    '''
    Initializes LDA model.

    get:

    response:
        'model initialized!'
    '''
    lda.init_random_topic_assign()
    lda.form_wt()
    lda.fit_cython(
        Nw=lda.Nw,
        wt=lda.wt,
        iterations=lda.iterations_init,
    )
    return jsonify('model initialized!')


@app.route('/re_initialize_model')
def re_initialize():
    '''
    Re-initializes LDA model with new number of topics.

    get:
        parameters:
            - name: numInitTopics
              type: integer

        response:
            'model re-initialized!'
    '''
    K = int(request.args.get('numInitTopics'))
    lda.K = K
    lda.re_initialize_corpus()
    lda.re_initialize_priors()
    lda.init_random_topic_assign()
    lda.form_wt()
    lda.fit_cython(
        Nw=lda.Nw,
        wt=lda.wt,
        iterations=lda.iterations_init,
    )
    return jsonify('model re-initialized!')


@app.route('/iterate_model', methods=['GET'])
def iterate_model():
    '''
    Iterates model specified number of times.

    get:
        parameters:
            - name: nIter
              type: integer

        response:
            'model iterated!'
    '''
    nIter = int(request.args.get('nIter'))
    print(f'nIter: {nIter}')
    lda.form_wt_copy()
    if nIter <= 10:
        lda.fit_cython(
            Nw=lda.Nw_copy,
            wt=lda.wt_copy,
            iterations=nIter,
        )
        print('create_topic_wv_dict')
        topic_wv_dict = lda.create_topic_wv_dict(top_n_words=50)
        lda.Nw_copy = lda.replace_words(
            topic_wv_dict=topic_wv_dict,
            Nw=lda.Nw_copy,
            Nt=lda.Nt,
            K=lda.K,
            kdtree=lda.intersection_kdtree,
            token2id=lda.intersection_token2id,
            w2v_id2token=lda.intersection_id2token,
            lambda_=0.2,
        )
        lda.form_wt_copy()
        print('fit_cython')
        lda.fit_cython(
            Nw=lda.Nw_copy,
            wt=lda.wt_copy,
            iterations=nIter,
        )

    elif nIter > 10:
        for i in range(nIter//10):
            lda.form_wt_copy()
            lda.fit_cython(
                Nw=lda.Nw_copy,
                wt=lda.wt_copy,
                iterations=10,
            )
            topic_wv_dict = lda.create_topic_wv_dict(top_n_words=50)
            lda.Nw_copy = lda.replace_words(
                topic_wv_dict=topic_wv_dict,
                Nw=lda.Nw_copy,
                Nt=lda.Nt,
                K=lda.K,
                kdtree=lda.intersection_kdtree,
                token2id=lda.intersection_token2id,
                w2v_id2token=lda.intersection_id2token,
                lambda_=0.2,
            )

    return jsonify('model iterated!')


@app.route('/n_topics', methods=['GET'])
def num_topics():
    '''
    Gets number of topics in LDA model.

    get:
        parameters:

        response:
            - description: number of topics in LDA model
              type: integer
    '''
    return jsonify(lda.K)


@app.route('/n_docs', methods=['GET'])
def num_docs():
    '''
    Gets number of documents in corpus.

    get:
        parameters:

        response:
            - description: number of documents in corpus. List of one element.
              type: list
    '''
    return jsonify([lda.D])


@app.route('/topic_top_words', methods=['GET'])
def topic_top_words():
    '''
    Gets the n most relevant words for a given topic.

    get:
        - parameters:
            - name: topic
              type: integer
              description: index of topic in LDA model
            - name: n_words
              type: integer
              description: number of most relevant words for topic

        - response:
            - description: list of the top n words for topic
              type: list of strings
    '''
    topic = int(request.args.get('topic'))
    n_words = int(request.args.get('n_words'))
    top_n_words = lda.get_top_n_words_for_topic(topic=topic, n=n_words)
    return jsonify(top_n_words)


@app.route('/topic_top_docs', methods=['GET'])
def topic_top_docs():
    '''
    Gets the n most relevant documents for a given topic.

    get:
        - parameters:
            - name: topic
              type: integer
              description: index of topic in LDA model
            - name: n_docs
              type: integer
              description: number of most relevant documents for a topic

        - response:
            - description: list of the top n documents for topic
              type: list of strings
    '''
    topic = int(request.args.get('topic'))
    n_docs = int(request.args.get('n_docs'))
    top_n_docs = lda.get_top_n_docs_for_topic(topic=topic, n=n_docs)
    return jsonify(top_n_docs)



@app.route('/add_to_stopwords', methods=['GET'])
def add_to_stopwords():
    '''
    Removes all occurrences of word in corpus.

    get:
        - parameters:
            - name: word
              type: string
              description: word in vocabulary to be removed from corpus

        - response:
            - description: 'removed <word>'
              type: string
    '''
    stopword = request.args.get('word')
    lda.add_to_stopwords(sw=stopword)
    return jsonify(f'removed {stopword}')



@app.route('/remove_doc_from_corpus', methods=['GET'])
def remove_doc_from_corpus():
    '''
    Removes document from corpus.

    get:
        - parameters:
            - name: docid
              type: integer
              description: index of document in corpus

        - response:
            - description: 'removed docid <docid>'
              type: string
    '''
    docid = int(request.args.get('docid'))
    lda.remove_doc_from_corpus(docid)
    return jsonify(f'removed docid {docid}')


@app.route('/merge_topics', methods=['GET'])
def merge_topics():
    '''
    Merges two topics. Modifies LDA model and assigns topic with higher index
    to topic with lower index.

    get:
        - parameters:
            - name: topic1
              type: integer
              description: first topic index to merge
            - name: topic2
              type: integer
              description: second topic index to merge

        - response:
            - description: topic <topic2> and topic <topic2> merged
              type: string
    '''
    topic1 = int(request.args.get('topic1'))
    topic2 = int(request.args.get('topic2'))
    print(topic1, topic2)
    lda.merge_topics(topic1=topic1, topic2=topic2)
    return jsonify(f'topic {topic1} and topic {topic2} merged')


@app.route('/get_num_docs_per_topic', methods=['GET'])
def num_docs_per_topic():
    '''
    Merges two topics. Modifies LDA model and assigns topic with higher index
    to topic with lower index.

    get:
        - parameters:
            - name: topic1
              type: integer
              description: first topic index to merge
            - name: topic2
              type: integer
              description: second topic index to merge

        - response:
            - description: 'topic <topic2> and topic <topic2> merged'
              type: string
    '''
    num_docs_per_topic = lda.get_num_docs_per_topic()
    return jsonify(num_docs_per_topic)


@app.route('/split_topic', methods=['GET'])
def split_topic():
    '''
    Splits a topic in two given seed words from a given topic. Modifies LDA
    model and creates new topic using seed words supplied.

    get:
        - parameters:
            - name: topic
              type: integer
              description: topic index to split
            - name: seed_words
              type: dictionary
              description: dictionary with single key of 'words', and value
              list of dictionaries, each with one key 'w', and key word (string)
              to use as seed words for new topic.

        - response:
            - description: 'split topic <topic>'
              type: string
    '''
    # topic = int(request.args.get('topic'))
    topic = int(request.json['topic'])
    seed_words = request.json['seed_words']
    seed_words = seed_words['words']
    seed_words = [sw['w'] for sw in seed_words]
    lda.split_topic(topic_to_split=topic, new_topic_seed_words=seed_words)
    return jsonify(f'split topic {topic}')



@app.route('/remove_word_from_topic', methods=['GET'])
def remove_word_from_topic():
    '''
    Removes a word from specified topic. LDA model is modified and topic
    assignments for the word are redrawn with small prior.

    get:
        - parameters:
            - name: topic
              type: integer
              description: topic index to remove word from
            - name: word
              type: string
              description: word to remove from topic

        - response:
            - description: 'removed <word> from topic <topic>'
              type: string
    '''
    topic = int(request.args.get('topic'))
    word = request.args.get('word')
    lda.remove_word_from_topic(topic=topic, word=word)
    return jsonify(f'removed {word} from topic {topic}')



@app.route('/add_word_to_topic', methods=['GET'])
def add_word_to_topic():
    '''
    Adds a word to specified topic. LDA model is modified and topic
    assignments for the word are redrawn with large prior.

    get:
        - parameters:
            - name: topic
              type: integer
              description: topic index to add word to
            - name: word
              type: string
              description: word to add to topic

        - response:
            - description: 'added <word> to topic <topic>'
              type: string
    '''
    topic = int(request.args.get('topic'))
    word = request.args.get('word')
    lda.add_word_to_topic(topic=topic, word=word)
    return jsonify(f'added {word} to topic {topic}')



@app.route('/remove_doc_from_topic', methods=['GET'])
def remove_doc_from_topic():
    '''
    Removes a document from specified topic. LDA model is modified and topic
    assignments for the document are redrawn with small prior.

    get:
        - parameters:
            - name: topic
              type: integer
              description: topic index to remove document from
            - name: doc_idx
              type: string
              description: document index to remove from topic

        - response:
            - description: 'removed doc idx <doc_idx> from topic <topic>'
              type: string
    '''
    topic = int(request.args.get('topic'))
    doc_idx = int(request.args.get('doc_idx'))
    lda.remove_doc_from_topic(topic=topic, doc_idx=doc_idx)
    return jsonify(f'removed doc idx {doc_idx} from topic {topic}')



@app.route('/add_doc_to_topic', methods=['GET'])
def add_doc_to_topic():
    '''
    Adds a document to specified topic. LDA model is modified and topic
    assignments for the document are redrawn with large prior.

    get:
        - parameters:
            - name: topic
              type: integer
              description: topic index to add document to
            - name: doc_idx
              type: string
              description: document index to add to topic

        - response:
            - description: 'added doc idx <doc_idx> to topic <topic>'
              type: string
    '''
    topic = int(request.args.get('topic'))
    doc_idx = int(request.args.get('doc_idx'))
    lda.add_doc_to_topic(topic=topic, doc_idx=doc_idx)
    return jsonify(f'added doc idx {doc_idx} to topic {topic}')


@app.route('/get_topic_name_dict', methods=['GET'])
def get_topic_name_dict():
    '''
    Gets dictionary of topic names. Keys are topic indexes and values are
    strings of topic names.

    get:
        - parameters:

        - response:
            - description: Dictionary containing topic names. Keys are topic
              indexes and values are strings of document names.
              type: dictionary
    '''
    topic_name_dict = {
        str(key): value
        for key, value in lda.topic_name_ddict.items()
    }
    topic_name_dict = dict(list(filter(lambda tpl: tpl[1], topic_name_dict.items())))
    # print('topic_name_dict: ', topic_name_dict)
    return jsonify(topic_name_dict)


@app.route('/name_topic', methods=['GET'])
def name_topic():
    '''
    Assigns a name to a topic given a specified topic index.

    get:
        - parameters:
            - name: topic
              type: integer
              description: topic index to be named
            - name: name
              type: string
              description: name to be used for specified topic

        - response:
            - description: Dictionary containing topic names. Keys are topic
              indexes and values are strings of document names.
              type: dictionary
    '''
    topic = int(request.args.get('topic'))
    name = request.args.get('name')
    lda.name_topic(topic=topic, name=name)
    return jsonify(f'topic {topic} named {name}')



@app.route('/save_model', methods=['POST'])
def save_model():
    '''
    Saves information of LDA model.

    post:
        - parameters:
            - name: saveName
              type: string
              description: name of model to be saved (user defined)

        - response:
            - description: 'model saved!'
              type: string
    '''
    save_name = request.args.get('saveName')
    lda.save_model(save_name=save_name)
    return jsonify('model saved!')


@app.route('/get_saved_model_names', methods=['GET'])
def get_saved_model_names():
    '''
    Gets save names of previously saved models.

    get:
        - parameters:

        - response:
            - description: list of strings. List contains save names
              type: list
    '''
    names = lda.get_save_names()
    return jsonify(names)


@app.route('/load_model_by_name', methods=['POST'])
def load_model_by_name():
    '''
    Loads previously saved LDA model.

    post:
        - parameters:
            - name: saveName
              type: string
              description: name of previously saved model to be loaded.

        - response:
            - description: 'model loaded'
              type: string
    '''
    save_name = request.args.get('saveName')
    lda.load_model(save_name=save_name)
    return jsonify('model loaded')


@app.route('/get_api_docstrings_dict', methods=['GET'])
def get_api_docstrings_dict():
    '''
    Gets docstrings of all API methods in this app.

    get:
        - parameters:

        - response:
            - description: dictionary with keys API routes and values
              docstrings for those endpoints
              type: dictionary
    '''
    func_dict = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint not in ['static', 'index']:
            func_dict[rule.rule] = eval(rule.endpoint).__doc__

    return jsonify(func_dict)


@app.route('/get_api_docstrings', methods=['GET'])
def get_api_docstrings():
    '''
    Gets docstrings of all API methods in this app.

    get:
        - parameters:

        - response:
            - description: string with docstrings for each API endpoint
              type: string
    '''
    docstrings = ''
    for rule in app.url_map.iter_rules():
        if rule.endpoint not in ['static', 'index']:
            docstrings += rule.rule
            docstrings += eval(rule.endpoint).__doc__
            # docstrings += '\n\n'

    return jsonify(docstrings)
