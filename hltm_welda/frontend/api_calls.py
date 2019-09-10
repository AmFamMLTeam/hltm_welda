import requests
import json


base_url = 'http://localhost:5000/'

#make a function for each major api call
#get all docs
def get_all_documentation():

    ext_url = 'get_api_docstrings_dict'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    api_calls = list(response.keys())
    return(api_calls, response)

#add_doc_to_topic
def add_doc_to_topic(topic, doc_index):

    ext_url = 'add_doc_to_topic'
    url = base_url + ext_url
    params = {
        'topic': topic,
        'doc_idx': doc_index
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#add word to stopwords
def add_word_to_stopwords(word):

    ext_url = 'add_to_stopwords'
    url = base_url + ext_url
    params = {
        'word': word
    }
    print(f'added "{word}" to stopwords')
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#add word to topic
def add_word_to_topic(topic, word):

    ext_url = 'add_word_to_topic'
    url = base_url + ext_url
    params = {
        'topic': topic,
        'word': word
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#get bubble chart data
def get_bubbles():

    ext_url = 'bubbles'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    return response

#merge two topics
def merge_topics(topic1, topic2):

    ext_url = 'merge_topics'
    url = base_url + ext_url
    params = {
        'topic1': topic1,
        'topic2': topic2
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#get saved models
def get_saved_model_names():

    ext_url = 'get_saved_model_names'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    return response

#get topic name dict
def get_topic_name_dict():

    ext_url = 'get_topic_name_dict'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    return response

#initialize model
def initialize_model():

    ext_url = 'initialize_model'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    # print(response)
    return response

#iterate_model
def iterate_model(nIter):

    ext_url = 'iterate_model'
    url = base_url + ext_url
    params = {
        'nIter':nIter
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#load model by name
def load_model(name):

    ext_url = 'load_model_by_name'
    url = base_url + ext_url
    params = {
        'saveName':name
    }
    response = requests.post(url=url, params=params)
    response = json.loads(response.text)
    return response

#get number of documents in corpus
def get_num_of_docs():

    ext_url = 'n_docs'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    return response

#get number of topics
def get_num_of_topics():

    ext_url = 'n_topics'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    return response

#assign name of topic
def assign_topic_name(topic, name):

    ext_url = 'name_topic'
    url = base_url + ext_url
    params = {
        'topic':topic,
        'name':name
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#reinitialize model
def reinitialize_model(num_init_topics):

    ext_url = 're_initialize_model'
    url = base_url + ext_url
    params = {
        'numInitTopics':num_init_topics
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#remove_doc_from_corpus
def remove_doc_from_corpus(docid):

    ext_url = 'remove_doc_from_corpus'
    url = base_url + ext_url
    params = {
        'docid':docid
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#remove doc from topic
def remove_doc_from_topic(topic, doc_index):

    ext_url = 'remove_doc_from_topic'
    url = base_url + ext_url
    params = {
        'topic':topic,
        'doc_idx':doc_index
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#remove word from topic
def remove_word_from_topic(topic, word):

    ext_url = 'remove_word_from_topic'
    url = base_url + ext_url
    params = {
        'topic':topic,
        'word':word
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#save current model
def save_model(name):

    ext_url = 'save_model'
    url = base_url + ext_url
    params = {
        'saveName':name
    }
    response = requests.post(url=url, params=params)
    response = json.loads(response.text)
    return response

#split topic based on seed words
def split_topic(topic, seed_words):
    # print(f'in split_topic; seed_words: {seed_words}')

    ext_url = 'split_topic'
    url = base_url + ext_url
    params = {
        "topic": topic,
        "seed_words": {
            "words": [
                {
                    "w": seed_word
                }
                for seed_word in seed_words
            ]
        }
    }
    # print(f'in split_topic; params: {params}')
    response = requests.get(url=url, json=params)
    response = json.loads(response.text)
    return response

#get topic top docs
def get_topic_top_docs(topic, n_docs):

    ext_url = 'topic_top_docs'
    url = base_url + ext_url
    params = {
        'topic': int(topic),
        'n_docs': int(n_docs)
    }
    response = requests.get(url=url, params=params)
    response = json.loads(response.text)
    return response

#drop from corpus
def get_topic_top_words(topic, n_words):

    ext_url = 'topic_top_words'
    url = base_url + ext_url
    params = {
        'n_words': int(n_words),
        'topic': int(topic)
    }
    response = requests.get(url=url, params=params)
    response_text = json.loads(response.text)

    words = [word_dict['w'] for word_dict in response_text]
    return words

def get_num_docs_per_topic():

    ext_url = 'get_num_docs_per_topic'
    url = base_url + ext_url
    response = requests.get(url=url)
    response = json.loads(response.text)
    return response

def get_num_docs_per_topic_json():

    ext_url = 'get_num_docs_per_topic'
    url = base_url + ext_url
    response = requests.get(url=url)
    #response = json.loads(response.text)
    return response
