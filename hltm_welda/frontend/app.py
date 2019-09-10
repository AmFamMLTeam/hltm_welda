import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import base64
import os
from flask import send_from_directory
import api_calls as api
import json
import time
from functools import partial
from textwrap3 import dedent


max_topics = 30
display_docs = 10

def convert_topic_name_to_index(topic_name):
    try:
        topic_ind = list(api.get_topic_name_dict().values()).index(topic_name)
    except ValueError:
        topic_ind = topic_name
    return topic_ind

def convert_topic_index_to_name(topic_ind, include_filler = False):
    try:
        topic_name_map = api.get_topic_name_dict()
        topic_name_dict = partial(topic_name_map_factory, topic_name_map)
        topic_name = topic_name_dict(topic_ind)
        # topic_name = api.get_topic_name_dict()[str(topic_ind)]
    except KeyError:
        if include_filler:
            topic_name = 'TOPIC {}'.format(topic_ind)
        else:
            topic_name = str(topic_ind)
    return topic_name

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI])

app.config['suppress_callback_exceptions'] = True

app.layout = dbc.Container(
children = [
    dbc.Row(
        style = {
            'backgroundColor':'#216278',
            'padding':'15px',
            'color':'rgb(280, 280, 280)'
        },
        justify = 'between',
        children = [
            dbc.Col(
                html.H1('Chat Aggregator'),
                width = 4
            ),
            dbc.Col(
                width = 2,
                children = dbc.DropdownMenu(
                    label="Model Options",
                    children = [
                        dbc.DropdownMenuItem("Save Model", id = "save_model"),
                        dbc.DropdownMenuItem("Load Model", id = "load_model"),
                        dbc.DropdownMenuItem("Reinitialize Model", id = "reinitialize_model"),
                        dbc.DropdownMenuItem("Refine Model", id = "refine_model")
                    ]
                ),
                align = 'center'
            )
        ]
    ),
    html.Button(
        'Initalize Model', id = 'init_button',
        style={'display':'none'}
    ),
    html.Div(id = 'model', style={'display':'none'}),

    #region main body
    dbc.Row([
        #topic overview column
        dbc.Col([
            dbc.Row(
                style = {
                    'padding':'10px',
                    'backgroundColor':'#04396c',
                    'color':'rgb(250,250,250)'
                },
                justify = 'between',
                children = [
                    dbc.Col(
                        children = html.H1('Topics')
                    ),
                    dbc.Col(
                        dbc.DropdownMenu(
                            label="Topic Options",
                            children = [
                                dbc.DropdownMenuItem("Merge Topics", id = 'merge_dropdown_button'),
                                dbc.DropdownMenuItem("Split Topics", id = 'split_dropdown_button'),
                                dbc.DropdownMenuItem("Rename Topic", id = 'rename_dropdown_button')
                            ]
                        ),
                        align = 'center'
                    )
                ]
            ),
            dcc.Loading(
                type = 'circle',
                children = [dbc.ListGroup(
                    id = 'topic_list',
                )]
            )
        ], width = 4,
        style = {
            'backgroundColor':'rgb(280, 280, 280)'
        }),

        #topic detail column
        dbc.Col(
            width = 8,
            className = 'cardBox',
            style = {
                'box-shadow': '3px 3px 3px rgb(240, 240, 240) inset',
                'padding':'25px',
                'backgroundColor':'rgb(260, 260, 260)'
            },
            children = [
                dbc.Container(
                    className = 'cardBox',
                    style = {
                        'backgroundColor':'rgb(280,280,280)',
                        'box-shadow':'12px 0 15px -4px rgba(240,240,240, 0.8), -12px 0 8px -4px rgba(240,240,240, 0.8)'
                    },
                    children = [
                        dcc.Loading(
                            type = 'circle',
                            children = [
                                dbc.Row(
                                    style = {
                                        'padding':'15px'
                                    },
                                    justify = 'between',
                                    align = 'center',
                                    children = [
                                        dbc.Col(
                                            id = 'topic_name',
                                            align = 'center'
                                        ),
                                        dbc.Col(
                                            dbc.DropdownMenu(
                                                    label = 'Word Actions',
                                                    children = [
                                                        dbc.DropdownMenuItem('Remove Words From Topic', id = 'remove_word_from_topic_dropdown'),
                                                        dbc.DropdownMenuItem('Move To Topic', id = 'move_word_topic_dropdown'),
                                                        dbc.DropdownMenuItem('Add Words To Stopwords', id = 'add_to_stopwords_dropdown')
                                                    ]
                                            ),
                                            width = 3,
                                            align = 'center'
                                        ),
                                        dbc.Col(
                                            dbc.DropdownMenu(
                                                    label = 'Document Actions',
                                                    children = [
                                                        dbc.DropdownMenuItem('Remove From Corpus', id = 'remove_doc_from_corpus_dropdown'),
                                                        dbc.DropdownMenuItem('Remove From Topic', id = 'remove_doc_from_topic_dropdown'),
                                                        dbc.DropdownMenuItem('Move To Topic', id = 'move_doc_to_topic_dropdown')
                                                    ]
                                            ),
                                            width = 3,
                                            align = 'center'
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Hr(),
                        #topic top words
                        dcc.Loading(
                            type = 'circle',
                            children = [
                                dbc.Row(
                                    id = 'topic_top_words',
                                    style = {
                                        'padding':'15px'
                                    }
                                )
                            ]
                        ),
                        html.Hr(),
                        #topic document details
                        dcc.Loading(
                            type = 'circle',
                            children = [
                                dbc.Container(
                                    id = 'topic_top_docs'
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]),
    #endregion

    #region topic action modals
    html.Div(
        id='merge_modal',
        children = [
            html.H3('Choose Topics To Merge'),
            html.Hr(),
            dbc.Row(
                id = 'merge-row',
                justify = 'center',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('First Topic'),
                        dcc.Dropdown(
                            id = 'first_topic_to_merge'
                        )],
                        width = 6
                    ),
                    dbc.Col([
                        html.H4('Second Topic'),
                        dcc.Dropdown(
                            id = 'second_topic_to_merge'
                        )],
                        width =6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-merge-modal',
                justify = 'end',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col(
                        dbc.Button('Merge', id = 'merge_button'),
                        width = 2
                    ),
                    dbc.Col(
                        dbc.Button('Cancel', id = 'cancel_merge_button'),
                        width = 2
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    html.Div(
        id='split_modal',
        children = [
            html.H3('Split Topic'),
            html.Hr(),
            dbc.Row(
                id = 'split-row',
                justify = 'center',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Choose Topic'),
                        dcc.Dropdown(
                            id = 'topic_to_split',
                            value = 0
                        )],
                        width = 6
                    ),
                    dbc.Col([
                        html.H4('Choose Seed Words'),
                        dcc.Dropdown(
                            id = 'split_seed_words',
                            multi=True
                        )],
                        width =6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-split-modal',
                justify = 'end',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col(
                        dbc.Button('Split', id = 'split_button'),
                        width = 2
                    ),
                    dbc.Col(
                        dbc.Button('Cancel', id = 'cancel_split_button'),
                        width = 2
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    html.Div(
        id='rename_modal',
        children = [
            html.H3('Rename Topic'),
            html.Hr(),
            dbc.Row(
                id = 'rename_row',
                justify = 'center',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Choose Topic'),
                        dcc.Dropdown(
                            id = 'topic_to_rename',
                            value = 0
                        )],
                        width = 6
                    ),
                    dbc.Col([
                        html.H4('New Name'),
                        dbc.Input(
                            id = 'rename_value'
                        )],
                        width =6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-rename-modal',
                justify = 'end',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col(
                        dbc.Button('Rename', id = 'rename_button'),
                        width = 2
                    ),
                    dbc.Col(
                        dbc.Button('Cancel', id = 'cancel_rename_button'),
                        width = 2
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    #endregion

    #region word action modals
    html.Div(
        id='remove_word_topic_modal',
        children = [
            html.H3('Remove From Topic'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Choose Words'),
                        dcc.Dropdown(
                            id = 'word_to_remove_topic',
                            multi = True
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-remove-word-topic-modal',
                children = [
                    dbc.Col(
                        dbc.Button('Remove', id = 'remove_word_topic_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Close', id = 'cancel_remove_word_topic_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container-bottom'
    ),
    html.Div(
        id='stopwords_modal',
        children = [
            html.H3('Add To Stopwords'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Choose Words'),
                        dcc.Dropdown(
                            id = 'add_to_stopwords',
                            multi = True
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-stopwords-modal',
                children = [
                    dbc.Col(
                        dbc.Button('Remove', id = 'stopwords_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Close', id = 'cancel_stopwords_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container-bottom'
    ),
    html.Div(
        id='move_word_modal',
        children = [
            html.H3('Move Word'),
            html.Hr(),
            dbc.Row(
                justify = 'center',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Move Topics'),
                        dcc.Dropdown(
                            id = 'word_to_move',
                            multi = True
                        )],
                        width = 6
                    ),
                    dbc.Col([
                        html.H4('Move To:'),
                        dcc.Dropdown(
                            id = 'move_word_to_topic'
                        )
                    ])
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-move-word-modal',
                children = [
                    dbc.Col(
                        dbc.Button('Move', id = 'move_word_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Close', id = 'cancel_move_word_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container-bottom'
    ),
    #endregion

    #region model option modals
    html.Div(
        id='refine_modal',
        children = [
            html.H3('Refine Model'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Refining Iterations'),
                        dcc.Input(
                            id = 'refine_model_iterations',
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-refine-modal',
                children = [
                    dbc.Col(
                        dbc.Button('Refine', id = 'refine_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Cancel', id = 'cancel_refine_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    html.Div(
        id='load_model_modal',
        children = [
            html.H3('Load Model'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Model To Load'),
                        dcc.Dropdown(
                            id = 'model_to_load'
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-load-modal',
                children = [
                    dbc.Col(
                        dbc.Button('Load', id = 'load_model_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Cancel', id = 'cancel_load_model_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    html.Div(
        id='save_model_modal',
        children = [
            html.H3('Save Model'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Model Name'),
                        dbc.Input(
                            id = 'save_model_name'
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-save-modal',
                children = [
                    dbc.Col(
                        dbc.Button('Save Model', id = 'save_model_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Cancel', id = 'cancel_save_model_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    html.Div(
        id='reinitialize_modal',
        children = [
            html.H3('Reinitialize Model'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Number Of Topics'),
                        dcc.Dropdown(
                            id = 'reinitialize_number_of_topics',
                            options = [{'value':i, 'label':i} for i in range(1,max_topics)]
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                id = 'close-cancel-reintialize-modal',
                children = [
                    dbc.Col(
                        dbc.Button('Reintialize', id = 'reinitialize_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Cancel', id = 'cancel_reinitialize_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    #endregion

    #region document option modals
    html.Div(
        id='remove_doc_corpus_modal',
        children = [
            html.H3('Remove Document From Corpus'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Choose Documents'),
                        dcc.Dropdown(
                            id = 'doc_to_remove_corpus',
                            multi = True
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                children = [
                    dbc.Col(
                        dbc.Button('Remove', id = 'remove_doc_corpus_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Close', id = 'cancel_remove_doc_corpus_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    html.Div(
        id='remove_doc_topic_modal',
        children = [
            html.H3('Remove Document From Topic'),
            html.Hr(),
            dbc.Row(
                justify = 'start',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Choose Documents'),
                        dcc.Dropdown(
                            id = 'doc_to_remove_topic',
                            multi = True
                        )],
                        width = 6
                    )
                ]
            ),
            html.Hr(),
            dbc.Row(
                children = [
                    dbc.Col(
                        dbc.Button('Remove', id = 'remove_doc_topic_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Close', id = 'cancel_remove_doc_topic_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    html.Div(
        id='move_doc_topic_modal',
        children = [
            html.H3('Move Document To Topic'),
            html.Hr(),
            dbc.Row(
                justify = 'center',
                align = 'center',
                style = {
                    'padding':'5px'
                },
                children = [
                    dbc.Col([
                        html.H4('Choose Documents'),
                        dcc.Dropdown(
                            id = 'doc_to_move_topic',
                            multi = True
                        )],
                        width = 6
                    ),
                    dbc.Col([
                        html.H4('Move To:'),
                        dcc.Dropdown(
                            id = 'move_doc_topic'
                        )
                    ])
                ]
            ),
            html.Hr(),
            dbc.Row(
                children = [
                    dbc.Col(
                        dbc.Button('Move', id = 'move_doc_topic_button'),
                        width = 6
                    ),
                    dbc.Col(
                        dbc.Button('Close', id = 'cancel_move_doc_topic_button'),
                        width = 6
                    )
                ]
            )
        ],
        className = 'modal-container'
    ),
    #endregion

    #region print statement testing -- remove before prod
    html.Div(id='save_success', style={'display':'none'}),
    html.Div(id='refine_success', style={'display':'none'}),
    html.Div(id='load_success', style={'display':'none'}),
    html.Div(id='reinitialize_success', style={'display':'none'}),

    html.Div(id='stopwords_success', style={'display':'none'}),
    html.Div(id='remove_word_success', style={'display':'none'}),
    html.Div(id='move_word_success', style={'display':'none'}),
    html.Div(id='remove_word_topic_success', style={'display':'none'}),

    html.Div(id='remove_doc_corpus_success', style={'display':'none'}),
    html.Div(id='remove_doc_topic_success', style={'display':'none'}),
    html.Div(id='move_doc_topic_success', style={'display':'none'}),

    html.Div(id='merge_success', style={'display':'none'}),
    html.Div(id='split_success', style={'display':'none'}),
    html.Div(id='rename_success', style={'display':'none'}),
    #endregion

    # hidden divs allow for objects to be dynamically updated
    # and passed to other callbacks
    html.Div(id='topics', style={'display':'none'}),
    html.Div(id='clicked_topic', style={'display':'none'})
])

#region Document Column Divs
@app.callback(
    Output('topic_name', 'children'),
    [Input('clicked_topic', 'children'),
    Input('topics', 'children'),
    Input('model', 'children')]
)
def print_clicked_topic_name(topic, *args):
    print(f'topic name is {topic}')
    if topic is None:
        topic = 0
    topic_name = convert_topic_index_to_name(topic)
    if len(str(topic_name)) <= 2:
        topic_name = 'TOPIC {}'.format(topic)
    return html.H1(topic_name.upper())


@app.callback(
    Output('topic_top_words', 'children'),
    [Input('clicked_topic', 'children'),
    Input('stopwords_success', 'children'),
    Input('move_word_button', 'n_clicks'),
    Input('remove_word_topic_button', 'n_clicks'),
    Input('remove_doc_corpus_success', 'children'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks'),
    Input('model', 'children'),
    Input('refine_success', 'children'),
    Input('reinitialize_success', 'children')])
def topic_detail_top_words(topic, *args):
    # print(f'in topic_detail_top_words, args: {args}')
    print('refreshed top words')
    if topic is None:
        topic = 0
    if topic is not None:
        print(f'topic {topic}')
        top_words = api.get_topic_top_words(topic,30)
        items = [
            html.Div(
                style = {
                    'backgroundColor':'rgb(220,220,220)',
                    'padding':'3px',
                    'border-radius':'7px',
                    'margin':'3px',
                    'font-size':'17px',
                    'font-style':'bold'
                },
                children = ' ' + word + ' '
            ) for word in top_words
        ]
        return items
    else:
        return dbc.Col(
            align ='center',
            children = 'Choose a topic on the left to get started',
            style = {
                'font-style':'italic',
                'font-size':'15px',
                'text-align':'center'
            }
        )

@app.callback(
    Output('topic_top_docs', 'children'),
    [Input('clicked_topic', 'children'),
    Input('stopwords_success', 'children'),
    Input('move_word_button', 'n_clicks'),
    Input('remove_word_topic_button', 'n_clicks'),
    Input('remove_doc_corpus_success', 'children'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks'),
    Input('model', 'children'),
    Input('refine_success', 'children'),
    Input('reinitialize_success', 'children')]
)
def topic_detail_top_docs(topic, *args):
    print('refreshed top docs')
    if topic is None:
        topic = 0

    if topic is not None:
        docs = api.get_topic_top_docs(topic, display_docs+1)

        div_list = []
        for i in range(len(docs)):
            doc = docs[i]['d']
            # doc_str = ' '.join([utterance['utterance'] for utterance in doc])
            doc_id = docs[i]['docid']
            # summary = doc_str[:60] + " ... "
            # summary = doc[:60] + ' ... '
            # summary = ' '.join([
            #     utterance['utterance']
            #     for utterance in doc
            #     if
            #         (
            #             utterance['speaker'] != 'sys'
            #             # and
            #             # utterance['speaker'] != 'IC'
            #         )
            # ])[:60] + ' ...'
            summary = doc[:60] + '...'
            # doc_display = ''
            # for element in doc:
            #     if element['speaker'] != 'sys':
            #         doc_display += '**'
            #         doc_display += element['speaker_name']
            #         # doc_display += ': '
            #         # doc_display += element['speaker']
            #         doc_display += '**: '
            #         doc_display += element['utterance']
            #         doc_display += '  \n'
            doc_display = doc

            div_list.append(
                dbc.Row(
                    style = {
                        'font-size':'16px',
                        'padding':'3px'
                    },
                    children = [
                        dbc.Col(
                            children=doc_id,
                            width=1
                        ),
                        dbc.Col(
                            html.Details([
                                html.Summary(summary),
                                html.Hr(),
                                html.Div([
                                    dcc.Markdown(dedent(doc_display))
                                    # dcc.Markdown(dedent(doc))
                                ])
                            ]),
                        )
                    ]
                )
            )
            div_list.append(html.Hr())
        return div_list


#endregion

#region Model Actions
#Intialize Model
@app.callback(
    Output('model', 'children'),
    [Input('init_button', 'n_clicks')])
def app_intialize_model(click):
    response = api.initialize_model()
    api.iterate_model(3)
    return response

#callbacks for save model modal -----------------------------------
@app.callback(
    Output('save_model_modal', 'style'),
    [Input('save_model', 'n_clicks'),
    Input('save_model_button', 'n_clicks'),
    Input('cancel_save_model_button', 'n_clicks')])
def display_save_modal(open_clicks, save_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if save_clicks is None:
        save_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + save_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('save_success','children'),
    [Input('save_model_button', 'n_clicks')],
    [State('save_model_name', 'value')])
def save_model(clicks, name):
    if name is not None and clicks is not None:
        response = api.save_model(name)
        return '{} as {}'.format(response, name)

#callbacks for load model modal -----------------------------------
@app.callback(
    Output('load_model_modal', 'style'),
    [Input('load_model', 'n_clicks'),
    Input('load_model_button', 'n_clicks'),
    Input('cancel_load_model_button', 'n_clicks')])
def display_load_model_modal(open_clicks, load_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if load_clicks is None:
        load_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + load_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('load_success','children'),
    [Input('load_model_button', 'n_clicks')],
    [State('model_to_load', 'value')])
def load_model(clicks, name):
    if clicks is not None and name is not None:
        response = api.load_model(name)
        return '{} {}'.format(name, response)

@app.callback(
    Output('model_to_load', 'options'),
    [Input('load_model', 'n_clicks')])
def generate_model_list_dropdown(clicks):
    model_names = api.get_saved_model_names()
    items = [
        {'label': model, 'value':model} for model in model_names
    ]
    return items

#callbacks for reinitialize modal -----------------------------------
@app.callback(
    Output('reinitialize_modal', 'style'),
    [Input('reinitialize_model', 'n_clicks'),
    Input('reinitialize_button', 'n_clicks'),
    Input('cancel_reinitialize_button', 'n_clicks')])
def display_reinitialize_modal(open_clicks, reinitialize_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if reinitialize_clicks is None:
        reinitialize_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + reinitialize_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('reinitialize_success','children'),
    [Input('reinitialize_button', 'n_clicks')],
    [State('reinitialize_number_of_topics', 'value')])
def reintialize_model(clicks, number):
    print(f'{clicks} and number is {number}')
    if clicks is not None and number is not None:
        response = api.reinitialize_model(number)
        return '{} with {} topics'.format(response, number)

#callbacks for refining modal -----------------------------------
@app.callback(
    Output('refine_modal', 'style'),
    [Input('refine_model', 'n_clicks'),
    Input('refine_button', 'n_clicks'),
    Input('cancel_refine_button', 'n_clicks')])
def display_refine_modal(open_clicks, refine_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if refine_clicks is None:
        refine_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + refine_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('refine_success','children'),
    [Input('refine_button', 'n_clicks')],
    [State('refine_model_iterations', 'value')])
def refine_model(clicks, number):
    if clicks is not None and number is not None:
        response = api.iterate_model(number)
        return f'{response} {number} times'

#endregion

#region Topic Actions
# Callbacks to create merge modal -------------------------------
@app.callback(
    Output('merge_modal', 'style'),
    [Input('merge_dropdown_button', 'n_clicks'),
    Input('merge_button', 'n_clicks'),
    Input('cancel_merge_button', 'n_clicks')])
def display_modal(open_clicks, merge_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if merge_clicks is None:
        merge_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + merge_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('merge_success','children'),
    [Input('merge_button', 'n_clicks')],
    [State('first_topic_to_merge', 'value'),
    State('second_topic_to_merge', 'value')])
def merge_topics(clicks, topic_1, topic_2):
    #change topic names to topic index
    if topic_1 is None:
        return 'Merged {} and {}'.format(topic_1, topic_2)
    elif topic_2 is None:
        return 'Merged {} and {}'.format(topic_1, topic_2)
    else:
        try:
            topic1_ind = list(api.get_topic_name_dict().values()).index(topic_1)
        except ValueError:
            topic1_ind = topic_1

        try:
            topic2_ind = list(api.get_topic_name_dict().values()).index(topic_2)
        except ValueError:
            topic2_ind = topic_2

        response = api.merge_topics(topic1_ind, topic2_ind)
        #
        # print('Merged {} and {}'.format(topic1_ind, topic2_ind))
        return response

@app.callback(
    Output('first_topic_to_merge', 'options'),
    [Input('topics', 'children')])
def generate_topic_list_merge1_dropdown(topics):
    topics = json.loads(topics)
    topics = list(topics.keys())
    items = []
    for topic in topics:
        try:
            topic_name = api.get_topic_name_dict()[str(topic)]
        except KeyError:
            topic_name = 'TOPIC {}'.format(topic)
        items.append(
            {'label': '{}'.format(topic_name), 'value':topic}
        )
    return items

@app.callback(
    Output('second_topic_to_merge', 'options'),
    [Input('topics', 'children')])
def generate_topic_list_merge2_dropdown(topics):
    topics = json.loads(topics)
    topics = list(topics.keys())
    items = []
    for topic in topics:
        try:
            topic_name = api.get_topic_name_dict()[str(topic)]
        except KeyError:
            topic_name = 'TOPIC {}'.format(topic)
        items.append(
            {'label': '{}'.format(topic_name), 'value':topic}
        )
    return items

# Callbacks to create split modal
@app.callback(
    Output('split_modal', 'style'),
    [Input('split_dropdown_button', 'n_clicks'),
    Input('split_button', 'n_clicks'),
    Input('cancel_split_button', 'n_clicks')])
def display_split_modal(open_clicks, split_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if split_clicks is None:
        split_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + split_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('split_success','children'),
    [Input('split_button', 'n_clicks')],
    [State('topic_to_split', 'value'),
    State('split_seed_words', 'value')])
def split_topics(clicks, topic, words):
    if topic is not None and words is not None:
        response = api.split_topic(topic, words)
        return response

@app.callback(
    Output('topic_to_split', 'options'),
    [Input('topics', 'children')])
def generate_topic_list_split_dropdown(topics):
    topics = json.loads(topics)
    topics = list(topics.keys())
    items = []
    for topic in topics:
        topic_name = convert_topic_index_to_name(topic)
        items.append(
            {'label': '{}'.format(topic_name), 'value':topic}
        )
    return items

@app.callback(
    Output('split_seed_words', 'options'),
    [Input('topic_to_split', 'value')])
def generate_split_seed_words_dropdown(topic):
    if topic is None:
        return None
    else:
        #get top words
        #print('getting seed words')

        words = api.get_topic_top_words(topic=topic,n_words=50)
        items = []
        for word in words:
            items.append({'label': word, 'value': word})
        return items

#callbacks for rename topic modal -----------------------------------
@app.callback(
    Output('rename_modal', 'style'),
    [Input('rename_dropdown_button', 'n_clicks'),
    Input('rename_button', 'n_clicks'),
    Input('cancel_rename_button', 'n_clicks')])
def display_rename_modal(open_clicks, rename_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if rename_clicks is None:
        rename_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + rename_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('rename_success','children'),
    [Input('rename_button', 'n_clicks')],
    [State('topic_to_rename', 'value'),
    State('rename_value', 'value')])
def rename_topics(clicks, topic, new_name):
    if topic is not None and new_name is not None:
        response = api.assign_topic_name(topic, new_name)
        return response

@app.callback(
    Output('topic_to_rename', 'options'),
    [Input('topics', 'children')])
def generate_topic_list_rename_dropdown(topics):
    topics = json.loads(topics)
    topics = list(topics.keys())
    items = []
    for topic in topics:
        topic_name = convert_topic_index_to_name(topic)
        items.append(
            {'label': '{}'.format(topic_name), 'value':topic}
        )
    return items


#endregion

#region Word Actions

#callbacks for stopwords modal -----------------------------------
@app.callback(
    Output('stopwords_modal', 'style'),
    [Input('add_to_stopwords_dropdown', 'n_clicks'),
    Input('stopwords_button', 'n_clicks'),
    Input('cancel_stopwords_button', 'n_clicks')])
def display_stopwords_modal(open_clicks, stopwords_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if stopwords_clicks is None:
        stopwords_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + stopwords_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('stopwords_success','children'),
    [Input('stopwords_button', 'n_clicks')],
    [State('add_to_stopwords', 'value')])
def add_to_stopwords(clicks, words):
    if clicks is not None and words is not None:
        for word in words:
            response = api.add_word_to_stopwords(word)
        return 'adding {} to stopwords'.format(words)

@app.callback(
    Output('add_to_stopwords', 'options'),
    [Input('clicked_topic', 'children'),
    Input('add_to_stopwords_dropdown', 'n_clicks')])
def generate_stopwords_dropdown(topic, *args):
    if topic is None:
        topic = 0
    words = api.get_topic_top_words(topic, 100)
    items = [
        {'label': word, 'value':word} for word in words
    ]
    return items

#callbacks for move word to topic modal -----------------------------------
@app.callback(
    Output('move_word_modal', 'style'),
    [Input('move_word_topic_dropdown', 'n_clicks'),
    Input('move_word_button', 'n_clicks'),
    Input('cancel_move_word_button', 'n_clicks')])
def display_move_word_modal(open_clicks, move_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if move_clicks is None:
        move_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + move_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('move_word_success','children'),
    [Input('move_word_button', 'n_clicks')],
    [State('word_to_move', 'value'),
    State('move_word_to_topic', 'value')])
def move_word(clicks, words, topic):
    if clicks is not None and words is not None and topic is not None:
        for word in words:
            response = api.add_word_to_topic(topic=topic, word=word)
        return 'moving {} to {}'.format(words, topic)

@app.callback(
    Output('word_to_move', 'options'),
    [Input('clicked_topic', 'children'),
    Input('move_word_topic_dropdown', 'n_clicks')])
def generate_move_word_dropdown(topic, *args):
    if topic is None:
        topic = 0
    words = api.get_topic_top_words(topic, 100)
    items = [
        {'label': word, 'value':word} for word in words
    ]
    return items

@app.callback(
    Output('move_word_to_topic', 'options'),
    [Input('topics', 'children'),
    Input('move_word_button', 'n_clicks')])
def generate_topic_move_word_dropdown(topics, *args):
    topics = json.loads(topics)
    topics = list(topics.keys())
    items = []
    for topic in topics:
        topic_name = convert_topic_index_to_name(topic)
        items.append(
            {'label': '{}'.format(topic_name), 'value':topic}
        )
    return items

#callbacks for remove word from topic modal -----------------------------------
@app.callback(
    Output('remove_word_topic_modal', 'style'),
    [Input('remove_word_from_topic_dropdown', 'n_clicks'),
    Input('remove_word_topic_button', 'n_clicks'),
    Input('cancel_remove_word_topic_button', 'n_clicks')])
def display_remove_word_topic_modal(open_clicks, remove_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if remove_clicks is None:
        remove_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + remove_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('remove_word_topic_success','children'),
    [Input('remove_word_topic_button', 'n_clicks')],
    [State('word_to_remove_topic', 'value'),
    State('clicked_topic', 'children')])
def remove_word_topic(clicks, words, topic):
    if topic is None:
        topic = 0
    if clicks is not None and words is not None and topic is not None:
        for word in words:
            response = api.remove_word_from_topic(topic=topic, word=word)
        return 'removing {} from {}'.format(words, topic)

@app.callback(
    Output('word_to_remove_topic', 'options'),
    [Input('clicked_topic', 'children'),
    Input('remove_word_from_topic_dropdown', 'n_clicks')])
def generate_remove_word_topic_dropdown(topic, *args):
    if topic is None:
        topic = 0
    words = api.get_topic_top_words(topic, 100)
    items = [
        {'label': word, 'value':word} for word in words
    ]
    return items

#endregion

#region Doc Actions
#callbacks for remove doc from corpus modal -----------------------------------
@app.callback(
    Output('remove_doc_corpus_modal', 'style'),
    [Input('remove_doc_from_corpus_dropdown', 'n_clicks'),
    Input('remove_doc_corpus_button', 'n_clicks'),
    Input('cancel_remove_doc_corpus_button', 'n_clicks')])
def display_remove_doc_corpus_modal(open_clicks, remove_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if remove_clicks is None:
        remove_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + remove_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('remove_doc_corpus_success','children'),
    [Input('remove_doc_corpus_button', 'n_clicks')],
    [State('doc_to_remove_corpus', 'value')])
def remove_doc_corpus(clicks, docids):
    if clicks is not None and docids is not None:
        for docid in docids:
            response = api.remove_doc_from_corpus(docid=docid)
        return 'removing {} from corpus'.format(docids)

@app.callback(
    Output('doc_to_remove_corpus', 'options'),
    [Input('clicked_topic', 'children'),
    Input('remove_doc_corpus_success', 'children'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks'),
    Input('remove_doc_from_corpus_dropdown','n_clicks')],
    [State('clicked_topic', 'children')])
def generate_remove_doc_corpus_dropdown(topic_1, click1, click2, click3, click4, topic):
    if topic is None:
        topic = 0
    docs = api.get_topic_top_docs(topic, display_docs+1)
    items = [
        {
            'label':'Document {}'.format(docs[i]['docid']),
            'value':docs[i]['docid']
        } for i in range(len(docs))
    ]
    return items

#callbacks for remove doc from topic modal -----------------------------------
@app.callback(
    Output('remove_doc_topic_modal', 'style'),
    [Input('remove_doc_from_topic_dropdown', 'n_clicks'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('cancel_remove_doc_topic_button', 'n_clicks')])
def display_remove_doc_topic_modal(open_clicks, remove_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if remove_clicks is None:
        remove_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + remove_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('remove_doc_topic_success','children'),
    [Input('remove_doc_topic_button', 'n_clicks')],
    [State('doc_to_remove_topic', 'value'),
    State('clicked_topic', 'children')])
def remove_doc_topic(clicks, docids, topic):
    if topic is None:
        topic = 0
    if clicks is not None and docids is not None and topic is not None:
        for docid in docids:
            response = api.remove_doc_from_topic(topic=topic, doc_index=docid)
        return 'removing {} from topic {}'.format(docids, topic)

@app.callback(
    Output('doc_to_remove_topic', 'options'),
    [Input('clicked_topic', 'children'),
    Input('remove_doc_corpus_success', 'children'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks'),
    Input('remove_doc_topic_success','children'),
    Input('remove_doc_from_topic_dropdown', 'n_clicks')])
def generate_remove_doc_topic_dropdown(topic, *args):
    if topic is None:
        topic = 0
    docs = api.get_topic_top_docs(topic, display_docs+1)
    items = [
        {
            'label':'Document {}'.format(docs[i]['docid']),
            'value':docs[i]['docid']
        } for i in range(len(docs))
    ]
    return items

#callbacks for move doc to topic modal -----------------------------------
@app.callback(
    Output('move_doc_topic_modal', 'style'),
    [Input('move_doc_to_topic_dropdown', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks'),
    Input('cancel_move_doc_topic_button', 'n_clicks')])
def display_move_doc_topic_modal(open_clicks, move_clicks, cancel_clicks):
    if open_clicks is None:
        open_clicks = 0
    if move_clicks is None:
        move_clicks = 0
    if cancel_clicks is None:
        cancel_clicks = 0
    total_clicks = open_clicks + move_clicks + cancel_clicks
    if total_clicks % 2 == 0:
        return {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(
    Output('move_doc_topic_success','children'),
    [Input('move_doc_topic_button', 'n_clicks')],
    [State('doc_to_move_topic', 'value'),
    State('clicked_topic', 'children'),
    State('move_doc_topic', 'value')])
def move_doc_topic(clicks, docids, topic, new_topic):
    if topic is None:
        topic = 0
    if clicks is not None and docids is not None and topic is not None and new_topic is not None:
        for docid in docids:
            response = api.add_doc_to_topic(topic=new_topic,doc_index=docid)
        return 'moving {} from topic {} to {}'.format(docids, topic, new_topic)

@app.callback(
    Output('doc_to_move_topic', 'options'),
    [Input('clicked_topic', 'children'),
    Input('remove_doc_corpus_success', 'children'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks'),
    Input('move_doc_to_topic_dropdown', 'n_clicks')])
def generate_move_doc_topic_dropdown(topic, *args):
    if topic is None:
        topic = 0
    docs = api.get_topic_top_docs(topic, display_docs+1)
    items = [
        {
            'label':'Document {}'.format(docs[i]['docid']),
            'value':docs[i]['docid']
        } for i in range(len(docs))
    ]
    return items

@app.callback(
    Output('move_doc_topic', 'options'),
    [Input('topics', 'children')])
def generate_doc_to_topic_dropdown(topics):
    topics = json.loads(topics)
    topics = list(topics.keys())
    items = []
    for topic in topics:
        topic_name = convert_topic_index_to_name(topic)
        items.append(
            {'label': '{}'.format(topic_name), 'value':topic}
        )
    return items

#endregion

#region Dynamic Topic Update
# Dynamically get up to date topic names/indexes
#  include all buttons that create topic changes as Input
@app.callback(
    Output('topics', 'children'),
    [Input('merge_success', 'children'),
    Input('split_success', 'children'),
    Input('rename_button', 'n_clicks'),
    Input('load_success', 'children'),
    Input('reinitialize_success', 'children'),
    Input('stopwords_success', 'children'),
    Input('move_word_button', 'n_clicks'),
    Input('remove_word_topic_button', 'n_clicks'),
    Input('remove_doc_corpus_success', 'children'),
    Input('model', 'children'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks')])
def get_topics(*args):
    topics = api.get_num_docs_per_topic_json()
    print('refreshed topic names to {}'.format(list(json.loads(topics.text).keys())))

    return(topics.text)

inputs = [
    Input('topic_{}_button'.format(topic), 'n_clicks_timestamp') for topic in range(max_topics)
]
@app.callback(
    Output('clicked_topic', 'children'),
    inputs
)
def list_success(*args):
   # print(args)
    stamps = [0 if time is None else time for time in args]
    topic = np.argmax(np.array(stamps))
    return topic




def topic_name_map_factory(topic_name_dict, topic_key):
    id_ = lambda x: x
    for typ in [id_, int, str]:
        if typ(topic_key) in topic_name_dict.keys():
            return topic_name_dict[typ(topic_key)]
    else:
        return topic_key





@app.callback(
    Output('topic_list', 'children'),
    [Input('topics', 'children'),
    Input('merge_success', 'children'),
    Input('stopwords_success', 'n_clicks'),
    Input('remove_word_topic_button', 'n_clicks'),
    Input('remove_doc_corpus_success', 'n_clicks'),
    Input('remove_doc_topic_button', 'n_clicks'),
    Input('move_doc_topic_button', 'n_clicks'),
    Input('refine_success', 'children'),
    Input('reinitialize_success', 'children'),
    Input('model', 'children'),
    Input('load_success', 'children')])
def app_get_topics(topics, *args):
    topics_dict = api.get_num_docs_per_topic()
    # print(f'topics_dict: {topics_dict}')
    topics = [str(x) for x in list(topics_dict.keys())]
    div_list = []
    print(f'refreshing topic list with topics: {topics}')
    for topic in range(max_topics):
        if str(topic) in topics:
            # print(f'topic {topic}')
            # print(f'type(topic) {type(topic)}')
            top_words = api.get_topic_top_words(topic=topic, n_words=4)
            # print(f'top_words {top_words}')
            top_words = ', '.join(top_words)

            # topic_name = api.get_topic_name_dict()[str(topic)]
            topic_name_map = api.get_topic_name_dict()
            # print(f'topic_name_map {topic_name_map}')
            topic_name_dict = partial(topic_name_map_factory, topic_name_map)
            # topic_name = api.get_topic_name_dict()[topic]
            topic_name = topic_name_dict(topic)
            # print(f'topic_name {topic_name}')
            if len(str(topic_name)) <= 2:
                topic_name = 'TOPIC {}'.format(topic)

            div_list.append(
                dbc.ListGroupItem(
                    style = {
                    'border-style':'hidden'
                    },
                    id = 'topic_{}_button'.format(topic),

                    action=True,
                    children = [
                        dbc.Row(
                            align = 'center',
                            justify = 'start',
                            children = [
                                dbc.Col(
                                    id=f'topic-{topic}',
                                    children=topic_name.upper(),
                                    style={
                                        'font-style': 'bold',
                                        'font-size': '20px'
                                    },
                                ),
                                dbc.Col(
                                    id=f'topic-{topic}-num_docs',
                                    children=str(topics_dict[str(topic)]),
                                    style={
                                        'font-style': 'italic',
                                        'font-size': '17px'
                                    },
                                ),
                                dbc.Col(
                                    id=f'topic-{topic}-top_words',
                                    children=top_words,
                                    style={
                                        'font-style': 'italic',
                                        'font-size': '17px'
                                    },
                                )
                            ]
                        )
                    ]
                )
            )
        else:
            div_list.append(
                html.Div(
                    style = {'display':'none'},
                    id = 'topic_{}_button'.format(topic)
                )
            )

    return(div_list)

#endregion
if __name__ == '__main__':
    app.run_server()
