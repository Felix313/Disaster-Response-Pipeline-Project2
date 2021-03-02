import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', con=engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    catergory_sums = df[['related', 'request',
           'offer', 'aid_related', 'medical_help', 'medical_products',
           'search_and_rescue', 'security', 'military', 'child_alone', 'water',
           'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
           'death', 'other_aid', 'infrastructure_related', 'transport',
           'buildings', 'electricity', 'tools', 'hospitals', 'shops',
           'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
           'storm', 'fire', 'earthquake', 'cold', 'other_weather',
           'direct_report']].sum().sort_values(ascending=False)
    
    catergory_names = list(catergory_sums.index)
    
    # message related to disaster or not
    related_messages = df['related'].value_counts()
    related_messages_labels = ['related messages', 'not related messages']
    
    #message 
    related = df[["aid_related", "infrastructure_related", "weather_related"]].sum().sort_values(ascending=False)
    related_name = list(related.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'titlefont' : {
                        'color':'grey',
                        'size':'18',
                    },
                'title': 'Channel popularity',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Channel",
                }
                
            }
        },
        {
            'data': [
                Bar(
                    x=catergory_names,
                    y=catergory_sums                )
            ],

            'layout': {
                'title': 'Message Volume per Topic',
                'titlefont' : {
                        'color':'grey',
                        'size':'18',
                    },
                'yaxis': {
                    'title': "Sum"
                },
                'xaxis': {
                    'tickangle': '-45',
                    'tickfont' : {
                        'size':'8',
                    },

                }
            }
        },
        {
            'data' : [
                Pie(labels=related_messages_labels,
                    values=related_messages,
                    textinfo='percent',
                    showlegend=False,
                    )
                ],
            'layout' : {
                'title' : 'Categorical Relevance of Messages',
                'titlefont' : {
                            'color':'grey',
                            'size' : '18'
                            }
                    }
        },
        {
            'data': [
                Pie(
                    labels=related_name,
                    values=related,
                    showlegend=False,
                    textinfo="label+percent",
                    textposition='outside'
                ),
            ],
            'layout': {
                'title': 'Related Categories',
                'titlefont' : {
                        'color':'grey',
                        'size':'18',
                        }
                }               
        },       
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(port=3001, debug=True)


if __name__ == '__main__':
    main()