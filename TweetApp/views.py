import csv
import json
import operator
import re
import collections
import hashlib
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import colorlover as cl
from plotly.offline.offline import _plot_html


from googletrans import Translator
from collections import Counter
import translate

import numpy as np
import pandas as pd
import vincent
from bokeh.embed import components
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render

from django.template import loader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

from TweetApp import models
from .forms import SignupForm
from nltk.tokenize import TweetTokenizer
from string import punctuation
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from aylienapiclient import textapi
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud

cache_english_stopwords = stopwords.words('english')  # type: object
cache_spanish_stopwords = stopwords.words('spanish')  # type: object
cache_portuguese_stopwords = stopwords.words('portuguese')  # type: object

client = textapi.Client('fa3ed458', 'f7904358a4d58ce3fa0d9415801bb481')

# punctuation = list(string.punctuation)
# stop = stopwords.words('spanish') + stopwords.words('english') + punctuation + ['rt', 'via', 'RT', '...', 'sa']
# com = defaultdict(lambda: defaultdict(int))

# tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)


# emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

'''def tokenize(s):
    return tokens_re.findall(s)'''

USER = ""
IDIOMA = ""
ESTACION = ""
HASHTAGS = []

py.sign_in('d.chichell', 'MmTDIsdVOIjwUMiSOCvM')

def signupform(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            return render(request, 'result.html', {
                'name': form.cleaned_data['name'],
                'email': form.cleaned_data['email'],
            })

    else:
        form = SignupForm()

    return render(request, 'signupform.html', {'form': form})

###############################################
'''Vistas pertenecientes a login del usuario'''


###############################################

def login(request):
    if request.method == 'POST':
        template = loader.get_template('index.html')
        context = {}
        email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']

        email = hashlib.sha256(email.encode('utf-8')).hexdigest()
        username = hashlib.sha256(username.encode('utf-8')).hexdigest()
        password = hashlib.sha256(password.encode('utf-8')).hexdigest()


        models.insertUser(email, username, password)

        return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('index.html')
        context = {}
        return HttpResponse(template.render(context, request))


def result(request):
    template = loader.get_template('result.html')
    #username = request.POST['username']
    #password = request.POST['password']
    #check = models.checkPassUser(username, password)

    context = {'username': USER}
    global IDIOMA
    IDIOMA = request.POST['drop1']
    global ESTACION
    ESTACION = request.POST['drop2']
    #global HASHTAGS
    #HASHTAGS = request.POST['drop2']
    return HttpResponse(template.render(context, request))

def config(request):
    if request.method == "POST":
        template = loader.get_template('config.html')
        username = request.POST['username']
        password = request.POST['password']
        check = models.checkPassUser(username, password)
        if check:
            global USER
            USER = request.POST['username']
            context = {'username': username}
            return HttpResponse(template.render(context, request))
        else:
            template = loader.get_template('index.html')
            context = {'error': True}
            return HttpResponse(template.render(context, request))
    else:
        template = loader.get_template('config.html')
        context = {'username':USER}
        return HttpResponse(template.render(context, request))

def mostrarRegistro(request):
    template = loader.get_template('registro.html')  # type: object
    context = {}
    return HttpResponse(template.render(context, request))


###############################################

def getJson(request):
    json_data = open("static/area.json").read()
    data = json.loads(json_data)
    return JsonResponse(data)


def getImage(request):
    with open("static/sentimientos.png", "rb") as f:
        return HttpResponse(f.read(), mimetype="image/png")


def index(request):
    word = "hola"
    context = {
        'palabra': word,
    }
    template = loader.get_template('index.html')
    return HttpResponse(template.render(context, request))

def inicio(request):
    return HttpResponse("¡¡Bienvenido!! Inicio de la web del sistema sanitario")


##############################NUEVO################################################

def wordcloud(text):
    tokens = [word for sent in text for word in sent.split()]
    text = ' '.join(tokens)
    wordcloud = WordCloud(max_font_size=40).generate(text)

    wordcloud.to_file('static/wordcloud.jpg')
    return ('wordcloud.jpg')


def word_frequency(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    freq = np.ravel(X.sum(axis=0))  # sum each columns to get total counts for each word

    # get vocabulary keys, sorted by value
    vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
    fdist = dict(zip(vocab, freq))  # return same format as nltk
    sorted_fdist = sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)[:5]

    x_data = [x[0] for x in sorted_fdist]
    y_data = [x[1] for x in sorted_fdist]

    # Set the x_range to the list of categories above
    p = figure(x_range=x_data, plot_height=300, plot_width=498, title="Palabras más frecuentes")

    # Categorical values can also be used as coordinates
    p.vbar(x=x_data, top=y_data, width=0.9)

    # Set some properties to make the plot look better
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.axis_label = "Palabras"
    p.yaxis.axis_label = "Nº de apariciones"

    script, div = components(p)  # plot
    contexto = {'script': script, 'div': div}

    return script, div
    # return render(request, 'graph.html', contexto)

def user_frequency(texts):
    counter = collections.Counter(texts)
    sorted_fdist = counter.most_common(4)

    x_data = [x[0] for x in sorted_fdist]
    y_data = [x[1] for x in sorted_fdist]

    p = figure(x_range=x_data, plot_height=300, plot_width=512, title="Nº de tweets por regiones (solo usuarios con localización)")

    p.vbar(x=x_data, top=y_data, width=0.9)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.axis_label = "Usuario"
    p.yaxis.axis_label = "Nº de Tweets publicados"

    script, div = components(p)  # plot
    contexto = {'script': script, 'div': div}

    return script, div
    # return render(request, 'graph.html', contexto)


def lda(texts):
    NUM_TOPICS = 10
    vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                                 stop_words='english', lowercase=True,
                                 token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vectorized = vectorizer.fit_transform(texts)

    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(n_topics=NUM_TOPICS, max_iter=10, learning_method='online')
    lda_Z = lda_model.fit_transform(data_vectorized)

    def print_topics(model, vectorizer, top_n=10):
        for idx, topic in enumerate(model.components_):
            print("Topic %d:" % (idx))
            print([(vectorizer.get_feature_names()[i], topic[i])
                   for i in topic.argsort()[:-top_n - 1:-1]])



    # plotear las palabras
    svd = TruncatedSVD(n_components=2)
    words_2d = svd.fit_transform(data_vectorized.T)

    df = pd.DataFrame(columns=['x', 'y', 'word'])
    df['x'], df['y'], df['word'] = words_2d[:, 0], words_2d[:, 1], vectorizer.get_feature_names()

    source = ColumnDataSource(ColumnDataSource.from_df(df))
    labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                      text_font_size="8pt", text_color="#555555",
                      source=source, text_align='center')

    plot = figure(plot_height=300, plot_width=498)
    plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
    plot.add_layout(labels)

    script, div = components(plot)  # plot
    contexto = {'script2': script, 'div2': div}

    return script, div
    # return render(request, 'graph.html', contexto)




def sentimental2(texts, idioma):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    freq = np.ravel(X.sum(axis=0))  # sum each columns to get total counts for each word

    # get vocabulary keys, sorted by value
    vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
    fdist = dict(zip(vocab, freq))  # return same format as nltk
    sorted_fdist = sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)[:500]
    print(sorted_fdist)
    diccionarioSentimientos = {}
    x = []
    y = []
    frequency = [x[1] for x in sorted_fdist]
    for key, value in sorted_fdist:
        if idioma != 'en':
            #traducido = Translator().translate(text=key,dest='en')
            #traducido = translate.Translator(idioma, 'en', key)
            diccionarioSentimientos[key] = analize_sentiment(key)
            '''translator = Translator()
            traducido = translator.translate(text=key,dest='en',src=idioma)
            diccionarioSentimientos[key] = analize_sentiment(str(traducido))'''
        else:
            diccionarioSentimientos[key] = analize_sentiment(key)
        #diccionarioSentimientos = dict(zip(key,value))
    print(diccionarioSentimientos)
    for key, value in diccionarioSentimientos.items():
        x.append(key)
        y.append(value)

    # Fixing random state for reproducibility
    colors = cl.scales['3']['seq']['Blues']
    data = {'Palabra': [1, 0, -1],
            'Polaridad': colors}
    df = pd.DataFrame(data)

    trace0 = go.Table(
        type='table',
        header=dict(
            values=["Palabra","Nº de apariciones","Polaridad [-1, 1]"],
            line=dict(color='black'),
            fill=dict(color='white'),
            align=['center'],
            font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[x,frequency,y],
            #line=dict(color=[df.Polaridad]),
            #fill=dict(color=[df.Polaridad]),
            fill=dict(color=['rgb(245,245,245)','rgb(245,245,245)',  # unique color for the first column
                             ['rgba(250,0,0, 0.8)' if val < 0 else ('rgb(245,245,245)' if val == 0  else 'rgba(0,250,0, 0.8)') for val in y]]),
                            #a = "neg" if b<0 else "pos" if b>0 else "zero"
            align='center',
            font=dict(color='black', size=11)
        ))
    data = [trace0]
    '''trace = go.Table(
        header=dict(values=['Palabra', 'Polaridad [-1, 1]']),
        #cells=dict(values=[[100, 90, 80, 90],                   [95, 85, 75, 95]]))
        cells = dict(values=[x,y])
    )
    data = [trace]'''

    '''plt.scatter(x,y)
    #plt.Axes.tick_params(labelsize='small')
    #plt.axes(labelsize='small')


    plt.tick_params(axis='x', labelsize=6)
    fig1 = plt.gcf()

    fig1.canvas()


    fig1.set_size_inches(10, 3, forward=False)
    fig1.savefig('static/sentimientos.png')
    sentimentalGraph = "sentimientos.png"
    fig1.clf()
    contexto = {'sentimientos': sentimentalGraph}'''

    #layout = go.Layout(width=1000, height=330)
    fig = go.Figure(data=data)

    plotly.offline.plot(fig, filename='static/table.html', auto_open=False)
    #sentimentalGraph = "a-simple-plot.png"
    return 0

    ################################################
def pieSentimental(texts):
    texto = texts
    data = pd.DataFrame(data=texto, columns=['Tweets'])

    #translator = Translator()
    #traducido = translator.translate(text=tweet, dest='en', src='auto')
    #data['SA'] = np.array([analize_sentiment(str(translator.translate(text=tweet, dest='en', src='auto'))) for tweet in data['Tweets']])
    data['SA'] = np.array([analize_sentiment(tweet) for tweet in data['Tweets']])
    pos_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
    neu_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
    neg_tweets = [tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

    pos = len(pos_tweets) * 100 / len(data['Tweets'])
    neg = len(neu_tweets) * 100 / len(data['Tweets'])
    neu = len(neg_tweets) * 100 / len(data['Tweets'])

    pos = float("{0:.1f}".format(pos))
    neg = float("{0:.1f}".format(neg))
    neu = float("{0:.1f}".format(neu))




    colors2 = ['green', 'red', 'grey']
    sizes2 = [pos, neg, neu]
    labels2 = 'Positivos ' + str(pos)+"%", 'Negativos ' + str(neg)+"%", 'Neutros ' + str(neu)+"%"

    ## use matplotlib to plot the chart
    plt.pie(
        x=sizes2,
        shadow=False,
        colors=colors2,
        #labels=labels2,
        startangle=90,
    )

    plt.title("Porcentaje de tweets por estado")
    plt.legend(labels2, loc="best")
    fig1 = plt.gcf()
    fig1.set_size_inches(4.7, 3, forward=False)
    fig1.savefig('static/PieSentimientos.png')
    pieSentimental = "PieSentimientos.png"
    # plt.show()
    fig1.clf()

    #contexto = {'sentimientos': sentimentalGraph}
    return pieSentimental
    #return 0

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(tweet)
    '''if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1'''
    return round(analysis.sentiment.polarity, 2)

def graphLine(estacion,idioma):
    models.dbToCsv(estacion,idioma)
    tweets = pd.read_csv('static/tweets.csv')
    tweets['created_at'] = pd.to_datetime(pd.Series(tweets['created_at']))

    tweets.set_index('created_at', drop=False, inplace=True)

    tweets_pm = tweets['created_at'].resample('M').count()

    # vincent.core.initialize_notebook()
    line = vincent.Line(tweets_pm)
    line.axis_titles(x='Meses', y='Nº Tweets')
    line.colors(brew='Spectral')
    line.to_json('static/area.json')
    return 0

def graphLineIdioma(lang, estacion):
    models.dbToCsv(estacion,lang)
    tweets = pd.read_csv('static/tweets.csv')
    #if(tweets['lang'] == lang):
    #tweets['created_at']['lang'] = pd.to_datetime(pd.Series(tweets['created_at'],tweets['lang']))
    lista = []
    for creacion, idioma in tweets.itertuples(index=False):
        if idioma == lang:
            tweets['created_at']= pd.to_datetime(pd.Series(tweets['created_at']))
            #tweets['created_at'] = pd.to_datetime(pd.Series(creacion))

    tweets.set_index('created_at', drop=False, inplace=True)

    tweets_pm = tweets['created_at'].resample('M').count()

    # vincent.core.initialize_notebook()
    line = vincent.Line(tweets_pm)
    line.axis_titles(x='Meses', y='Nº Tweets')
    line.colors(brew='Spectral')
    line.to_json('static/area.json')
    return 0

def graficos(request):
        a = IDIOMA
        b = ESTACION
        if (b=="primavera"):
            tweets = models.searchTweetPrimavera()
        elif (b == "invierno"):
            tweets = models.searchTweetInvierno()
        elif (b == "verano"):
            tweets = models.searchTweetVerano()
        elif (b == "otono"):
            tweets = models.searchTweetOtono()
        else:
            tweets = models.searchTweet()
        idiom = clean = tweets
        if (a == "all"):

            #tweets = models.searchTweet()

            texts, image2 = clean_tweet(clean)
            #image2 = idiomasFrecuentes(idiom)
            textUser = obtain_user(a,b)
            script1, div1 = lda(texts)
            script2, div2 = word_frequency(texts)

            script3, div3 = user_frequency(textUser)


            image3 = wordcloud(texts)
            image = sentimental2(texts,a)
            graphLine(b,"")
            imagePieSentimental = pieSentimental(texts)

            contexto = {'script3': script3,'div3':div3,'idioma': "Todos", 'username': USER, 'script1': script1, 'div1': div1, 'script2': script2,
                        'div2': div2, 'image2': image2,
                        'image': image, 'image3': image3,'estacion':b,'pieSentimental':imagePieSentimental}

        elif (a == "en"):
            #tweets = models.searchTweet()
            #texts = clean_tweet_languaje(clean,a)
            texts, image2 = clean_tweet_languaje(clean,a)
            #textUser = obtain_user_lang(a)
            textUser = obtain_user(a, b)
            script1, div1 = lda(texts)
            script2, div2 = word_frequency(texts)

            script3, div3 = user_frequency(textUser)

            #image2 = idiomasFrecuentes(idiom)
            image3 = wordcloud(texts)
            image = sentimental2(texts,a)
            #graphLine()
            graphLineIdioma(a,b)
            imagePieSentimental = pieSentimental(texts)
            contexto = {'script3': script3,'div3':div3,'idioma': "Inglés", 'username': USER, 'script1': script1, 'div1': div1, 'script2': script2,
                        'div2': div2, 'image2': image2,
                        'image': image, 'image3': image3,'estacion':b,'pieSentimental':imagePieSentimental}


        elif (a == "es"):
            #tweets = models.searchTweet()
            #texts = clean_tweet_languaje(clean, a)
            texts, image2 = clean_tweet_languaje(clean,a)
            #textUser = obtain_user_lang(a)
            textUser = obtain_user(a, b)
            script1, div1 = lda(texts)
            script2, div2 = word_frequency(texts)
            script3, div3 = user_frequency(textUser)
            #image2 = idiomasFrecuentes(idiom)
            image3 = wordcloud(texts)
            image = sentimental2(texts,a)
            graphLineIdioma(a,b)
            imagePieSentimental = pieSentimental(texts)
            contexto = {'script3': script3,'div3':div3,'idioma': "Español", 'username': USER, 'script1': script1, 'div1': div1, 'script2': script2,
                        'div2': div2, 'image2': image2,
                        'image': image, 'image3': image3,'estacion':b, 'pieSentimental':imagePieSentimental}


        elif (a == "pt"):
            #tweets = models.searchTweet()
            #texts = clean_tweet_languaje(clean, a)
            texts, image2 = clean_tweet_languaje(clean,a)
            #textUser = obtain_user_lang(a)
            textUser = obtain_user(a, b)
            script1, div1 = lda(texts)
            script2, div2 = word_frequency(texts)
            script3, div3 = user_frequency(textUser)
            #image2 = idiomasFrecuentes(idiom)
            image3 = wordcloud(texts)
            image = sentimental2(texts,a)
            graphLineIdioma(a,b)
            imagePieSentimental = pieSentimental(texts)
            contexto = {'script3': script3,'div3':div3,'idioma': "Portugués", 'username': USER, 'script1': script1, 'div1': div1, 'script2': script2,
                        'div2': div2, 'image2': image2,
                        'image': image, 'image3': image3,'estacion':b, 'pieSentimental':imagePieSentimental}

        return render(request, 'graph.html', contexto)

def idiomasFrecuentes(file):

    data = file['lang'].tolist()

    es = 0
    pt = 0
    en = 0
    others = 0
    for x in data:
        if x == 'es':
            es = es + 1
        elif x == 'en':
            en = en + 1
        elif x == 'pt':
            pt = pt + 1
        else:
            others = others + 1

    colors2 = ['blue', 'red', 'green', 'grey']
    sizes2 = [en, es, pt, others]
    labels2 = 'Inglés ' + str(en), 'Español ' + str(es), 'Portugués ' + str(pt), 'Otros ' + str(others)

    ## use matplotlib to plot the chart
    plt.pie(
        x=sizes2,
        shadow=False,
        colors=colors2,
        #labels=labels2,
        startangle=90
    )

    plt.title("Nº de tweets por idioma")
    # sentimentalGraph = figure.savefig('static/sentimientos.png', bbox_inches='tight')
    plt.legend(labels2, loc="best")
    fig2 = plt.gcf()
    fig2.set_size_inches(4.7, 3, forward=True)
    fig2.savefig('static/idiomas.png')
    idioma = "idiomas.png"
    fig2.clf()
    plt.close(fig2)

    return (idioma)


def clean_tweet(file):
    #dataFrame = pd.read_json(file, orient='columns') ORIGINAL
    #data = dataFrame['text'].tolist() ORIGINAl
    df = pd.DataFrame(list(file))

    data = df['text'].tolist()
    lista = []
    for tweet in data:
        lista.append(text_clean(tweet))


    return lista, idiomasFrecuentes(df)


def obtain_user(a,b):
    data = models.searchUbicacion(a,b)
    df = pd.DataFrame(list(data))
    lista = []
    for index, row in df.iterrows():
        if row['user']['location'] != '':
            lista.append(row['user']['location'])


    return lista

def obtain_user_lang(lang):
    file = 'tweets/PruebaSanidad_1000.json'
    with open(file) as data_file:
        data = json.load(data_file)
        list = []
        for v in data:
            if v['lang'] == lang and v['user']['location']!='':
                list.append(v['user']['location'])
    print(list)
    return list

def clean_tweet_languaje(file, lang):
    # Load the first sheet of the JSON file into a data frame
    #df = pd.read_json(file, orient='columns')
    df = pd.DataFrame(list(file))
    if (lang == 'en'):
        columna = df[(df['lang'] == 'en')]
    elif (lang == 'es'):
        columna = df[(df['lang'] == 'es')]
    else:
        columna = df[(df['lang'] == 'pt')]

    data = columna.text.tolist()

    l = []
    for tweet in data:
        s = text_clean(tweet)

        l.append(s)

    return (l), idiomasFrecuentes(df)


def text_clean(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet_no_special_entities = re.sub(r'\&\w*;', '', tweet)
    # Remove tickers
    tweet_no_tickers = re.sub(r'\$\w*', '', tweet_no_special_entities)
    # Remove hyperlinks
    tweet_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', tweet_no_tickers)
    # Remove hashtags
    tweet_no_hashtags = re.sub(r'#\w*', '', tweet_no_hyperlinks)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet_no_punctuation = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet_no_hashtags)
    # Remove https
    tweet_no_https = re.sub(r'http', '', tweet_no_punctuation)
    # Remove words with 2 or fewer letters
    tweet_no_small_words = re.sub(r'\b\w{1,2}\b', '', tweet_no_https)
    # Remove whitespace (including new line characters)
    tweet_no_whitespace = re.sub(r'\s\s+', ' ', tweet_no_small_words)
    tweet_no_whitespace = tweet_no_whitespace.lstrip(' ')  # Remove single space remaining at the front of the tweet.
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet_no_emojis = ''.join(c for c in tweet_no_whitespace if
                              c <= '\uFFFF')  # Apart from emojis (plane 1), this also removes historic scripts and mathematical alphanumerics (also plane 1), ideographs (plane 2) and more.
    # Tokenize: Change to lowercase, reduce length and remove handles
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True,
                           strip_handles=True)  # reduce_len changes, for example, waaaaaayyyy to waaayyy.
    tw_list = tknzr.tokenize(tweet_no_emojis)
    # Remove stopwords
    list_no_stopwords = [i for i in tw_list if i not in cache_english_stopwords]
    list_no_stopwords = [i for i in list_no_stopwords if i not in cache_spanish_stopwords]
    list_no_stopwords = [i for i in list_no_stopwords if i not in cache_portuguese_stopwords]
    # Final filtered tweet
    tweet_filtered = ' '.join(list_no_stopwords)
    return (tweet_filtered)
