import hashlib

from bson import json_util
from bson.json_util import dumps
from django.db import models
import json
import pymongo
import sys
import twitter
import csv

# Create your models here.
def insertUser(email,username,password):
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.usu
    record.insert({"email":email,"user":username,"pass":password})
    return 0

def checkPassUser(username,password):
    username = hashlib.sha256(username.encode('utf-8')).hexdigest()
    password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.usu
    try:
        userExists = False
        if record.find({"user":username,"pass":password}).count() > 0:
            userExists = True

            return userExists
    except Exception as e:
        return userExists

def file_to_mongodb(files):
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.tweets_collection_geo          #cambiar collection si se desea
    #page = open("PruebaSanidad_10.json", 'r') #modo lectura
    page = open(files, 'r')
    parsed = json.loads(page.read())

    for item in parsed:
        record.insert(item)

def api_to_mongodb(hashtag):
    api = twitter.Api(consumer_key='g5HJFxdsuDJ7LRadAd7oY7WQs',
                      consumer_secret='Lsu6unYx3kc2WfxPH8qguIS0xwMXudQ3nw5feOwEPvpNzP5LYb',
                      access_token_key='2757961272-otTy91oLvomHBJ2n6s3TJthBkkr1ZlHTRzembG6',
                      access_token_secret='FRsgmSsLYKIBUtPKL8yl7lN94NtTybwkZGp3tQ2RwSEgU')

    search = api.GetSearch(raw_query="l=es&q=%23" + hashtag + "%20since%3A2017-07-04%20until%3A2018-02-20&count=99")

    # search = api.GetSearch('#'+hashtag)
    file = open('tweetsFromApi.json', 'w')
    file.write('[')
    length = len(search)
    il = 0

    for tweet in search:
        if il < length:
            textTweet = str(tweet)
            file.write(textTweet)
            file.write(',')
            il += 1
    else:
        textTweet = str(tweet)
        file.write(textTweet)

    file.write(']')
    file.close()
    file_to_mongodb('tweetsFromApi.json')


def GeoLocatedRead():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    # db.tweets_collection_api.ensureIndex({"coordinates.coordinates": "2d"})
    record = db.tweets_collection_api.find({"coordinates.coordinates": {"$within":{ "$centerSphere": [ [40.4167,-3.70325], 650/6378.1] }}})
    #record = db.tweets_collection_api.find()

    return record

def searchTweetByLanguage(lan):
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.tweets_collection_api.find({ "lang": lan })
    return record

def dbToCsv(database,idioma):
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets

    with open('static/tweets.csv', 'w') as outfile:
        fieldnames = ['created_at','lang']
        writer = csv.DictWriter(outfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        if database =="primavera" and idioma == "":
            for data in db.tweets_primavera.find():
                writer.writerow({


                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        elif database =="invierno" and idioma == "":
            for data in db.tweets_invierno.find():
                writer.writerow({


                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        elif database =="primavera" and idioma != "":
            for data in db.tweets_primavera.find({"lang": idioma}):
                writer.writerow({


                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        elif database == "invierno" and idioma != "":
            for data in db.tweets_invierno.find({"lang": idioma}):
                writer.writerow({

                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        ########################################################
        elif database =="verano" and idioma == "":
            for data in db.tweets_verano.find():
                writer.writerow({


                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        elif database =="otono" and idioma == "":
            for data in db.tweets_otonyo.find():
                writer.writerow({


                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        elif database =="verano" and idioma != "":
            for data in db.tweets_verano.find({"lang": idioma}):
                writer.writerow({


                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        elif database == "otono" and idioma != "":
            for data in db.tweets_otonyo.find({"lang": idioma}):
                writer.writerow({

                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()
        ##########################################################
        elif idioma != "":
            for data in db.tweets_todo.find({"lang": idioma}):
                writer.writerow({

                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()

        else:
            for data in db.tweets_todo.find():
                writer.writerow({
                    'created_at': data['created_at'],
                    'lang': data['lang'],

                })

            outfile.close()

def searchTweet():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    #record = db.tweets_collection_geo.find()
    record = db.tweets_todo.find()
    #json_docs = [json.dumps(doc, default=json_util.default) for doc in record]
    #print(json_docs)
    #record = json.dumps(record)

    return record

def searchUbicacion(lang, periodo):
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets

    if periodo == "primavera" and lang == "all":
        consulta = db.tweets_primavera.find()
    elif periodo == "invierno" and lang == "all":
        consulta = db.tweets_invierno.find()

    elif periodo == "primavera" and lang != "all":
        consulta = db.tweets_primavera.find({"lang": lang})

    elif periodo == "invierno" and lang != "all":
        consulta = db.tweets_invierno.find({"lang": lang})
########################################################3
    elif periodo == "verano" and lang == "all":
        consulta = db.tweets_verano.find()
    elif periodo == "otono" and lang == "all":
        consulta = db.tweets_otonyo.find()

    elif periodo == "verano" and lang != "all":
        consulta = db.tweets_verano.find({"lang": lang})

    elif periodo == "otono" and lang != "all":
        consulta = db.tweets_otonyo.find({"lang": lang})
    ####################################################333


    elif lang != "all":
        consulta = db.tweets_todo.find({"lang": lang})


    else:
        consulta = db.tweets_todo.find()

    print("la consulta")
    print(consulta)
    return consulta

def searchTweetPrimavera():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.tweets_primavera.find()
    return record
def searchTweetVerano():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.tweets_verano.find()
    return record
def searchTweetOtono():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.tweets_otonyo.find()
    return record

def searchTweetInvierno():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.tweets_invierno.find()
    return record

# file_to_mongodb(sys.argv[1])
# api_to_mongodb(sys.argv[1])
# GeoLocatedRead()
# dbToCsv()
