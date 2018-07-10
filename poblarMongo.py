#!/usr/bin/python
#-*- encoding: utf-8 -*-
####################################################USO: Pasar como agumento el hashtag que se quiera descargar#######################333333
import json
import pymongo
import sys
import twitter
import csv

def file_to_mongodb():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    record = db.tweets_todo          #cambiar collection si se desea
    page = open("PruebaSanidad_1000.json", 'r') #modo lectura
    #page = open(files, 'r')
    parsed = json.loads(page.read())

    for item in parsed:
        record.insert(item)
        
def api_to_mongodb(hashtag):
    api = twitter.Api(consumer_key='XXXXXXXXXXXXXXXXXXXXX',
    consumer_secret='XXXXXXXXXXXXXXXXXXXXXXXXXXX',
    access_token_key='XXXXXXXXXX-XXXXXXXXXXXXX',
    access_token_secret='XXXXXXXXXXXXXXXXXXXXXXX')
    
    search = api.GetSearch(raw_query="l=es&q=%23"+hashtag+"%20since%3A2017-07-04%20until%3A2018-02-20&count=99")
    
    #search = api.GetSearch('#'+hashtag)
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
    #db.tweets_collection_api.ensureIndex({"coordinates.coordinates": "2d"})
    #record = db.tweets_collection_api.find({"coordinates.coordinates": {"$within":{ "$centerSphere": [ [40.4167,-3.70325], 650/6378.1] }}})
    record = db.tweets_collection_api.find()
    
    return record

def dbToCsv():
    connection = pymongo.MongoClient("mongodb://localhost:27017")
    db = connection.tweets
    with open('tweets.csv', 'w') as outfile:
      fieldnames = ['text', 'user', 'created_at', 'geo']
      writer = csv.DictWriter(outfile, delimiter=',', fieldnames=fieldnames)
      writer.writeheader()

      for data in db.tweets_collection_geo.find():
        writer.writerow({ 
          'text': json.dumps(data['text']), 
          'user': json.dumps(data['user']), 
          'created_at': data['created_at'],
          'geo': data['geo']
        })

      outfile.close()
#file_to_mongodb()
api_to_mongodb(sys.argv[1])
#GeoLocatedRead()
#dbToCsv()



