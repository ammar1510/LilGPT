import requests
from bs4 import BeautifulSoup
import re
import sys

api_key = "7u3iSB_yrE9yEIbg_wbVjWyLg_byy4TwpzwbIbLLhMpho07tMt-fHJ-3rAhOBzg3"
web_url = "https://genius.com"
api_url = "https://api.genius.com/search/"
song_url = "https://api.genius.com/"
file_path = "songs.txt"

artist_name = "drake"
if len(sys.argv)>1:
    artist_name = sys.argv[1]

print("artist: ",artist_name)
error = "error.txt"
err_f = open(error,"w")

params = {"q":artist_name,
          "access_token":api_key
          }


def fetch(url,params=None,api=False,web=True):
    if api:
        url = api_url+url
    elif web:
        url = web_url+url
    else:
        url = song_url+url

    if params is None:
        params = {}

    session=requests.Session()
    cnt = 3
    response = None
    
    while response is None and cnt>0:
        cnt-=1
        response = session.get(url,timeout=10,params=params)

    if response is None:
        print("Retry limit exceeded")
    elif response.status_code==200:
        if web:
            return response.text
        else:
            return response.json()
    print(f"Error: {response.status_code}")


def get_songs(params,file):

    titles = []
    
    data = fetch("",params=params,api=True,web=False)
    if data is not None:
        artist_id = data['response']['hits'][0]['result']['primary_artist']['id']
    else:
        print(f"Error")
        
    prev = None

    for i in range(100):
        params['page']=i+1

        song_data = fetch(f"artists/{artist_id}/songs",params=params,api=False,web=False)
        if song_data is None:
            print("Unable to fetch songs")
            continue
        songs = [song['title'] for song in song_data['response']['songs']]
    
        for song in songs:
            if song == prev:
                continue
            prev=song
            file.write(song+"\n")
    
    

def get_lyrics(songs,params,write_to):#songs is a file
    
    song_list=songs.readlines()
    #print(song_list) 

    for song in song_list:
        try:
           params['q']=song
           #print(song)
           song_data = fetch("",params,api=True,web=False)
           #print(song_data)
           song_path = song_data["response"]["hits"][0]["result"]["path"]
           
           lyrics_data = fetch(song_path,params,api=False)
           html = BeautifulSoup(lyrics_data.replace('<br/>','\n'),"html.parser")
           divs = html.find_all("div", class_=re.compile("^lyrics$|Lyrics__Container"))
           
           if divs is None or len(divs)==0:
               print(f"Couldn't find lyrics to {song}")
               continue

           lyrics = "\n".join([div.get_text() for div in divs])
           write_to.write(lyrics + "\n")
        except Exception as e:
            
            err_f.write(song+"/n")
            continue

if __name__ == "__main__":
    
    with open(file_path,"w") as file:
        get_songs(params,file)

    lyrics_path = "lyrics.txt"
    with open(lyrics_path,"w") as file: 
        song_file = open(file_path,"r")
        get_lyrics(song_file,params,file)
        song_file.close()
    
