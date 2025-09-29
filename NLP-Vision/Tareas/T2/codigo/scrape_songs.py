import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import argparse
import re

load_dotenv('./.env')
GENIUS_API_TOKEN = os.getenv('GENIUS_API_TOKEN')

# Obtener artista de Genius.com
def request_artist_info(artist_name, page):
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + GENIUS_API_TOKEN}
    search_url = base_url + '/search?per_page=10&page=' + str(page)
    data = {'q': artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    return response

# Obtener URL's de las canciones del artista, hasta song_cap.
def request_song_url(artist_name, song_cap):
    page = 1
    songs = []
    
    while True:
        response = request_artist_info(artist_name, page)
        json = response.json()
        # Collect up to song_cap song objects from artist
        song_info = []
        for hit in json['response']['hits']:
            if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
                song_info.append(hit)
    
        # Collect song URL's from song objects
        for song in song_info:
            if (len(songs) < song_cap):
                url = song['result']['url']
                songs.append(url)
            
        if (len(songs) == song_cap):
            break
        else:
            page += 1
        
    print(f'Se encontraron {len(songs)} canciones de {artist_name}.')
    return songs

# Scrapear lyrics de la pagina de la cancion
def scrape_song_lyrics(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics_containers = html.find_all(attrs={"data-lyrics-container": "true"})
    if not lyrics_containers:
        print("No se encontraron las letras en la página:", url)
        return ""

    lyrics_parts = []
    for container in lyrics_containers:
        lyrics_parts.append(container.get_text(separator='\n', strip=True))
    raw_lyrics = ' '.join(lyrics_parts)

    clean_lyrics = ' '.join(line.strip() for line in raw_lyrics.splitlines() if line.strip())

    idx = clean_lyrics.find('Read More')

    # Remove "Read More" and everything before it (metadata of song)
    clean_lyrics = clean_lyrics[idx+10:] if idx != -1 else clean_lyrics 

    # Remove annotations like [Chorus], [Verse 1], etc.
    clean_lyrics = re.sub(r'\[.*?\]', '', clean_lyrics)

    # Song boundary markers
    clean_lyrics = f"<|song_start|>{clean_lyrics.strip()}<|song_end|>"

    return clean_lyrics

def write_lyrics_to_file(artist_name: str, song_count: int):
    f = open('lyrics/' + artist_name.lower() + '.txt', 'wb')
    urls = request_song_url(artist_name, song_count)
    for i, url in enumerate(urls):
        print(f'Scrapeando canciones {i+1} de {len(urls)}: {url}')
        
        lyrics = scrape_song_lyrics(url)
        if i != len(urls) - 1:
            lyrics += '\n'
            
        f.write(lyrics.encode("utf8"))
    f.close()
    num_lines = sum(1 for line in open('../data/lyrics/' + artist_name.lower().replace(' ', '_') + '.txt', 'rb'))
    print(f'Se escribieron {num_lines} al archivo {artist_name.lower().replace(" ", "_")}.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrapea letras de canciones de un artista desde Genius.com")
    parser.add_argument("artist", type=str, help="Nombre del artista")
    parser.add_argument("song_count", type=int, help="Cantidad de canciones a scrapear")
    args = parser.parse_args()

    write_lyrics_to_file(args.artist, args.song_count)