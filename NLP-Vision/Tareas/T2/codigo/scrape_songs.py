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
    # Create the data/lyrics directory if it doesn't exist
    os.makedirs('/data/lyrics', exist_ok=True)
    
    filename = artist_name.lower().replace(' ', '_') + '.txt'
    filepath = f'/data/lyrics/{filename}'
    
    f = open(filepath, 'wb')
    urls = request_song_url(artist_name, song_count)
    for i, url in enumerate(urls):
        print(f'Scrapeando canciones {i+1} de {len(urls)}: {url}')
        
        lyrics = scrape_song_lyrics(url)
        if i != len(urls) - 1:
            lyrics += '\n'
            
        f.write(lyrics.encode("utf8"))
    f.close()
    
    # Count lines in the file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for line in f)
    except Exception:
        num_lines = 0
    
    print(f'Se escribieron {num_lines} líneas al archivo {filename}')

def scrape_multiple_artists():
    """Scrape lyrics from 3 artists with specific song counts to total 100 songs"""
    # Define artists and their song counts (33, 33, 34 = 100 total)
    artists_config = [
        {"name": "Kanye West", "song_count": 33},
        {"name": "Jay-Z", "song_count": 33},
        {"name": "Kendrick Lamar", "song_count": 34}
    ]
    
    total_songs = 0
    print("=== Iniciando scraping de múltiples artistas ===")
    
    for config in artists_config:
        artist_name = config["name"]
        song_count = config["song_count"]
        
        print(f"\n--- Procesando {artist_name} ({song_count} canciones) ---")
        try:
            write_lyrics_to_file(artist_name, song_count)
            total_songs += song_count
            print(f"✓ Completado: {artist_name}")
        except Exception as e:
            print(f"✗ Error procesando {artist_name}: {e}")
    
    print("\n=== Scraping completado ===")
    print(f"Total de canciones procesadas: {total_songs}")
    print("Archivos generados en ../data/lyrics/")

def scrape_custom_artists(artists_list):
    """Scrape lyrics from custom list of artists with song counts"""
    total_songs = 0
    print("=== Iniciando scraping de artistas personalizados ===")
    
    for artist_config in artists_list:
        if isinstance(artist_config, dict):
            artist_name = artist_config.get("name", "")
            song_count = artist_config.get("song_count", 33)
        else:
            # If it's just a string, use default song count
            artist_name = artist_config
            song_count = 33
        
        if not artist_name:
            print("⚠ Nombre de artista vacío, saltando...")
            continue
            
        print(f"\n--- Procesando {artist_name} ({song_count} canciones) ---")
        try:
            write_lyrics_to_file(artist_name, song_count)
            total_songs += song_count
            print(f"✓ Completado: {artist_name}")
        except Exception as e:
            print(f"✗ Error procesando {artist_name}: {e}")
    
    print("\n=== Scraping completado ===")
    print(f"Total de canciones procesadas: {total_songs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrapea letras de canciones desde Genius.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplos de uso:
  # Scrapear un solo artista:
  python scrape_songs.py "Taylor Swift" 50
  
  # Scrapear 3 artistas predefinidos (33, 33, 34 canciones = 100 total):
  python scrape_songs.py --multiple
  
  # Modo interactivo para definir artistas personalizados:
  python scrape_songs.py --interactive"""
    )
    
    # Make artist and song_count optional when using --multiple or --interactive
    parser.add_argument("artist", type=str, nargs='?', help="Nombre del artista")
    parser.add_argument("song_count", type=int, nargs='?', help="Cantidad de canciones a scrapear")
    parser.add_argument("--multiple", action="store_true", 
                       help="Scrapear 3 artistas predefinidos (Taylor Swift: 33, Drake: 33, Kendrick Lamar: 34)")
    parser.add_argument("--interactive", action="store_true",
                       help="Modo interactivo para definir artistas personalizados")
    
    args = parser.parse_args()
    
    if args.multiple:
        scrape_multiple_artists()
    elif args.interactive:
        print("=== Modo Interactivo ===")
        print("Define los artistas que quieres scrapear.")
        print("Presiona Enter sin escribir nada para terminar.\n")
        
        artists_list = []
        while True:
            artist_name = input("Nombre del artista: ").strip()
            if not artist_name:
                break
                
            try:
                song_count = int(input(f"Cantidad de canciones para {artist_name} [33]: ") or "33")
            except ValueError:
                song_count = 33
                
            artists_list.append({"name": artist_name, "song_count": song_count})
            print(f"✓ Agregado: {artist_name} ({song_count} canciones)\n")
        
        if artists_list:
            scrape_custom_artists(artists_list)
        else:
            print("No se agregaron artistas.")
    else:
        # Single artist mode (original behavior)
        if not args.artist or not args.song_count:
            parser.error("Se requieren 'artist' y 'song_count' para el modo de un solo artista.")
        
        write_lyrics_to_file(args.artist, args.song_count)