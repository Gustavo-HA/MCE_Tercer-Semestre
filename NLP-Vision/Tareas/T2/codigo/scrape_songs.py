import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
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

# Extract song metadata and lyrics separately for LLM training
def scrape_song_with_metadata(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    
    # Extract song title and artist from page
    song_title = ""
    try:
        title_elem = html.find('h1', class_='SongHeaderdesktop__HiddenMask-sc-1effuo1-11')
        if not title_elem:
            title_elem = html.find('h1')
        if title_elem:
            song_title = title_elem.get_text(strip=True)
    except Exception:
        pass
    
    # Extract album info if available
    album_info = ""
    try:
        album_elem = html.find('a', href=re.compile(r'/albums/'))
        if album_elem:
            album_info = album_elem.get_text(strip=True)
    except Exception:
        pass
    
    # Extract lyrics
    lyrics_containers = html.find_all(attrs={"data-lyrics-container": "true"})
    if not lyrics_containers:
        print("No se encontraron las letras en la página:", url)
        return "", "", "", ""

    lyrics_parts = []
    for container in lyrics_containers:
        lyrics_parts.append(container.get_text(separator='\n', strip=True))
    raw_lyrics = ' '.join(lyrics_parts)

    clean_lyrics = ' '.join(line.strip() for line in raw_lyrics.splitlines() if line.strip())

    # Remove "Read More" and everything before it (this is always junk)
    idx = clean_lyrics.find('Read More')
    if idx != -1:
        clean_lyrics = clean_lyrics[idx+10:]
    
    # Only remove web-specific junk, preserve musical metadata
    web_junk_patterns = [
        r'\d+\s+Contributors?\s+',
        r'Translations?\s+[A-Z][a-z]+\s+',
        r'Embed.*?Copy.*?',
        r'You might also like',
        r'Get tickets as low as \$\d+',
        r'EmbedShare URLCopyEmbedCopy',
        r'See .*? Lyrics Meaning',
        r'\d+Embed$'
    ]
    
    for pattern in web_junk_patterns:
        clean_lyrics = re.sub(pattern, '', clean_lyrics, flags=re.IGNORECASE)
    
    # Keep song structure annotations like [Verse 1], [Chorus], etc.
    # Keep producer credits and featured artists
    # Only remove excessive repetition of metadata
    
    # Split into lyrics and metadata
    lyrics_content = clean_lyrics
    metadata_content = f"Title: {song_title}" + (f" | Album: {album_info}" if album_info else "")
    
    # Clean up extra spaces
    lyrics_content = re.sub(r'\s+', ' ', lyrics_content).strip()
    
    return song_title, album_info, metadata_content, lyrics_content

def write_lyrics_to_tsv(artist_name: str, song_count: int, tsv_writer):
    """Write lyrics with metadata for one artist to the TSV file for LLM training"""
    urls = request_song_url(artist_name, song_count)
    
    for i, url in enumerate(urls):
        print(f'Scrapeando {artist_name} - canción {i+1} de {len(urls)}: {url}')
        
        song_title, album_info, metadata, lyrics = scrape_song_with_metadata(url)
        
        if lyrics.strip():  # Only write if we got actual lyrics
            # Format for LLM training: include metadata as context
            # This helps the model learn about song structure and music industry
            
            # Clean up for TSV format (replace tabs/newlines with spaces)
            lyrics_clean = re.sub(r'[\t\n\r]+', ' ', lyrics)
            lyrics_clean = re.sub(r'\s+', ' ', lyrics_clean).strip()
            
            metadata_clean = re.sub(r'[\t\n\r]+', ' ', metadata)
            metadata_clean = re.sub(r'\s+', ' ', metadata_clean).strip()
            
            # Write to TSV: Artist \t Metadata \t Lyrics
            tsv_writer.writerow([artist_name, metadata_clean, lyrics_clean])
    
    print(f'✓ Completado: {artist_name} ({len(urls)} canciones procesadas)')

def scrape_multiple_artists():
    """Scrape lyrics from 3 artists with specific song counts to total 100 songs and save as TSV"""
    import csv
    
    # Define artists and their song counts (33, 33, 34 = 100 total)
    artists_config = [
        {"name": "Kanye West", "song_count": 33},
        {"name": "Jay-Z", "song_count": 33},
        {"name": "Kendrick Lamar", "song_count": 34}
    ]
    
    # Create the data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Output file path
    output_file = './data/lyrics_dataset.tsv'
    
    total_songs = 0
    print("=== Iniciando scraping de múltiples artistas ===")
    print(f"Generando archivo TSV: {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        # Create TSV writer
        tsv_writer = csv.writer(tsvfile, delimiter='\t')
        
        # Write header for LLM training format
        tsv_writer.writerow(['Artist', 'Metadata', 'Lyrics'])
        
        for config in artists_config:
            artist_name = config["name"]
            song_count = config["song_count"]
            
            print(f"\n--- Procesando {artist_name} ({song_count} canciones) ---")
            try:
                write_lyrics_to_tsv(artist_name, song_count, tsv_writer)
                total_songs += song_count
            except Exception as e:
                print(f"✗ Error procesando {artist_name}: {e}")
    
    print("\n=== Scraping completado ===")
    print(f"Total de canciones procesadas: {total_songs}")
    print(f"Archivo generado: {output_file}")
    print("\nFormato del archivo (optimizado para fine-tuning de LLM):")
    print("Columna 1: Artist (nombre del artista)")
    print("Columna 2: Metadata (título, álbum, y contexto musical)")
    print("Columna 3: Lyrics (letras con anotaciones estructurales preservadas)")

if __name__ == "__main__":
    print("=== Scraper de Letras para Fine-tuning de LLM ===")
    print("Modo: 3 artistas predefinidos (Kanye West: 33, Jay-Z: 33, Kendrick Lamar: 34)")
    scrape_multiple_artists()