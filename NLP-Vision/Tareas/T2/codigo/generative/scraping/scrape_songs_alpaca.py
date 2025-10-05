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
        return "", "", "", ""

    lyrics_parts = []
    for container in lyrics_containers:
        lyrics_parts.append(container.get_text(separator='\n', strip=True))
    raw_lyrics = '\n'.join(lyrics_parts)

    # Clean each line but preserve line structure
    lines = raw_lyrics.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line:  # Only keep non-empty lines
            clean_lines.append(line)
    
    clean_lyrics = '\n'.join(clean_lines)

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
        clean_lyrics = re.sub(pattern, '', clean_lyrics, flags=re.IGNORECASE | re.MULTILINE)
    
    # Keep song structure annotations like [Verse 1], [Chorus], etc.
    # Keep producer credits and featured artists
    # Preserve line breaks for proper verse structure
    
    # Split into lyrics and metadata
    lyrics_content = clean_lyrics
    metadata_content = f"Title: {song_title}" + (f" | Album: {album_info}" if album_info else "")
    
    # Fix line breaks in tags and parentheses, add spacing between sections
    lyrics_content = format_lyrics_structure(lyrics_content)
    
    return song_title, album_info, metadata_content, lyrics_content

def format_lyrics_structure(lyrics_text: str) -> str:
    """Format lyrics with proper line breaks for tags, parentheses, and section transitions"""
    
    # First, clean up extra whitespace while preserving single line breaks
    lyrics_text = re.sub(r'[ \t]+', ' ', lyrics_text).strip()
    
    # Remove line breaks within square brackets (tags like [Chorus: Artist])
    # Use re.DOTALL to handle multiline cases within brackets
    lyrics_text = re.sub(r'\[([^\]]*?)\n([^\]]*?)\]', r'[\1 \2]', lyrics_text, flags=re.DOTALL)
    # Continue until no more line breaks in brackets
    while re.search(r'\[[^\]]*\n[^\]]*\]', lyrics_text):
        lyrics_text = re.sub(r'\[([^\]]*?)\n([^\]]*?)\]', r'[\1 \2]', lyrics_text, flags=re.DOTALL)
    
    # Remove line breaks within parentheses
    # Use re.DOTALL to handle multiline cases within parentheses
    lyrics_text = re.sub(r'\(([^\)]*)\n([^\)]*)\)', r'(\1 \2)', lyrics_text, flags=re.DOTALL)
    # Continue until no more line breaks in parentheses
    while re.search(r'\([^\)]*\n[^\)]*\)', lyrics_text):
        lyrics_text = re.sub(r'\(([^\)]*)\n([^\)]*)\)', r'(\1 \2)', lyrics_text, flags=re.DOTALL)
    
    # Add double line breaks before section markers (but not at the start)
    # Common section markers: [Verse], [Chorus], [Bridge], [Outro], [Intro], etc.
    section_pattern = r'(?<!^)(?<!\n\n)(\[[A-Za-z][^\]]*\])'
    lyrics_text = re.sub(section_pattern, r'\n\n\1', lyrics_text)
    
    # Clean up excessive empty lines (more than 2 consecutive)
    lyrics_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', lyrics_text)
    
    # Clean up any leading/trailing whitespace
    lyrics_text = lyrics_text.strip()
    
    return lyrics_text

def write_lyrics_to_alpaca(artist_name: str, song_count: int, dataset_list: list):
    """Write lyrics with metadata in Alpaca instruction format for LLM training"""
    urls = request_song_url(artist_name, song_count)
    
    # Variation in instructions to make the model more flexible
    instruction_templates = [
        "You are a hip-hop artist helping people write lyrics.",
        "As a rap artist, help write song lyrics.",
        "You are a skilled hip-hop lyricist. Write rap lyrics.",
        "Acting as a hip-hop artist, create song lyrics.",
        "You are an experienced rapper. Help write lyrics."
    ]
    
    for i, url in enumerate(urls):
        print(f'Scrapeando {artist_name} - canción {i+1} de {len(urls)}: {url}')
        
        song_title, album_info, metadata, lyrics = scrape_song_with_metadata(url)
        
        if lyrics.strip():  # Only write if we got actual lyrics
            # Preserve line breaks for proper verse structure
            # Don't convert to single line - keep the song structure
            lyrics_clean = lyrics.strip()
            
            # Create Alpaca format entry with proper separation:
            # instruction = the general task (what to do)
            # input = specific constraints/context (artist style, album context)
            # output = the actual lyrics with preserved structure
            
            # Rotate through instruction templates for variety
            instruction = instruction_templates[i % len(instruction_templates)]
            
            # Input contains the query for the artist style
            input_context = f"Help me write a song in the style of {artist_name}."
            
            # Create the Alpaca format dictionary
            alpaca_entry = {
                "instruction": instruction,
                "input": input_context,
                "output": lyrics_clean
            }
            
            dataset_list.append(alpaca_entry)
    
    print(f'Completado: {artist_name} ({len(urls)} canciones procesadas)')

def scrape_multiple_artists():
    """Scrape lyrics from 3 artists with specific song counts to total 100 songs and save in Alpaca format"""
    import json
    import random
    
    # Define artists and their song counts (100 each)
    artists_config = [
        {"name": "Kanye West", "song_count": 100},
        {"name": "Jay-Z", "song_count": 100},
        {"name": "Kendrick Lamar", "song_count": 100}
    ]
    
    # Create the data directory if it doesn't exist
    os.makedirs('./data/text_gen', exist_ok=True)
    
    # Output file paths
    train_json = './data/text_gen/train_lyrics_alpaca.json'
    test_json = './data/text_gen/test_lyrics_alpaca.json'

    # List to hold all dataset entries
    dataset = []
    
    total_songs = 0
    print("=== Iniciando scraping de múltiples artistas ===")
    
    for config in artists_config:
        artist_name = config["name"]
        song_count = config["song_count"]
        
        print(f"\n--- Procesando {artist_name} ({song_count} canciones) ---")
        try:
            write_lyrics_to_alpaca(artist_name, song_count, dataset)
            total_songs += song_count
        except Exception as e:
            print(f"✗ Error procesando {artist_name}: {e}")
    
    # Shuffle dataset for random train/test split
    random.seed(42)  # For reproducibility
    random.shuffle(dataset)

    # Split into train (80%) and test (20%)
    split_idx = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    # Save train set
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=2)
    
    # Save test set
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)
    
    print("\n=== Scraping completado ===")
    print(f"Total de canciones procesadas: {total_songs}")
    print("\nArchivos generados:")
    print(f"  - Train: {train_json}")
    print(f"  - Test:  {test_json}")

if __name__ == "__main__":
    print("=== Scraper de Letras para Fine-tuning de LLM ===")
    print("Formato: Alpaca instruction-following (compatible con HuggingFace)")
    print("Estructura: instruction (tarea) + input (estilo/contexto) -> output (lyrics)")
    print("Artistas: Kanye West (100), Jay-Z (100), Kendrick Lamar (100)")
    print("Total: 100 canciones | Split: 80% train, 20% test\n")
    scrape_multiple_artists()