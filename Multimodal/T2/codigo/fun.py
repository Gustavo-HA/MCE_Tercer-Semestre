from music21 import converter, note, chord
import os
import pandas as pd
import re


def get_all_notes(midi_files, folder_path, prefix_file="notes"):
    """
    Extrae notas, acordes y silencios de archivos MIDI en un directorio y los guarda en archivos de texto.

    Para cada archivo MIDI en el directorio especificado, la función analiza el archivo, extrae todas las notas,
    acordes y silencios, y los escribe como una cadena separada por espacios en un archivo de texto en la carpeta destino.
    Además, recopila metadatos de los archivos procesados y los retorna en un DataFrame de pandas.

    Parámetros
    ----------
    midi_files : str
        Ruta al directorio que contiene los archivos MIDI a procesar.
    folder_path : str
        Ruta al directorio donde se guardarán los archivos de texto generados.
    prefix_file : str, opcional
        Prefijo para los nombres de los archivos de texto generados (por defecto es 'notes').

    Retorna
    -------
    df : pandas.DataFrame
        DataFrame con metadatos de cada archivo MIDI procesado, con las columnas:
        - 'output_file': Nombre del archivo de texto generado.
        - 'file_name': Nombre del archivo MIDI original correspondiente.

    Notas
    -----
    - Solo se procesan archivos con extensión '.mid'.
    - Si un archivo MIDI no puede ser analizado, se omite y se imprime un mensaje de error.
    - Cada archivo de texto contiene una lista separada por espacios de notas, acordes y silencios extraídos del archivo MIDI.
    - Las notas se representan como "<pitch>.<duration>", los acordes como números de pitch separados por puntos, y los silencios como "<rest>.<duration>".
    """

    temp = 0
    records = []
    for file in os.scandir(midi_files):
        notes = []
        print("Analizando %s" % file.name)

        try:
            midi = converter.parse(file.path)
        except Exception:  # Catches any other unexpected exceptions
            print("Midi Error and skip %s" % file.name)
            continue

        notes_to_parse = None

        # en esta parte, se extraen todas las notas del archivo midi
        # como 'midi' es un objeto Score (que a su vez hereda de un Stream),
        # pueden accederse a todas las partes de la partitura (Score)
        notes_to_parse = midi.flat.notesAndRests
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch) + "." + str(element.duration.type))
            elif isinstance(element, chord.Chord):
                notes.append(".".join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest):
                notes.append(str(element.name) + "." + str(element.duration.type))

        output_file = output_file = os.path.join(
            folder_path, prefix_file + "_" + str(temp) + ".txt"
        )
        with open(output_file, "w") as f:
            f.write(" ".join(str(n) for n in notes))
        records.append(
            {
                "output_file": prefix_file + "_" + str(temp) + ".txt",
                "file_name": file.name,
            }
        )
        temp += 1

    df = pd.DataFrame(records)
    return df


# mi tokenizador
def my_tokenizer(text):
    return re.split("\\s+", text)
