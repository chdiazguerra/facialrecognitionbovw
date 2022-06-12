"""Descarga la base de datos utilizada para probar el modelo creado.
Este corresponde a Pain expressions de http://pics.psych.stir.ac.uk/2D_face_sets.htm
"""

import os
import argparse
import zipfile
import requests

def download(args: argparse.Namespace):
    """Descarga el dataset

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos entregados por el usuario
    """

    URL = "http://pics.psych.stir.ac.uk/zips/pain.zip"
    PATH_DOWNLOAD = os.path.join(args.path, "data.zip")
    PATH_UNZIP = os.path.join(args.path, "pain")

    if not os.path.isdir(args.path):
        os.mkdir(args.path)

    if (not os.path.isfile(PATH_DOWNLOAD) and not os.path.isdir(PATH_UNZIP)) or args.f:
        print("Downloading. It may takes a few minutes.")

        response = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'})

        with open(PATH_DOWNLOAD, "wb") as file:
            file.write(response.content)

        print("Downloaded.")

    if (os.path.isfile(PATH_DOWNLOAD) and not os.path.isdir(PATH_UNZIP)) or args.f:
        print("Decompressing.")
        if not os.path.isdir(PATH_UNZIP):
            os.mkdir(PATH_UNZIP)

        with zipfile.ZipFile(PATH_DOWNLOAD,"r") as zip_ref:
            zip_ref.extractall(PATH_UNZIP)

    if args.e and os.path.isfile(PATH_DOWNLOAD):
        print("Removing downloaded file.")
        os.remove(PATH_DOWNLOAD)

    print("Terminated.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        help="Path de la carpeta en donde ser guarda el archivo descargado",
                        nargs="?", type=str, default="data")
    parser.add_argument("-e", help="Elimina el zip descargado y deja el descomprimido",
                        action="store_true")
    parser.add_argument("-f", help="Fuerza la descarga", action="store_true")

    parse_args = parser.parse_args()

    download(parse_args)
