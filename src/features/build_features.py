""" Genera las features de todas las imagenes.
"""

import argparse
import os
import sys
import pickle

import json
import yaml
from tqdm import tqdm
from feature_extraction import get_extractor

def extraer(args):
    if not os.path.isdir(args.folder_imgs):
        print("La carpeta de imagenes no existe")
        sys.exit()

    if not os.path.isfile(args.dataset_file):
        print("No se ha encontrado el archivo con los datos de la imagenes")
        sys.exit()

    if not os.path.isfile(args.config_file):
        print("No se ha encontrado el archivo con la configuracion")
        sys.exit()

    extractor = get_extractor(args.config_file, args.features_method)

    with open(args.dataset_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("Extrayendo features")

    features = {}
    for file in tqdm(data):
        nombre = file["nombre"]
        archivo = os.path.join(args.folder_imgs, nombre)
        feature = extractor.extract(archivo)
        features[nombre] = feature

    OUT = os.path.join(args.out, "features_"+args.features_method+".pkl")
    with open(OUT, "wb") as file:
        pickle.dump(features, file)

    print("Terminado")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_imgs", help="Path de la carpeta con las imagenes",
                        nargs="?", type=str, default="data/pain")
    parser.add_argument("--out", help="Carpeta en donde se guarda el archivo de salida",
                        nargs="?", type=str, default="data")
    parser.add_argument("--dataset_file", help="Archivo json con los datos del dataset",
                        nargs="?", type=str, default="data/data.json")
    parser.add_argument("--features_method",
                        help="Metodo para extraer las features de las imagenes",
                        nargs="?", type=str,
                        choices=["sift", "brisk", "orb", "daisy"], default="sift")
    parser.add_argument("--config_file",
                        help="Archivo con la configuracion del extractor de features escogido",
                        nargs="?", type=str, default="src/features/config.yaml")

    args = parser.parse_args()

    extraer(args)
