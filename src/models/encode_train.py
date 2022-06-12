"""Realiza la codificacion de las imagenes de train de acuerdo a KMeans
"""

import os
import sys
import argparse
import pickle

import pandas
import numpy
from clustering import KMeansFeatures

def encode_train(args):
    data = pandas.read_json(args.dataset_file)
    data_train = data[data.split=="train"]
    nombres_train = list(data_train.nombre)

    with open(args.file_features, "rb") as file:
        features = pickle.load(file)

    X_train = []

    for nombre in nombres_train:
        f = features[nombre]
        X_train.append(f)

    X_train = numpy.vstack(X_train)

    method = args.file_features.split("_")[1]

    kmeans = KMeansFeatures(args.clusters)

    if not args.file_kmeans:
        print("Entrenando modelo")
        kmeans.train(X_train)
        path = os.path.join(args.out_model, "kmeans_"+method)
        kmeans.save_model(path)
        args.file_kmeans = path

    kmeans.load_model(args.file_kmeans)

    encoded = {}

    print("Generando encodings")
    for nombre in nombres_train:
        f = features[nombre]
        encoded[nombre] = kmeans.encode(f, args.normalize)

    if not os.path.isdir(args.out_encoding):
        os.mkdir(args.out_encoding)

    with open(os.path.join(args.out_encoding, "encoding_train_"+method), "wb") as file:
        pickle.dump(encoded, file)

    print("Terminado")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_features",
                        help="Path del archivo con las features (features_method.pkl)",
                        type=str, required=True)
    parser.add_argument("--out_encoding", help="Carpeta en donde se guarda el archivo de salida",
                        nargs="?", type=str, default="data")
    parser.add_argument("--out_model",
                        help="Carpeta en donde se guarda el archivo del model KMeans",
                        nargs="?", type=str, default="models")
    parser.add_argument("--dataset_file", help="Archivo json con los datos del dataset",
                        nargs="?", type=str, default="data/data.json")
    parser.add_argument("--clusters",
                        help="Cantidad de clusters a utilizar para KMeans",
                        nargs="?", type=int, default=256)
    parser.add_argument("--file_kmeans",
                        help="Path del archivo con el modelo KMeans ya entrenado",
                        nargs="?", type=str)
    parser.add_argument("--normalize",
                        help="Si se deben normalizar los encodings",
                        action="store_true")

    args = parser.parse_args()

    if not os.path.isfile(args.file_features):
        print("No existe el archivo con las features")
        sys.exit()

    if not os.path.isfile(args.dataset_file):
        print("No existe el archivo con los datos")
        sys.exit()

    if args.file_kmeans and not os.path.isfile(args.file_kmeans):
        print("No existe el archivo con el modelo KMeans entrenado")
        sys.exit()

    encode_train(args)
