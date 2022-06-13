"""Realiza la codificacion de las imagenes de test de acuerdo
a un model K-nearest neighbor"""

import os
import sys
import argparse
import pickle

import pandas
import numpy
from clustering import KNNFeatures

def encode_test(args):
    data = pandas.read_json(args.dataset_file)
    data_test = data[data.split=="test"]
    nombres_test = list(data_test.nombre)

    with open(args.file_features, "rb") as file:
        features = pickle.load(file)

    X_test = []

    for nombre in nombres_test:
        f = features[nombre]
        X_test.append(f)

    X_test = numpy.vstack(X_test)

    method = args.file_features.split("_")[1]

    knn = KNNFeatures()

    if not args.file_knn:
        print("Entrenando modelo")
        knn.train(args.path_kmeans)
        path = os.path.join(args.out_model, "knn_"+method)
        knn.save_model(path)
        args.file_knn = path

    knn.load_model(args.file_knn)

    encoded = {}

    print("Generando encodings")
    for nombre in nombres_test:
        f = features[nombre]
        encoded[nombre] = knn.encode(f, args.normalize)

    if not os.path.isdir(args.out_encoding):
        os.mkdir(args.out_encoding)

    with open(os.path.join(args.out_encoding, "encoding_test_"+method), "wb") as file:
        pickle.dump(encoded, file)

    print("Terminado")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_features",
                        help="Path del archivo con las features (features_method.pkl)",
                        type=str, required=True)
    parser.add_argument("--path_kmeans", help="Path del modelo KMeans entrenado",
                        type=str, required=True)
    parser.add_argument("--out_encoding", help="Carpeta en donde se guarda el archivo de salida",
                        nargs="?", type=str, default="data")
    parser.add_argument("--out_model",
                        help="Carpeta en donde se guarda el archivo del model KNN",
                        nargs="?", type=str, default="models")
    parser.add_argument("--dataset_file", help="Archivo json con los datos del dataset",
                        nargs="?", type=str, default="data/data.json")
    parser.add_argument("--file_knn",
                        help="Path del archivo con el modelo KNN ya entrenado",
                        nargs="?", type=str)
    parser.add_argument("--normalize",
                        help="Si se deben normalizar los encodings",
                        action="store_true")

    args = parser.parse_args()

    if not os.path.isfile(args.file_features):
        print("No existe el archivo con las features")
        sys.exit()
    
    if args.file_knn and not os.path.isfile(args.path_kmeans):
        print("No existe el archivo con el modelo KMeans")
        sys.exit()

    if not os.path.isfile(args.dataset_file):
        print("No existe el archivo con los datos")
        sys.exit()

    if args.file_knn and not os.path.isfile(args.file_knn):
        print("No existe el archivo con el modelo KNN entrenado")
        sys.exit()

    encode_test(args)
