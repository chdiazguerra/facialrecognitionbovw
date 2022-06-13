"""Entrena el modelo de clasificacion
"""

import os
import sys
import argparse
import pickle
import numpy

import pandas
from classifier import Classifier


def entrenar(args):
    method = args.file_encoding.split("_")[2].split(".")[0]

    with open(args.file_encoding, "rb") as file:
        encodings = pickle.load(file)

    data = pandas.read_json(args.dataset_file)
    data = data[data.split=="train"]

    X_train = []
    y_train = []

    for i, (nombre, clase) in data[["nombre", "clase"]].iterrows():
        X_train.append(encodings[nombre])
        y_train.append(clase)

    X_train = numpy.vstack(X_train)
    y_train = numpy.array(y_train)

    classifier = Classifier(args.model_type, args.config)

    if not args.file_model:
        print("Entrenando modelo")
        classifier.train_model(X_train, y_train)
        path = os.path.join(args.out_model, f"{args.model_type}_{method}.pkl")

        if not os.path.isdir(args.out_model):
            os.mkdir(args.out_model)

        classifier.save_model(path)
        args.file_model = path

    classifier.load_model(args.file_model)
    pred = classifier.predict(X_train)
    acc = sum(y_train==pred)/len(y_train)
    print("Accuracy train:")
    print(acc*100,"%")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_encoding",
                        help="Path del archivo con los encoding de train (encoding_train_method.pkl)",
                        type=str, required=True)
    parser.add_argument("--dataset_file", help="Archivo json con los datos del dataset",
                        nargs="?", type=str, default="data/data.json")
    parser.add_argument("--out_model",
                        help="Carpeta en donde se guarda el archivo del modelo entrenado",
                        nargs="?", type=str, default="models")
    parser.add_argument("--file_model",
                        help="Path del archivo con el modelo ya entrenado",
                        nargs="?", type=str)
    parser.add_argument("--config", help="Path del archivo de configuracion del modelo",
                        nargs="?", type=str, default="src/models/config.yaml")
    parser.add_argument("--model_type", help="Escoge el tipo del modelo de clasificacion",
                        nargs="?", type=str, choices=["RF", "SVM", "XGBoost"], default="RF")

    args = parser.parse_args()

    if not os.path.isfile(args.dataset_file):
        print("No existe el archivo con los datos")
        sys.exit()

    if not os.path.isfile(args.file_encoding):
        print("No existe el archivo con los encodings")
        sys.exit()

    if args.file_model and not os.path.isfile(args.file_model):
        print("No existe el archivo con el modelo entrenado")
        sys.exit()

    entrenar(args)
