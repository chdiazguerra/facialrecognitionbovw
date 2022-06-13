"""Realiza la prediccion sobre el conjunto de test, entregando el accuracy
y guardando las predicciones en un archivo json
"""

import json
import os
import sys
import argparse
import pickle
import numpy

import pandas
from classifier import Classifier

def predict(args):
    method = os.path.split(args.file_model)[-1].split(".")[0]

    with open(args.file_encoding, "rb") as file:
        encodings = pickle.load(file)

    data = pandas.read_json(args.dataset_file)
    data = data[data.split=="test"]

    X_test = []
    y_test = []

    for _, (nombre, clase) in data[["nombre", "clase"]].iterrows():
        X_test.append(encodings[nombre])
        y_test.append(clase)

    X_test = numpy.vstack(X_test)
    y_test = numpy.array(y_test)

    classifier = Classifier(use_predict=True)
    classifier.load_model(args.file_model)

    pred = classifier.predict(X_test)

    preds_dict = {}
    for nombre, p in zip(data.nombre, pred):
        preds_dict[nombre] = int(p)

    with open(os.path.join(args.out_pred, f"pred_{method}.json"), "w", encoding="utf-8") as file:
        json.dump(preds_dict, file)

    acc = sum(y_test==pred)/len(y_test)
    print("Accuracy test:")
    print(f"{acc*100:.2f}%")

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_encoding",
                        help="Path del archivo con los encoding de test (encoding_test_method.pkl)",
                        type=str, required=True)
    parser.add_argument("--file_model",
                        help="Path del archivo con el modelo ya entrenado (type_method.pkl)",
                        type=str, required=True)
    parser.add_argument("--dataset_file", help="Archivo json con los datos del dataset",
                        nargs="?", type=str, default="data/data.json")
    parser.add_argument("--out_pred", help="Carpeta en donde se guardan las predicciones",
                        nargs="?", type=str, default="data")

    args = parser.parse_args()

    if not os.path.isfile(args.dataset_file):
        print("No existe el archivo con los datos")
        sys.exit()

    if not os.path.isfile(args.file_encoding):
        print("No existe el archivo con los encodings")
        sys.exit()

    if not os.path.isfile(args.file_model):
        print("No existe el archivo con el modelo entrenado")
        sys.exit()

    predict(args)
