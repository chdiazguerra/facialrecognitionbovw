"""Genera la particion para entrenamiento y test
"""

import argparse
import os
import sys
import re

import json
from sklearn.model_selection import train_test_split

def create_dataset(args: argparse.Namespace):
    """Crea las particiones de train y test, generando un json con el contenido

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos entregados por el usuario
    """
    if not os.path.isdir(args.path):
        print("No existe la carpeta con las imagenes.")
        sys.exit()

    OUT = os.path.join(args.out, "data.json")

    if args.f or not os.path.isfile(OUT):
        name_imgs = os.listdir(args.path)

        i = 0
        clases_dict = {}
        clases = []
        names = []
        REGEX = '(\d+)'

        for name in name_imgs:
            sep = re.split(REGEX, name)
            clase = sep[0] + sep[1]

            if "-" in sep[2]:
                continue

            if not clase in clases_dict:
                clases_dict[clase] = i
                i += 1

            clases.append(clases_dict[clase])
            names.append(name)

        id2s = {v: k for k, v in clases_dict.items()}

        X_train, X_test, y_train, y_test = train_test_split(
                                            names, clases, test_size=args.test_split,
                                            stratify=clases,
                                            random_state=0 if not args.f else None)

        dataset = []
        for nombre, clase in zip(X_train, y_train):
            datos = {"nombre": nombre,
                    "clase": clase,
                    "clase_str": id2s[clase],
                    "split": "train"}
            dataset.append(datos)

        for nombre, clase in zip(X_test, y_test):
            datos = {"nombre": nombre,
                    "clase": clase,
                    "clase_str": id2s[clase],
                    "split": "test"}
            dataset.append(datos)

        if not os.path.isdir(args.out):
            print("Creando directorio", args.out)
            os.mkdir(args.out)

        print("Guardando datos")

        with open(OUT, "w", encoding="utf-8") as file:
            json.dump(dataset, file)

        print("Terminado")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path de la carpeta con las imagenes",
                        nargs="?", type=str, default="data/pain")
    parser.add_argument("-f",
                        help="Forzar particion aleatoria",
                        action="store_true")
    parser.add_argument("--test_split",
                        help="Porcentaje total de los archivos que se usa como test",
                        nargs="?", type=float, default=0.2)
    parser.add_argument("--out", help="Carpeta en donde se guarda el archivo de salida",
                        nargs="?", type=str, default="data")

    parse_args = parser.parse_args()

    create_dataset(parse_args)
