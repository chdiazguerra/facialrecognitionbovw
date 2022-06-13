"""Permite realizar la prediccion de solo una imagen
"""

import os
import sys
import argparse

from src.features.feature_extraction import get_extractor
from src.models.clustering import KNNFeatures
from src.models.classifier import Classifier


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_img", help="Path de la imagen",
                            type=str, required=True)
    parser.add_argument("--features_method",
                        help="Metodo para extraer las features de las imagenes",
                        nargs="?", type=str,
                        choices=["sift", "brisk", "orb", "daisy"], required=True)
    parser.add_argument("--knn_model",
                        help="Path del modelo KNN",
                        type=str, required=True)
    parser.add_argument("--classifier", help="Path del modelo de clasificacion",
                        type=str, required=True)
    parser.add_argument("--config_features",
                        help="Archivo de configuracion del metodo de caracteristicas",
                        nargs="?", type=str, default="src/features/config.yaml")

    args = parser.parse_args()

    if not os.path.isfile(args.path_img):
        print("No existe la imagen")
        sys.exit()

    if not os.path.isfile(args.knn_model):
        print("No existe el modelo KNN")
        sys.exit()

    if not os.path.isfile(args.classifier):
        print("No existe el clasificador")
        sys.exit()

    if not os.path.isfile(args.config_features):
        print("No existe el archivo de configuracion")
        sys.exit()

    extractor = get_extractor(args.config_features, args.features_method)

    encoder = KNNFeatures()
    encoder.load_model(args.knn_model)

    classifier = Classifier(use_predict=True)
    classifier.load_model(args.classifier)

    features = extractor.extract(args.path_img)
    encoding = encoder.encode(features)
    pred = classifier.predict(encoding.reshape(1,-1), name=True)[0]

    print("La imagen corresponde a:")
    print(pred)
