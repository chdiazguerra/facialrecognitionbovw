"""Contiene las clases necesarias para realizar el clustering de las features
"""

from dataclasses import dataclass
import pickle

import numpy
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class KMeansFeatures:
    """Permite entrenar un modelo KMeans y extraer el encoding de una imagen

    Attributes
    ----------
    n_clusters : int, default=256
    """
    n_clusters : int = 256

    def __post_init__(self):
        self.kmeans = KMeans(n_clusters=self.n_clusters)

    def train(self, X: numpy.ndarray):
        """Entrena el clusterizador

        Parameters
        ----------
        X : numpy.ndarray
            Vectores para el entrenamiento
        """
        self.kmeans.fit(X)

    def save_model(self, path: str):
        """Guarda el modelo entrenado

        Parameters
        ----------
        path : str
            Path donde guardar el modelo
        """
        with open(path, "wb") as file:
            pickle.dump(self.kmeans, file)

    def load_model(self, path: str):
        """Carga el modelo

        Parameters
        ----------
        path : str
            Path del modelo guardado
        """
        try:
            with open(path, "rb") as file:
                self.kmeans = pickle.load(file)
        except FileNotFoundError:
            print("No existe el modelo guardado")

    def encode(self, x: numpy.ndarray, normalize: bool = False) -> numpy.ndarray:
        """Realiza la codificacion de las caracteristicas en un vector de frecuencias
        de los centroides del KMeans

        Parameters
        ----------
        x : numpy.ndarray
            Features de la imagen a codificar
        normalize : bool
            Aplicar normalizacion L2

        Returns
        -------
        numpy.ndarray
            Codificacion de las caracteristicas de la imagen
        """
        largo = len(self.kmeans.cluster_centers_)
        pred = self.kmeans.predict(x)
        categorical = numpy.eye(largo)[pred]
        encoding = numpy.sum(categorical, axis=0)

        if normalize:
            encoding = encoding/numpy.sqrt(numpy.sum(encoding**2))

        return encoding


@dataclass
class KNNFeatures:
    """Permite entrenar el modelo de K-nearest neighbors y
    extraer el encoding de una imagen
    """
    def __post_init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=1)

    def train(self, path_kmeans_model: str):
        """Entrena el modelo de acuerdo a clusters de KMeans

        Parameters
        ----------
        path_kmeans_model : str
            Path del modelo KMeans desde el que se entrena
        """
        with open(path_kmeans_model, "rb") as file:
            kmeans_model = pickle.load(file)

        Xc = kmeans_model.cluster_centers_
        y = numpy.arange(len(Xc))
        self.knn.fit(Xc, y)

    def save_model(self, path: str):
        """Guarda el modelo entrenado

        Parameters
        ----------
        path : str
            Path donde guardar el modelo
        """
        with open(path, "wb") as file:
            pickle.dump(self.knn, file)

    def load_model(self, path: str):
        """Carga el modelo

        Parameters
        ----------
        path : str
            Path del modelo guardado
        """
        try:
            with open(path, "rb") as file:
                self.knn = pickle.load(file)
        except FileNotFoundError:
            print("No existe el modelo guardado")

    def encode(self, x: numpy.ndarray, normalize: bool = False) -> numpy.ndarray:
        """Realiza la codificacion de las caracteristicas en un vector de frecuencias
        de los centroides del KNN

        Parameters
        ----------
        x : numpy.ndarray
            Features de la imagen a codificar
        normalize : bool
            Aplicar normalizacion L2

        Returns
        -------
        numpy.ndarray
            Codificacion de las caracteristicas de la imagen
        """
        largo = len(self.knn.classes_)
        pred = self.knn.predict(x)
        categorical = numpy.eye(largo)[pred]
        encoding = numpy.sum(categorical, axis=0)

        if normalize:
            encoding = encoding/numpy.sqrt(numpy.sum(encoding**2))

        return encoding
