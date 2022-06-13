"""Este modulo implementa las clases necesarias para el calculo
de las features de cada imagen.
"""

from abc import abstractmethod
from dataclasses import dataclass
import sys

import yaml
import numpy
import cv2
import skimage
import matplotlib.pyplot as plt

class Extractor:
    @abstractmethod
    def extract(self, img_path: str) -> numpy.ndarray:
        """Entrega el feature de la imagen

        Parameters
        ----------
        img_path : str
            Path de la imagen a convertir

        Returns
        -------
        numpy.ndarray
            Feature de la imagen
        """
        pass

    @abstractmethod
    def visualize(self, img_path: str):
        """Muestra los keypoints utilizados por el descriptor

        Parameters
        ----------
        img_path : str
            Path de la imagen a visualizar
        """
        pass

@dataclass
class SIFT(Extractor):
    """Implementa la extraccion por caracteristicas SIFT

    Attributes
    ----------

    nfeatures: int, default=0
        Numero de features a retener en el vector final de la imagen. Si es 0, retiene todos

    nOctaveLayers: int, default=3
        Numero de capas en cada octava

    contrastThreshold: float, default=0.04
        El umbral para filtrar regiones de poco contraste

    edgeThreshold: float, default=10
        Umbral usado para filtrar edge-like features

    sigma: float, default=1.6
        El sigma de la Guassiana ocupada en la octava #0

    """
    nfeatures: int = 0
    nOctaveLayers: int = 3
    contrastThreshold: float = 0.04
    edgeThreshold: float = 10
    sigma: float = 1.6

    def __post_init__(self):
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures,
                                    nOctaveLayers=self.nOctaveLayers,
                                    contrastThreshold=self.contrastThreshold,
                                    edgeThreshold=self.edgeThreshold,
                                    sigma=self.sigma)


    def extract(self, img_path: str) -> numpy.ndarray:
        """Entrega el SIFT de la imagen

        Parameters
        ----------
        img_path : str
            Path de la imagen a convertir

        Returns
        -------
        numpy.ndarray
            SIFT de la imagen, de tamanho (nfeatures, 128)
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, des = self.sift.detectAndCompute(img, mask=None)
        return des

    def visualize(self, img_path: str):
        """Muestra los keypoints utilizados por el descriptor

        Parameters
        ----------
        img_path : str
            Path de la imagen a visualizar
        """

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, _ = self.sift.detectAndCompute(img, mask=None)

        img = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0), flags=0)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


@dataclass
class BRISK(Extractor):
    """Implementa la extraccion por caracteristica BRISK

    Attributes
    ----------
    thresh: int, default = 30
        Umbral AGAST

    octaves: int, default = 3
        Octavas de deteccion. 0 para una sola escala

    patterScale: float, default = 1.0
        Escala del patron usado para samplear la vecindad de los keypoints
    """

    thresh: int = 30
    octaves: int = 3
    patternScale: float = 1.0

    def __post_init__(self):
        self.brisk = cv2.BRISK_create(thresh=self.thresh,
                                        octaves=self.octaves,
                                        patternScale=self.patternScale)

    def extract(self, img_path: str) -> numpy.ndarray:
        """Entrega el brisk de la imagen

        Parameters
        ----------
        img_path : str
            Path de la imagen a convertir

        Returns
        -------
        numpy.ndarray
            BRISK de la imagen
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, des = self.brisk.detectAndCompute(img, mask=None)
        return des

    def visualize(self, img_path: str):
        """Muestra los keypoints utilizados por el descriptor

        Parameters
        ----------
        img_path : str
            Path de la imagen a visualizar
        """

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, _ = self.brisk.detectAndCompute(img, mask=None)

        img = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0), flags=0)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


@dataclass
class ORB(Extractor):
    """Implementa la extraccion de caracteristicas ORB

    Attributes
    ----------
    nfeatures : int, default=500
        Numero maximo de features a retener

    scaleFactor : float, default=1.2
        Ratio de decimacion de la piramide (mayor a 1)

    nlevels : int, default=8
        Cantidad de niveles en la piramide

    edgeThreshold : int, default=31
        Tamanho del borde donde las features no son detectadas. Debe ser similar a patchSize

    patchSize : int, default=31
        Tamanho del patch para el calculo de las caracteristicas
    """

    nfeatures: int = 500
    scaleFactor: float = 1.2
    nlevels: int = 8
    edgeThreshold: int = 31
    patchSize: int = 31

    def __post_init__(self):
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures,
                                    scaleFactor=self.scaleFactor,
                                    nlevels=self.nlevels,
                                    edgeThreshold=self.edgeThreshold,
                                    patchSize=self.patchSize)

    def extract(self, img_path: str) -> numpy.ndarray:
        """Entrega el ORB de la imagen

        Parameters
        ----------
        img_path : str
            Path de la imagen a convertir

        Returns
        -------
        numpy.ndarray
            ORB de la imagen
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, des = self.orb.detectAndCompute(img, mask=None)
        return des

    def visualize(self, img_path: str):
        """Muestra los keypoints utilizados por el descriptor

        Parameters
        ----------
        img_path : str
            Path de la imagen a visualizar
        """

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        keypoints, _ = self.orb.detectAndCompute(img, mask=None)

        img = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0), flags=0)
        plt.imshow(img)
        plt.axis("off")
        plt.show()


@dataclass
class DAISY(Extractor):
    """Implementa la extraccion de caracteristicas Daisy

    Attributes
    ----------
    step : int, default=4
        Distancia, en pixeles, entre los descriptores

    radius : int, default=15
        Radio, en pixeles, del anillo exterio

    rings : int, default=3
        Cantidad de anillos

    histograms : int, default=8
        Numero de histogramas por anillo

    orientations : int, default=8
        Numero de bins (orientaciones) por histograma
    """

    step: int = 4
    radius: int = 15
    rings: int = 3
    histograms: int = 8
    orientations: int = 8

    def extract(self, img_path: str) -> numpy.ndarray:
        """Entrega las caracteristicas daisy de la imagen

        Parameters
        ----------
        img_path : str
            Path de la imagen a convertir

        Returns
        -------
        numpy.ndarray
            Daisy de la imagen
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        des = skimage.feature.daisy(img,
                                        step=self.step,
                                        radius=self.radius,
                                        rings=self.rings,
                                        histograms=self.histograms,
                                        orientations=self.orientations)
        n_vectors = des.shape[2]
        des = numpy.reshape(des, (-1, n_vectors))
        return des

    def visualize(self, img_path: str):
        """Muestra los keypoints utilizados por el descriptor

        Parameters
        ----------
        img_path : str
            Path de la imagen a visualizar
        """

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, keypoints = skimage.feature.daisy(img,
                                        step=self.step,
                                        radius=self.radius,
                                        rings=self.rings,
                                        histograms=self.histograms,
                                        orientations=self.orientations,
                                        visualize=True)

        plt.imshow(keypoints)
        plt.axis("off")
        plt.show()


def get_extractor(config_file: str, features_method: str) -> Extractor:
    """Entrega el metodo de extraccion de caracteristicas correspondiente

    Parameters
    ----------
    config_file : str
        Archivo de configuracion del metodo escogido
    features_method : str
        Metodo de extraccion

    Returns
    -------
    Extractor
        Instancia del metodo de extraccion escogido
    """
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    method = features_method

    config_method = config.get(method, None)

    if config_method is None:
        print("No existe la configuracion para el metodo escogido")
        sys.exit()

    if method=="sift":
        extractor = SIFT(**config_method)
    if method=="brisk":
        extractor = BRISK(**config_method)
    if method=="orb":
        extractor = ORB(**config_method)
    if method=="daisy":
        extractor = DAISY(**config_method)

    return extractor
