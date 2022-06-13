"""Contiene la clase del clasificador utilizado
"""

from dataclasses import dataclass
import os
import numpy
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import yaml


@dataclass
class Classifier:
    """Modelo de clasificacion

    Attributes
    ----------
    model_type : str, default='RF'
        Modelo de clasificacion utilizado. Puede ser 'RF' (Random Forest),
        'SVM' o 'XGBoost'
    config_file : str, default='config.yaml'
        Archivo de configuracion del modelo
    use_predict : bool, default=False
        True si se utilizara para predecir y por lo tanto, se debe cargar el modelo
    """
    model_type: str = "RF"
    config_file: str = "config.yaml"
    use_predict: bool = False

    def __post_init__(self):
        if not self.use_predict:
            if not os.path.isfile(self.config_file):
                raise Exception("No existe el archivo de configuracion")

            with open(self.config_file, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            config = config.get(self.model_type, None)

            if config is None:
                raise Exception("No existe la configuracion para el metodo escogido")

            if self.model_type=="RF":
                self.model = RandomForestClassifier(**config)
            elif self.model_type=="SVM":
                self.model = SVC(**config)
            elif self.model_type=="XGBoost":
                self.model = XGBClassifier(**config)
            else:
                raise Exception("No existe el tipo de modelo especificado")

    def train_model(self, X_train: numpy.ndarray, y_train: numpy.ndarray):
        """Entrena el modelo de clasificacion escogido

        Parameters
        ----------
        X_train : numpy.ndarray
            Features de entrenamiento
        y_train : numpy.ndarray
            Clases de entrenamiento
        """
        if self.model_type=="XGBoost":
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                            test_size=0.2, shuffle=True, stratify=y_train)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
        else:
            self.model.fit(X_train, y_train)
    
    def save_model(self, path: str):
        """Guarda el modelo entrenado

        Parameters
        ----------
        path : str
            Path del archivo
        """
        with open(path, "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path: str):
        """Carga el modelo

        Parameters
        ----------
        path : str
            Path del modelo guardado
        """
        try:
            with open(path, "rb") as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            print("No existe el modelo guardado")
        
    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        """Realiza la prediccion de la feature de entrada

        Parameters
        ----------
        x : numpy.ndarray
            Feature de entrada

        Returns
        -------
        numpy.ndarray
            Prediccion
        """
        return self.model.predict(x)
