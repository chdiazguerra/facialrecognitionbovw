Facial Recognition BoVW
==============================

Implementación de un algoritmo de reconocimiento facial basado en bag-of-visual-words.

Se proveen una serie de scripts para su ejecución, los cuales deben ser utilizados desde el directorio raíz de este proyecto. Para más información de ellas, utilizar

```bash
python script.py -h
```

## Dataset
---------

Este proyecto es probado en la base de datos <a target="_blank" href="http://pics.psych.stir.ac.uk/">PICS</a>, para su dataset <a target="_blank" href="http://pics.psych.stir.ac.uk/zips/pain.zip">Pain</a>. Se provee un script para realizar su descarga, el cual puede ser ejecutado utilizando

```bash
python src/data/download_dataset.py
```
Este crea una carpeta *data* y descomprime el zip descargado.

Para la creación de los splits de train y test, se debe ejecutar

```bash
python src/data/make_dataset.py
```
El cual lee los datos descargados con el script anterior y crea un archivo json con los datos de las imágenes, de la forma

    [
        "nombre": nombre_imagen,
        "clase": clase_imagen,
        "split": "train"/"test"
    ]

## Feature Extraction
---------

Para la extracción de las features de cada imagen, se tienen cuatro métodos implementados: SIFT, BRISK, ORB y DAISY. Para el primer caso, se debe ejecutar

```bash
python src/features/build_features.py --features_method sift
```
La configuración de este método (y los demás) puede ser configurada con el archivo *config.yaml* de la carpeta *src/features*.

## Encoding
---------

Para realizar la codificación de este, primero se debe ejecutar para el dataset de entrenamiento. Esto se realiza utilizando

```bash
python src/models/encode_train.py --file_features data/features_sift.pkl
```
En la carpeta *models* se entregan modelos preentrenados con la configuración utilizada en nuestro caso. Este puede ser utilizado indicandolo en el script anterior agregando al final *--file_kmeans \[FILE\]*.

Para realizar la codificacion para test, se utiliza

```bash
python src/models/encode_test.py --file_features data/features_sift.pkl
```
Aquí también se puede utilizar el modelo preentrenado entregado.

## Classification
---------

Para clasificar cada imagen, primeramente se crea el clasificador. Este puede ser Random Forest (RF), SVM o XGBoost, cuyas configuraciones pueden ser cambiadas desde el archivo *config.yaml* de la carpeta *src/models*. Para su entrenamiento se debe ejecutar

```bash
python src/models/train_model.py --file_encoding data/encoding_train_{method}.pkl
```
donde *method* depende de la extraccion de características utilizada, por defecto, *sift*. En este caso también se entregan modelos preentrenados.

Para la predicción en el dataset de test, se utiliza

```bash
python src/models/predict_model.py --file_encoding data/encoding_test_{method}.pkl --file_model models/RF_{method}.pkl
```
Esto mostrará el accuracy alcanzado, además de generar un json con las predicciones realizadas.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
