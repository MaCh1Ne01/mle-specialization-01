# Especialización MLE - Proyecto 01

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este proyecto corresponde al curso MLE 1 de la Especialización MLE.

## Estructura del proyecto

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         package-mles-01 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── package-mles-01   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes package-mles-01 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Problemática
Se requiere implementar un modelo de regresión para predecir el costo de un vuelo de avión y competir en el mercado de aerolíneas en India.

## Diagrama de flujo
<img src="https://drive.google.com/uc?export=view&id=1SLlZ6p5BR_8izMYnEFX-__W597HC_uX1" allow="autoplay">

## Descripción del dataset
El conjunto de datos fue extraído de la página web "EaseMyTrip", la cual contiene información sobre reserva de vuelos para viajes entre las 6 principales ciudades metropolitanas de la India. Dicha recopilación se realizo en 2 partes, una para vuelos de clase económica y otra para vuelos de clase ejecutiva, recolectando un total de 300261 registros y 11 características, las cuales se detallan a continuación:

| **Feature**      | **Descripción**                                                                                                        |
|:----------------:|:----------------------------------------------------------------------------------------------------------------------:|
| Airline          | Nombre de la aerolínea, es una feature categórica y tiene 6 valores diferentes.                                        |
| Flight           | Código del vuelo de avión, es una feature categórica.                                                                  |
| Source City      | Ciudad de origen del vuelo, es una feature categórica y tiene 6 valores diferentes.                                    |
| Departure Time   | Intervalo del día que representa la hora de salida del vuelo, es una feature categórica y tiene 6 valores diferentes.  |
| Stops            | Número de paradas durante el vuelo, es una feature categórica y tiene 3 valores diferentes.                            |
| Arrival Time     | Intervalo del día que representa la hora de llegada del vuelo, es una feature categórica y tiene 6 valores diferentes. |
| Destination City | Ciudad de destino del vuelo, es una feature categórica y tiene 6 valores diferentes.                                   |
| Class            | Clase del asiento del vuelo, es una feature categórica y tiene 2 valores diferentes.                                   |
| Duration         | Tiempo de duración del vuelo en horas, es una feature numérica.                                                        |
| Days Left        | Días restantes para la fecha del vuelo contabilizados desde la fecha de reservación, es una feature numérica.          |
| Price            | Precio en rupias indias del ticket de avión, es la variable a predecir.                                                |

## Model Card
<img src="https://drive.google.com/uc?export=view&id=1-BJGFQr32BeK_ASnEyaYxt8Uw__ezk_S" allow="autoplay">

## Resultados con métricas de evaluación offline
<img src="https://drive.google.com/uc?export=view&id=19u7vcUN5YNxUc9f8FGA2tGisMSFEud1t" allow="autoplay">

## Conclusiones
PENDING