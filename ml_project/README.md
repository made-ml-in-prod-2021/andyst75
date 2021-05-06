![CI workflow](https://github.com/made-ml-in-prod-2021/andyst75/actions/workflows/homework1.yml/badge.svg?branch=homework1)

## Data

Download data from [heart-disease-uci](https://www.kaggle.com/ronitf/heart-disease-uci) and extract into folder `data/raw`

```bash
mkdir -p data/raw && unzip archive.zip -d data/raw
```

## Preresquistes

* [`Python 3.7`](https://www.python.org/)
* `virtualenv` (`pip install virtualenv`)

Create and activate virtual environment

```bash
virtualenv venv
. venv/bin/activate
```

Finally, install the module

```bash
pip install .
```

## EDA

Open Jupyter notebook from `notebooks/EDA.ipynb`

## Tests

```bash
pip install .
python -m pytest . -v --cov --cov-fail-under=80
```

## Code linter:

```bash
pylint --output-format=colorized -v src
```

## Usage

Installation

```bash
pip install .
```

### Training

```bash
python -m src.train
```
this a same as 
```bash
python -m src.train --config-name=train
```
or use other model
```bash
python -m src.train --config-name=train_nb
```

### Predict

```bash
python -m src.predict
```
this with full path
```bash
python -m src.predict --config-path=configs/predict.yaml
```
or other model
```bash
python -m src.predict --config-path=configs/predict_nb.yaml
```

## Roadmap

№ | Описание | Баллы
--- | --- | ---
-2 | ~~Назовите ветку homework1~~ | 1
-1 | ~~Положите код в папку ml_project~~ | -
0 | ~~В описании к пулл реквесту описаны основные &quot;архитектурные&quot; и тактические решения, которые сделаны в вашей работе.~~ | 3
1 | ~~Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками~~ | 3
2 | ~~Проект имеет модульную структуру(не все в одном файле =) )~~ | 3
3 | ~~Использованы логгеры~~ | 2
4 | ~~Написаны тесты на отдельные модули и на прогон всего пайплайна~~ | 5
5 | ~~Для тестов генерируются синтетические данные, приближенные к реальным~~ | 5
6 | ~~Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing)~~ | 2
7 | ~~Используются датаклассы для сущностей из конфига, а не голые dict~~ | 3
8 | ~~Используйте кастомный трансформер(написанный своими руками) и протестируйте его~~ | 3
9 | ~~Обучите модель, запишите в readme как это предлагается~~ | 3
10 | ~~Напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку(без меток) и запишет предикт, напишите в readme как это сделать~~ | 3
11 | ~~Используется hydra  (https://hydra.cc/docs/intro/)~~ | 3 (доп баллы)
12 | ~~Настроен CI(прогон тестов, линтера) на основе github actions~~  | 3 балла (доп баллы)
13 | ~~Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему~~ | 1 (доп баллы)


