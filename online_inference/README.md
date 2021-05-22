![CI workflow](https://github.com/made-ml-in-prod-2021/andyst75/actions/workflows/homework2.yml/badge.svg?branch=homework2)

[Online inference for Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

# Usage
## python:
how to run app:
~~~
python -m src.src.main_app
~~~
or
~~~
uvicorn src.main_app:app
~~~
how to run request script (on running service):
~~~
python -m src.get_predict
~~~
tests:
~~~
pytest --cov
~~~
code linter:
~~~
pylint --output-format=colorized -v src
~~~

# Docker:
Build command:
~~~
docker build -t andyst75/online_inference:v2 .
~~~
Run command:
~~~
docker run --network host andyst75/online_inference:v2
~~~
Test running image:
~~~
curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'accept: application/json' -d '{"features":["slope", "oldpeak", "sex", "trestbps", "ca", "restecg", "chol", "age", "thal", "thalach", "fbs", "cp", "exang"], "data":[[1,1,1,1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0,0,0,0]]}'
~~~

Login to docker:
~~~
docker login
~~~

Push command:
~~~
docker push andyst75/online_inference:v2
~~~
Pull command:
~~~
docker pull andyst75/online_inference:v2
~~~
