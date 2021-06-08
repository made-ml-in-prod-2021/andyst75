# Homework 3: airflow

## Configure
Configure [`docker-compose.yml`](./docker-compose.yml) for paths.  
Configure [`airflow parameters`](dags/dag_constants.py).

## Start docker
```bash
docker-compose up --build
```

## View airflow
With browser, [URL](http://localhost:8080/).

## Stop docker
```bash
docker-compose down
```

## Clean after run
```bash
yes | docker system prune && yes | docker volume prune && yes | docker network prune | docker rmi airflow-docker airflow-predict airflow-preprocess airflow-download airflow-ml-base
```

## Run tests

1. Test structure dag
```bash
pytest tests
```

2. Test available dags (in docker)

- run `docker airflow`
- run `docker ps`
- find names `airflow_ml_dags_scheduler_1`
- give `IMAGE ID` for this process
- run `docker exec -it image_id bash`
- in docker process run `pytest tests`
