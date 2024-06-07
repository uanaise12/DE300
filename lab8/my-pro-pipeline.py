from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import tomli
import pathlib
import pandas as pd
import boto3
from io import StringIO

# Your bucket URI
BUCKET_URI = "s3://de300spring2024/anaise/lab8/"

# read the parameters from toml
CONFIG_FILE = "/root/configs/wine_config.toml"

def read_config() -> dict:
    path = pathlib.Path(CONFIG_FILE)
    with path.open(mode="rb") as param_file:
        params = tomli.load(param_file)
    return params

PARAMS = read_config()

def create_db_connection():
    from sqlalchemy import create_engine
    conn = BaseHook.get_connection(PARAMS['db']['db_connection'])
    engine = create_engine(conn.get_uri())
    return engine.connect()

def download_data_from_s3(**kwargs):
    """
    Download data from S3 bucket to a pandas DataFrame and write to RDS
    """
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket='de300spring2024', Key='anaise/lab8/wine.csv')
    data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    conn = create_db_connection()
    df.to_sql(PARAMS['db']['table_name'], conn, if_exists="replace", index=False)
    conn.close()

# Remaining functions should be modified similarly to interact with the database

# Define the default args dictionary for DAG
default_args = {
    'owner': 'johndoe',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 0,
}

# Instantiate the DAG
dag = DAG(
    'Pro-Classify',
    default_args=default_args,
    description='Classify with feature engineering and model selection',
    schedule_interval=PARAMS['workflow']['workflow_schedule_interval'],
    tags=["de300"]
)

download_data = PythonOperator(
    task_id="download_data_from_s3",
    python_callable=download_data_from_s3,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

add_data_to_table = PythonOperator(
    task_id='add_data_to_table',
    python_callable=add_data_to_table_func,  # Assumes this function is defined to load CSV data into RDS
    provide_context=True,
    queue=PARAMS['workflow']['sequential_queue'],
    dag=dag
)

# Assume further PythonOperators here for clean_data, normalize_data, etc.
# Each task should be similarly adapted to use the create_db_connection for database operations

# Set task dependencies as logical in the workflow
download_data >> add_data_to_table >> clean_data >> normalize_data
clean_data >> eda
normalize_data >> [fe_max, fe_product]
fe_product >> product_train
fe_max >> max_train
normalize_data >> production_train
[product_train, max_train, production_train] >> model_selection
model_selection >> [dummy_task, *evaluation_tasks]
