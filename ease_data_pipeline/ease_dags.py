from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'Cartelle',
    'depends_on_past': False,
    'start_date': datetime(2022, 7, 26),
    'email': ['cartelle.fotsing@kaeyros-analytics.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=3)
}

dag = DAG(dag_id="ease",
          default_args=default_args,
          #start_date=datetime(2022,7,26),
          #schedule_interval=timedelta(hours=24))
          schedule_interval=timedelta(minutes=10))


# Create a templated command to execute
# 'bash cleandata.sh datestring'
templated_command = """
  python3 {{ params.filename }} 
"""

task1 = BashOperator(task_id='extract',
                    bash_command=templated_command,
                    params={'filename': '/c/Users/user/dags/ease_data_pipeline/extract.py'},
                    dag=dag)


task2 = BashOperator(task_id='transform',
                    bash_command=templated_command,
                    params={'filename': '/c/Users/user/dags/ease_data_pipeline/transform.py'},
                    dag=dag)

task3 = BashOperator(task_id='load',
                    bash_command=templated_command,
                    params={'filename': '/c/Users/user/dags/ease_data_pipeline/load.py'},
                    dag=dag)

task1 >> task2 >> task3



