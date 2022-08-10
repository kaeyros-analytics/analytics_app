from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
# import

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
          schedule_interval=timedelta(minutes=2),
          #schedule_interval='0 5 * * *' ,
          catchup=False)


# Create a templated command to execute
# 'bash cleandata.sh datestring'
templated_command = """
  python3 {{ params.filename }} 
"""

task1 = BashOperator(task_id='get_data_from_mysqldb',
                    bash_command=templated_command,
                    params={'filename': '/c/Users/user/dags/analytics_app/ease_data_pipeline/get_data_from_mysqldb.py'},
                    dag=dag)


task2 = BashOperator(task_id='transform',
                    bash_command=templated_command,
                    params={'filename': '/c/Users/user/dags/analytics_app/ease_data_pipeline/transform.py'},
                    dag=dag)

task3 = BashOperator(task_id='save_data_to_mysqldb',
                    bash_command=templated_command,
                    params={'filename': '/c/Users/user/dags/analytics_app/ease_data_pipeline/save_data_to_mysqldb.py'},
                    dag=dag)

task1 >> task2 >> task3 





