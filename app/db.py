# %%
from os import environ as env
from dotenv import load_dotenv
import snowflake.connector
import os
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col
import json
from snowflake.snowpark.types import ArrayType, FloatType
from snowflake.snowpark.functions import udf

load_dotenv()
# print(f"User is : {os.getenv('SF_USER')}")
# print(f"Password is : {os.getenv('SF_PASSWORD')}")
# print(f"Account is : {os.getenv('SF_ACCOUNT')}")
# print(f"Database is : {os.getenv('SF_DATABASE')}")
'''
create connection
'''
def create_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=os.getenv('SF_USER'),
            password=os.getenv('SF_PASSWORD'),
            account=os.getenv('SF_ACCOUNT'),
            warehouse=os.getenv('SF_WAREHOUSE'),
            database=os.getenv('SF_DATABASE'),
            schema=os.getenv('SF_SCHEMA') 
            #role='<your_role>'
        )
        print("Connection established successfully")
        return conn
    except Exception as e:
        print("Error in connection : ",e)
        return None

# Execute SQL query in Snowflake
def run_query(query, conn):
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()

#create snowpark session
def create_snowpark_session():
    
    try:
        with open('connection_parameters.json') as f:
            connection_properties=json.load(f)

        session = Session.builder.configs(connection_properties).create()

        return session
    except Exception as e:
        print("Error in connection : ",e)
        return None
# session=create_snowpark_session()
# stage_location = '@udf_stage'
# @udf(name='VECTOR_ADD', is_permanent=True, replace=True, session=session, stage_location=stage_location)
# def vector_add(vec1: list, vec2: list, vec3: list) -> list:
#     try:
#         return [x + y + z for x, y, z in zip(vec1, vec2, vec3)]
#     except Exception as e:
#         print("Error in vector_add : ",e)
#         return None
    


# %%
# conn=create_snowflake_connection()
# cursor=conn.cursor()
# print(cursor.execute("select current_version()"))
# session=create_snowpark_session()
# print(session)
