# %%
import streamlit as st
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from transformers import pipeline
from db import create_snowflake_connection, run_query

import logging
import os
#import sys
import datetime

dtm=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
root_dir = os.getcwd()
log_file_path = os.path.join(root_dir, f"logs/test_group_{dtm}.log")
# Load Hugging Face GPT-2 model for text generation

os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path,level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.info(f"Log file created: {log_file_path}")
generator = pipeline('text-generation', model='gpt2')


def assign_group_id(df):
    group_id = 0
    df['group_id'] = -1  # Initialize group_id column

    def assign_group(task_df, index, group_id):
        stack = [index]
        while stack:
            current_index = stack.pop()
            if df.at[current_index, 'group_id'] == -1:  # If not already assigned to a group
                df.at[current_index, 'group_id'] = group_id
                for i, r in task_df.iterrows():
                    if i != current_index and df.at[i, 'group_id'] == -1:
                        # Add your matching condition here
                        if df.at[current_index, 'Match Type'] == r['Match Type']:
                            #df.at[i, 'group_id'] = group_id
                            stack.append(i)

    for task_id in df['task_id'].unique():
        task_df = df[df['task_id'] == task_id]
        for index, row in task_df.iterrows():
            if df.at[index, 'group_id'] == -1:  # If not already assigned to a group
                group_id += 1
                assign_group(task_df, index, group_id)

    # Assign singleton group_id to unmatched records
    for index, row in df.iterrows():
        if df.at[index, 'group_id'] == -1:
            group_id += 1
            df.at[index, 'group_id'] = group_id

    return df

try:

    original_df=pd.read_csv('input_records.csv')
    # print(original_df.head())
    st.write("original df loaded successfully")
    st.write(original_df)
    # Read CSV file into a DataFrame
    matching_df = pd.read_csv('matching_data.csv')
    st.write("matching_df loaded successfully")
    st.write(matching_df)
    #print(matching_df.head())

    result_df = assign_group_id(matching_df)
    st.write("Records grouped successfully")
    st.write(result_df) 
except Exception as e:
    logging.error(f"Error in group_records: {e}")
    st.write("Error in group_records")
    st.write(e)  

#print(result_df.head())

#task_df = matching_df[matching_df['task_id']==333]

#print(task_df)
