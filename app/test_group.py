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

# Function to assign group_id to records
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

#Function to group records based on task_id
def group_records(df):
    grouped_records = []
    used_ids = set()
    try:
        for task_id, group in df.groupby('task_id'):
        
            group['fn_similarity_score'] = group['fn_similarity_score'].astype(float)
            group['ln_similarity_score'] = group['ln_similarity_score'].astype(float)
            group['adrs_similarity_score'] = group['adrs_similarity_score'].astype(float)
        # Sort by similarity scores and Match Type
            group['total_score'] = (group['fn_similarity_score'] + 
                                    group['ln_similarity_score'] + 
                                    group['adrs_similarity_score'])
            
            # Sort groups by total score
            sorted_group = group.sort_values(by='total_score', ascending=False)

            # for _, row in sorted_group.iterrows():
            #     if row['id_a'] not in used_ids and row['id_b'] not in used_ids:
            #         grouped_records.append({
            #             'task_id': task_id,
            #             'group': [row['id_a'], row['id_b']],
            #             'Match Type': row['Match Type'],
            #             'total_score': row['total_score']
            #         })
            #         used_ids.update([row['id_a'], row['id_b']])

            for _, row in sorted_group.iterrows():
                if row['id_a'] not in used_ids and row['id_b'] not in used_ids:
                    current_group_id = row['group_id']
                    group_members = group[group['group_id'] == current_group_id]
                    
                    group_ids = list(group_members['id_a']) + list(group_members['id_b'])
                    group_ids = list(set(group_ids))  # Remove duplicates
                    
                    grouped_records.append({
                        'task_id': task_id,
                        'group': group_ids,
                        'Match Type': row['Match Type'],
                        'total_score': row['total_score'],
                        'group_id': current_group_id
                    })
                    used_ids.update(group_ids)
            # st.write(f"task_id: {task_id}")
            # st.write(f"grouped records B4 singleton:")
            # st.write(pd.DataFrame(grouped_records))
            # Handle singleton groups for unmatched records
            unmatched = group[~group['id_a'].isin(used_ids) & ~group['id_b'].isin(used_ids)]
            for _, row in unmatched.iterrows():
                grouped_records.append({
                    'task_id': task_id,
                    'group': [row['id_a']],
                    'Match Type': 'Singleton',
                    'total_score': 0,
                    'group_id': -1
                })
            # st.write(f"grouped records After singleton:")
            # st.write(pd.DataFrame(grouped_records))

            # Handle the unmatched records (non-matching records)
            unmatched_a = group[~group['id_a'].isin(used_ids)]
            unmatched_b = group[~group['id_b'].isin(used_ids)]

            # Collect the unmatched records
            for _, row in unmatched_a.iterrows():
                if row['id_a'] not in used_ids:
                    grouped_records.append({
                        'task_id': task_id,
                        'group': [row['id_a']],
                        'Match Type': 'Singleton',
                        'total_score': 0,
                        'group_id': -1
                    })
                    used_ids.add(row['id_a'])

            for _, row in unmatched_b.iterrows():
                if row['id_b'] not in used_ids:
                    grouped_records.append({
                        'task_id': task_id,
                        'group': [row['id_b']],
                        'Match Type': 'Singleton',
                        'total_score': 0,
                        'group_id': -1})
                    used_ids.add(row['id_b'])

        logging.info(f"Records grouped successfully")
        return pd.DataFrame(grouped_records)
        
    except Exception as e:
        #print(f"Error grouping records: {e}")
        logging.error(f"Error grouping records: {e}")  


original_df=pd.read_csv('input_records.csv')
#print(original_df.head())
st.write("original df loaded successfully")
st.write(original_df)
# Read CSV file into a DataFrame
matching_df = pd.read_csv('matching_data.csv')
st.write("matching_df loaded successfully")
st.write(matching_df)
#print(matching_df.head())

matching_grp_id_df = assign_group_id(matching_df)
# Display the first few rows of the DataFrame
#print(matching_df.head())
st.write(matching_grp_id_df)   
result_df = group_records(matching_grp_id_df)
st.write("Records grouped successfully")
st.write(result_df)   
#print(result_df.head())


# Explode the group column to get one row per ID

result_df['group'] = result_df['group'].apply(lambda x: x if isinstance(x, list) else [x])
exploded_df = result_df.explode('group')

#Reset the index
exploded_df.reset_index(drop=True, inplace=True)
st.write("Grouped recored exploded")
st.write(exploded_df)

#merge with original_df on task_id and id and drop the column group
merged_df = exploded_df.merge(original_df, left_on=['task_id', 'group'], right_on=['TASK_ID', 'ID'], how='left', suffixes=('', '_original'))
#merged_df_final = merged_df.drop(columns=['group'])
st.write("Merged records")
st.write(merged_df)
# Ensure all records are included by concatenating unmatched records
# used_ids = original_df['group'].unique()
# unmatched = matching_df[~matching_df['id_a'].isin(used_ids) & ~matching_df['id_b'].isin(used_ids)]
# final_df = pd.concat([merged_df, unmatched], ignore_index=True)
#exploded_df = exploded_df.merge(original_df, left_on='group', right_on='id', how='left', suffixes=('', '_original'))
# %%
# Function to highlight matching patterns
def highlight_matching_patterns(row):
    if row['group'] in singlton_grpoup_ids:
        return ['background-color: #00FFFF'] * len(row)
    elif row['Match Type'] == 'Exact firstname+lastname+dob+gender':
        return ['background-color: #D4EDDA'] * len(row)
    elif row['Match Type'] == 'hc-firstname+lc-lastname | Exact DOB+gender':
        return ['background-color: #CCE5FF'] * len(row)
    elif row['Match Type'] == 'lc firstname+lastname + hc-address  | Exact DOB+gender':
        return ['background-color: #FFF3CD'] * len(row)
    else:
        return ['background-color: #F8D7DA'] * len(row)

# Get unique task_ids
task_ids = result_df['task_id'].unique()

# Create tabs for each task_id
tabs = st.tabs([f"Task ID {task_id}" for task_id in task_ids])

for tab, task_id in zip(tabs, task_ids):
    with tab:
        st.write(f"Data for Task ID {task_id}")
        filtered_df = merged_df[merged_df['task_id'] == task_id]
        # Identify records with mach_type='singleton' and highlight them with separate colour
        singlton_grpoup_ids = filtered_df[filtered_df['Match Type'] == 'Singleton']['group'].unique()
        #filtered_df_final = filtered_df.drop(columns=['group'])
        # if 'group' in filtered_df.columns:
        #     filtered_df_final = filtered_df.drop(columns=['group'])
        # else:
        #     filtered_df_final = filtered_df
        #styled_df = filtered_df_final.style.apply(highlight_matching_patterns, axis=1)
        styled_df = filtered_df.style.apply(highlight_matching_patterns, axis=1)
        st.dataframe(styled_df)
# Display the legend
st.markdown("""
### Legend
- <span style="background-color: #D4EDDA;">Exact firstname+lastname+dob+gender</span>
- <span style="background-color: #CCE5FF;">hc-firstname+lc-lastname | Exact DOB+gender</span>
- <span style="background-color: #FFF3CD;">lc firstname+lastname + hc-address  | Exact DOB+gender</span>
- <span style="background-color: #F8D7DA;">Other</span>
""", unsafe_allow_html=True)