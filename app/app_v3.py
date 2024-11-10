#%%
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
log_file_path = os.path.join(root_dir, f"logs/app_{dtm}.log")
# Load Hugging Face GPT-2 model for text generation

os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path,level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logging.info(f"Log file created: {log_file_path}")
generator = pipeline('text-generation', model='gpt2')

def upload_data_to_snowflake(df, conn, table_name):
    # Add an 'id' column if not present
    try:
        #if 'id' not in df.columns:
        df['id'] = range(1, 1 + len(df))
            #df.insert(0, 'id', range(1, 1 + len(df)))
        # Upload data #ignore the header
        success, nchunks, nrows, _ = write_pandas(conn, df, table_name, overwrite=True, quote_identifiers=False)
        if success:
            st.write(f"Successfully uploaded {nrows} rows to '{table_name}' table in Snowflake.")
            logging.info(f"Successfully uploaded {nrows} rows to '{table_name}' table in Snowflake.")
        else:
            st.write("Failed to upload data to Snowflake.")
            logging.error("Failed to upload data to Snowflake.")
    except Exception as e:
        logging.error(f"Error uploading data to Snowflake: {e}")

def add_embeddings_to_table(conn, table_name):
    try:
        check_columns_query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        AND column_name IN ('FIRSTNAME_VECTOR', 'LASTNAME_VECTOR', 'ADDRESS_VECTOR');
        """
        existing_columns = set()
        with conn.cursor() as cursor:
            cursor.execute(check_columns_query)
            for row in cursor:
                existing_columns.add(row[0].upper())
        # Add columns for embeddings
        columns_to_add = [
                ('FIRSTNAME_VECTOR', 'VECTOR(FLOAT, 768)'),
                ('LASTNAME_VECTOR', 'VECTOR(FLOAT, 768)'),
                ('ADDRESS_VECTOR', 'VECTOR(FLOAT, 768)')
            ]
        # Generate ALTER TABLE statements for columns that do not exist
        cursor = conn.cursor()
        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                alter_query=f"""Alter table {table_name} ADD COLUMN {column_name} {column_type}"""
                #print(f"Alter Query: {alter_query}")
                logging.info(f"Alter Query: {alter_query}")
                cursor.execute(alter_query)
                #print(f"Added column{column_name} for embeddings to '{table_name}' table.")
                logging.info(f"Added column{column_name} for embeddings to '{table_name}' table.")
            else:
                #print("Column - {column_name}  already exist in the table.")
                logging.info("Column - {column_name}  already exist in the table.") 
        # Update embeddings
        update_query = f"""
        UPDATE {table_name}
        SET
            firstname_vector = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', firstname),
            lastname_vector = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', lastname),
            address_vector = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m',address);
        """
        #print(f"Update Query: {update_query}")
        logging.info(f"Update Query: {update_query}")
        cursor.execute(update_query)
        #print(f"Embeddings calculated and stored back in '{table_name}' table.")
        logging.info(f"Embeddings calculated and stored back in '{table_name}' table.")
        # st.write(f"Embeddings calculated and stored back in '{table_name}' table.")
    except Exception as e:
        #print(f"Error adding embeddings to table: {e}")
        logging.error(f"Error adding embeddings to table: {e}")
        st.write(f"Error adding embeddings to table: {e}")
        logging.error(f"Error adding embeddings to table: {e}")
def apply_matching_rules_in_snowflake(conn):
    try:
        query = """
        WITH VECTOR_DATA AS (
        SELECT
            id,
            task_id,
            firstname,
            lastname,
            dob,
            GENDER,
            COUNTRY,
            firstname_vector,
            lastname_vector,
            address_vector
        FROM INPUT_RECORDS
        )
        SELECT
                a.task_id as task_id,
                A.id AS id_a,
                B.id AS id_b,
                A.DOB as dob_a,
                B.DOB as dob_b,
                A.GENDER as gender_a,
                B.GENDER as gender_b,
                A.firstname as firstname_a,
                B.firstname as firstname_b,
                A.lastname as lastname_a,
                B.lastname as lastname_b,
                A.firstname || ' ' || A.lastname || ' ' || A.DOB || ' ' || A.Gender || ' ' || A.COUNTRY AS first_record,
                B.firstname || ' ' || B.lastname || ' ' || B.DOB || ' ' || B.Gender || ' ' || B.COUNTRY AS second_record,
                CASE
                WHEN A.FIRSTNAME = B.FIRSTNAME
                    AND A.LASTNAME = B.LASTNAME
                    AND COALESCE(A.DOB,'11110101') = COALESCE(B.DOB,'11110101')
                    AND COALESCE(A.GENDER,'UNK') = COALESCE(B.GENDER,'UNK')
                THEN 'Exact firstname+lastname+dob+gender'
                when VECTOR_COSINE_SIMILARITY(A.firstname_vector,B.firstname_vector) > 0.95
                    AND VECTOR_COSINE_SIMILARITY(A.lastname_vector,B.lastname_vector) > 0.95
                    AND COALESCE(A.DOB,'11110101') = COALESCE(B.DOB,'11110101')
                    AND COALESCE(A.GENDER,'UNK') = COALESCE(B.GENDER,'UNK')
                THEN 'hc-firstname+hc-lastname | Exact DOB+gender'
                WHEN VECTOR_COSINE_SIMILARITY(A.ADDRESS_VECTOR, B.ADDRESS_VECTOR) > 0.9
                    AND VECTOR_COSINE_SIMILARITY(A.firstname_vector,B.firstname_vector) > 0.9
                    AND VECTOR_COSINE_SIMILARITY(A.lastname_vector,B.lastname_vector) > 0.9
                    AND COALESCE(A.DOB,'11110101') = COALESCE(B.DOB,'11110101')
                THEN 'lc firstname+lastname + hc-address  | Exact DOB+gender'
                WHEN EDITDISTANCE(A.LASTNAME, B.LASTNAME) < 3 
                THEN 'Fuzzy Match on Lastname'
                ELSE 'No Match'
                END AS MATCH_TYPE,
                VECTOR_COSINE_SIMILARITY(A.firstname_vector,B.firstname_vector) as fn_similarity_score,
                VECTOR_COSINE_SIMILARITY(A.lastname_vector,B.lastname_vector) as ln_similarity_score,
                VECTOR_COSINE_SIMILARITY(A.ADDRESS_VECTOR, B.ADDRESS_VECTOR) as adrs_similarity_score
            FROM VECTOR_DATA A
            JOIN VECTOR_DATA B ON A.id < B.id
            and A.task_id = B.task_id;
            """
        #print(f"Matching Rules Query: {query}")
        logging.info(f"Matching Rules Query: {query}")
        return pd.DataFrame(run_query(query, conn), columns=['task_id','id_a','id_b','dob_a','dob_b','gender_a','gender_b','firstname_a','firstname_b','lastname_a','lastname_b','First Record', 'Second Record', 'Match Type', 'fn_similarity_score', 'ln_similarity_score', 'adrs_similarity_score'])
    except Exception as e:
        #print(f"Error applying matching rules: {e}")
        logging.error(f"Error applying matching rules: {e}")
        return None
def get_matching_group_ids(matching_df):
    try:
        result = matching_df.groupby(['Match Type','task_id']).apply(
            lambda g: list(set(g['id_a']).union(set(g['id_b'])))
            ).reset_index(name='distinct_ids')
        #print(f"matching groups are calculated successfully")
        logging.info(f"matching groups are calculated successfully at task_id level")
        #print(result)
        logging.info(result)
        return result
    except Exception as e:
        #print(f"Error calculating matching groups: {e}")
        logging.error(f"Error calculating matching groups: {e}")
        return None

def split_matching_ids_to_rows(matching_df,groups):
    data_rows = []
    try:
    #iterate over the groups dataframe
        for _, row in groups.iterrows():
            #print(f"Row: {row}")
            logging.info(f"Row: {row}")
            match_type = row['Match Type']
            task_id = row['task_id']
            match_group_ids = row['distinct_ids']
            fn_similarity_score = matching_df[matching_df['task_id'] == task_id]['fn_similarity_score'].iloc[0]
            ln_similarity_score = matching_df[matching_df['task_id'] == task_id]['ln_similarity_score'].iloc[0]
            adrs_similarity_score = matching_df[matching_df['task_id'] == task_id]['adrs_similarity_score'].iloc[0]
            ##print(f"Match pattern: {match_type}")
            logging.info(f"Match pattern for task_id {task_id} is: {match_type}")
            ##print(f"match group ids : {match_group_ids}")
            logging.info(f"match group ids for task_id {task_id} is : {match_group_ids}")
            for record_id in match_group_ids:
                #print(f"Peocessing Record ID: {record_id}")
                logging.info(f"Peocessing Record ID: {record_id}")
                # Filter DataFrame for matching records
                record = matching_df[(matching_df['id_a'] == record_id) | (matching_df['id_b'] == record_id)].iloc[0]
                data_rows.append({
                    'ID': record_id,
                    'Task ID': task_id,
                    'First Name': record['firstname_a'] if record['id_a'] == record_id else record['firstname_b'],
                    'Last Name': record['lastname_a'] if record['id_a'] == record_id else record['lastname_b'],
                    'DOB': record['dob_a'] if record['id_a'] == record_id else record['dob_b'],
                    'First Record': record['First Record'] if record['id_a'] == record_id else record['Second Record'],
                    'Match Type': match_type, 
                    'Similarity Score FN': fn_similarity_score,
                    'Similarity Score LN': ln_similarity_score,
                    'Similarity Score Adrs': adrs_similarity_score
                })
        return pd.DataFrame(data_rows)
        #print(pd.DataFrame(data_rows))
    except Exception as e:
        #print(f"Error splitting matching records: {e}")
        logging.error(f"Error splitting matching records: {e}")
        return None
#def display_patterns_with_rules(df):
# Function to apply conditional formatting based on matching patterns
def highlight_matching_patterns(row):
    if row['Match Type'] == 'Exact firstname+lastname+dob+gender':
        return ['background-color: #D4EDDA'] * len(row)
    elif row['Match Type'] == 'hc-firstname+lc-lastname | Exact DOB+gender':
        return ['background-color: #CCE5FF'] * len(row)
    elif row['Match Type'] == 'lc firstname+lastname + hc-address  | Exact DOB+gender':
        return ['background-color: #FFF3CD'] * len(row)
    else:
        return ['background-color: #F8D7DA'] * len(row)


def create_matching_groups(df):
    group_id = 0
    df['group_id'] = -1  # Initialize group_id column
    for task_id in df['task_id'].unique():
        task_df = df[df['task_id'] == task_id]
        for index, row in task_df.iterrows():
            if df.at[index, 'group_id'] == -1:  # If not already assigned to a group
                group_id += 1
                df.at[index, 'group_id'] = group_id
                # Find matching records and assign the same group_id
                for i, r in task_df.iterrows():
                    if i != index and df.at[i, 'group_id'] == -1:
                        # Add your matching condition here
                        if row['Match Type'] == r['Match Type']:
                            df.at[i, 'group_id'] = group_id
    return df
# Function to generate complex SQL using Hugging Face GPT-2 model with specific table and column context
def generate_complex_sql_from_input(user_input):
    # Define your table name and column names
    table_name = "INPUT_RECORDS"
    column_names = ["id", "firstname", "lastname", "dob", "GENDER", "COUNTRY", "firstname_vector", "lastname_vector", "address_vector"]

    # Provide context about the table and column names in the prompt
    prompt = f"""
    You are an expert SQL generator. The table is named `{table_name}`, and the columns are {', '.join(column_names)}.
    
    The columns `firstname`, `lastname`, `dob` store names and dates, and columns `firstname_vector` and `lastname_vector` store vector representations for similarity matching.
    
    I need a complex SQL query based on this table structure. Here are some rules:
    - Use `VECTOR_COSINE_SIMILARITY` for comparing vector columns with a threshold.
    - Use `COALESCE` to handle nulls in `dob` and `gender`.
    - Use `CASE` to return different types of matches based on conditions.
    - Join records where `id < id` to avoid self-joins.
    - Support both exact matches and fuzzy matches.

    Example:
    Input: "Exact match on firstname and DOB, similar last name and country"
    SQL: "SELECT * FROM {table_name} A JOIN {table_name} B ON A.id < B.id
           WHERE A.firstname = B.firstname
           AND VECTOR_COSINE_SIMILARITY(A.lastname_vector, B.lastname_vector) > 0.85
           AND COALESCE(A.DOB, '11110101') = COALESCE(B.DOB, '11110101')
           AND A.COUNTRY = B.COUNTRY;"

    Example:
    Input: "Firstname is similar, last name is exact match"
    SQL: "SELECT * FROM {table_name} A JOIN {table_name} B ON A.id < B.id
           WHERE VECTOR_COSINE_SIMILARITY(A.firstname_vector, B.firstname_vector) > 0.9 AND lastname = lastname;"

    Now, generate the SQL for: {user_input}
    """
    
    # Use Hugging Face to generate the SQL query
    generated_sql = generator(prompt, max_new_tokens=300, num_return_sequences=1)[0]['generated_text']
    return generated_sql

# Function to validate SQL by running it with LIMIT 0
def validate_sql(conn, sql_query):
    try:
        # Run the query with LIMIT 0 to validate syntax without fetching data
        validation_query = f"{sql_query.strip().rstrip(';')} LIMIT 0"
        cur = conn.cursor()
        cur.execute(validation_query)
        return True, None  # If no exception, the SQL is valid
    except Exception as e:
        logging.error(f"Error validating SQL: {e}")
        return False, str(e)  # Return the error message if SQL is invalid

# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

try:

    if uploaded_file is not None:
        # Read CSV data into DataFrame
        data = pd.read_csv(uploaded_file)
        #convert DOB to date
        #data['DOB'] = pd.to_datetime(data['DOB'], errors='coerce')
        st.write("Data Preview:")
        st.write(data.head())
        # Step 2: Connect to Snowflake      
        conn = create_snowflake_connection()

        # Check if required columns exist
        #required_columns = ['firstname','lastname', 'address', 'DOB', 'gender', 'country']
        #if all(column in data.columns for column in required_columns):
        if st.button("Process Data"):
            # Step 3: Upload Data to Snowflake
            table_name = 'INPUT_RECORDS'
            upload_data_to_snowflake(data, conn, table_name)

            # Step 4: Calculate Embeddings and Update Table
            add_embeddings_to_table(conn, table_name)

            # Step 5: apply matching rules
            matching_df = apply_matching_rules_in_snowflake(conn)
            #print(f"Matching Data: {matching_df}")
            st.write("Matching Data:")
            st.write(matching_df)
            if matching_df is not None:
                # Identify groups of matching records
                groups = get_matching_group_ids(matching_df)
                st.write(groups)
                #st.write(f"Matching Groups: {groups}")
                # Reshape the DataFrame
                #matche_group_df = split_matching_ids_to_rows(matching_df, groups)
                match_group_df = create_matching_groups(matching_df)
                
                
                
                st.write("Reshaped Data:")
                st.write(match_group_df)

                st.dataframe(match_group_df)

                # get unique task_ids
                task_ids = match_group_df['Task ID'].unique()

                #Create tabs for each task_id
                tabs = st.tabs([f"Task ID {task_id}" for task_id in task_ids])

                for tab, task_id in zip(tabs, task_ids):
                    with tab:
                        st.write(f"Data for Task ID {task_id}")
                        #filtered_df = matching_df[matching_df['task_id'] == task_id]
                        filtered_df = match_group_df[match_group_df['Task ID'] == task_id]
                        styled_df = filtered_df.style.apply(highlight_matching_patterns, axis=1)
                        st.dataframe(styled_df)

                # # Display patterns based on the matching rules
                # st.write("Matching Patterns Based on Rules:")
                # # Apply conditional formatting
                # styled_df = matche_group_df.style.apply(highlight_matching_patterns, axis=1)
                # st.dataframe(styled_df)
                #st.dataframe(  matching_df.style.apply(highlight_matching_patterns, axis=1))
                #matching_df.style.apply(highlight_matching_patterns, axis=1)
                        # Step 2: If the user is not happy with the predefined results, allow custom rule input
        st.write("If you are not happy with the above results, provide a custom matching rule.")
        custom_rule_input = st.text_area("Enter custom matching rule (e.g., 'Exact match on firstname, similar last name and same DOB')")

            # Button to apply the custom rule
        if st.button("Apply Custom Matching Rule"):
            generated_sql = generate_complex_sql_from_input(custom_rule_input)
            st.write(f"Generated SQL: {generated_sql}")

            # Step 3: Validate the generated SQL before execution
            is_valid, validation_error = validate_sql(conn, generated_sql)

            if is_valid:
                st.write("SQL is valid!")
                try:
                    # Execute the valid SQL query
                    cur = conn.cursor()
                    cur.execute(generated_sql)
                    custom_results = cur.fetchall()
                    custom_df = pd.DataFrame(custom_results, columns=[desc[0] for desc in cur.description])
                    st.write("Results from Custom Rule:")
                    st.write(custom_df)
                except Exception as e:
                    st.write(f"Error executing custom SQL: {e}")
            else:
                # Show the SQL validation error
                st.write(f"SQL is invalid: {validation_error}")

        conn.close()
        # else:
        #     st.write(f"CSV file must contain the following columns: {required_columns}")
    else:       
        st.write("Please upload a CSV file.")
except Exception as e:
    #print(f"Error processing data: {e}")
    st.write(f"Error processing data: {e}")

#%%
# from itertools import combinations
# from collections import defaultdict
# conn = create_snowflake_connection()
# matching_df = apply_matching_rules_in_snowflake(conn)
# print(f"Matching Data: {matching_df.columns}")
# groups = get_matching_groups(matching_df)
# # print(f"type(groups): {type(groups)}")
# # print(f"Matching Groups: {groups}")
# # get the mathcing records from matching_df 
# data_rows = []
# try:
#     #iterate over the groups dataframe
#     for _, row in groups.iterrows():
#         print(f"Row: {row}")
#         match_type = row['Match Type']
#         match_group_ids = row['distinct_ids']
#         print(f"Match pattern: {match_type}")
#         print(f"match group ids : {match_group_ids}")
#         for record_id in match_group_ids:
#             print(f"Peocessing Record ID: {record_id}")
#             # Filter DataFrame for matching records
#             record = matching_df[(matching_df['id_a'] == record_id) | (matching_df['id_b'] == record_id)].iloc[0]
#             data_rows.append({
#                 'ID': record_id,
#                 'First Record': record['First Record'] if record['id_a'] == record_id else record['Second Record'],
#                 'Match Type': match_type, 
#                 'similarity_score_fn_ln_adrs': record['similarity_score_fn_ln_adrs']
#             })
#     print(pd.DataFrame(data_rows))
# except Exception as e:
#     print(f"Error reshaping data: {e}")





# %%
# data = [
#     (1, {2, 4, 5, 6, 'No Match'}),
#     (1, {3, 'hc-firstname+lc-lastname | Exact DOB+gender'}),
#     (5, {'lc firstname+lastname + hc-address | Exact DOB+gender', 6})
# ]

# # Convert to dictionary if needed
# data_dict = {}
# for key, value_set in data:
#     if key not in data_dict:
#         data_dict[key] = set()
#     data_dict[key].update(value_set)

# print(data_dict)
# %%
# import pandas as pd

# # Sample DataFrame
# data = {
#     'id_a': [1, 1, 5],
#     'id_b': [2, 3, 6],
#     'First Record': ['Record1', 'Record2', 'Record5'],
#     'Second Record': ['RecordX', 'RecordY', 'RecordZ'],
#     'Match Type': ['TypeA', 'TypeA', 'TypeC'],
#     'similarity_score_fn_ln_adrs': [0.9, 0.8, 0.7]
# }

# df = pd.DataFrame(data)

# print(df)

# # result = df.groupby('id_a').agg({
# #     'Match Type': lambda x: list(x.unique()),
# #     'id_b': lambda x: list(x)
# # }).reset_index()

# # Group by 'Match Type' and aggregate pairs of [id_a, id_b]
# #result = df.groupby('Match Type').apply(lambda g: g[['id_a', 'id_b']].values.tolist()).reset_index(name='id_pairs')

# result = df.groupby('Match Type').apply(
#     lambda g: list(set(g['id_a']).union(set(g['id_b'])))
# ).reset_index(name='distinct_ids')
# print(result)

# %%
