#%%
import streamlit as st
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from db import create_snowflake_connection, run_query

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
        else:
            st.write("Failed to upload data to Snowflake.")
    except Exception as e:
        print(f"Error uploading data to Snowflake: {e}")

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
                print(f"Alter Query: {alter_query}")
                cursor.execute(alter_query)
                print(f"Added column{column_name} for embeddings to '{table_name}' table.")
            else:
                print("Column - {column_name}  already exist in the table.")
        # Update embeddings
        update_query = f"""
        UPDATE {table_name}
        SET
            firstname_vector = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', firstname),
            lastname_vector = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', lastname),
            address_vector = SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m',address);
        """
        print(f"Update Query: {update_query}")
        cursor.execute(update_query)
        print(f"Embeddings calculated and stored back in '{table_name}' table.")
        # st.write(f"Embeddings calculated and stored back in '{table_name}' table.")
    except Exception as e:
        print(f"Error adding embeddings to table: {e}")
        st.write(f"Error adding embeddings to table: {e}")
def apply_matching_rules_in_snowflake(conn):
    try:
        query = """
        WITH VECTOR_DATA AS (
        SELECT
            id,
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
                A.id AS id_a,
                B.id AS id_b,
                A.firstname || ' ' || A.lastname || ' ' || A.DOB || ' ' || A.Gender || ' ' || A.COUNTRY AS first_record,
                B.firstname || ' ' || B.lastname || ' ' || B.DOB || ' ' || B.Gender || ' ' || B.COUNTRY AS second_record,
                -- A.DOB as DOB_A,
                -- B.DOB as DOB_B,
                -- A.GENDER as GENDER_A,
                -- B.GENDER as GENDER_B,
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
                THEN 'hc-firstname+lc-lastname | Exact DOB+gender'
                WHEN VECTOR_COSINE_SIMILARITY(A.ADDRESS_VECTOR, B.ADDRESS_VECTOR) > 0.9
                    AND VECTOR_COSINE_SIMILARITY(A.firstname_vector,B.firstname_vector) > 0.9
                    AND VECTOR_COSINE_SIMILARITY(A.lastname_vector,B.lastname_vector) > 0.9
                    AND COALESCE(A.DOB,'11110101') = COALESCE(B.DOB,'11110101')
                THEN 'lc firstname+lastname + hc-address  | Exact DOB+gender'
                WHEN EDITDISTANCE(A.LASTNAME, B.LASTNAME) < 3 
                THEN 'Fuzzy Match on Lastname'
                ELSE 'No Match'
                END AS MATCH_TYPE,
                concat(VECTOR_COSINE_SIMILARITY(A.firstname_vector,B.firstname_vector)||'|'||
                VECTOR_COSINE_SIMILARITY(A.lastname_vector,B.lastname_vector)||'|'||
                VECTOR_COSINE_SIMILARITY(A.ADDRESS_VECTOR, B.ADDRESS_VECTOR)) as similarity_score_fn_ln_adrs
                -- VECTOR_COSINE_SIMILARITY(A.firstname_vector,B.firstname_vector) as FNAME_COSINE_SIMILARITY,
                -- VECTOR_COSINE_SIMILARITY(A.lastname_vector,B.lastname_vector) as LNAME_COSINE_SIMILARITY,
                -- VECTOR_COSINE_SIMILARITY(A.ADDRESS_VECTOR, B.ADDRESS_VECTOR) as ADDRESS_COSINE_SIMILARITY
            FROM VECTOR_DATA A
            JOIN VECTOR_DATA B ON A.id < B.id;
            """
        print(f"Matching Rules Query: {query}")
        return pd.DataFrame(run_query(query, conn), columns=['id_a','id_b', 'First Record', 'Second Record', 'Match Type', 'similarity_score_fn_ln_adrs'])
    except Exception as e:
        print(f"Error applying matching rules: {e}")
        return None
def get_matching_group_ids(matching_df):
    try:
        result = matching_df.groupby('Match Type').apply(
            lambda g: list(set(g['id_a']).union(set(g['id_b'])))
            ).reset_index(name='distinct_ids')
        print(f"matching groups are calculated successfully")
        print(result)
        return result
    except Exception as e:
        print(f"Error calculating matching groups: {e}")
        return None

def split_matching_ids_to_rows(matching_df,result):
    data_rows = []
    try:
    #iterate over the groups dataframe
        for _, row in groups.iterrows():
            print(f"Row: {row}")
            match_type = row['Match Type']
            match_group_ids = row['distinct_ids']
            print(f"Match pattern: {match_type}")
            print(f"match group ids : {match_group_ids}")
            for record_id in match_group_ids:
                print(f"Peocessing Record ID: {record_id}")
                # Filter DataFrame for matching records
                record = matching_df[(matching_df['id_a'] == record_id) | (matching_df['id_b'] == record_id)].iloc[0]
                data_rows.append({
                    'ID': record_id,
                    # 'First Name': record['firstname_a'] if record['id_a'] == record_id else record['firstname_b'],
                    # 'Last Name': record['lastname_a'] if record['id_a'] == record_id else record['lastname_b'],
                    # 'DOB': record['dob_a'] if record['id_a'] == record_id else record['dob_b'],
                    'First Record': record['First Record'] if record['id_a'] == record_id else record['Second Record'],
                    'Match Type': match_type, 
                    'similarity_score_fn_ln_adrs': record['similarity_score_fn_ln_adrs']
                })
        return pd.DataFrame(data_rows)
        #print(pd.DataFrame(data_rows))
    except Exception as e:
        print(f"Error splitting matching records: {e}")
        return None
#def display_patterns_with_rules(df):
def highlight_matching_patterns(row):
    if row['Match Type'] == 'Exact firstname+lastname+dob+gender':
        return ['background-color: #D4EDDA'] * len(row)
    elif row['Match Type'] == 'hc-firstname+lc-lastname | Exact DOB+gender':
        return ['background-color: #CCE5FF'] * len(row)
    elif row['Match Type'] == 'lc firstname+lastname + hc-address  | Exact DOB+gender':
        return ['background-color: #FFF3CD'] * len(row)
    else:
        return ['background-color: #F8D7DA'] * len(row)


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

        # Check if required columns exist
        #required_columns = ['firstname','lastname', 'address', 'DOB', 'gender', 'country']
        #if all(column in data.columns for column in required_columns):
        if st.button("Process Data"):
            # Step 2: Connect to Snowflake
            conn = create_snowflake_connection()

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
                st.write(f"Matching Groups: {groups}")

                # Reshape the DataFrame
                matche_group_df = split_matching_ids_to_rows(matching_df, groups)
                st.write("Reshaped Data:")
                st.dataframe(matche_group_df)
                # Display patterns based on the matching rules
                st.write("Matching Patterns Based on Rules:")
                # Apply conditional formatting
                styled_df = matche_group_df.style.apply(highlight_matching_patterns, axis=1)
                st.dataframe(styled_df)
                #st.dataframe(  matching_df.style.apply(highlight_matching_patterns, axis=1))
                #matching_df.style.apply(highlight_matching_patterns, axis=1)
            conn.close()
        # else:
        #     st.write(f"CSV file must contain the following columns: {required_columns}")
    else:       
        st.write("Please upload a CSV file.")
except Exception as e:
    print(f"Error processing data: {e}")
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
