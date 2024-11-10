
import streamlit as st
import pandas as pd
#import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import seaborn as sns
import matplotlib.pyplot as plt
from db import create_snowflake_connection, run_query


# Connect to Snowflake
#conn = create_snowflake_connection()
# Step 1: Read data from CSV file
def read_data_from_csv(file):
    try:
        df = pd.read_csv(file)
        st.write(f"Successfully read {len(df)} records from the uploaded CSV file.")
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

# Step 2: Insert records into Snowflake
def insert_records_into_snowflake(df, conn):
    try:
        write_pandas(conn, df, 'INPUT_RECORDS')
        print("Records inserted successfully.")
    except Exception as e:
        print("Error in inserting records: ", e)

# Step 3: Compute similarity in Snowflake
def compute_similarity_in_snowflake(conn):
    query = f"""
    CREATE OR REPLACE TEMPORARY TABLE SIMILARITY AS
    SELECT 
        A.FIRSTNAME AS FIRST_RECORD, 
        B.FIRSTNAME AS SECOND_RECORD,
        VECTOR_DOT_PRODUCT(A.FULL_VECTOR, B.FULL_VECTOR) AS COSINE_SIMILARITY
    FROM (SELECT
        FIRSTNAME,
        LASTNAME,
        SENTENCE_EMBEDDING(ADDRESS) AS ADDRESS_VECTOR,
        VECTOR_ADD(FIRSTNAME, LASTNAME) AS NAME_VECTOR
    FROM INPUT_RECORDS) A, 
    (SELECT
        FIRSTNAME,
        LASTNAME,
        SENTENCE_EMBEDDING(ADDRESS) AS ADDRESS_VECTOR,
        VECTOR_ADD(FIRSTNAME, LASTNAME) AS NAME_VECTOR
    FROM INPUT_RECORDS) B
    WHERE A.FIRSTNAME <> B.FIRSTNAME;
    """
    try:
        run_query(query, conn)
        print("Similarity computed successfully.")
    except Exception as e:
        print("Error in computing similarity: ", e)

# Step 4: Fetch all matching patterns and group by similarity
def fetch_and_group_patterns(conn):
    query = """
    SELECT FIRST_RECORD, SECOND_RECORD, COSINE_SIMILARITY
    FROM SIMILARITY
    WHERE COSINE_SIMILARITY > 0.8
    ORDER BY COSINE_SIMILARITY DESC;
    """
    results = run_query(query, conn)
    df = pd.DataFrame(results, columns=['First Record', 'Second Record', 'Similarity Score'])
    # Cluster records by using a simple linkage method, or use an algorithm like DBSCAN or Agglomerative Clustering
    # For simplicity, here we will use a placeholder for clustering logic
    df['Pattern Group'] = df['Similarity Score'].apply(lambda x: 'High' if x > 0.9 else 'Medium')
    return df

# Step 5: Display patterns in Streamlit UI with color coding
def display_patterns(df):
    st.write("Matching Patterns:")
    if df.empty:
        st.write("No matching patterns found.")
    else:
        pattern_groups = df.groupby('Pattern Group')
        for name, group in pattern_groups:
            st.write(f"## Pattern Group: {name}")
            sns.pairplot(group, hue='Pattern Group', vars=['First Record', 'Second Record', 'Similarity Score'])
            plt.show()
            st.pyplot()
            for _, row in group.iterrows():
                color = "#FFC107" if name == 'High' else "#FF5722"
                st.markdown(
                    f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                    f"<p><strong>Record 1:</strong> {row['First Record']}</p>"
                    f"<p><strong>Record 2:</strong> {row['Second Record']}</p>"
                    f"<p><strong>Similarity Score:</strong> {row['Similarity Score']:.2f}</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
def main():
    # Step 1: Input Data via Streamlit
    st.title("User Data Similarity App")
    st.write("Enter user data to find similar patterns.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df =read_data_from_csv(uploaded_file)
        st.write(df)
    #step 2: Insert records into Snowflake
        if df is not None:
            try:
                conn = create_snowflake_connection()
                insert_records_into_snowflake(df, conn)
                compute_similarity_in_snowflake(conn)
                patterns_df = fetch_and_group_patterns(conn)
                display_patterns(patterns_df)
            except Exception as e:
                st.error(f"An error occurred in main method: {e}")
            finally:
                conn.close()

if __name__ == "__main__":
    main()
#streamlit run app.py

/**********************************/

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

        # Display patterns based on the matching rules
        #display_patterns_with_rules(matching_df)
        st.write("Matching Patterns Based on Rules:")
        
        # Highlight matching patterns based on the match type
        st.dataframe(  matching_df.style.apply(highlight_matching_patterns, axis=1))
        #matching_df.style.apply(highlight_matching_patterns, axis=1)
        conn.close()
    # else:
    #     st.write(f"CSV file must contain the following columns: {required_columns}")
else:
    st.write("Please upload a CSV file.")

# #%%
# #check python snowflake connector version
# import snowflake.connector
# print(snowflake.connector.__version__)

# %%

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

    Now, generate the SQL for: {user_input}
    """

    # Use the Hugging Face model to generate the SQL query
    generated_sql = generator(prompt, max_length=300, num_return_sequences=1)[0]['generated_text']
    return generated_sql


create or replace TABLE LLM_DB.LLM_DEV.INPUT_RECORDS (
	ID NUMBER(38,0),
	FIRSTNAME VARCHAR(16777216),
	LASTNAME VARCHAR(16777216),
	ADDRESS VARCHAR(16777216),
	DOB VARCHAR(16777216),
	GENDER VARCHAR(16777216),
	COUNTRY VARCHAR(16777216),
	FIRSTNAME_VECTOR VECTOR(FLOAT, 768),
	LASTNAME_VECTOR VECTOR(FLOAT, 768),
	ADDRESS_VECTOR VECTOR(FLOAT, 768)
);