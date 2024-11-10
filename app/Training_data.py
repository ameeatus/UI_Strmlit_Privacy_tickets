# %%
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from transformers import pipeline
from db import create_snowflake_connection, run_query
import pandas as pd
demo_data = pd.read_csv('Use_case_FP_FN.csv')
print(f" unique values are : {demo_data['ID'].unique().size}")
#print(demo_data.head())

#print(demo_data.head())

# Function to fill missing values with mode at the ID level
def fill_missing_with_mode(df, group_col):
    def fill_group(group):
        for col in group.columns:
            if group[col].isnull().any():
                mode_series = group[col].mode()
                if not mode_series.empty:
                    mode_value = mode_series.iloc[0]
                    group[col].fillna(mode_value, inplace=True)
        return group

    return df.groupby(group_col).apply(fill_group)

# Fill missing values
demo_data_filled = fill_missing_with_mode(demo_data, 'ID')
from snowflake.connector.pandas_tools import write_pandas
from db import create_snowflake_connection, run_query
conn = create_snowflake_connection()
#load the data into the snowflake
write_pandas(conn, demo_data_filled, 'JOINED_DATA_DRVD', overwrite=True, quote_identifiers=False)


# %%
