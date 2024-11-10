--Functions:

CREATE OR REPLACE FUNCTION LLM_DB.LLM_DEV.HAMMING_DISTANCE("STR1" VARCHAR(16777216), "STR2" VARCHAR(16777216))
RETURNS NUMBER(38,0)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'hamming_distance'
AS '
def hamming_distance(str1: str, str2: str) -> int:
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))
';

CREATE OR REPLACE FUNCTION LLM_DB.LLM_DEV.JACCARD_DISTANCE("SET1" VARCHAR(16777216), "SET2" VARCHAR(16777216))
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'jaccard_distance'
AS '
def jaccard_distance(set1, set2):
    # Handle None inputs
    if set1 is None and set2 is None:
        return 0.0  # Both sets are None, consider them equal (distance = 0)
    elif set1 is None or set2 is None:
        return 1.0  # One set is None, distance = 1 (completely dissimilar)

    # Convert strings to sets of elements
    set1 = set(set1.split('',''))
    set2 = set(set2.split('',''))

    # Calculate intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Handle the case where both sets are empty
    if union == 0:
        return 0.0  # Return 0.0 for empty sets

    # Calculate Jaccard distance
    return 1 - (intersection / union)
';


CREATE OR REPLACE FUNCTION LLM_DB.LLM_DEV.JARO_WINKLER_DISTANCE("S1" VARCHAR(16777216), "S2" VARCHAR(16777216))
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'jaro_winkler'
AS '
def jaro_winkler(s1, s2):
    if s1 == s2:
        return 1.0

    len_s1 = len(s1)
    len_s2 = len(s2)
    max_dist = (max(len_s1, len_s2) // 2) - 1

    match_s1 = [0] * len_s1
    match_s2 = [0] * len_s2

    matches = 0
    for i in range(len_s1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len_s2)
        for j in range(start, end):
            if match_s2[j] == 0 and s1[i] == s2[j]:
                match_s1[i] = 1
                match_s2[j] = 1
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Calculate the number of transpositions
    t = 0
    m = 0
    j = 0
    for i in range(len_s1):
        if match_s1[i] == 1:
            while j < len_s2 and match_s2[j] == 0:
                j += 1
            if j < len_s2 and s1[i] != s2[j]:
                t += 1
            j += 1
            m += 1

    t //= 2

    jaro_distance = (matches / len_s1 + matches / len_s2 + (matches - t) / matches) / 3.0

    # Calculate the Jaro-Winkler distance
    prefix_length = 0
    max_prefix_length = 4  # The maximum length of common prefix
    for i in range(min(len(s1), len(s2), max_prefix_length)):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break

    jaro_winkler_distance = jaro_distance + (prefix_length * 0.1 * (1 - jaro_distance))

    return jaro_winkler_distance
';

CREATE OR REPLACE FUNCTION LLM_DB.LLM_DEV.SUPPRESS_VOWELS("NAME" VARCHAR(16777216))
RETURNS VARCHAR(16777216)
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'suppress_vowels'
AS '
def suppress_vowels(name):
    vowels = ''aeiouAEIOU''
    return ''''.join([char for char in name if char not in vowels])
';

/*********************************/
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
	ADDRESS_VECTOR VECTOR(FLOAT, 768),
	TASK_ID NUMBER(38,0)
);
select * from LLM_DB.LLM_DEV.INPUT_RECORDS;

show classes in database snowflake;
show classes in schema  snowflake.ML;
--show functions in SNOWFLAKE.CORTEX;
--show snowflake.ML.classification;

CREATE OR REPLACE TABLE ds_inpt_lvld_data (
    TASK_ID VARCHAR(16777216),
	MCID VARCHAR(16777216),
	FIRST_NAME VARCHAR(16777216),
	LAST_NAME VARCHAR(16777216),
	MIDDLE_INITIAL VARCHAR(16777216),
	GENDER VARCHAR(16777216),
	DOB DATE,
	SUBSCRIBER_ID VARCHAR(16777216),
	ZIP_CODE VARCHAR(16777216),
	COUNTRY VARCHAR(16777216),
	RESPONSE_FINAL NUMBER(38,0)
);

select * from ds_inpt_lvld_data 
where TASK_ID IN (200001422,200001672,200001972) order by TASK_ID,MCID;

select distinct TASK_ID,RESPONSE_FINAL from ds_inpt_lvld_data 
--where TASK_ID IN (200001422,200001672,200001972);

select * from ds_inpt_lvld_data 
--where TASK_ID IN (200001422,200001672,200001972);

CREATE OR REPLACE TABLE ds_inpt_lvld_data_drvd (
    TASK_ID VARCHAR(16777216),
	MCID VARCHAR(16777216),
    FN_LN_MI STRLLM_DB.LLM_DEV.JACCARD_DISTANCELLM_DB.LLM_DEV.SUPPRESS_VOWELSING,
    FN_LN_MI_VECTOR VECTOR(FLOAT, 768),
    FN_LN_MI_SPRSD_VWL STRING,
    GENDER STRING,
    DOB DATE,
    RESPONSE_FINAL NUMBER(38,0)
);


INSERT INTO ds_inpt_lvld_data_drvd
select distinct 
 TASK_ID, MCID  ,FIRST_NAME||''||LAST_NAME||''||MIDDLE_INITIAL as FN_LN_MI ,
 SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m',FIRST_NAME||''||LAST_NAME||''||MIDDLE_INITIAL )as FN_LN_MI_VECTOR ,
 suppress_vowels(FIRST_NAME||''||LAST_NAME||''||MIDDLE_INITIAL) as FN_LN_MI_SPRSD_VWL,
 GENDER,
    DOB ,      
 RESPONSE_FINAL
 from ds_inpt_lvld_data;


 select * from ds_inpt_lvld_data_drvd order by task_id;

 Create or replace table ds_inpt_data_fetaure_comp as
SELECT
    A.TASK_ID AS TASK_ID,
    A.MCID AS MCID_a,
    B.MCID AS MCID_b,
    A.FN_LN_MI AS FN_LN_MI_A,
    B.FN_LN_MI AS FN_LN_MI_B,
    VECTOR_COSINE_SIMILARITY(A.FN_LN_MI_VECTOR, B.FN_LN_MI_VECTOR) AS fn_ln_mi_smlrty_score,
    JARO_WINKLER_DISTANCE(A.FN_LN_MI, B.FN_LN_MI) AS FN_LN_MI_JW,
    JACCARD_DISTANCE(A.FN_LN_MI, B.FN_LN_MI) AS FN_LN_MI_JD,
    EDITDISTANCE(A.FN_LN_MI, B.FN_LN_MI) AS FN_LN_MI_ED,
    EDITDISTANCE(A.FN_LN_MI_SPRSD_VWL, B.FN_LN_MI_SPRSD_VWL) AS FN_LN_MI_SPRS_VWL_ED,
    JARO_WINKLER_DISTANCE(A.FN_LN_MI_SPRSD_VWL, B.FN_LN_MI_SPRSD_VWL) AS FN_LN_MI_SPRS_VWL_JW,
    A.DOB AS DOB_A,
    B.DOB AS DOB_B,
    CASE WHEN EXTRACT(MONTH FROM A.DOB) = EXTRACT(MONTH FROM B.DOB) THEN 1 ELSE 0 END AS Month_Match,
    CASE WHEN EXTRACT(YEAR FROM A.DOB) = EXTRACT(YEAR FROM B.DOB) THEN 1 ELSE 0 END AS Year_Match,
    CASE WHEN EXTRACT(DAY FROM A.DOB) = EXTRACT(DAY FROM B.DOB) THEN 1 ELSE 0 END AS Day_Match,
    (CASE WHEN EXTRACT(MONTH FROM A.DOB) = EXTRACT(MONTH FROM B.DOB) THEN 1 ELSE 0 END +
     CASE WHEN EXTRACT(YEAR FROM A.DOB) = EXTRACT(YEAR FROM B.DOB) THEN 1 ELSE 0 END +
     CASE WHEN EXTRACT(DAY FROM A.DOB) = EXTRACT(DAY FROM B.DOB) THEN 1 ELSE 0 END) AS Total_Match_Score
FROM 
    ds_inpt_lvld_data_drvd A
JOIN 
    ds_inpt_lvld_data_drvd B ON A.TASK_ID = B.TASK_ID

WHERE 
    A.MCID < B.MCID;


    select * from ds_inpt_data_fetaure_comp where TASK_ID IN (200001422,200001672,200001972);





/*******************************/






    

CREATE OR REPLACE TABLE ds_inpt_data_fetaure_agg AS
with RESP as (select distinct task_id,response_final from ds_inpt_lvld_data) 
SELECT
    A.TASK_ID,
    AVG(fn_ln_mi_smlrty_score) AS avg_fn_ln_mi_smlrty_score,  -- Mean cosine similarity
    MEDIAN(fn_ln_mi_smlrty_score) AS median_fn_ln_mi_smlrty_score,  -- Median cosine similarity
    MAX(fn_ln_mi_smlrty_score) AS max_fn_ln_mi_smlrty_score,  -- Maximum cosine similarity
    MIN(fn_ln_mi_smlrty_score) AS min_fn_ln_mi_smlrty_score,  -- Minimum cosine similarity
    Coalesce(STDDEV(fn_ln_mi_smlrty_score),0) AS stddev_fn_ln_mi_smlrty_score,  -- Standard deviation of cosine similarity
    AVG(FN_LN_MI_JW) AS avg_FN_LN_MI_JW,  -- Mean Jaro-Winkler distance
    MEDIAN(FN_LN_MI_JW) AS median_FN_LN_MI_JW,  -- Median Jaro-Winkler distance
    MAX(FN_LN_MI_JW) AS max_FN_LN_MI_JW,  -- Maximum Jaro-Winkler distance
    MIN(FN_LN_MI_JW) AS min_FN_LN_MI_JW,  -- Minimum Jaro-Winkler distance
    coalesce(STDDEV(FN_LN_MI_JW),0) AS stddev_FN_LN_MI_JW,  -- Standard deviation of Jaro-Winkler distance
    AVG(FN_LN_MI_SPRS_VWL_JW) AS avg_FN_LN_MI_SPRS_VWL_JW,  -- Mean Jaro-Winkler distance on sprsdvowel
    MEDIAN(FN_LN_MI_SPRS_VWL_JW) AS median_FN_LN_MI_SPRS_VWL_JW,  -- Median Jaro-Winkler distance on sprsdvowel
    MAX(FN_LN_MI_SPRS_VWL_JW) AS max_FN_LN_MI_SPRS_VWL_JW,  -- Maximum Jaro-Winkler distance on sprsdvowel
    MIN(FN_LN_MI_SPRS_VWL_JW) AS min_FN_LN_MI_SPRS_VWL_JW,  -- Minimum Jaro-Winkler distance on sprsdvowel
    coalesce(STDDEV(FN_LN_MI_SPRS_VWL_JW),0) AS stddev_FN_LN_MI_SPRS_VWL_JW,  -- Standard deviation of Jaro-Winkler distance on sprsdvowel
    AVG(FN_LN_MI_JD) AS avg_FN_LN_MI_JD,  -- Mean Jaccard distance
    MEDIAN(FN_LN_MI_JD) AS median_FN_LN_MI_JD,  -- Median Jaccard distance
    MAX(FN_LN_MI_JD) AS max_FN_LN_MI_JD,  -- Maximum Jaccard distance
    MIN(FN_LN_MI_JD) AS min_FN_LN_MI_JD,  -- Minimum Jaccard distance
    Coalesce(STDDEV(FN_LN_MI_JD),0) AS stddev_FN_LN_MI_JD,  -- Standard deviation of Jaccard distance
    AVG(FN_LN_MI_ED) AS avg_FN_LN_MI_ED,  -- Mean Edit distance
    MEDIAN(FN_LN_MI_ED) AS median_FN_LN_MI_ED,  -- Median Edit distance
    MAX(FN_LN_MI_ED) AS max_FN_LN_MI_ED,  -- Maximum Edit distance
    MIN(FN_LN_MI_ED) AS min_FN_LN_MI_ED,  -- Minimum Edit distance
    coalesce(STDDEV(FN_LN_MI_ED),0) AS stddev_FN_LN_MI_ED,  -- Standard deviation of Edit distance
    AVG(FN_LN_MI_SPRS_VWL_ED) AS avg_FN_LN_MI_SPRS_VWL_ED,  -- Mean Edit distance with suppressed vowels
    MEDIAN(FN_LN_MI_SPRS_VWL_ED) AS median_FN_LN_MI_SPRS_VWL_ED,  -- Median Edit distance with suppressed vowels
    MAX(FN_LN_MI_SPRS_VWL_ED) AS max_FN_LN_MI_SPRS_VWL_ED,  -- Maximum Edit distance with suppressed vowels
    MIN(FN_LN_MI_SPRS_VWL_ED) AS min_FN_LN_MI_SPRS_VWL_ED,  -- Minimum Edit distance with suppressed vowels
    coalesce(STDDEV(FN_LN_MI_SPRS_VWL_ED),0) AS stddev_FN_LN_MI_SPRS_VWL_ED,  -- Standard deviation of Edit distance with suppressed vowels
    RESP.response_final
FROM
    ds_inpt_data_fetaure_comp  a-- This is the table with your comparison results
    inner join RESP 
    on a.task_id = resp.task_id
GROUP BY
    A.TASK_ID,RESP.response_final;

    select * from ds_inpt_data_fetaure_agg;


CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION model_fn_identification(
    INPUT_DATA => SYSTEM$REFERENCE('table', 'ds_inpt_data_fetaure_agg'),
    TARGET_COLNAME => 'RESPONSE_FINAL',
    CONFIG_OBJECT => {'evaluate': TRUE});


Create  or replace view ds_inpt_data_fetaure_agg_test as
select * exclude(response_final) from ds_inpt_data_fetaure_agg limit 10


create or replace table model_output_predict as
select *, 
  model_fn_identification!PREDICT(
    input_data => object_construct(*))
    as prediction
from ds_inpt_data_fetaure_agg_test;


select * from model_output_predict;


select *,
  prediction:"class"::boolean as class,
  prediction:"probability"."0" as probability_false,
  prediction:"probability"."1" as probability_true
from model_output_predict;



CALL model_fn_identification!SHOW_EVALUATION_METRICS();
CALL model_fn_identification!SHOW_GLOBAL_EVALUATION_METRICS();
CALL model_fn_identification!SHOW_THRESHOLD_METRICS();

CALL model_fn_identification!SHOW_CONFUSION_MATRIX();
CALL model_fn_identification!SHOW_FEATURE_IMPORTANCE();
CALL model_fn_identification!SHOW_TRAINING_LOGS();
