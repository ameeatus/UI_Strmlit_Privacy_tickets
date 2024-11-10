# SF siilarity sreahc function

SELECT
  issue,
  VECTOR_COSINE_SIMILARITY(
    issue_vec,
    SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', 'User could not install Facebook app on his phone')
  ) AS similarity
FROM issues
ORDER BY similarity DESC
LIMIT 5
WHERE DATEDIFF(day, CURRENT_DATE(), issue_date) < 90 AND similarity > 0.7

   'dob_a','dob_b','gender_a','gender_b','firstname_a','firstname_b','lastname_a','lastname_b'


   /***********************/

   FN1_FN2_Mean,FN1_LN2_Mean,FN2_LN1_Mean,LN1_LN2_Mean,FN_LN_MI1_FN_LN_MI2_Mean,FN1_FN2_Min,FN1_LN2_Min,FN2_LN1_Min,LN1_LN2_Min,FN_LN_MI1_FN_LN_MI2_Min,FN1_FN2_Max,FN1_LN2_Max,FN2_LN1_Max,LN1_LN2_Max,FN_LN_MI1_FN_LN_MI2_Max,FN1_FN2_Sd,FN1_LN2_Sd,FN2_LN1_Sd,LN1_LN2_Sd,FN_LN_MI1_FN_LN_MI2_Sd,FN1_FN2_JD_Mean,FN1_LN2_JD_Mean,FN2_LN1_JD_Mean,LN1_LN2_JD_Mean,FN_LN_MI1_FN_LN_MI2_JD_Mean,FN1_FN2_JD_Min,FN1_LN2_JD_Min,FN2_LN1_JD_Min,LN1_LN2_JD_Min,FN_LN_MI1_FN_LN_MI2_JD_Min,FN1_FN2_JD_Max,FN1_LN2_JD_Max,FN2_LN1_JD_Max,LN1_LN2_JD_Max,FN_LN_MI1_FN_LN_MI2_JD_Max,FN1_FN2_JD_Sd,FN1_LN2_JD_Sd,FN2_LN1_JD_Sd,LN1_LN2_JD_Sd,FN_LN_MI1_FN_LN_MI2_JD_Sd,FN1_FN2_JW_Mean,FN1_LN2_JW_Mean,FN2_LN1_JW_Mean,LN1_LN2_JW_Mean,FN_LN_MI1_FN_LN_MI2_JW_Mean,FN1_FN2_JW_Min,FN1_LN2_JW_Min,FN2_LN1_JW_Min,LN1_LN2_JW_Min,FN_LN_MI1_FN_LN_MI2_JW_Min,FN1_FN2_JW_Max,FN1_LN2_JW_Max,FN2_LN1_JW_Max,LN1_LN2_JW_Max,FN_LN_MI1_FN_LN_MI2_JW_Max,FN1_FN2_JW_Sd,FN1_LN2_JW_Sd,FN2_LN1_JW_Sd,LN1_LN2_JW_Sd,FN_LN_MI1_FN_LN_MI2_JW_Sd

   /**************/
  FN_LN_MI_cosine_smiliarity --> MIN , MAX, SD,MEan
  FN_LN_MI_JD --> MIN , MAX, SD,MEan
  FN_LN_MI_JW --> MIN , MAX, SD,MEan
  
