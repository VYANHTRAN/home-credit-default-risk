import numpy as np
import pandas as pd


def _safe_div(a, b, fill=np.nan):
    """
    Divide a / b, replacing zero denominators with `fill`.
    """
    denom = b.replace(0, np.nan)
    result = a / denom

    if not np.isnan(fill):
        result = result.fillna(fill)

    return result

def add_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to a DataFrame (either train or test).
    """
    df = df.copy()
    cols = set(df.columns)

    # 1. EXT_SOURCE interactions                                          
    # EXT_SOURCE_{1,2,3} are the strongest predictors in the dataset.
    # Defaulters score systematically lower on all three. 

    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in cols]

    if ext:
        df["EXT_SOURCE_MEAN"]   = df[ext].mean(axis=1)
        df["EXT_SOURCE_MIN"]    = df[ext].min(axis=1)
        df["EXT_SOURCE_MAX"]    = df[ext].max(axis=1)
        df["EXT_SOURCE_STD"]    = df[ext].std(axis=1)  
        df["EXT_SOURCE_RANGE"]  = df["EXT_SOURCE_MAX"] - df["EXT_SOURCE_MIN"]

    if {"EXT_SOURCE_1", "EXT_SOURCE_2"}.issubset(cols):
        df["EXT_12_PRODUCT"]  = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]

    if {"EXT_SOURCE_1", "EXT_SOURCE_3"}.issubset(cols):
        df["EXT_13_PRODUCT"]  = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]

    if {"EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(cols):
        df["EXT_23_PRODUCT"]  = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    if len(ext) == 3:
        df["EXT_123_PRODUCT"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # EXT_SOURCE combined with credit amount => high loan + low score = danger
    if {"EXT_SOURCE_MEAN", "AMT_CREDIT"}.issubset(df.columns):
        df["EXT_MEAN_X_CREDIT"] = df["EXT_SOURCE_MEAN"] * df["AMT_CREDIT"]


    # 2. Credit affordability                                             
    # A borrower who must spend a large share of income on debt service
    # is at much higher risk of default.

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(cols):
        # Debt-service-to-income ratio
        df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(cols):
        # Total loan vs annual income
        df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])

    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(cols):
        # Implicit loan term in years — longer term = greater repayment risk
        df["CREDIT_TERM_YEARS"] = _safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"] * 12)

    if {"AMT_INCOME_TOTAL", "AMT_ANNUITY"}.issubset(cols):
        # Residual income after annuity: how much is left over each month?
        df["RESIDUAL_INCOME"] = df["AMT_INCOME_TOTAL"] / 12 - df["AMT_ANNUITY"]

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(cols):
        # Per-capita income
        df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])


    # 3. Loan structure / LTV proxy                                       
    # Borrowing more than the good is worth is a red flag; it may signal
    # the applicant is using the loan for cash extraction.

    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(cols):
        df["CREDIT_GOODS_RATIO"]   = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
        # Over-financing flag: credit exceeds goods price
        df["OVER_FINANCED"]        = (df["AMT_CREDIT"] > df["AMT_GOODS_PRICE"]).astype(np.int8)
        df["CREDIT_GOODS_DIFF"]    = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]

    if {"AMT_ANNUITY", "AMT_GOODS_PRICE"}.issubset(cols):
        # Annuity relative to goods
        df["ANNUITY_GOODS_RATIO"]  = _safe_div(df["AMT_ANNUITY"], df["AMT_GOODS_PRICE"])


    # 4. Employment stability                                            
    # Borrowers who have been employed longer, especially relative to their
    # age, are far more stable credits.

    if "DAYS_EMPLOYED" in cols:
        # Boolean: is this person currently employed (vs pensioner/unemployed)?
        df["IS_EMPLOYED"] = (df["DAYS_EMPLOYED"] != 365243).astype(np.int8)

    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(cols):
        employed_adj = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        # Years employed as a share of working-age life
        # DAYS_BIRTH is negative (days before application), so age = -DAYS_BIRTH
        df["EMPLOYED_FRACTION_OF_LIFE"]  = _safe_div(-employed_adj, -df["DAYS_BIRTH"])
        # Career seniority: years employed / age in years
        df["EMPLOYMENT_YEARS"] = -employed_adj / 365.25


    # 5. Age & life-stage                                                 
    # Very young applicants (< 25) and very old ones default at higher rates.

    if "DAYS_BIRTH" in cols:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

        # Risk-relevant age bands (from EDA: young adults default more)
        age = df["AGE_YEARS"]
        df["AGE_BAND"] = pd.cut(
            age,
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype("float32")

        df["IS_YOUNG_ADULT"]   = (age < 30).astype(np.int8)
        df["IS_NEAR_RETIREMENT"] = (age > 58).astype(np.int8)


    # 6. Social / family & housing risk                                   

    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(cols):
        # Children as a share of household 
        df["CHILD_DEPENDENCY_RATIO"] = _safe_div(
            df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"]
        )
        # Flag: more children than income can reasonably support
        df["MANY_CHILDREN"] = (df["CNT_CHILDREN"] >= 3).astype(np.int8)

    if {"DAYS_REGISTRATION", "DAYS_BIRTH"}.issubset(cols):
        # How recently (relative to age) did the applicant register their address?
        # Late / recent registration may indicate instability or fraud.
        df["REGISTRATION_AGE_RATIO"] = _safe_div(
            df["DAYS_REGISTRATION"], df["DAYS_BIRTH"]
        )

    if {"DAYS_ID_PUBLISH", "DAYS_BIRTH"}.issubset(cols):
        # Time since ID was last published relative to age
        df["ID_PUBLISH_AGE_RATIO"] = _safe_div(
            df["DAYS_ID_PUBLISH"], df["DAYS_BIRTH"]
        )

    if {"DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH"}.issubset(cols):
        # Frequent phone changes are an identity instability signal
        df["PHONE_CHANGE_AGE_RATIO"] = _safe_div(
            df["DAYS_LAST_PHONE_CHANGE"], df["DAYS_BIRTH"]
        )

    # 7. Information transparency                                         
    # Applicants who submit fewer documents and provide fewer contact
    # details are harder to verify, correlating with higher default.

    doc_cols = [c for c in cols if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCS_SUBMITTED"]     = df[doc_cols].sum(axis=1)
        df["NO_DOCS_SUBMITTED"]  = (df["DOCS_SUBMITTED"] == 0).astype(np.int8)

    contact_flags = [c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                                  "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"] if c in cols]
    if contact_flags:
        df["CONTACT_REACHABILITY"] = df[contact_flags].sum(axis=1)


    # 8. Asset ownership                                                
    # Car ownership can indicate wealth, but an old car relative to the
    # owner's age may signal persistent low wealth accumulation.

    if {"OWN_CAR_AGE", "DAYS_BIRTH"}.issubset(cols):
        df["CAR_AGE_TO_OWNER_AGE"] = _safe_div(
            df["OWN_CAR_AGE"], df["AGE_YEARS"] if "AGE_YEARS" in df.columns
            else -df["DAYS_BIRTH"] / 365.25
        )
        # Flag: applicant owns no car (OWN_CAR_AGE is NaN when no car)
        df["HAS_CAR"] = df["OWN_CAR_AGE"].notna().astype(np.int8)

    # 9. Credit bureau enquiry recency                                   
    # Many recent enquiries signal desperation for credit,
    # a well-known predictor of near-term default.

    enq_cols = {
        "AMT_REQ_CREDIT_BUREAU_HOUR":  1,
        "AMT_REQ_CREDIT_BUREAU_DAY":   7,
        "AMT_REQ_CREDIT_BUREAU_WEEK":  1,
        "AMT_REQ_CREDIT_BUREAU_MON":   1,
        "AMT_REQ_CREDIT_BUREAU_QRT":   1,
        "AMT_REQ_CREDIT_BUREAU_YEAR":  1,
    }
    present_enq = [c for c in enq_cols if c in cols]
    if present_enq:
        df["TOTAL_ENQUIRIES"] = df[present_enq].sum(axis=1)
        df["RECENT_ENQUIRIES_30D"] = df[
            [c for c in ["AMT_REQ_CREDIT_BUREAU_HOUR",
                          "AMT_REQ_CREDIT_BUREAU_DAY",
                          "AMT_REQ_CREDIT_BUREAU_WEEK",
                          "AMT_REQ_CREDIT_BUREAU_MON"] if c in cols]
        ].sum(axis=1)
        df["CREDIT_HUNTING_FLAG"] = (df["RECENT_ENQUIRIES_30D"] >= 3).astype(np.int8)


    # 10. Composite risk score                  
    # A hand-crafted composite that multiplies the three most potent
    # signals: external score quality, debt burden, and employment.

    if "EXT_SOURCE_MEAN" in df.columns and "ANNUITY_INCOME_RATIO" in df.columns:
        # Higher EXT score = safer; higher annuity ratio = riskier
        # so we invert: low EXT * high burden = high risk composite
        df["RISK_COMPOSITE"] = (1 - df["EXT_SOURCE_MEAN"]) * df["ANNUITY_INCOME_RATIO"]

    if "EXT_SOURCE_MEAN" in df.columns and "CREDIT_INCOME_RATIO" in df.columns:
        df["RISK_COMPOSITE_CREDIT"] = (1 - df["EXT_SOURCE_MEAN"]) * df["CREDIT_INCOME_RATIO"]

    return df