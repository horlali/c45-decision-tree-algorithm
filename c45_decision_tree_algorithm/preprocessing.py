import pandas as pd


def preprocess_data(input_csv, output_csv):
    # Load the data from the input CSV file
    data = pd.read_csv(input_csv)

    # Definition of rules to apply to the data
    def apply_rules(row):
        household_size = row["household_size"]
        income_20k_25k = ((data["income"] >= 20000) & (data["income"] <= 25000)).mean()
        caucasian_percentage = (data["race"] == "caucasian").mean()
        males_age_5_or_less = ((data["sex"] == "male") & (data["age"] <= 5)).mean()
        income_45k_50k = ((data["income"] >= 45000) & (data["income"] <= 50000)).mean()
        age_15_20 = ((data["age"] >= 15) & (data["age"] <= 20)).mean()

        if household_size > 4139 and income_20k_25k > 0.11:
            return "NO"

        elif (
            household_size > 4139
            and income_20k_25k <= 0.11
            and caucasian_percentage > 0.116
            and males_age_5_or_less > 0.054
        ):
            return "NO"

        elif (
            household_size > 4139
            and income_20k_25k <= 0.11
            and caucasian_percentage > 0.116
            and males_age_5_or_less <= 0.054
            and income_45k_50k > 0.076
        ):
            return "NO"

        elif (
            household_size > 4139
            and income_20k_25k <= 0.11
            and caucasian_percentage > 0.116
            and males_age_5_or_less <= 0.054
            and income_45k_50k <= 0.076
            and age_15_20 > 0.028
        ):
            return "YES"

        elif (
            household_size > 4139
            and income_20k_25k <= 0.11
            and caucasian_percentage <= 0.116
        ):
            return "NO"

        else:
            return "NO"

    # Apply the rules to each row of data
    data["decision"] = data.apply(apply_rules, axis=1)

    # Write the transformed data to the output CSV file
    data.to_csv(output_csv, index=False)

    return pd.read_csv(output_csv)
