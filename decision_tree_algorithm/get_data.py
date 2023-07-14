import csv
import random

import pandas as pd

from decision_tree_algorithm.directories import household_data


def generate_data(number_of_entries: int, output_file: str):
    demographic_data = []

    for _ in range(1, number_of_entries + 1):
        household_size = random.randint(1, 10)
        income = random.randint(9000, 100000)
        race = random.choice(
            ["caucasian", "african american", "hispanic", "asian", "other"]
        )
        age = random.randint(0, 60)
        sex = random.choice(["male", "female"])
        zipcode = random.randint(10000, 99999)
        demographic_data.append([household_size, income, race, age, sex, zipcode])

    # Write data to CSV file
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["household_size", "income", "race", "age", "sex", "zipcode"])
        writer.writerows(demographic_data)

    return pd.read_csv(output_file)


# if __name__ == "__main__":
#     generate_data(3000, household_data)
