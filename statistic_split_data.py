import pandas as pd
import numpy as np
import json
import os
import hashlib


PASSWORD = str(os.environ.get("SALT_ANON_PATIENT"))


def clean_text(string: str) -> str:
    # clean and standardize text descriptions, which makes searching files easier
    forbidden_symbols = [
        "*",
        ".",
        "+",
        ",",
        '"',
        "\\",
        "/",
        "|",
        "[",
        "]",
        ":",
        ";",
        " ",
        "-",
    ]
    for symbol in forbidden_symbols:
        string = string.replace(symbol, "_")  # replace everything with an underscore
    return string.lower()


def sanitize_output_name_for_excel(forname: str, name: str) -> str:
    return clean_text(name + " " + forname).replace("_", " ").replace("-", " ").strip()


def load_list_patients() -> pd.DataFrame:
    dfListPatients = pd.read_excel(
        io="../../../data/raw/liste_patients_finale.xlsx", sheet_name="Sheet1"
    )
    for index, row in dfListPatients.iterrows():
        forname = (
            clean_text(row["Prénom"])
            .replace("_", " ")
            .replace("-", " ")
            .strip()
            .title()
        )
        name = (
            clean_text(row["Nom"]).replace("_", " ").replace("-", " ").strip().title()
        )
        dfListPatients.loc[index, "clean_name"] = sanitize_output_name_for_excel(
            row["Prénom"], (row["Nom"])
        )
        dfListPatients.loc[index, "anon_name"] = hashlib.sha256(
            sanitize_output_name_for_excel(row["Prénom"], (row["Nom"])).encode("utf-8")
            + PASSWORD.encode("utf-8")
        ).hexdigest()

    return dfListPatients


def filter_data(detail_df, localisation):
    filtered_df = detail_df[detail_df["Localisation"] == localisation]
    filtered_df["Age_Group"] = pd.cut(
        filtered_df["Age"],
        bins=range(0, 101, 10),
        right=False,
        labels=[f"{i}-{i+9}" for i in range(0, 100, 10)],
    )

    pivot_table = pd.pivot_table(
        filtered_df, index=["Age_Group"], columns=["Sexe"], aggfunc="size"
    )  # values='Sexe',
    distribution_df = pivot_table.reset_index()
    distribution_df.drop(columns="O", inplace=True)
    distribution_df["Total"] = distribution_df.iloc[:, :].sum(axis=1)
    distribution_df.sort_index(inplace=True)

    return distribution_df


def generate_groups(filtered_df):

    total_population = filtered_df["Total"].sum()
    test_set_size = int(total_population * 0.2)
    train_set_size = total_population - test_set_size
    proportions = filtered_df["Total"] / total_population
    group_size = train_set_size // 5
    print(
        "total",
        total_population,
        "train",
        train_set_size,
        group_size,
        "test",
        test_set_size,
    )

    groups = [{"Female": [], "Male": []} for _ in range(5)]
    distribution = []

    for i in range(len(filtered_df)):
        female_proportion = filtered_df.iloc[i]["F"] / filtered_df.iloc[i]["Total"]
        male_proportion = filtered_df.iloc[i]["M"] / filtered_df.iloc[i]["Total"]

        for group in groups:
            group["Female"].append(
                round(group_size * proportions.iloc[i] * female_proportion)
            )
            group["Male"].append(
                round(group_size * proportions.iloc[i] * male_proportion)
            )

    # Create a separate group for the test set
    test_group = {"Female": [], "Male": []}
    for i in range(len(filtered_df)):
        female_proportion = filtered_df.iloc[i]["F"] / filtered_df.iloc[i]["Total"]
        male_proportion = filtered_df.iloc[i]["M"] / filtered_df.iloc[i]["Total"]
        test_group["Female"].append(
            round(test_set_size * proportions.iloc[i] * female_proportion)
        )
        test_group["Male"].append(
            round(test_set_size * proportions.iloc[i] * male_proportion)
        )
    # print("test", test_group)
    groups.append(test_group)

    for i in range(len(filtered_df)):
        total_female_in_groups = sum(group["Female"][i] for group in groups)
        total_male_in_groups = sum(group["Male"][i] for group in groups)
        # print("Female", total_female_in_groups, "Male", total_male_in_groups)
        if total_female_in_groups > filtered_df.iloc[i]["F"]:
            excess_female = total_female_in_groups - filtered_df.iloc[i]["F"]
            for group in groups:
                if group["Female"][i] > 0:
                    group["Female"][i] -= round(excess_female / len(groups))

        if total_male_in_groups > filtered_df.iloc[i]["M"]:
            excess_male = total_male_in_groups - filtered_df.iloc[i]["M"]
            for group in groups:
                if group["Male"][i] > 0:
                    group["Male"][i] -= round(excess_male / len(groups))
    groups_list = []
    for dictionary in groups:
        female_list = dictionary["Female"]
        male_list = dictionary["Male"]
        groups_list.append([(f, m) for f, m in zip(female_list, male_list)])
    # print("Final", groups_list)
    return groups_list


def extract_data(detail_df, distribution_df, groups, localisation):
    data_groups = [[] for _ in range(6)]
    detail_df = detail_df[detail_df["Localisation"] == localisation]

    age_intervals = distribution_df["Age_Group"].str.split("-", expand=True).astype(int)
    for idx, row in distribution_df.iterrows():
        age_range_min, age_range_max = age_intervals.loc[idx]

        subset_df = detail_df[
            (detail_df["Age"] >= age_range_min) & (detail_df["Age"] <= age_range_max)
        ]

        for gender in ["F", "M"]:
            gender_subset_df = subset_df[subset_df["Sexe"] == gender]

            ##Treat by gender to identify correctly when group is full
            for _, data_row in gender_subset_df.iterrows():
                # Group choosen for data
                if gender == "F":
                    tot = sum(group[idx][0] for group in groups)
                    if tot > 0:
                        p = [(group[idx][0]) / tot for group in groups]
                        # print("P", p, "distribu", tot)
                        group_idx = np.random.choice(
                            range(6), p=[(group[idx][0]) / tot for group in groups]
                        )
                        data_groups[group_idx].append(data_row["anon_name"])
                        groups[group_idx][idx] = (
                            groups[group_idx][idx][0] - 1,
                            groups[group_idx][idx][1],
                        )

                else:
                    tot = sum(group[idx][1] for group in groups)
                    if tot > 0:
                        p = [(group[idx][1]) / tot for group in groups]
                        # print("P", p, "distribu", tot)
                        group_idx = np.random.choice(
                            range(6), p=[(group[idx][1]) / tot for group in groups]
                        )
                        data_groups[group_idx].append(data_row["anon_name"])
                        groups[group_idx][idx] = (
                            groups[group_idx][idx][0],
                            groups[group_idx][idx][1] - 1,
                        )

    return data_groups


detail_df = load_list_patients()
localisation = "Abdomen"
distribution_df = filter_data(detail_df, localisation)

# Step 2: Generate groups
groups = generate_groups(distribution_df)

# Step 3: Extract data
data_groups = extract_data(detail_df, distribution_df, groups, localisation)

# Step 4: Save data groups to separate JSON files
for idx, data_group in enumerate(data_groups):
    with open(f"data_group_{idx+1}.json", "w") as json_file:
        json.dump(data_group, json_file, indent=4)
    print(f"Data group {idx+1} saved as data_group_{idx+1}.json")

print("Data extraction completed.")
