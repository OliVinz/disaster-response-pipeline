import sys

# import libraries
import os
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(messages_filepath, categories_filepath):
    """
    Function:
    loads data from 2 csv files + merge them

    Args:
    messages_filepath (str): the file path of messages csv file
    categories_filepath (str): the file path of categories csv file

    Return:
    df (DataFrame): A dataframe of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath, encoding="latin-1", low_memory=False)
    df = pd.merge(messages, categories, on="id", how="outer")
    return df


def clean_data(df):
    """
    Function:
    Cleaning the dataframe named df

    Args:
    df (DataFrame): A dataframe of messages and categories earlier imported

    Return:
    df (DataFrame): returning a cleaned dataframe
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[[1]]
    category_colnames = [category_name.split("-")[0] for category_name in row.values[0]]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)

    df.drop(["categories"], axis=1, inplace=True)
    df = pd.concat([df, categories], join="inner", axis=1)
    
    # remove rows where the values are not 0 or 1 in the categorical values
    for col in list(category_colnames):
        df = df.drop(df[(df[col] != 0) & (df[col] != 1)].index)

    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Function:
    Store the dataframe df into an sqlite database

    Args:
    df (DataFrame): A dataframe of messages and categories
    database_filename (str): The file name of the database

    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("messages", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
