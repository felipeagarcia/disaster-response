import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Reads the data files and merges it into a df

    INPUTS
    messages_filepath - string with messages dataset path
    categories_filepath - string with categories dataset path

    OUTPUTS
    df - resulting dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    Clean and prepare the data, preparing the categories column and dropping duplicates

    INPUTS
    df - pandas dataframe to be cleanned

    OUTPUTS
    df - cleanned df
    '''
    # Split the categories columns on ';'
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))

    # Get the categories names
    colnames = df['categories'].str.split(';', expand=True).iloc[0]\
                                    .apply(lambda x: x.split('-')[0])
    categories.columns = colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df = pd.concat([df.drop(columns=['categories']),
                    categories], axis=1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    Saves the dataframe into sqlite database

    INPUTS
    df - pandas dataframe to be saved
    database_filename - path to save the db
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()