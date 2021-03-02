import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    """ 
    This function loads messages and category data and 
    merges them together for later cleaning.
    
    Parameters: 
        messages_filepath:      File path of the messages.csv data file.
        categories_filepath:    File path of the catefories.csv data file.
  
    Returns: 
        df: A pandas dataframe of the whole data. 
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories, how='inner', on="id")
    
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories =  df.categories.str.split(pat=";", n=-1, expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from df
    df.drop('categories', axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates in the data
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """ 
    This function saves the data into a SQL database.
    
    Parameters: 
        df:                The pandas dataframe of the cleaned and processed data.
        database_filename: The name of the database to create wth the data
  
    Returns: 
        SQL Database
    """
    
    # Create SQL engine
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filename))
    
    # Write to an SQL database
    df.to_sql('df', engine, index=False)  


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