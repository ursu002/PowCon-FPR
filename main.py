# Data packages
import os
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
# Plotting packages
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.simplefilter("ignore", ValueWarning)


def main():
    sb.set()
    # set the maximum number of images to be opened at once
    plt.rcParams['figure.max_open_warning'] = 50
    # import MinMaxScaler for data normalization
    # create a directory for the cleaned CSV files, if it doesn't already exist
    cleaned_data_dir_1 = './clean-data/minute'
    if not os.path.exists(cleaned_data_dir_1):
        os.makedirs(cleaned_data_dir_1)

    # create an empty list to store the DataFrames for each cleaned CSV file
    dfs = []
    # loop through each CSV file
    for file in os.listdir('./data'):
        # check if the file is a CSV file
        if os.path.splitext(file)[1] == '.csv':
            try:
                # read the CSV file into a DataFrame, specifying the encoding
                df = pd.read_csv('./data/' + file, encoding='utf-8')

                # clean the DataFrame
                # remove duplicates and deal with missing values
                # fill in the missing values with forward fill
                df.fillna(method='ffill', inplace=True)

                # drop duplicates
                df.drop_duplicates(inplace=True)

                # convert the 'timestamp' column to datetime format
                df['datetime'] = pd.to_datetime(
                    df['timestamp'], errors='coerce')

                # floor to nearest minute
                freq = '1min'
                df['datetime'] = df['datetime'].dt.floor(freq)

                # separate power by minute
                df = df.groupby(['datetime'], as_index=False)[
                    'power'].sum()

                # create a column = 'power' for the sum of power per minute
                df = df.groupby(['datetime'], as_index=False)[
                    'power'].sum()
                # remove outliers
                df = df[np.abs(
                    df['power']-df['power'].mean()) <= (3*df['power'].std())]

                # add minute column
                df['minute'] = df.index - df.index.min()

                # save the cleaned DataFrame to a new CSV file with the same name as the original file
                cleaned_file_path = os.path.join(
                    cleaned_data_dir_1, file.replace('.csv', '_per_min.csv'))
                df.to_csv(cleaned_file_path, index=False)

                # add the cleaned DataFrame to the list of DataFrames
                dfs.append(df)

            except UnicodeDecodeError:
                # print the name of the file that caused the decoding error
                print(f"Error decoding file: {file}")

    # create an empty list to store the DataFrames for each cleaned CSV file
    dfs = []

    # loop through each cleaned CSV file
    for file in os.listdir(cleaned_data_dir_1):
        # check if the file is a CSV file
        if os.path.splitext(file)[1] == '.csv':
            # read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(cleaned_data_dir_1, file))
            # append the DataFrame to the list of DataFrames
            dfs.append(df)

    # combine all the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # group the data by minute and calculate the sum of the power
    summary_df = combined_df.groupby(
        ['datetime'], as_index=False)['power'].sum()
    # add minute column
    summary_df['minute'] = summary_df.index - summary_df.index.min()
    # save the summary DataFrame to a new CSV file
    summary_file_path = os.path.join(cleaned_data_dir_1, 'summary.csv')
    summary_df.to_csv(summary_file_path, index=False)

    # create a directory for the cleaned CSV files, if it doesn't already exist
    cleaned_data_dir_2 = './clean-data/day'
    if not os.path.exists(cleaned_data_dir_2):
        os.makedirs(cleaned_data_dir_2)
    # create an empty list to store the DataFrames for each cleaned CSV file
    dfs = []
    # loop through each CSV file
    for file in os.listdir('./data'):
        # check if the file is a CSV file
        if os.path.splitext(file)[1] == '.csv':
            try:
                # read the CSV file into a DataFrame, specifying the encoding
                df = pd.read_csv(
                    './data/' + file,
                    encoding='utf-8')

                # fill in the missing values with forward fill
                df.fillna(method='ffill', inplace=True)

                # drop duplicates
                df.drop_duplicates(inplace=True)

                # convert the 'timestamp' column to datetime format
                df['datetime'] = pd.to_datetime(
                    df['timestamp'], errors='coerce')

                # floor to nearest  day
                freq = '1D'
                df['datetime'] = df['datetime'].dt.floor(freq)

                # separate power by day
                df = df.groupby(['datetime'], as_index=False)[
                    'power'].sum()

                # create a column = 'power' for the sum of power per day
                df = df.groupby(['datetime'], as_index=False)[
                    'power'].sum()
                # remove outliers
                df = df[np.abs(
                    df['power']-df['power'].mean()) <= (3*df['power'].std())]

                # add day column
                df['day'] = df.index - df.index.min()

                # save the cleaned DataFrame to a new CSV file with the same name as the original file
                cleaned_file_path = os.path.join(
                    cleaned_data_dir_2, file.replace('.csv', '_per_day.csv'))
                df.to_csv(cleaned_file_path, index=False)

                # add the cleaned DataFrame to the list of DataFrames
                dfs.append(df)

            except UnicodeDecodeError:
                # print the name of the file that caused the decoding error
                print(f"Error decoding file: {file}")

    # create an empty list to store the DataFrames for each cleaned CSV file
    dfs = []

    # loop through each cleaned CSV file
    for file in os.listdir(cleaned_data_dir_2):
        # check if the file is a CSV file
        if os.path.splitext(file)[1] == '.csv':
            # read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(cleaned_data_dir_2, file))
            # append the DataFrame to the list of DataFrames
            dfs.append(df)

    # combine all the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # group the data by day and calculate the sum of the power
    summary_df = combined_df.groupby(
        ['datetime'], as_index=False)['power'].sum()
    # add day column
    summary_df['day'] = summary_df.index - summary_df.index.min()
    # save the summary DataFrame to a new CSV file
    summary_file_path = os.path.join(cleaned_data_dir_2, 'summary.csv')
    summary_df.to_csv(summary_file_path, index=False)

    directory = './clean-data/day'
    directory_graphs = './graphs/day'
    directory_forecasts = './forecast/day'

    if not os.path.exists(directory_graphs):
        os.makedirs(directory_graphs)

    if not os.path.exists(directory_forecasts):
        os.makedirs(directory_forecasts)

    # get a list of all csv files in the directory
    csv_files = [file for file in os.listdir(
        directory) if file.endswith('.csv')]

    for file in csv_files:
        # read the csv file into a pandas dataframe
        df = pd.read_csv(os.path.join(directory, file))

        # create a graph for the current file
        fig = go.Figure([go.Scatter(x=df['day'], y=df['power'])])

        # set the title of the graph to the name of the current file
        title = os.path.splitext(file)[0]
        fig.update_layout(title=title)

        # save the graph as a JPEG image in the same directory as the csv file
        image_path = os.path.join(directory_graphs, f"{title}.jpg")
        fig.write_image(image_path, format='jpeg')

    directory_2 = './clean-data/minute'
    directory_graphs_2 = './graphs/minute'

    if not os.path.exists(directory_graphs_2):
        os.makedirs(directory_graphs_2)

    # get a list of all csv files in the directory
    csv_files = [file for file in os.listdir(
        directory_2) if file.endswith('.csv')]

    for file in csv_files:
        # read the csv file into a pandas dataframe
        df = pd.read_csv(os.path.join(directory_2, file))

        # create a graph for the current file
        fig = go.Figure([go.Scatter(x=df['minute'], y=df['power'])])

        # set the title of the graph to the name of the current file
        title = os.path.splitext(file)[0]
        fig.update_layout(title=title)

        # save the graph as a JPEG image in the same directory as the csv file
        image_path = os.path.join(directory_graphs_2, f"{title}.jpg")
        fig.write_image(image_path, format='jpeg')

    # 1 start linear regression
    # loop through each CSV file
    for file in os.listdir(directory):
        # check if the file is a CSV file
        if os.path.splitext(file)[1] == '.csv':
            try:
                # read the csv file into a pandas dataframe
                df = pd.read_csv(os.path.join(directory, file))

                # convert the 'datetime' column to datetime object
                df['datetime'] = pd.to_datetime(df['datetime'])

                # set the 'datetime' column as the index
                df.set_index('datetime', inplace=True)

                # resample the data to hourly frequency and forward fill missing values
                df = df.resample('H').ffill()

                # resample the data to daily frequency, averaging the power values
                df = df.resample('D').mean()

                # create a new column for the hour
                df['hour'] = df.index.hour
                # create a pivot table to reshape the data
                df = df.pivot_table(index=df.index.date,
                                    columns='hour', values='power')
                # recreate the 'power' column as the mean of all hourly power values
                df['power'] = df.mean(axis=1)

                # reset the index and set the 'datetime' column as the index
                df.reset_index(inplace=True)
                df['datetime'] = pd.to_datetime(df['index'])
                df.set_index('datetime', inplace=True)
                df.drop('index', axis=1, inplace=True)
                # split data into training and testing set
                X = df['power'].values
                size = int(len(X) * 0.66)
                train, test = X[0:size], X[size:len(X)]

                # fit the ARIMA model
                model = ARIMA(df['power'], order=(5, 1, 0))
                model_fit = model.fit()
                # make predictions for the next 24*30 hours (one month)
                forecast = model_fit.forecast(steps=24*30)[0]
                # print the forecast
                print(f"Forecast for {file}: {forecast}")
                # calculate the R-squared score for the test set
                preds = []
                for t in range(len(test)):
                    model = ARIMA(train, order=(5, 1, 0))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    preds.append(yhat)
                    obs = test[t]
                    train = np.append(train, [obs])
                r2 = r2_score(test, preds)
                print(f"R-squared score for {file}: {r2}")
                # save the forecast results in a new CSV file in the  directory
                forecast_df = pd.DataFrame({'forecast': [forecast]})
                forecast_df.to_csv(os.path.join(
                    directory_forecasts, f"{os.path.splitext(file)[0]}_forecast.csv"), index=False)
            except UnicodeDecodeError:
                # print the name of the file that caused the decoding error
                print(f"Error decoding file: {file}")


if __name__ == "__main__":
    main()
