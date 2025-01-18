import os
import pandas as pd
import numpy as np
import clustering.clustering as clustering
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def remove_trailing_spaces_from_labels(basepath, filename='labeled_data_all_new.csv', column='activity_label'):
    """
    This function will remove the trailing spaces from the activity labels in the data files.
    :param basepath: The path to the directory containing the data files
    :return: None
    """
    df = pd.read_csv(f'{basepath}/NIH/{filename}')

    df['activity_label'] = df['activity_label'].str.strip()
    df.to_csv(basepath +'/NIH/'+ filename, index=False)

def remove_unlabeled_rows(df):
    """
    This function removes the rows that have the activity label as _unknown_
    """
    fildf = df[df.activity_label != '_unknown_']
    return fildf


def min_max_scale_columns(df, columns):
    """
    This function applies Min-Max scaling to the specified columns of the DataFrame.
    :param df: DataFrame to scale
    :param columns: List of columns to scale
    :return: DataFrame with scaled columns
    """
    if not df.empty:
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
    return df


def create_labeled_windows(roughpath, min_max_scale=False):
    """
    this file take the rough file of each user (it has labelled and unlablled both data)
    and creates only a labeled file for each user

    :param roughpath: The path that creates the rough
    :return:
    """
    files = os.listdir(roughpath)
    #print(files)
    labeled_path = roughpath + '../labeled/'
    files.sort()
    columns_to_scale = ['distance_from_cp', 'avg_altitude']

    for filename in files:
        if '.csv' not in filename:
            continue

        user_id = filename[filename.find('participant_') + len('participant_'):filename.rfind('.csv')]
        df = pd.read_csv(roughpath + filename)

        try:
            fildf = remove_unlabeled_rows(df)
            n = fildf.shape[0]
            user_arr = [user_id for i in range(n)]
            fildf = fildf.assign(user_id=user_arr)
            if min_max_scale:
                fildf = min_max_scale_columns(fildf, columns_to_scale)
            fildf.to_csv(labeled_path + 'labeled_features_user_{}.csv'.format(user_id))

            # comment the following line --> this is for testing purpose
            # break
        except AttributeError:
            print(filename, 'is problematic')


def merge_labeled_files(labeled_path):
    """
    This function will merge all the labeled files into one file.
    :param labeled_path: The directory containing the labeled files for individual users.
    :return: None, but creates the CSV file in the parent directory to the labeled path
    """

    files = os.listdir(labeled_path)
    files.sort()

    basefilename = 'labeled_features_user_{}.csv'

    df = pd.DataFrame()
    for user in range(1, 40):
        filename = basefilename.format(user)
        '''
        Example filename: labeled_features_user_1.csv
        '''
        print('Now merging:', filename)
        thisdf = pd.read_csv(labeled_path + filename)
        df = pd.concat([df, thisdf], axis=0)

    '''
    The "../" ensures that the file is created in the parent directory of the labeled_path
    '''
    df.to_csv(labeled_path + '../labeled_data_all_new.csv')




def create_cluster_dict(clusters):
    """
    This function will create a dictionary that maps the real labels to the clusters they belong to.
    :param clusters:
    :return:
    """
    clusterdict = {}
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            clusterdict[clusters[i][j]] = i

    return clusterdict


def get_cluster(activity, clusters):
    for idx, cluster in enumerate(clusters):
        if activity in cluster:
            return idx
    return -1  # In case the activity is not found in any cluster

def create_input_based_on_clusters(basepath, output_path, clusters, K, llambda):
    """
    This function will create the input for the classification algorithm and also convert the real labels to the clusters they belong to.
    :param basepath: basepath is the path to the root directory containing the data
    :param clusters: clusters containing the labels belonging to each cluster that was created by the clustering algorithm
    :return:
    """
    #df = pd.read_csv(f'{basepath}/NIH/labeled_data_all.csv')
    df = pd.read_csv(f'{basepath}/NIH/labeled_data_all_new.csv')  # this new file contains the movement features
    # df.dropna(inplace=True)
    cols = df.columns.tolist()
    # start = cols.index('monday')
    start = cols.index('mean_accelX') # when we are not using any timestamps
    end = cols.index('gyroZ4')
    feature_list = cols[start:end + 1]
    features = df[feature_list].to_numpy().astype(float)
    labels = df['activity_label'].to_numpy()
    # print(f'label : {labels}')
    # print(clusters)
    # clusterdict = create_cluster_dict(clusters)
    # print(f'Cluster dict: {clusterdict}')
    # tmp_label= 'Getting Hair Cut'
    # print(clusterdict.get(tmp_label))
    # print(tmp_label.strip(), -1)
    # print(clusterdict.get(tmp_label.strip(), -1))
    # print(clusterdict.get(tmp_label.strip()))
    # print(clusterdict.get(tmp_label))
    # clustered_labels = [clusterdict.get(label) for label in labels]
    # df['clustered_labels'] = clustered_labels  # adding the clustered labels to the dataframe
    df['clustered_labels'] = df['activity_label'].apply(lambda x: get_cluster(x, clusters))
    clustered_labels = df['clustered_labels'].to_numpy()
    user_ids = df['user_id'].to_numpy()
    # print(f'length of labels: {len(labels)}, length of clustered_labels: {len(clustered_labels)}')
    clustering.create_directory_if_not_exists(output_path, 'clustered-input+labels-no-headers')

    if not os.path.exists(f'{basepath}/NIH/clustered-input+labels+headers/'):
        os.makedirs(f'{basepath}/NIH/clustered-input+labels+headers/')

    df.to_csv(f'{basepath}/NIH/clustered-input+labels+headers/clustered_input_lambda_{llambda}_K_{K}[HEADINGs].csv')
    return np.array(features), np.array(labels), np.array(clustered_labels), np.array(user_ids)


def add_headers_to_merged_numpy_clustered_data(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Clean and split each row
    cleaned_data = []
    for row in data:
        if row.strip():  # Ensure the row is not just whitespace
            # Strip the brackets and split by comma
            cleaned_row = row.strip('[]\n').split(',')
            # Further split each element if needed (e.g., by tabs or additional commas)
            cleaned_data.append([x.strip() for x in cleaned_row])

    # Create a DataFrame from the cleaned data
    df = pd.DataFrame(cleaned_data)
    # Optionally, assign column names if known, or generate generic ones
    num_cols = len(cleaned_data[0])
    # print(f'Number of columns: {num_cols}')
    df.columns = [f'Feature_{i}' for i in range(num_cols - 3)] + ['Activity_Label', 'Clustered_Label', 'User_ID']

    df[f'Feature_{num_cols-3-1}'] = df['Feature_86'].str.replace(r'\]$', '', regex=True) # Remove the trailing bracket from the last column Feature_86 or Feature_88(if GPS)

    return df


def export_dataframes_details_into_textfile(df, base_csv_file, output_path):
    base_file = base_csv_file.replace('.csv', '')

    # Count how many times each label appears per user for String Labels
    user_label_counts = df.groupby(['User_ID', 'Activity_Label']).size().reset_index(name='Count')

    # Count how many times each label appears per user for Numeric Labels
    user_clustered_label_counts = df.groupby(['User_ID', 'Clustered_Label']).size().reset_index(name='Count')

    # Aggregate to find total occurrences of each label across all users
    total_clustered_label_counts = user_clustered_label_counts.groupby('Clustered_Label')['Count'].sum()
    total_label_counts = user_label_counts.groupby('Activity_Label')['Count'].sum()

    # Find the most common activity across all users
    most_common_activity = total_label_counts.idxmax()

    # Path to save the text file
    file_path = f'{output_path}/{base_file}_summary.txt'  # Change to your desired file path

    # Write the results to a text file
    with open(file_path, 'w') as file:
        file.write("User-Specific Label Counts (String Labels):\n")
        file.write(user_label_counts.to_string())
        file.write("\n\nUser-Specific Label Counts (Numeric Labels):\n")
        file.write(user_clustered_label_counts.to_string())
        file.write("\n\nTotal Occurrences of Each Numeric Label Across All Users:\n")
        file.write(total_clustered_label_counts.to_string())
        file.write("\n\nTotal Occurrences of Each String Label Across All Users:\n")
        file.write(total_label_counts.to_string())
        file.write(f"\n\nMost Common Activity Across All Users: {most_common_activity}")


def export_peruser_csv_files(df, base_csv_file, base_path):
    """
    Export per-user CSV files from a DataFrame
    :param df: DataFrame containing the data for all users, this dataframe represents the data of all users for a particular lambda and K
    :param base_csv_file: The base filename of the original CSV file, including .csv extension
    :param base_path: The base path where the base_csv_file is located
    """
    unique_users = df['User_ID'].unique()  # Get all unique user IDs
    output_directory = base_path + '/per-user-data/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    base_filename = base_csv_file.replace('.csv', '')

    for user_id in unique_users:
        user_data = df[df['User_ID'] == user_id]  # Filter the DataFrame for the current user_id
        filename = f'{base_filename}_user_{user_id}.csv'  # Define the filename incorporating the user ID
        # Full path for the file
        file_path = os.path.join(output_directory, filename)  # Using os.path.join for better path handling
        user_data.to_csv(file_path, index=False, header=False)  # Save the filtered data to a CSV file
        # Optionally print the path to confirm where the file is saved
        # print(f'Data for User {user_id} saved to: {file_path}')


def process_clustered_input_data(base_path, output_path):
    """
    Process the clustered input data and export per-user CSV files and summary text files
    """

    #- Create the output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(base_path)
    files = [f for f in files if f.endswith('.csv')]
    files.sort()
    for filename in files:
        file_path = base_path + filename

        df = add_headers_to_merged_numpy_clustered_data(file_path)
        export_peruser_csv_files(df, filename, output_path)
        export_dataframes_details_into_textfile(df, filename, output_path)


def apply_ADASYN_for_oversampling(x, y, min_samples_required=4):
    class_counts = pd.Series(y).value_counts()  # Analyze class distribution
    classes_to_oversample = class_counts[
        class_counts >= min_samples_required].index  # Identify classes that meet the requirement for oversampling

    adasyn_for_above_minimum = ADASYN(random_state=42, n_neighbors=min_samples_required, sampling_strategy='minority')
    adsyn_for_below_minimum = ADASYN(random_state=42, n_neighbors=2, sampling_strategy='minority')

    # Separate data by classes that will be oversampled and those that will not
    X_to_oversample = x[np.isin(y, classes_to_oversample)]
    y_to_oversample = y[np.isin(y, classes_to_oversample)]

    print(f'Classes to oversample: {classes_to_oversample}, classes count: {class_counts}')
    print(f'Classes not to oversample: {set(y) - set(classes_to_oversample)}')
    print(f'X_to_oversample shape: {X_to_oversample.shape}, y_to_oversample shape: {y_to_oversample.shape}')

    X_not_oversampled = x[~np.isin(y, classes_to_oversample)]
    y_not_oversampled = y[~np.isin(y, classes_to_oversample)]

    # Apply ADASYN to the selected classes
    if len(y_to_oversample) > 0:
        X_resampled, y_resampled = adasyn_for_above_minimum.fit_resample(X_to_oversample, y_to_oversample)
        X_resampled_below, y_resampled_below = adsyn_for_below_minimum.fit_resample(X_to_oversample, y_to_oversample)
    else:
        X_resampled, y_resampled = X_to_oversample, y_to_oversample
        X_resampled_below, y_resampled_below = X_not_oversampled, y_not_oversampled

    X_over_sampled = np.vstack([X_resampled, X_resampled_below])
    y_over_sampled = np.concatenate([y_resampled, y_resampled_below])
    return X_over_sampled, y_over_sampled


def save_oversampled_data_into_csv(df, min_samples_required, output_path, filename):
    X = df.drop(['Activity_Label', 'Clustered_Label', 'User_ID'], axis=1)
    y = df['Clustered_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_resampled, y_train_resampled = apply_ADASYN_for_oversampling(X_train, y_train, min_samples_required)

    # Create a DataFrame from the combined data
    df_train_resampled = pd.DataFrame(X_train_resampled, columns=[f'Feature_{i + 1}' for i in range(X.shape[1])])
    df_train_resampled['Clustered_Label'] = y_train_resampled

    # df_train_resampled = pd.concat([X_train_resampled, y_train_resampled], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_train_resampled.to_csv(f'{output_path}/{filename}_train.csv', index=False)
    df_test.to_csv(f'{output_path}/{filename}_test.csv', index=False)


def save_oversampled_data_into_csv_for_each_user(df, min_samples_required, output_path, filename):
    X = df.drop(['Activity_Label', 'Clustered_Label', 'User_ID'], axis=1)
    y = df['Clustered_Label']
    X_train_resampled, y_train_resampled = apply_ADASYN_for_oversampling(X, y, min_samples_required)

    # Create a DataFrame from the combined data
    df_X_resampled = pd.DataFrame(X_train_resampled, columns=[f'Feature_{i + 1}' for i in range(X.shape[1])])
    df_X_resampled['Clustered_Label'] = y_train_resampled

    df_X_resampled.to_csv(f'{output_path}/{filename}_resampled.csv', index=False)


def oversampling_clustered_input(base_path, min_samples_required=4):
    """
    This function will oversample the clustered input data (all users combined) for the classification task.
    it will be used for random train-test split, we will only oversample the training data and keep the test data as it is.
    For oversampling the training data, we will use ADASYN. The minimum number of samples required for a class to be oversampled is 5.
    if any class has less than 5 samples, we will oversample it by using 2 nearest neighbors. [check 'adsyn_for_below_minimum' variable on code]
    """
    output_path = base_path + '/oversampled/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(base_path)
    files = [f for f in files if f.endswith('.csv')]
    files.sort()

    for filename in files:
        file_path = base_path + filename
        df = add_headers_to_merged_numpy_clustered_data(file_path)
        # print(df.head(5))
        filename = filename.replace('.csv', '')
        print(f'Processing file: {filename}')
        save_oversampled_data_into_csv(df, min_samples_required, output_path, filename)
    print('Oversampling completed for all files.')


def oversampling_clustered_input_each_user(base_path, min_samples_required=5):
    """
    This function will oversample the clustered input data of each user individually for LOSO operation
    """
    output_path = base_path + '/oversampled/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(base_path)
    files = [f for f in files if f.endswith('.csv')]
    files.sort()
    lambda_values = [10, 20, 100, 200, 400, 500]
    k_values = [2, 3, 4, 5]  # Replace with your K values

    for file in files:
        for llambda in lambda_values:
            for K in k_values:
                if f"lambda_{llambda}_K_{K}" in file:
                    new_folder = f"lambda_{llambda}_K_{K}"
                    oversample_output_path = f'{output_path}/{new_folder}/'
                    if not os.path.exists(oversample_output_path):
                        os.makedirs(oversample_output_path)
                    file_path = base_path + file
                    df = add_headers_to_merged_numpy_clustered_data(file_path)
                    file = file.replace('.csv', '')
                    save_oversampled_data_into_csv_for_each_user(df, min_samples_required, oversample_output_path,
                                                                 filename=file)
        print(f'===>>>>Oversampling completed for all users for lambda:{llambda}, K:{K}. <<<<===\n')
