#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import math
import os
from timeit import default_timer as timer


def string_to_stamp(string_):
    """
    This function converts a date time string to a timestamp. The date time string is in the format of 'YYYY-MM-DD HH:MM:SS', or 'YYYY-MM-DD HH:MM:SS.ffffff' if there are microseconds.
    Examples:
    '2019-01-01 00:00:00' -> 1546300800.0
    '2019-01-01 00:00:00.123456' -> 1546300800.123456

    :param string_: The date and time in a string
    :return: a timestamp extracted from the date time string
    """
    if '.' in string_:
        return datetime.datetime.strptime(string_, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    return datetime.datetime.strptime(string_, '%Y-%m-%d %H:%M:%S').timestamp()


def check_continuity(timestrings):
    """
    This function checks if the timestamps are consecutive by following the sampling rate.
    If they are, it returns a list of timestamps. If not, it returns None.
    Explanation:
        here the sampling rate is 0.5 Hz. So, the difference between two consecutive timestamps should be not more than 2 (=1/0.5) second.
        and, not less than 0.5 second as we can not sample faster than sampling rate = 0.5 Hz.

    :param timestrings: A list of date time strings that are converted to timestamps first before checking the continuity.
    :return: a list of timestamps if the timestamps are consecutive, or None if not.
    """
    allstamps = [string_to_stamp(timestrings[i]) for i in range(len(timestrings))]
    for i in range(len(allstamps) - 1):
        diff = allstamps[i + 1] - allstamps[i]
        #print(diff)
        if diff > 1.9 or diff < 0.5:
            return None
    return allstamps


def get_csv_header():
    """
    This function returns the header of the csv file that contains the features.
    :return: a string that will go to the top of the csv file and serve as the column headers
    """
    csv_header = 'start_time_local,start_timestamp,duration,monday,tuesday,wednesday,thursday,friday,saturday,sunday'
    csv_header = csv_header + ''.join([',hour' + str(i) for i in range(24)])

    csv_header = csv_header + ',mean_accelX,mean_accelY,mean_accelZ'
    csv_header = csv_header + ',median_accelX,median_accelY,median_accelZ,std_accelX,std_accelY,std_accelZ,mean_gyroX,mean_gyroY,mean_gyroZ'
    csv_header = csv_header + ',median_gyroX,median_gyroY,median_gyroZ,std_gyroX,std_gyroY,std_gyroZ,ai_accel,vi_accel,sma_accel'
    csv_header = csv_header + ',ai_gyro,vi_gyro,sma_gyro,avg_movement,avg_accel,distance_from_cp,avg_altitude' + ''.join(
        [',accelX' + str(i) for i in range(5)]) + ''.join(
        [',accelY' + str(i) for i in range(5)])
    csv_header = csv_header + ''.join([',accelZ' + str(i) for i in range(5)]) + ''.join(
        [',gyroX' + str(i) for i in range(5)]) + ''.join([',gyroY' + str(i) for i in range(5)])
    csv_header = csv_header + ''.join([',gyroZ' + str(i) for i in range(5)])
    csv_header += ',activity_label'
    return csv_header


def weekday_and_hr(timestring):
    """
    Day of the week and hour of the day are extracted from the date time string and then converted to one-hot encoding
    :param timestring: date time string. Example '2019-01-01 00:00:00'
    :return: day of the week and hour (one-hot encoded) in  string format
    """
    date, time = timestring.split()
    yy, mm, dd = date.split('-')
    hr = time.split(':')[0]

    weekday = datetime.date(int(yy), int(mm), int(dd)).weekday()
    weekarr = [0 for i in range(7)]
    weekarr[weekday] = 1
    hourarr = [0 for i in range(24)]
    hourarr[int(hr)] = 1

    weektxt = ''.join([',' + str(weekarr[i]) for i in range(7)])
    hourtxt = ''.join([',' + str(hourarr[i]) for i in range(24)])

    return weektxt, hourtxt


def mean_median_std_creator(arrX, arrY, arrZ):
    """
    This function creates the mean, median, and standard deviation features from the 5-second window data of a tri-axial sensor.
    :param arrX:
    :param arrY:
    :param arrZ:
    :return: a concatenated string of the three features, mean, median, and standard deviation from the 5-second window
    """
    txt = str(np.mean(arrX)) + ',' + str(np.mean(arrY)) + ',' + str(np.mean(arrZ)) + ','
    txt = txt + str(np.median(arrX)) + ',' + str(np.median(arrY)) + ',' + str(np.median(arrZ)) + ','
    txt = txt + str(np.std(arrX)) + ',' + str(np.std(arrY)) + ',' + str(np.std(arrZ))
    return txt


def extract_movement_from_accelerometer(arrX, arrY, arrZ, cur_time_stamp, pre_time_stamp,
                                        avg_signal_pre):
    """
    Extracts movement from accelerometer data provided in separate arrays for X, Y, Z axes, calculates the average signal value,
    and applies a sanity check on time differences to reset the average signal if needed.

    Args:
    - arrX (np.ndarray): Array of X-axis accelerometer data.
    - arrY (np.ndarray): Array of Y-axis accelerometer data.
    - arrZ (np.ndarray): Array of Z-axis accelerometer data.
    - time_stamps (list of str): List of time stamps corresponding to each data point.
    - avg_signal_pre (float): Previous value of the averaged signal.

    Returns:
    - float: The new average signal value.
    - np.ndarray: The array of average accelerometer values.
    """

    acc = np.stack((arrX, arrY, arrZ), axis=1)  # Shape should be (window_size, 3)

    # Initialize the movement list with the maximum of the absolute initial values
    movement = [max(abs(arrX[0]), abs(arrY[0]), abs(arrZ[0]))]

    # Convert timestamps to pandas datetime objects for easier manipulation
    pre_time_stamp = pd.to_datetime(pre_time_stamp)
    cur_time_stamp = pd.to_datetime(cur_time_stamp)

    for i in range(1, len(arrX)):
        '''
        Check time difference, if different day or time different between current and prev window > 30 minutes,
        reset avg_signal_pre = 0
        '''
        if ((cur_time_stamp - pre_time_stamp) > pd.Timedelta(minutes=30) or
                cur_time_stamp.date() is not pre_time_stamp.date()):
            avg_signal_pre = 0  # Reset avg_signal_pre if the condition is met

        x_now, y_now, z_now = arrX[i], arrY[i], arrZ[i]
        x_pre, y_pre, z_pre = arrX[i - 1], arrY[i - 1], arrZ[i - 1]
        movement.append(max(abs(x_now - x_pre), abs(y_now - y_pre), abs(z_now - z_pre)))

    # Calculate the new average signal value
    avg_signal = avg_signal_pre * 0.9 + 0.1 * (np.mean(movement))

    avg_signal_pre = avg_signal

    # Calculate the average values for accelerometer data in the current window
    acc_avg = np.mean(acc)

    return avg_signal, acc_avg, avg_signal_pre


def ai_vi_sma_creator(arrX, arrY, arrZ):
    """
    This function creates the ai (Average Intensity), vi (Variance Intensity), and sma (Normalized signal magnitude area) features from the data of a tri-axial sensor.
    :param arrX: list of X axis data of the sensor
    :param arrY: list of Y axis data of the sensor
    :param arrZ: list of Z axis data of the sensor
    :return: a concatenated string of the three features, average intensity, variance intensity, and normalized signal magnitude area
    """
    mi = [0 for i in range(len(arrX))]
    for i in range(len(arrX)):
        mi[i] = math.sqrt(arrX[i] * arrX[i] + arrY[i] * arrY[i] + arrZ[i] * arrZ[i])
    ai = np.mean(np.array(mi))

    sum_for_vi = 0
    for i in range(len(mi)):
        sum_for_vi += (mi[i] - ai) * (mi[i] - ai)

    vi = sum_for_vi / len(mi)

    sum_for_sma = 0
    for i in range(len(arrX)):
        sum_for_sma += abs(arrX[i]) + abs(arrY[i]) + abs(arrZ[i])
    sma = sum_for_sma / len(arrX)

    return '{},{},{}'.format(ai, vi, sma)


def sensor_txt_creator(arrX, arrY, arrZ):
    """
    Concatenate the data from the three axes of a sensor into a string. A sensor can be an accelerometer or a gyroscope (for this dataset)
    or any other tri-axial sensor.

    :param arrX: X-axis data for the sensor
    :param arrY: Y-axis data for the sensor
    :param arrZ: Z-axis data for the sensor
    :return: a string concatenating all three axes data
    """
    txt = ''.join([',' + str(arrX[i]) for i in range(len(arrX))])
    txt = txt + ''.join([',' + str(arrY[i]) for i in range(len(arrY))])
    txt = txt + ''.join([',' + str(arrZ[i]) for i in range(len(arrZ))])
    return txt


def activity_finder(acts):
    """
    Better implementation of this function may be possible. But it is a data processing function,
    so it is not a priority as the data has been processed already.
    :param acts: list of activities (may contain nan)
    :return:
    """
    n = len(acts)
    for i in range(n - 1, -1, -1):  # iterating from n-1 to 0 (backward, decreasing 1 step each time)
        try:
            if np.isnan(acts[i]):
                continue
            else:
                return acts[i]
        except TypeError:
            return acts[i]

    return '_unknown_'


def row_creator(timestrings, timestamps, acts, accelX, accelY, accelZ,
                gyroX, gyroY, gyroZ, geo, curr_timestamp, prev_timestamp, avg_signal_pre):
    """
    This function creates a row of the csv file that contains the features from a 5-second window.
    :param timestrings: date and time in string format
    :param timestamps: timestamps in numeric format
    :param acts: activity labels (may contain nan)
    :param accelX:
    :param accelY:
    :param accelZ:
    :param gyroX:
    :param gyroY:
    :param gyroZ:
    :param geo: dictionary containing latitude, longitude, and altitude
    :return:
    """

    weektxt, hourtxt = weekday_and_hr(timestrings[0])
    rowtxt = '{},{},5{}{}'.format(timestrings[0], timestamps[0], weektxt, hourtxt)
    rowtxt = rowtxt + ',' + mean_median_std_creator(accelX, accelY, accelZ)
    rowtxt = rowtxt + ',' + mean_median_std_creator(gyroX, gyroY, gyroZ)
    rowtxt = rowtxt + ',' + ai_vi_sma_creator(accelX, accelY, accelZ)
    rowtxt = rowtxt + ',' + ai_vi_sma_creator(gyroX, gyroY, gyroZ)

    avg_signal, acc_avg, avg_signal_pre = extract_movement_from_accelerometer(accelX, accelY, accelZ, curr_timestamp,
                                                                              prev_timestamp, avg_signal_pre)
    avg_latitude, avg_longitude, avg_altitude, distance = calculate_geo_features(geo)
    rowtxt = rowtxt + ',' + str(avg_signal) + ',' + str(acc_avg) + ',' + str(distance) + ',' + str(avg_altitude)

    rowtxt = rowtxt + sensor_txt_creator(accelX, accelY, accelZ)
    rowtxt = rowtxt + sensor_txt_creator(gyroX, gyroY, gyroZ)
    rowtxt = rowtxt + ',' + activity_finder(acts)
    return rowtxt, avg_signal_pre


# Haversine formula to calculate distance between two lat/lon points
def haversine(lon1, lat1, fixed_ref_point):
    lon2, lat2 = fixed_ref_point[0], fixed_ref_point[1]
    R = 6371  # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # in kilometers
    return distance / 1000


def calculate_geo_features(geo):
    """
    This function calculates the average latitude, longitude, altitude, and distance from a central point.
    :param geo: dictionary containing latitude, longitude, and altitude
    :param central_point: a tuple containing the latitude and longitude of the central point
    :return: a tuple containing the average latitude, average longitude, average altitude, and distance from the central point
    """
    central_point = geo['central_point']
    avg_latitude = np.mean(geo['latitude'])
    avg_longitude = np.mean(geo['longitude'])
    avg_altitude = np.mean(geo['altitude'])
    distance = haversine(avg_longitude, avg_latitude, central_point)

    return avg_latitude, avg_longitude, avg_altitude, distance


def extract_windows(df, filename, participant, roughpath):
    """
    This function extracts the features from the raw data and creates a 'rough' csv file for one participant.
    The 'rough' csv file contains the features extracted in 5-second windows. Also, it contains both labeled and unlabeled data.

    :param df: The dataframe containing the raw data
    :param filename: The name of the raw file
    :param participant: The participant id/sequence no. (processed)
    :param roughpath: The location where the rough files will be stored
    :return: None, but creates the rough csv file for the participant
    """
    central_point = (33.64605575, -117.84934249)  # Central point of (802 Medical Science Ct, Irvine, CA 92617, USA)

    i = 0
    n = df.shape[0]

    txt = get_csv_header() + '\n'
    start = timer()
    last_print = -20000
    rowsToWrite = 0
    avg_signal_pre = 0
    prev_timestamp = None
    current_timestamp = None
    #clear the file first so that appending does not create duplicate rows
    with open(roughpath + 'temp_{}_participant_{}.csv'.format(filename, participant), 'a') as filewriter:
        filewriter.write("")

    while i + 4 < n:
        if i - last_print >= 5000:
            ### print the progress every 5000 rows
            nowtime = timer()
            elapsed = nowtime - start
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            print('Now processing row {} out of {}. Time elapsed: {} min {} sec'.format(i, n, minutes, seconds))
            if i > 0:
                eta = elapsed / i * (n - i)
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                print('ETA: {} min {} sec'.format(eta_min, eta_sec))
            last_print = i

        timestrings = [df.iloc[j]['Sensor Data Time (Local)'] for j in range(i, i + 5)]

        allstamps = check_continuity(timestrings)

        if allstamps is None:
            #print('found inconsistency')
            i += 1

        else:

            acts = [df.iloc[j]['Activity Label Provided By User'] for j in range(i, i + 5)]
            accelX = [df.iloc[j]['User Acceleration X (m/s^2)'] for j in range(i, i + 5)]
            accelY = [df.iloc[j]['User Acceleration Y (m/s^2)'] for j in range(i, i + 5)]
            accelZ = [df.iloc[j]['User Acceleration Z (m/s^2)'] for j in range(i, i + 5)]

            gyroX = [df.iloc[j]['Rotation Rate X (rad/s)'] for j in range(i, i + 5)]
            gyroY = [df.iloc[j]['Rotation Rate Y (rad/s)'] for j in range(i, i + 5)]
            gyroZ = [df.iloc[j]['Rotation Rate Z (rad/s)'] for j in range(i, i + 5)]

            latitude = [df.iloc[j]['Latitude'] for j in range(i, i + 5)]
            longitude = [df.iloc[j]['Longitude'] for j in range(i, i + 5)]
            altitude = [df.iloc[j]['Altitude (m)'] for j in range(i, i + 5)]

            geo = {
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude,
                'central_point': central_point
            }

            if i < 5 or prev_timestamp is None:
                # first window of a file
                # prev_timestamp = timestrings[-1]
                current_timestamp = timestrings[0]
                prev_timestamp = current_timestamp
                # print(f'1st window ======\nprev_timestamp: {prev_timestamp}, current_timestamp: {current_timestamp}')
            else:
                # all other windows
                prev_timestamp = current_timestamp
                current_timestamp = timestrings[0]
                # print(f'prev_timestamp: {prev_timestamp}, current_timestamp: {current_timestamp}')

            rowtxt, avg_signal_pre = row_creator(timestrings, allstamps, acts, accelX, accelY, accelZ,
                                                 gyroX, gyroY, gyroZ, geo, current_timestamp, prev_timestamp,
                                                 avg_signal_pre)
            txt = txt + rowtxt + '\n'
            rowsToWrite += 1
            i += 5

            if rowsToWrite >= 8192:
                with open(roughpath + 'temp_{}_participant_{}.csv'.format(filename, participant), 'a') as filewriter:
                    filewriter.write(txt)
                txt = ''
                rowsToWrite = 0

    if rowsToWrite > 0:
        with open(roughpath + 'temp_{}_participant_{}.csv'.format(filename, participant), 'a') as filewriter:
            filewriter.write(txt)


def adjust_sensor_data(df):
    """
    This function adjusts the sensor data so that each Sensor Data Window ID has 5 records.
    If a Sensor Data Window ID has less than 5 records, the records are repeated to make the total count 5.
    If a Sensor Data Window ID has only 1 record, the record is repeated 4 more times to make the total count 5.
    :param df: The dataframe containing the raw data
    :return: a new dataframe where each Sensor Data Window ID has 5 records
    """
    freq = df['Sensor Data Window ID'].value_counts()

    # Create a list to store new records
    new_rows = []

    # Iterate through the unique Sensor Data Window IDs
    for sensor_id, count in freq.items():
        # print(sensor_id, count)
        # Filter out rows with this sensor ID
        records = df[df['Sensor Data Window ID'] == sensor_id]
        df['AugmentedSensorData'] = 0

        if 5 > count > 1:
            # Take the average of the records and repeat it to make the total count 5
            avg_record = records.mean(numeric_only=True)
            avg_record['Sensor Data Window ID'] = int(sensor_id)
            avg_record['Activity Label Provided By User'] = records.iloc[0]['Activity Label Provided By User']
            avg_record['AugmentedSensorData'] = 1
            #if we dont want to use the augmented data,
            # we need to find out the Sensor Ids that are augmented (=1) and remove them from the dataframe

            rows_to_add = 5 - count  # How many rows need to be added
            # Repeat the averaged row
            new_rows.extend([avg_record] * rows_to_add)

        elif count == 1:
            # Repeat the single record 4 more times to make the total count 5
            new_rows.extend([records.iloc[0]] * 4)

    # Add the new rows to the dataframe
    df_extended = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    # sort the dataframe by Sensor Data Window ID
    df_extended = df_extended.sort_values(by='Sensor Data Window ID')
    df_extended['Sensor Data Window ID'] = df_extended['Sensor Data Window ID'].astype(int)

    # if df_extended has at least 1 record then excecute the following code
    if len(df_extended) > 0:
        # --------- Lowering the precision of GPS to 4 decimal places and finding the most frequent pair of GPS coordinates
        df_extended['Latitude_rounded'] = df_extended['Latitude'].round(4)
        df_extended['Longitude_rounded'] = df_extended['Longitude'].round(4)

        # Group by the rounded Latitude and Longitude and find the most frequent pair
        most_frequent_pair = df_extended.groupby(['Latitude_rounded', 'Longitude_rounded']).size().idxmax()
        frequency = df_extended.groupby(['Latitude_rounded', 'Longitude_rounded']).size().max()

        df_extended['most_frequent_Latitude'] = most_frequent_pair[0]
        df_extended['most_frequent_Longitude'] = most_frequent_pair[1]

    return df_extended


def save_rough_files_that_contains_activity(import_path, output_path):
    """
    This function saves the rough files that contain activity labels.
    :param df: The dataframe containing the raw data
    :param output_path: The location where the rough files will be stored
    :return: None, but creates the rough csv files
    """
    files = os.listdir(import_path)
    files.sort()

    for i in range(len(files)):
        file = files[i]
        if '.csv' in file:
            print(file)
            df = pd.read_csv(import_path + file)
            df = df[df['Activity Label Provided By User'].notnull()]
            df = adjust_sensor_data(df)
            if len(df) > 0:
                # user 130/27 doesnot has any activity label so it will be skipped
                df.to_csv(output_path + f'{i + 1}_activity_' + file, index=False)


def process_all_raw_files(datapath):
    """
    This function processes all the raw files in the datapath and creates a csv file for each participant.
    The rough files have features extracted in 5-second windows. Also, they have both labeled and unlabeled data.
    The naming of the rough files is done in a way so that the mapping of the original user id (from the raw file) and the
    processed user id (from the rough file) is preserved.
    :param datapath:
    :return:
    """

    print(
        "WARNING-- The function is highly dependent on the file names and presence of junk files. Please check the code and the files before running.")
    print(
        "The function is kept in the code to show the process of data preparation. It is not recommended to run this function without checking the code and the files.")

    files = os.listdir(datapath)
    files.sort()
    roughpath = datapath + '../rough/'
    # print(files)
    #return

    output_path = f'{datapath}' + '../activity_anonymized_GPS+no_NULL/'
    # save_rough_files_that_contains_activity(datapath, output_path=output_path)

    for i in range(len(files)):
        '''
        During the processing of the raw files, the following files were found to be problematic: (the first file)
        The reason was that the timestamps were not written properly.
        Please be careful when processing the raw files.
        '''


        # if i in [0]:  #, 37, 39]: #problematic files
        #     continue
        file = files[i]
        # file = files[1]
        if '.csv' in file:
            df = pd.read_csv(datapath + file)
            print(file)

            extract_windows(df, file, i, roughpath)

        #delete it later
        # if i > 0:
        #     break
