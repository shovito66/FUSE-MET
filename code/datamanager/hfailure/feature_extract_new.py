#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import math
import os
from timeit import default_timer as timer


def get_csv_header():
    """
    This function returns the header of the csv file that contains the features.
    :return: a string that will go to the top of the csv file and serve as the column headers
    """
    # csv_header = 'start_time_local,start_timestamp,duration,monday,tuesday,wednesday,thursday,friday,saturday,sunday'
    # csv_header = csv_header + ''.join([',hour' + str(i) for i in range(24)])

    csv_header = 'mean_accelX,mean_accelY,mean_accelZ'
    csv_header = csv_header + (',median_accelX,median_accelY,median_accelZ,std_accelX,std_accelY,'
                               'std_accelZ,mean_gyroX,mean_gyroY,mean_gyroZ')
    csv_header = csv_header + (',median_gyroX,median_gyroY,median_gyroZ,std_gyroX,std_gyroY,'
                               'std_gyroZ,ai_accel,vi_accel,sma_accel')
    csv_header = (csv_header + ',ai_gyro,vi_gyro,sma_gyro')
    csv_header = (csv_header + ',distance_from_cp,avg_altitude')
    csv_header = (csv_header + ''.join([',accelX' + str(i) for i in range(5)])
                  + ''.join([',accelY' + str(i) for i in range(5)]))
    csv_header = (csv_header + ''.join([',accelZ' + str(i) for i in range(5)])
                  + ''.join([',gyroX' + str(i) for i in range(5)])
                  + ''.join([',gyroY' + str(i) for i in range(5)]))
    csv_header = csv_header + ''.join([',gyroZ' + str(i) for i in range(5)])
    csv_header += ',activity_label'
    return csv_header


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


def row_creator(acts, accelX, accelY, accelZ, gyroX, gyroY, gyroZ, geo):
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

    # weektxt, hourtxt = weekday_and_hr(timestrings[0])
    # rowtxt = '{},{},5{}{}'.format(timestrings[0], timestamps[0], weektxt, hourtxt)
    rowtxt = mean_median_std_creator(accelX, accelY, accelZ)
    rowtxt = rowtxt + ',' + mean_median_std_creator(gyroX, gyroY, gyroZ)
    rowtxt = rowtxt + ',' + ai_vi_sma_creator(accelX, accelY, accelZ)
    rowtxt = rowtxt + ',' + ai_vi_sma_creator(gyroX, gyroY, gyroZ)

    # avg_signal, acc_avg, avg_signal_pre = extract_movement_from_accelerometer(accelX, accelY, accelZ, curr_timestamp,
    #                                                                           prev_timestamp, avg_signal_pre)
    avg_latitude, avg_longitude, avg_altitude, distance = calculate_geo_features(geo)
    # print(distance)
    # print(avg_altitude)
    rowtxt = rowtxt + ',' + str(distance) + ',' + str(avg_altitude)


    rowtxt = rowtxt + sensor_txt_creator(accelX, accelY, accelZ)
    rowtxt = rowtxt + sensor_txt_creator(gyroX, gyroY, gyroZ)
    rowtxt = rowtxt + ',' + activity_finder(acts)
    return rowtxt


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


def extract_windows_without_timestamp(df, filename, participant, roughpath):
    """
    This function extracts the features from the raw data and creates a 'rough' csv file for one participant.
    The 'rough' csv file contains the features extracted in 5-second windows. Also, it contains both labeled and unlabeled data.

    :param df: The dataframe containing the raw data
    :param filename: The name of the raw file
    :param participant: The participant id/sequence no. (processed)
    :param roughpath: The location where the rough files will be stored
    :return: None, but creates the rough csv file for the participant
    """
    # central_point = (33.64605575, -117.84934249)
    central_point = (df.iloc[0]['most_frequent_Latitude'], df.iloc[0]['most_frequent_Longitude'])
    n = df.shape[0]
    txt = get_csv_header() + '\n'
    start = timer()
    last_print = -20000
    rowsToWrite = 0
    i = 0

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

        rowtxt = row_creator(acts, accelX, accelY, accelZ, gyroX, gyroY, gyroZ, geo)
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


    roughpath = datapath + '../rough/'
    output_path = f'{datapath}' + '../activity_anonymized_GPS+no_NULL/'
    # save_rough_files_that_contains_activity(datapath, output_path=output_path)

    files = os.listdir(output_path)
    files.sort()
    # print(files)

    for i in range(len(files)):
        '''
        During the processing of the raw files, the following files were found to be problematic: (the first file)
        The reason was that the timestamps were not written properly.
        Please be careful when processing the raw files.
        '''

        file = files[i]
        # file = files[1]
        if '.csv' in file:
            df = pd.read_csv(output_path + file)
            print(file)

            extract_windows_without_timestamp(df, file, i, roughpath)

        #delete it later
        # if i > 0:
        #     break
