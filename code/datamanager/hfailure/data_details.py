"""
To print the details of the data, call the following function.
print_data_summary(datapath)
"""

import pandas as pd
import numpy as np
import os

def get_userlist(datapath):
    """
    This function will look for data files in the datapath return the list of users and the list of files associated with each user.
    The userid is included into the filename, so we can extract the userid from the filename.
    :param datapath:
    :return:
    """
    allusers = []
    allfiles = []
    files = os.listdir(datapath)
    files.sort()
    for file in files:
        token = 'anonymized_user_data_epsl'
        if token not in file:
            continue
        user = file[len(token):file.find('.')]
        allusers.append(user)
        allfiles.append(file)
    return allusers, allfiles


def find_frequency(timestamps):
    """
    This was created to find the frequency of the data. However, it was not needed because the data is not continuous,
    and sampled at constant 1 Hz frequency when collected.
    :param timestamps:
    :return:
    """
    pass


def get_info(tempdf):
    ##tempdf = pd.read_csv(filepath)
    n = tempdf.shape[0]
    labels = tempdf['Activity Label Provided By User'].unique().astype(str)
    rawlabels = tempdf['Activity Label Provided By User'].to_numpy().astype(str)
    
    #g.labels
    
    labels = np.delete(labels, np.where(labels == 'nan')[0])
    rawlabels = np.delete(rawlabels, np.where(rawlabels=='nan')[0])
    labels.sort()
    info = Info(n, len(rawlabels), labels)
    return info

def get_info_string(info):
    return "Info(n = {}, n_labeled_records = {}, 1 label for {} records, n_unique_labels = {})".format(info.n, info.n_labeled_records, info.inverse_label_ratio, info.n_unique_labels)



def print_all_info(allusers, allinfo, printLabels = True):
    
    if printLabels:
        for i in range(len(allusers)):
            print('User:', allusers[i],';', allinfo[i])
    else:
        for i in range(len(allusers)):
            print('User:', allusers[i],';', get_info_string(allinfo[i]))


def print_all_labels(allusers, allinfo):
    
    all_labels = []
    for i in range(len(allusers)):
        all_labels = all_labels + allinfo[i].labels.tolist()
    
    all_labels = np.unique(np.array(all_labels))
    all_labels.sort()
    
    print("All labels:")
    print(all_labels)
    print()
    for i in range(len(allusers)):
        print('User',allusers[i], '; Unique count:', len(allinfo[i].labels), '; Unique labels:',allinfo[i].labels)


class FeatureGroup:
    def __init__(self):
        pass

"""
This class contains one user's data summary. How many data points are available, how many of them are labeled,
and what are the unique labels available for that user.
"""
class Info:
    def __init__(self, n=None, n_labeled_records = None, labels = None):
        self.n = n
        self.n_labeled_records = n_labeled_records
        self.labels = labels
        if labels is None:
            self.n_unique_labels = None
        else:
            self.n_unique_labels = len(labels)
        if n is None or n_labeled_records is None:
            self.inverse_label_ratio = None
        elif n_labeled_records == 0:
            self.inverse_label_ratio = 'INF'
        else:
            self.inverse_label_ratio = n/n_labeled_records
            
    def __str__(self):
        return "Info(n = {}, n_labeled_records = {}, 1 label for {} records, n_unique_labels = {}, labels = {})".format(self.n, self.n_labeled_records, self.inverse_label_ratio, self.n_unique_labels, self.labels)
        

def aggregate_values(allusers, allinfo):
    """
    This function will aggregate the values from all the users into a single object.
    :param allusers:
    :param allinfo:
    :return:
    """
    g = FeatureGroup()
    sample_counts = []
    #sample_freqs = []
    all_label_counts = []
    unique_label_counts = []
    all_unique_labels = []
    
    for i in range(len(allusers)):
        sample_counts.append(allinfo[i].n)
        #sample_freqs.append(allinfo[i].f)
        all_label_counts.append(allinfo[i].n_labeled_records)
        unique_label_counts.append(allinfo[i].n_unique_labels)
        all_unique_labels += allinfo[i].labels.tolist()
    
    
    all_unique_labels = np.unique(np.array(all_unique_labels))
    
    sample_counts = np.array(sample_counts)
    all_label_counts = np.array(all_label_counts)
    all_unique_labels = np.array(all_unique_labels)
    
    
    #all_unique_labels.sort()
    #=================================================================================================
    # The stats related the total users, total records, total labeled records and average per user
    #==================================================================================================
    g.total_users = len(allusers)
    g.total_records = np.sum(sample_counts)
    g.total_labeled_records = np.sum(all_label_counts)
    g.avg_frequency = "1 Hz (for every consecutive 5 seconds available, the next 5 seconds are missing)"
    g.mean_records = np.mean(sample_counts)
    g.std_records = np.std(sample_counts)
    
    g.mean_labeled_records = np.mean(all_label_counts)
    g.std_labeled_records = np.std(all_label_counts)
    
    g.sizeperlabel = g.total_records/g.total_labeled_records
    
    # =========== Total and average unique labels =====================
    g.total_unique_labels = len(all_unique_labels)
    g.mean_unique_labels = np.mean(unique_label_counts)
    g.std_unique_labels = np.std(unique_label_counts)
    
    #=================================================================================================
    # Users with minimum and maximum data points/labeled data points/unique labels 
    #==================================================================================================
    g.minuser = allusers[np.argmin(sample_counts)]
    g.minval = np.min(sample_counts)
    g.maxuser = allusers[np.argmax(sample_counts)]
    g.maxval = np.max(sample_counts)
    
    g.minlabeluser = allusers[np.argmin(all_label_counts)]
    g.minlabel =  np.min(all_label_counts)
    g.maxlabeluser = allusers[np.argmax(all_label_counts)]
    g.maxlabel = np.max(all_label_counts)
    
    g.minuniqueuser = allusers[np.argmin(unique_label_counts)]
    g.minunique =  np.min(unique_label_counts)
    g.maxuniqueuser = allusers[np.argmax(unique_label_counts)]
    g.maxunique = np.max(unique_label_counts)
    
    all_inverse_ratios = sample_counts/all_label_counts
    g.min_inverse_ratio_user = allusers[np.argmin(all_inverse_ratios)]
    g.min_inverse_ratio =  np.min(all_inverse_ratios)
    g.max_inverse_ratio_user = allusers[np.argmax(all_inverse_ratios)]
    g.max_inverse_ratio = np.max(all_inverse_ratios)
    
    
    return g



def print_data_summary(datapath, test=False):
    """
    We want this function to print the number of users available, number of samples, sampling rate, etc.
    :param datapath: The path to the data folder
    :return: None
    """
    
    allusers, allfiles = get_userlist(datapath)
    
    if test:
        allusers = allusers[:2]
        allfiles = allfiles[:2]
    
    allinfo= []
    for file in allfiles:
        tempdf = pd.read_csv(datapath+file)
        allinfo.append(get_info(tempdf))
        
    
    g = aggregate_values(allusers, allinfo)
    #return
    print('Number of users: {}'.format(g.total_users))
    print('Average Sampling rate:', g.avg_frequency)
    
    print('Total number of records:', g.total_records)
    print('Mean and std of samples per user:{} +/- {}'.format(g.mean_records, g.std_records))
    print('Total number of labeled records: {}'.format(g.total_labeled_records))
    print('Mean and std of labeled records per user:{} +/- {}'.format(g.mean_labeled_records, g.std_labeled_records))

    print('Total number of unique labels: {}'.format(g.total_unique_labels))
    print('Mean and std of unique labels per user:{} +/- {}'.format(g.mean_unique_labels, g.std_unique_labels))

    print('Average duration of sensor data per label:{}'.format(g.sizeperlabel))
    
    print('User with minimum number of records: {} ({})'.format(g.minuser, g.minval))
    print('User with maximum number of records: {} ({})'.format(g.maxuser, g.maxval))
    
    print('User with minimum number of labeled records: {} ({})'.format(g.minlabeluser, g.minlabel))
    print('User with maximum number of labeled records: {} ({})'.format(g.maxlabeluser, g.maxlabel))
    
    print('User with minimum number of unique labels: {} ({})'.format(g.minuniqueuser, g.minunique))
    print('User with maximum number of unique labels: {} ({})'.format(g.maxuniqueuser, g.maxunique))
    
    print('User with minimum label ratio: {} (1 label for {} records)'.format(g.max_inverse_ratio_user, g.max_inverse_ratio))
    print('User with maximum label ratio: {} (1 label for {} records)'.format(g.min_inverse_ratio_user, g.min_inverse_ratio))
    #print_all_labels(allusers, allinfo)
    print()
    print_all_info(allusers, allinfo, False)
    
    print_all_labels(allusers, allinfo)






