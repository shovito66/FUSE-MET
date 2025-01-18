# -*- coding: utf-8 -*-
import numpy as np
import argparse
import clustering.clustering as clustering
import pandas as pd
import warnings
import datamanager.hfailure.data_visualization as dataviz
import datamanager.hfailure.data_preparation as data_preparation
import label_merger.classifiers as classifiers
import label_merger.utils as utils
import label_merger.controller as controller
import datamanager.hfailure.feature_extract as f_extract
import datamanager.hfailure.feature_extract_new as f_extract_new
from sklearn.utils import shuffle
import os
from semi_supervised import label_propagation


def setup():
    """
    the function to initialize any basic configurations, such as output configurations,
    initializing seeds for random number generations, etc.
    """
    warnings.filterwarnings('ignore')


def run_full_pipeline(allwords, metdf, basepath, output_path, results_df, args, run_baseline, K=2, llambda=1, ):
    print(f"output_path: {output_path}")
    # cluster_list = clustering.clusterCombined(allwords, metdf, basepath, output_path, K,
    #                                           llambda=llambda, embedding = args.word_embedding)  # from the output of these method: we need to select the best 'K' value

    cluster_list = clustering.clusterCombined_with_autoencoder(allwords, metdf, basepath, output_path, K,
                                                               llambda=llambda, embedding=args.word_embedding,
                                                               norm_before=True, add_embeddings=True)
    print("Combined clustering working")
    # num_classes = len(clusters)
    # print("Number of classes: ", num_classes)
    K = 2
    # print(f'cluster list: {cluster_list}')
    base_path_llm_clustered = f'{basepath}/baseline-cluster-llm/'
    base_llm_result = []
    pred_df_llm = []
    output_path_train_test_split = f'{base_path_llm_clustered}/results_train-test-split/'
    no_header_input_path = f'{output_path}/clustered-input+labels-no-headers'
    train_test_split_path = f'{no_header_input_path}/train-test-split/'

    for clusters in cluster_list:  # cluster_list==> represent K values
        print(f"lambda: {llambda}, K: {K} ===> Result Generation started\n")
        features, labels, clustered_labels, user_ids = data_preparation.create_input_based_on_clusters(basepath,
                                                                                                       output_path,
                                                                                                       clusters, K,
                                                                                                       llambda)
        features = np.array(features).astype(float)
        # print(features.shape, labels.shape, clustered_labels.shape)
        txt = ''
        for i in range(len(features)):
            txt += f'{features[i].tolist()},{labels[i]},{clustered_labels[i]}, {user_ids[i]}\n'

        # ----uncomment this line to save the file
        with open(f'{no_header_input_path}/clustered_input_lambda_{llambda}_K_{K}.csv', 'w') as file:
            file.write(txt)

        # print("Converting the labels to one-hot encoding")
        # # print(labels[:5])
        # # labels = pd.get_dummies(labels).values.astype(int)
        # # print(labels[:5])
        clustered_labels = pd.get_dummies(clustered_labels).values.astype(int)

        features, clustered_labels, labels = shuffle(features, clustered_labels, labels, random_state=0)
        np.nan_to_num(features, copy=False)

        # pivot = int(len(features) * 0.8)
        # Xtrain = features[:pivot]
        # Ytrain = clustered_labels[:pivot]
        # Xtest = features[pivot:]
        # Ytest = clustered_labels[pivot:]

        Xtrain, Ytrain, Xtest, Ytest = utils.stratified_train_test_split(features, clustered_labels, train_ratio=0.8,
                                                                         random_state=0)
        Xtrain, Ytrain = utils.generate_synthetic_data(Xtrain, Ytrain, 42)
        Ytrain_plain = np.argmax(Ytrain, axis=1)
        Ytest_plain = np.argmax(Ytest, axis=1)
        utils.save_label_counts(Ytrain_plain, Ytest_plain, train_test_split_path, K, llambda)

        if run_baseline:
            print("Running baseline model, for K=", K)
            _, _, baseline_clustered_labels = clustering.create_llm_clusters_for_same_dataset(basepath,
                                                                                              clustered_labels, labels,
                                                                                              K)
            print(baseline_clustered_labels[:5])
            Xtrain_llm, Ytrain_llm, Xtest_llm, Ytest_llm = utils.stratified_train_test_split(features,
                                                                                             baseline_clustered_labels,
                                                                                             train_ratio=0.8,
                                                                                             random_state=0)
            Ytrain_llm_plain = np.argmax(Ytrain_llm, axis=1)
            Ytest_llm_plain = np.argmax(Ytest_llm, axis=1)
            utils.save_label_counts(Ytrain_llm_plain, Ytest_llm_plain, train_test_split_path, K, llambda=-999,
                                    output_file='label_counts_llm.txt', isbaseline=True)

            pred_df_llm = controller.generate_llm_results_from_models(Xtrain_llm, Xtest_llm, Ytrain_llm, Ytest_llm,
                                                                      output_path_train_test_split,
                                                                      base_llm_result, K, pred_df=pred_df_llm)


        # dataviz.plot_tsne(Xtrain, Ytrain_plain,llambda, K, export_path=train_test_split_path)
        pred_df = []

        acc_rf, pr_rf, rec_rf, f1_rf, Ypred_rf = classifiers.train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest)
        results_df.append(
            {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'RF', 'Accuracy': acc_rf, 'Precision': pr_rf,
             'Recall': rec_rf, 'F1': f1_rf}, )

        acc_1nn, pr_1nn, rec_1nn, f1_1nn, Ypred_1nn = classifiers.train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest)
        results_df.append(
            {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': '1NN', 'Accuracy': acc_1nn, 'Precision': pr_1nn,
             'Recall': rec_1nn, 'F1': f1_1nn}, )

        acc_svm, pr_svm, rec_svm, f1_svm, Ypred_svm = classifiers.train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest)
        results_df.append(
            {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'SVM', 'Accuracy': acc_svm, 'Precision': pr_svm,
             'Recall': rec_svm, 'F1': f1_svm}, )

        acc_nn, pr_nn, rec_nn, f1_nn, Ypred_nn = classifiers.train_and_test_neuralnet(Xtrain, Ytrain, Xtest, Ytest,
                                                                                      output_path,
                                                                                      K, llambda)
        results_df.append(
            {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'NN', 'Accuracy': acc_nn, 'Precision': pr_nn,
             'Recall': rec_nn, 'F1': f1_nn}, )

        acc_xgb, pr_xgb, rec_xgb, f1_xgb, Ypred_xgb = classifiers.train_and_test_xgboost(Xtrain, Ytrain, Xtest, Ytest)
        results_df.append(
            {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'XGB', 'Accuracy': acc_xgb, 'Precision': pr_xgb,
             'Recall': rec_xgb, 'F1': f1_xgb}, )

        acc_danets, pr_danets, rec_danets, f1_danets, Ypred_danets = classifiers.train_and_test_danets(Xtrain, Ytrain,
                                                                                                       Xtest, Ytest)
        results_df.append(
            {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'DANets', 'Accuracy': acc_danets,
             'Precision': pr_danets,
             'Recall': rec_danets, 'F1': f1_danets}, )

        # acc_tabnet, pr_tabnet, rec_tabnet, f1_tabnet, Ypred_tabnet = classifiers.train_and_test_tabnet(Xtrain, Ytrain,
        #                                                                                                Xtest, Ytest,
        #                                                                                                epochs=100)
        # results_df.append(
        #     {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'TabNet', 'Accuracy': acc_tabnet,
        #      'Precision': pr_tabnet,
        #      'Recall': rec_tabnet, 'F1': f1_tabnet}, )

        acc_cnn, pr_cnn, rec_cnn, f1_cnn, Ypred_cnn = classifiers.train_and_test_ConvNet(Xtrain, Ytrain, Xtest, Ytest,
                                                                                         output_path, K, llambda)
        results_df.append(
            {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'CNN', 'Accuracy': acc_cnn, 'Precision': pr_cnn,
             'Recall': rec_cnn, 'F1': f1_cnn}, )

        pred_df.append(
            {
                'Test_labels': Ytest_plain,
                f'RF_K_{K}': Ypred_rf,
                f'1NN_K_{K}': Ypred_1nn,
                f'SVM_K_{K}': Ypred_svm,
                f'NN_K_{K}': Ypred_nn,
                f'XGB_K_{K}': Ypred_xgb,
                f'DANets_K_{K}': Ypred_danets,
                # f'TabNet_K_{K}': Ypred_tabnet,
                f'CNN_K_{K}': Ypred_cnn
            }
        )

        pred_df = pd.DataFrame(pred_df)
        pred_df.to_csv(f'{train_test_split_path}/model_prediction_lambda_{llambda}_K_{K}.csv', index=False)

        K = K + 1
        print(f"====================>>>> Result Generation Finished==========>\n\n")
    if run_baseline:
        results = pd.DataFrame(base_llm_result)
        results.to_csv(f'{output_path_train_test_split}/model_results_baseline.csv')
        pred_df_llm = pd.DataFrame(pred_df_llm)
        pred_df_llm.to_csv(f'{train_test_split_path}/model_prediction_baseline.csv', index=False)
        run_baseline = False
    return run_baseline


def test_run(basepath, output_path, args):
    """
        This function is for performing a test run.
        :param basepath: the root-level directory that contains the data
        """
    # dd.print_data_summary(datapath) #uncomment this to print the summary of the data
    metdf = pd.read_csv(f'{basepath}met_values.csv')  # met values of the activity labels
    labelsdf = pd.read_csv(
        f'{basepath}all_labels.csv')  # csv file containing all the activity labels only, no signals/features
    allwords = labelsdf.label.to_numpy().astype(str)  # converting the labels to numpy array

    # Perform K-means clustering with MET values only
    clusters = clustering.clusterMET(allwords, metdf, output_path)  # prev output path: basepath+'output/regenerated/'
    print("MET-based clustering working")
    num_classes = len(clusters)
    print("Number of classes: ", num_classes)

    # Perform K-means clustering with GLOVE(word) embeddings only
    clusters = clustering.clusterGlove(allwords, basepath, output_path)
    print("Glove-based clustering working")
    num_classes = len(clusters)
    print("Number of classes: ", num_classes)

    #---> Perform K-means clustering with combined (MET+Word) embeddings
    results_df = []
    train_test_split_path = f'{output_path}/clustered-input+labels-no-headers/train-test-split/'
    print(f"train_test_split_path: {train_test_split_path}")
    if not os.path.exists(train_test_split_path):
        os.makedirs(train_test_split_path)
    with open(f'{train_test_split_path}/label_counts.txt', 'w') as f:
        f.write('Label Counts for train-test Split\n\n')

    '''
    this run_baseline, flag ensures that baseline will be run for each K value only once and 
    not for each lambda value; it will run only for the first lambda value and 
    then set to false for the rest of the lambda values
    '''
    run_baseline = True
    # [0,0.1, 0.5, 0.8, 1]
    for lamb in [0, 0.1, 0.2, 0.4, 0.5, 0.8, 1]:  # Iman : [1, 5, 10, 15, 20, 30, 40], [0, 0.1, 0.2, 0.4, 0.5, 0.8, 1]
        print("Lambda: ", lamb)
        run_baseline = run_full_pipeline(allwords, metdf, basepath, output_path=output_path, K=3, llambda=lamb,
                                         results_df=results_df,
                                         args=args, run_baseline=run_baseline)


    # ------- uncomment the following line
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(f'{output_path}/model_results.csv')


def warn_about_data_processing_risks():
    print(
        "WARNING: Data processing takes a long time and can be affected by presence of junk files. Override in the main function to run these functions anyway.")

def train_and_test_without_clustering(basepath, outputpath, args):
    df = pd.read_csv(f'{basepath}/NIH/labeled_data_all_new.csv')
    df.fillna(0, inplace=True)
    # Take all the columns starting from mean_accelX and ending at gyroZ4
    #X = df.iloc[:, 2:-2].values
    y = df.iloc[:, -2].values

    np.set_printoptions(suppress=True)

    labels, counts = np.unique(y, return_counts=True)
    print(labels, counts)
    labels_dict = {}
    ind = 0
    for i in range(len(labels)):
        present = labels_dict.get(labels[i], False)
        if not present:
            labels_dict[labels[i]] = ind
            ind += 1

    print(labels_dict)
    from sklearn.utils import shuffle
    trainX, trainY, testX, testY = [], [], [], []
    for label in labels:
        fildf = df[df['activity_label'] == label]
        print('Label:', label, 'Size:', fildf.shape[0])
        X = fildf.iloc[:, 2:-2].values
        y = fildf.iloc[:, -2].values
        if fildf.shape[0] == 0:
            continue
        elif fildf.shape[0] == 1:
            testX.append(X[0])
            testY.append(labels_dict[label])
        else:
            X, y = shuffle(X, y, random_state=0)
            pivot = max(1, int(0.2 * fildf.shape[0]))

            testX.extend(X[:pivot].tolist())
            test_y_list = []
            for i in range(pivot):
                test_y_list.append(labels_dict[label])

            testY.extend(test_y_list)

            trainX.extend(X[pivot:].tolist())

            train_y_list = []
            for i in range(fildf.shape[0] - pivot):
                train_y_list.append(labels_dict[label])

            trainY.extend(train_y_list)

    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)
    print('Train size:', trainX.shape, trainY.shape)
    print('Test size:', testX.shape, testY.shape)

    trainX, trainY = shuffle(trainX, trainY, random_state=42)

    # make one_hot_encoding

    trainY_onehot = one_hot(trainY, 36)
    testY_onehot = one_hot(testY, 36)

    print(trainY_onehot.shape, testY_onehot.shape)
    # RF
    classifiers.train_and_test_random_forest(trainX, trainY_onehot, testX, testY_onehot)

    # 1NN
    classifiers.train_and_test_1NN(trainX, trainY_onehot, testX, testY_onehot)

    # SVM
    classifiers.train_and_test_svm(trainX, trainY_onehot, testX, testY_onehot)

    # NN
    classifiers.train_and_test_neuralnet(trainX, trainY_onehot, testX, testY_onehot, outputpath, 36, -999)

    # DANets
    classifiers.train_and_test_danets(trainX, trainY_onehot, testX, testY_onehot)

    # XGB
    classifiers.train_and_test_xgboost(trainX, trainY_onehot, testX, testY_onehot)

    # CNN
    classifiers.train_and_test_ConvNet(trainX, trainY_onehot, testX, testY_onehot, outputpath, 36, -999)



def one_hot(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels.astype(int)


def execute(basepath, outputpath, args):
    """
    This function calls different functions based on the option.
    :param basepath: the root-level directory that contains the data
    :param data_path: the path to the data directory
    :param outputpath: the directory where the output files will be stored
    :param task: the task to be performed
    :param override: whether to override the warnings and run the functions anyway
    """

    match args.task:

        case 'baseline_no_cluster':
            train_and_test_without_clustering(basepath, outputpath, args)


        case 'test_run':
            test_run(basepath, outputpath, args)
        case 'summary2csv':
            print("Creating csv from summary")
            if args.override:
                dataviz.create_csv_from_summary(f'{basepath}/data_summary.txt')
            else:
                warn_about_data_processing_risks()
        case 'process_raw_files':
            '''
            Here is where the processing of the data files starts. First we get the raw files from the data/NIH/acitivity_anonymized folder.
            Then the process_all_raw_files function will create the rough files in the data/NIH/rough folder.
            '''
            print('Processing raw files')
            if args.override:
                f_extract_new.process_all_raw_files(f'{basepath}/NIH/activity_with_gps/')
                # f_extract.process_all_raw_files(
                #     f'{basepath}/NIH/activity_with_gps/')  #activity_anonymized ==> use it for the anonymized data (w/o gps)
            else:
                warn_about_data_processing_risks()
        case 'create_labeled_windows':
            print('creating labeled windows')
            if args.override:
                data_preparation.create_labeled_windows(f'{basepath}/NIH/rough/', min_max_scale=True)
            else:
                warn_about_data_processing_risks()
        case 'merge_labeled_files':
            print('merging labeled files')
            if args.override:
                # data_preparation.merge_labeled_files(f'{basepath}/NIH/labeled/')
                data_preparation.remove_trailing_spaces_from_labels(f'{basepath}')
            else:
                warn_about_data_processing_risks()
        case 'process_clustered_labels':
            print('processing clustered labels')
            if args.override:
                data_preparation.process_clustered_input_data(base_path=outputpath, output_path=f'{outputpath}')
                if args.oversample:
                    print('oversampling clustered input')
                    # data_preparation.oversampling_clustered_input(base_path=outputpathtpath)
                    #-----Dont uncomment the following line, as it is not needed
                    ## data_preparation.oversampling_clustered_input_each_user(base_path=outputpath) # No need to oversample user data for LOSO
            else:
                warn_about_data_processing_risks()
        case 'model_run_on_oversampled_data':
            if args.override:
                controller.model_run_on_oversampled_data([1, 5, 10, 15, 20, 30, 40], base_path=outputpath)
            else:
                warn_about_data_processing_risks()
        case 'run_loso':
            if args.override:
                print("Running LOSO")
                if args.loso_full_pipeline:
                    print("Running full pipeline for LOSO")
                    # not recommended to run this function, as it takes a long time to run
                    # it will generate the results for all the lambda and K values and for all the models
                    controller.run_full_pipeline_for_LOSO(outputpath, lambda_list=[0, 0.1, 0.2, 0.4, 0.5, 0.8, 1],
                                                          k_list=args.k_values)
                    utils.export_model_wise_LOSO_to_csv(outputpath, lambda_list=[0, 0.1, 0.2, 0.4, 0.5, 0.8, 1],
                                                        k_list=args.k_values)
                if args.loso_specific_parameter:
                    # 'RF', '1NN', 'NN',
                    # 'SVM', 'XGB', 'DANets' 'TabNet'
                    for model in ['TabNet']:  # change here if you dont want to pass theses values from the console
                        print(f"==>Running LOSO for specific parameter for model: {model}\n ")
                        controller.run_loso_for_specific_parameter(outputpath,
                                                                   # lambda_list=args.lambda_values,  # uncomment here if you  want to pass theses values from the console
                                                                   lambda_list=[0, 0.1, 0.2, 0.4, 0.5, 0.8, 1],
                                                                   # comment here if you dont want to pass theses values from the console
                                                                   k_list=args.k_values,
                                                                   # model_to_run=args.model_to_run, # uncomment here if you want to pass theses values from the console
                                                                   model_to_run=model)  # comment here if you dont want to pass theses values from the console
                        # utils.export_model_wise_LOSO_to_csv(outputpath, lambda_list=args.lambda_values,
                        #                                     k_list=args.k_values)
                    print("LOSO for specific parameter done")
                if args.export_model_wise_loso:
                    utils.export_model_wise_LOSO_to_csv(outputpath, lambda_list=[0, 0.1, 0.2, 0.4, 0.5, 0.8, 1],
                                                        k_list=args.k_values)
            else:
                warn_about_data_processing_risks()

        case 'baseline_model':
            if args.override:
                results = []
                base_path_llm_clustered = f'{basepath}/baseline-cluster-llm/'  #base_path also
                output_path_train_test_split = f'{base_path_llm_clustered}/results_train-test-split/'
                if not os.path.exists(output_path_train_test_split):
                    os.makedirs(output_path_train_test_split)

                for k in args.k_values:
                    clustering.create_clusters_from_llm(basepath=basepath, K=k)
                    controller.run_model_for_baseline_case(base_path=base_path_llm_clustered,
                                                           output_path=output_path_train_test_split,
                                                           results_df=results, K=k)
                results = pd.DataFrame(results)
                results.to_csv(f'{output_path_train_test_split}/model_results_baseline.csv')

                # if args.loso_for_baseline:
                #     base_path_llm_clustered = f'{basepath}/baseline-cluster-llm/'
                #     # 'RF', '1NN', 'NN',
                #     # 'SVM', 'XGB', 'DANets' 'TabNet'
                #     for model in ['TabNet']:
                #         controller.run_loso_for_baseline_case(base_path_llm_clustered, args.k_values,
                #                                               # model_to_run = args.model_to_run,
                #                                               model_to_run=model)
            else:
                warn_about_data_processing_risks()

        case 'plot_figures':
            if args.override:
                utils.export_all_figures_train_test_split(basepath, result_dir=outputpath)
            else:
                warn_about_data_processing_risks()


def main(args):
    task = args.task
    data_tag = args.data_tag
    backbone = args.backbone
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    dataset = args.dataset
    basepath = args.basepath
    output_folder = args.output_folder
    exp_name = args.exp_name
    override = args.override

    #pointing the data path to basepath/data_folder. For hfailure dataset, the data folder is NIH.
    # Add entries to teh data_path_dict for other datasets
    data_path_dict = {'hfailure': 'NIH'}
    data_path = os.path.join(basepath, data_path_dict.get(dataset, None))

    output_path = os.path.join(basepath, output_folder, exp_name)
    # add '/' to the end of the output path if it is not already there
    if output_path[-1] != '/':
        output_path += '/'
    print(f"Output path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    if override:
        print("WARNING- Override enabled! Data processing functions will be allowed to run")

    setup()
    if task:
        execute(basepath, output_path, args)


def parse_int_list(value):
    # Strip brackets if present
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
    # Split the string into components based on comma and convert each to int
    return list(map(int, value.split(',')))


def comma_separated_strings(value):
    # Your custom parsing logic
    return value.split(',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the LabelMerger algorithm')

    parser.add_argument('--dataset',
                        default='hfailure',
                        type=str)

    parser.add_argument('--basepath',
                        default='../data/',
                        help='The base path of where the data is stored. The data should be stored in a folder named after the dataset.',
                        type=str)

    parser.add_argument('--output_folder',
                        default='output',
                        help='The folder where the output will be stored',
                        type=str)

    parser.add_argument('--exp_name',
                        default='default_exp_name',
                        help='A unique name for the experiment. If not unique, the existing experiment will be overwritten.',
                        type=str)

    parser.add_argument('--task',
                        default='None',
                        help='Choose from label_propagation, cluster, classify, baseline_no_cluster',
                        type=str)

    parser.add_argument('--pipeline_task',
                        default='None',
                        help='Choose from ....',
                        type=str)

    parser.add_argument('--data_tag',
                        default='None',
                        help='Choose from (TBD)',
                        type=str)

    parser.add_argument('--backbone',
                        default='mlp',
                        help='Options: mlp, rforest, xgboost, nn',
                        type=str)

    parser.add_argument('--num_epochs',
                        default=100,
                        help='Number of epochs to train the model',
                        type=int)

    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)

    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    parser.add_argument('--override',
                        default=False,
                        type=bool)
    parser.add_argument('--oversample',
                        default=False,
                        type=bool)

    parser.add_argument('--loso_full_pipeline',
                        default=False,
                        help='run the full pipeline for LOSO and export the result individually for each model',
                        type=bool)
    parser.add_argument('--loso_specific_parameter',
                        default=False,
                        help='run the LOSO pipeline for some specific lambda/K/Model and export the result individually for each parameter/model',
                        type=bool)

    parser.add_argument('--export_model_wise_loso',
                        default=False,
                        help='export the result individually for each model, ONLY RUN THIS COMMAND AFTER RUNNING THE FULL PIPELINE as you need the merged results',
                        type=bool)

    parser.add_argument('--loso_for_baseline',
                        default=False,
                        type=bool)

    parser.add_argument('--lambda_values',
                        help='Provide a list of integers in the format 10,20,30',
                        type=parse_int_list,
                        default=[])

    parser.add_argument('--k_values',
                        help='Provide a list of integers in the format 10,20,30',
                        type=parse_int_list,
                        default=[])

    parser.add_argument('--model_to_run',
                        help='Options: RF, 1NN, NN, SVM',
                        type=str)

    parser.add_argument('--word_embedding',
                        default='glove',
                        help='Options: glove, bert',
                        type=str)

    #------use the following command if you want to pass a list of strings
    # parser.add_argument('--model_to_run',
    #                     default="one,two,three",
    #                     help='A comma-separated list of strings',
    #                     type=comma_separated_strings)

    main(args=parser.parse_args())

'''
how to run the code:
from the terminal provide the following command
python main.py --task test_run --output_folder output --exp_name default_exp_name --override True

* After running the above command, the output will be stored in the output folder with the name default_exp_name
* Now run the following command to preprocess the output files and also oversample the dataset
    python main.py --task process_clustered_labels --output_folder output --exp_name default_exp_name/clustered-input+labels-no-headers --override True --oversample True
    
* Now run the following command to train the model on the oversampled data (before doing that you must need to have oversampled data saved 
in this directory: output/default_exp_name/clustered-input+labels-no-headers/oversampled/)
    python main.py --task model_run_on_oversampled_data --output_folder output --exp_name default_exp_name/clustered-input+labels-no-headers/oversampled --override True
'''

'''
Task to do for LOSO:
for each user data [for different K and different lambda values]
python main.py --task run_loso --output_folder output --exp_name default_exp_name/clustered-input+labels-no-headers --override True --loso_full_pipeline True
For a specific Lambda, specific K and for specific model [these are user input/parameters taken from the console]

#-------------------    generating result for Imans work   -------------------#
1. python main.py --task test_run --output_folder output --exp_name imans-work --override True
'''

'''
BASELINE:
python main.py --task baseline_model --output_folder output --k_values 2,3,4,5 --loso_for_baseline True --override True --model_to_run RF

'''
