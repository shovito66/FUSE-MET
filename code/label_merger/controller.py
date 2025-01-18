import os
import pandas as pd
import numpy as np
from .classifiers import train_and_test_random_forest, train_and_test_1NN, train_and_test_svm, train_and_test_neuralnet, train_and_test_ConvNet
from .classifiers import train_and_test_xgboost, train_and_test_tabnet, train_and_test_danets
from datamanager.hfailure.data_preparation import add_headers_to_merged_numpy_clustered_data
from sklearn.model_selection import train_test_split


def model_run_on_oversampled_data(lambda_list, base_path):
    output_path = f'{base_path}/results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_df = []
    for llambda in lambda_list:
        for K in range(2, 6):
            print(f"====================>>>> Running for Lambda: {llambda}, K: {K}==========>\n\n")
            # load train  data frame from the basepath
            train_dataframe_name = f'clustered_input_lambda_{llambda}_K_{K}_train.csv'
            test_dataframe_name = f'clustered_input_lambda_{llambda}_K_{K}_test.csv'
            Xtrain = pd.read_csv(f'{base_path}/{train_dataframe_name}')
            Xtest = pd.read_csv(f'{base_path}/{test_dataframe_name}')
            Ytrain = Xtrain['Clustered_Label'].to_numpy()
            Ytest = Xtest['Clustered_Label'].to_numpy()

            Ytrain = pd.get_dummies(Ytrain).values.astype(int)
            Ytest = pd.get_dummies(Ytest).values.astype(int)

            Xtrain = Xtrain.drop(columns=['Clustered_Label'])
            Xtest = Xtest.drop(columns=['Clustered_Label'])
            Xtrain = np.array(Xtrain.to_numpy().astype(float))
            Xtest = np.array(Xtest.to_numpy().astype(float))
            # print(Xtrain[:10])
            # print(f"Xtrain: {Xtrain.shape}, Ytrain: {Ytrain.shape}, Xtest: {Xtest.shape}, Ytest: {Ytest.shape}\n")

            acc_rf, pr_rf, rec_rf, f1_rf = train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest)
            results_df.append(
                {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'RF', 'Accuracy': acc_rf, 'Precision': pr_rf,
                 'Recall': rec_rf, 'F1': f1_rf}, )

            acc_1nn, pr_1nn, rec_1nn, f1_1nn = train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest)
            results_df.append(
                {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': '1NN', 'Accuracy': acc_1nn, 'Precision': pr_1nn,
                 'Recall': rec_1nn, 'F1': f1_1nn}, )

            acc_svm, pr_svm, rec_svm, f1_svm = train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest)
            results_df.append(
                {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'SVM', 'Accuracy': acc_svm, 'Precision': pr_svm,
                 'Recall': rec_svm, 'F1': f1_svm}, )

            acc_nn, pr_nn, rec_nn, f1_nn = train_and_test_neuralnet(Xtrain, Ytrain, Xtest, Ytest, output_path, K,
                                                                    llambda)
            results_df.append(
                {'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'NN', 'Accuracy': acc_nn, 'Precision': pr_nn,
                 'Recall': rec_nn, 'F1': f1_nn}, )

    print(f"====================>>>> Result Generation Finished for Oversampled Data==========>\n\n")
    # ------- uncomment the following line for exporting the results to a csv file----------------
    results_df = pd.DataFrame(results_df)
    results_df.to_csv(f'{output_path}/model_results_oversampled.csv', index=False)


def get_result_for_a_model(model_to_run, Xtrain, Ytrain, Xtest, Ytest, user, llambda, K, output_path):
    results_row = pd.DataFrame()
    if model_to_run == 'RF':
        acc_rf, pr_rf, rec_rf, f1_rf = train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest)
        results_row = pd.DataFrame(
            [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'RF', 'Accuracy': acc_rf,
              'Precision': pr_rf, 'Recall': rec_rf, 'F1': f1_rf}])

    elif model_to_run == '1NN':
        acc_1nn, pr_1nn, rec_1nn, f1_1nn = train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest)
        results_row = pd.DataFrame(
            [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': '1NN',
              'Accuracy': acc_1nn,
              'Precision': pr_1nn, 'Recall': rec_1nn, 'F1': f1_1nn}])

    elif model_to_run == 'SVM':
        acc_svm, pr_svm, rec_svm, f1_svm = train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest)
        results_row = pd.DataFrame(
            [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'SVM',
              'Accuracy': acc_svm,
              'Precision': pr_svm, 'Recall': rec_svm, 'F1': f1_svm}])

    elif model_to_run == 'NN':
        acc_png_filename = f'accuracy_K_{K}_lambda_{llambda}_user_{user}'
        loss_png_filename = f'loss_K_{K}_lambda_{llambda}_user_{user}'
        acc_nn, pr_nn, rec_nn, f1_nn = train_and_test_neuralnet(Xtrain, Ytrain, Xtest, Ytest, output_path, K,
                                                                llambda, acc_filename=acc_png_filename,
                                                                loss_filename=loss_png_filename)
        results_row = pd.DataFrame(
            [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'NN', 'Accuracy': acc_nn,
              'Precision': pr_nn, 'Recall': rec_nn, 'F1': f1_nn}])

    elif model_to_run == 'XGB':
        acc_xgb, pr_xgb, rec_xgb, f1_xgb = train_and_test_xgboost(Xtrain, Ytrain, Xtest, Ytest)
        results_row = pd.DataFrame(
            [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'XGB', 'Accuracy': acc_xgb,
              'Precision': pr_xgb, 'Recall': rec_xgb, 'F1': f1_xgb}])

    elif model_to_run == 'DANets':
        acc_danets, pr_danets, rec_danets, f1_danets = train_and_test_danets(Xtrain, Ytrain, Xtest, Ytest)
        results_row = pd.DataFrame(
            [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'DANets', 'Accuracy': acc_danets,
              'Precision': pr_danets, 'Recall': rec_danets, 'F1': f1_danets}])

    elif model_to_run == 'TabNet':
        acc_tabnet, pr_tabnet, rec_tabnet, f1_tabnet = train_and_test_tabnet(Xtrain, Ytrain, Xtest, Ytest)
        results_row = pd.DataFrame(
            [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'TabNet', 'Accuracy': acc_tabnet,
              'Precision': pr_tabnet, 'Recall': rec_tabnet, 'F1': f1_tabnet}])

    return results_row


def export_model_wise_LOSO_to_csv(output_path, df, llambda, K):
    for classifier, group in df.groupby('Classifier'):
        csv_file_path = os.path.join(output_path, f'model_results_LOSO_lambda_{llambda}_K_{K}_model_{classifier}.csv')
        group.to_csv(csv_file_path, index=False)


def run_loso_for_specific_parameter(base_path, lambda_list, k_list, model_to_run):
    # base = ....../clustered-input+labels-no-headers/per-user-data

    output_path = f'{base_path}/results_loso/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(base_path)
    files = [f for f in files if f.endswith('.csv')]
    files.sort()

    for K in k_list:
        for llambda in lambda_list:
            final_output_path = f'{output_path}/K_{K}_lambda_{llambda}_{model_to_run}/'
            if not os.path.exists(final_output_path):
                os.makedirs(final_output_path)

            filename = f'clustered_input_lambda_{llambda}_K_{K}.csv'
            csv_file_path = os.path.join(final_output_path, f'model_results_LOSO_lambda_{llambda}_K_{K}_merged.csv')

            file_path = base_path + filename
            df = add_headers_to_merged_numpy_clustered_data(file_path)

            # results_df = []
            results_df = pd.DataFrame(
                columns=['Test_UID', 'Lambda', 'Cluster No (K)', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1'])
            results_df.to_csv(csv_file_path, index=False, mode='w')

            log_file_path = os.path.join(final_output_path, f'loso_data_summary_K_{K}_lambda_{llambda}.txt')
            log_message = f'<<====Total Class Distribution for K={K}, lambda={llambda}:\n{df["Clustered_Label"].value_counts()}====>>\n'
            with open(log_file_path, 'w') as log_file:
                log_file.write(log_message)

            for user in range(1, 40):  # this user is the one that will be left out for testing
                df_train = df[df['User_ID'] != str(user)]
                df_test = df[df['User_ID'] == str(user)]

                if df_test.shape[0] == 0:
                    # -----Sanity check to see if the user has any data in the test set
                    # if not, then skip the user (No user with user id:26)
                    continue
                print(f"====================>>>> Running for User: {user}, Lambda: {llambda}, K: {K}==========>\n\n")
                log_message = (
                    f'\n===>Running for user {user} with lambda {llambda} and K {K}\n'
                    f'DataFrame shapes: full={df.shape}, train={df_train.shape}, test={df_test.shape}\n'
                    'Training set class distribution:\n'
                    f'{df_train["Clustered_Label"].value_counts()}\n'
                    'Testing set class distribution:\n'
                    f'{df_test["Clustered_Label"].value_counts()}\n'
                )
                # print(log_message)
                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_message)

                Ytrain = df_train['Clustered_Label'].to_numpy().astype('int64')
                Ytest = df_test['Clustered_Label'].to_numpy().astype('int64')
                Ytrain = pd.get_dummies(Ytrain).values.astype(int)
                Ytest = pd.get_dummies(Ytest).values.astype(int)

                Xtrain = df_train.drop(columns=['Activity_Label', 'Clustered_Label', 'User_ID'], axis=1)
                Xtest = df_test.drop(columns=['Activity_Label', 'Clustered_Label', 'User_ID'], axis=1)
                Xtrain = np.array(Xtrain.to_numpy().astype('float32'))
                Xtest = np.array(Xtest.to_numpy().astype('float32'))

                results_row = get_result_for_a_model(model_to_run, Xtrain, Ytrain, Xtest, Ytest, user, llambda, K,
                                                     final_output_path)
                # Concatenate results and append to CSV
                results_df = pd.concat([results_row])
                results_df.to_csv(csv_file_path, index=False, mode='a', header=False)
            export_model_wise_LOSO_to_csv(output_path, results_df, llambda, K)
    print(f"====================>>>> Result Generation Finished for LOSO Data==========>\n\n")


def run_full_pipeline_for_LOSO(base_path, lambda_list, k_list):
    # base = ....../clustered-input+labels-no-headers/per-user-data

    output_path = f'{base_path}/results_loso/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(base_path)
    files = [f for f in files if f.endswith('.csv')]
    files.sort()

    for K in k_list:
        for llambda in lambda_list:
            filename = f'clustered_input_lambda_{llambda}_K_{K}.csv'
            csv_file_path = os.path.join(output_path, f'model_results_LOSO_lambda_{llambda}_K_{K}_merged.csv')

            file_path = base_path + filename
            df = add_headers_to_merged_numpy_clustered_data(file_path)
            # results_df = []
            results_df = pd.DataFrame(
                columns=['Test_UID', 'Lambda', 'Cluster No (K)', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1'])
            results_df.to_csv(csv_file_path, index=False, mode='w')

            log_file_path = os.path.join(output_path, f'loso_data_summary_K_{K}_lambda_{llambda}.txt')
            log_message = f'<<====Total Class Distribution for K={K}, lambda={llambda}:\n{df["Clustered_Label"].value_counts()}====>>\n'
            with open(log_file_path, 'w') as log_file:
                log_file.write(log_message)

            for user in range(1, 40):  # this user is the one that will be left out for testing
                df_train = df[df['User_ID'] != str(user)]
                df_test = df[df['User_ID'] == str(user)]

                if df_test.shape[0] == 0:
                    #-----Sanity check to see if the user has any data in the test set
                    # if not, then skip the user (No user with user id:26)
                    continue
                print(f"====================>>>> Running for User: {user}, Lambda: {llambda}, K: {K}==========>\n\n")
                log_message = (
                    f'\n===>Running for user {user} with lambda {llambda} and K {K}\n'
                    f'DataFrame shapes: full={df.shape}, train={df_train.shape}, test={df_test.shape}\n'
                    'Training set class distribution:\n'
                    f'{df_train["Clustered_Label"].value_counts()}\n'
                    'Testing set class distribution:\n'
                    f'{df_test["Clustered_Label"].value_counts()}\n'
                )
                # print(log_message)
                with open(log_file_path, 'a') as log_file:
                    log_file.write(log_message)

                Ytrain = df_train['Clustered_Label'].to_numpy().astype('int64')
                Ytest = df_test['Clustered_Label'].to_numpy().astype('int64')
                Ytrain = pd.get_dummies(Ytrain).values.astype(int)
                Ytest = pd.get_dummies(Ytest).values.astype(int)

                Xtrain = df_train.drop(columns=['Activity_Label', 'Clustered_Label', 'User_ID'], axis=1)
                Xtest = df_test.drop(columns=['Activity_Label', 'Clustered_Label', 'User_ID'], axis=1)
                Xtrain = np.array(Xtrain.to_numpy().astype('float32'))
                Xtest = np.array(Xtest.to_numpy().astype('float32'))

                # acc_rf, pr_rf, rec_rf, f1_rf = train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest)
                # results_df.append(
                #     {'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'RF', 'Accuracy': acc_rf,
                #      'Precision': pr_rf,
                #      'Recall': rec_rf, 'F1': f1_rf}, )
                #
                # acc_1nn, pr_1nn, rec_1nn, f1_1nn = train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest)
                # results_df.append(
                #     {'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': '1NN', 'Accuracy': acc_1nn,
                #      'Precision': pr_1nn,
                #      'Recall': rec_1nn, 'F1': f1_1nn}, )
                #
                # acc_svm, pr_svm, rec_svm, f1_svm = train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest)
                # results_df.append(
                #     {'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'SVM', 'Accuracy': acc_svm,
                #      'Precision': pr_svm,
                #      'Recall': rec_svm, 'F1': f1_svm}, )
                #
                # acc_png_filename = f'accuracy_K_{K}_lambda_{llambda}_user_{user}'
                # loss_png_filename = f'loss_K_{K}_lambda_{llambda}_user_{user}'
                # acc_nn, pr_nn, rec_nn, f1_nn = train_and_test_neuralnet(Xtrain, Ytrain, Xtest, Ytest, output_path, K,
                #                                                         llambda, acc_filename=acc_png_filename,
                #                                                         loss_filename=loss_png_filename)
                #     results_df.append(
                #         {'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'NN', 'Accuracy': acc_nn,
                #          'Precision': pr_nn,
                #          'Recall': rec_nn, 'F1': f1_nn})
                # results_df = pd.DataFrame(results_df)
                # results_df.to_csv(f'{output_path}/model_results_LOSO_lambda_{llambda}_K_{K}', index=False)

                acc_rf, pr_rf, rec_rf, f1_rf = train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest)
                results_row_rf = pd.DataFrame(
                    [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'RF', 'Accuracy': acc_rf,
                      'Precision': pr_rf, 'Recall': rec_rf, 'F1': f1_rf}])

                acc_1nn, pr_1nn, rec_1nn, f1_1nn = train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest)
                results_row_1nn = pd.DataFrame(
                    [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': '1NN',
                      'Accuracy': acc_1nn,
                      'Precision': pr_1nn, 'Recall': rec_1nn, 'F1': f1_1nn}])

                acc_svm, pr_svm, rec_svm, f1_svm = train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest)
                results_row_svm = pd.DataFrame(
                    [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'SVM',
                      'Accuracy': acc_svm,
                      'Precision': pr_svm, 'Recall': rec_svm, 'F1': f1_svm}])

                acc_png_filename = f'accuracy_K_{K}_lambda_{llambda}_user_{user}'
                loss_png_filename = f'loss_K_{K}_lambda_{llambda}_user_{user}'
                acc_nn, pr_nn, rec_nn, f1_nn = train_and_test_neuralnet(Xtrain, Ytrain, Xtest, Ytest, output_path, K,
                                                                        llambda, acc_filename=acc_png_filename,
                                                                        loss_filename=loss_png_filename)
                results_row_nn = pd.DataFrame(
                    [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'NN', 'Accuracy': acc_nn,
                      'Precision': pr_nn, 'Recall': rec_nn, 'F1': f1_nn}])

                acc_xgb, pr_xgb, rec_xgb, f1_xgb = train_and_test_xgboost(Xtrain, Ytrain, Xtest, Ytest)
                results_row_xgb = pd.DataFrame(
                    [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'XGB',
                      'Accuracy': acc_xgb,
                      'Precision': pr_xgb, 'Recall': rec_xgb, 'F1': f1_xgb}])

                acc_danets, pr_danets, rec_danets, f1_danets = train_and_test_danets(Xtrain, Ytrain, Xtest, Ytest)
                results_row_danets = pd.DataFrame(
                    [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'DANets',
                      'Accuracy': acc_danets,
                      'Precision': pr_danets, 'Recall': rec_danets, 'F1': f1_danets}])

                acc_tabnet, pr_tabnet, rec_tabnet, f1_tabnet = train_and_test_tabnet(Xtrain, Ytrain, Xtest, Ytest)
                results_row_tabnet = pd.DataFrame(
                    [{'Test_UID': user, 'Lambda': llambda, 'Cluster No (K)': K, 'Classifier': 'TabNet',
                      'Accuracy': acc_tabnet,
                      'Precision': pr_tabnet, 'Recall': rec_tabnet, 'F1': f1_tabnet}])

                # Concatenate results and append to CSV
                results_df = pd.concat(
                    [results_row_rf, results_row_1nn, results_row_svm, results_row_nn, results_row_xgb,
                     results_row_danets, results_row_tabnet])
                results_df.to_csv(csv_file_path, index=False, mode='a', header=False)
            export_model_wise_LOSO_to_csv(output_path, results_df, llambda, K)
    print(f"====================>>>> Result Generation Finished for LOSO Data==========>\n\n")


def generate_llm_results_from_models(Xtrain, Xtest, Ytrain, Ytest, output_path, results_df, K, pred_df=None):
    print(f"====================>>>> Running Baseline for K: {K}==========>\n\n")
    acc_rf, pr_rf, rec_rf, f1_rf, Ypred_rf = train_and_test_random_forest(Xtrain, Ytrain, Xtest, Ytest)
    results_df.append(
        {'Cluster No (K)': K, 'Classifier': 'RF', 'Accuracy': acc_rf, 'Precision': pr_rf,
         'Recall': rec_rf, 'F1': f1_rf}, )

    acc_1nn, pr_1nn, rec_1nn, f1_1nn, Ypred_1nn = train_and_test_1NN(Xtrain, Ytrain, Xtest, Ytest)
    results_df.append(
        {'Cluster No (K)': K, 'Classifier': '1NN', 'Accuracy': acc_1nn, 'Precision': pr_1nn,
         'Recall': rec_1nn, 'F1': f1_1nn}, )

    acc_svm, pr_svm, rec_svm, f1_svm, Ypred_svm = train_and_test_svm(Xtrain, Ytrain, Xtest, Ytest)
    results_df.append(
        {'Cluster No (K)': K, 'Classifier': 'SVM', 'Accuracy': acc_svm, 'Precision': pr_svm,
         'Recall': rec_svm, 'F1': f1_svm}, )

    acc_nn, pr_nn, rec_nn, f1_nn, Ypred_nn = train_and_test_neuralnet(Xtrain, Ytrain, Xtest, Ytest, output_path, K,
                                                                      000)
    results_df.append(
        {'Cluster No (K)': K, 'Classifier': 'NN', 'Accuracy': acc_nn, 'Precision': pr_nn,
         'Recall': rec_nn, 'F1': f1_nn}, )

    acc_xgb, pr_xgb, rec_xgb, f1_xgb, Ypred_xgb = train_and_test_xgboost(Xtrain, Ytrain, Xtest, Ytest)
    results_df.append(
        {'Cluster No (K)': K, 'Classifier': 'XGB', 'Accuracy': acc_xgb, 'Precision': pr_xgb,
         'Recall': rec_xgb, 'F1': f1_xgb}, )

    acc_danets, pr_danets, rec_danets, f1_danets, Ypred_danets = train_and_test_danets(Xtrain, Ytrain, Xtest, Ytest)
    results_df.append(
        {'Cluster No (K)': K, 'Classifier': 'DANets', 'Accuracy': acc_danets, 'Precision': pr_danets,
         'Recall': rec_danets, 'F1': f1_danets}, )
    #
    # acc_tabnet, pr_tabnet, rec_tabnet, f1_tabnet, Ypred_tabnet = train_and_test_tabnet(Xtrain, Ytrain, Xtest, Ytest)
    # results_df.append(
    #     {'Cluster No (K)': K, 'Classifier': 'TabNet', 'Accuracy': acc_tabnet, 'Precision': pr_tabnet,
    #      'Recall': rec_tabnet, 'F1': f1_tabnet}, )

    acc_cnn, pr_cnn, rec_cnn, f1_cnn, Ypred_cnn = train_and_test_ConvNet(Xtrain, Ytrain, Xtest, Ytest, output_path, K, -999)
    results_df.append(
        {'Cluster No (K)': K, 'Classifier': 'CNN', 'Accuracy': acc_cnn, 'Precision': pr_cnn,
         'Recall': rec_cnn, 'F1': f1_cnn}, )

    pred_df.append(
        {
            'Test_labels': np.argmax(Ytest, axis=1),
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
    return pred_df


def run_model_for_baseline_case(base_path, output_path, results_df, K):
    file_path = f'{base_path}base_clustered_input_K_{K}.csv'
    df = pd.read_csv(file_path)

    y = df['clustered_labels']
    X = df.drop(['activity_label', 'clustered_labels', 'user_id'], axis=1)
    # print(X)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    Ytrain = Ytrain.to_numpy()
    Ytest = Ytest.to_numpy()

    Ytrain = pd.get_dummies(Ytrain).values.astype(int)
    Ytest = pd.get_dummies(Ytest).values.astype(int)

    Xtrain = np.array(Xtrain.to_numpy().astype(float))
    Xtest = np.array(Xtest.to_numpy().astype(float))

    generate_llm_results_from_models(Xtrain, Xtest, Ytrain, Ytest, output_path, results_df, K)


def run_loso_for_baseline_case(base_path, k_list, model_to_run):
    output_path = f'{base_path}/results_loso/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(base_path)
    files = [f for f in files if f.endswith('.csv')]
    files.sort()

    for K in k_list:
        filename = f'base_clustered_input_K_{K}.csv'
        df = pd.read_csv(f'{base_path}/{filename}')
        csv_file_path = os.path.join(output_path, f'model_results_baseline_LOSO_K_{K}_{model_to_run}.csv')

        results_df = pd.DataFrame(
            columns=['Test_UID', 'Lambda', 'Cluster No (K)', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1'])
        results_df.to_csv(csv_file_path, index=False, mode='w')

        log_file_path = os.path.join(output_path, f'baseline_loso_data_summary_K_{K}_{model_to_run}.txt')
        log_message = f'<<====Total Class Distribution for K={K}:\n{df["clustered_labels"].value_counts()}====>>\n'
        with open(log_file_path, 'w') as log_file:
            log_file.write(log_message)

        for user in range(1, 40):
            df_train = df[df['user_id'] != user]
            df_test = df[df['user_id'] == user]

            if df_test.shape[0] == 0:
                continue
            print(f"====================>>>> Running for User: {user}, K: {K}==========>\n\n")
            log_message = (
                f'\n===>Running for user {user} with K {K}\n'
                f'DataFrame shapes: full={df.shape}, train={df_train.shape}, test={df_test.shape}\n'
                'Training set class distribution:\n'
                f'{df_train["clustered_labels"].value_counts()}\n'
                'Testing set class distribution:\n'
                f'{df_test["clustered_labels"].value_counts()}\n'
            )
            with open(log_file_path, 'a') as log_file:
                log_file.write(log_message)

            Ytrain = df_train['clustered_labels'].to_numpy().astype('int64')
            Ytest = df_test['clustered_labels'].to_numpy().astype('int64')
            Ytrain = pd.get_dummies(Ytrain).values.astype(int)
            Ytest = pd.get_dummies(Ytest).values.astype(int)

            Xtrain = df_train.drop(['activity_label', 'clustered_labels', 'user_id'], axis=1)
            Xtest = df_test.drop(['activity_label', 'clustered_labels', 'user_id'], axis=1)
            Xtrain = np.array(Xtrain.to_numpy().astype('float32'))
            Xtest = np.array(Xtest.to_numpy().astype('float32'))

            results_row = get_result_for_a_model(model_to_run, Xtrain, Ytrain, Xtest, Ytest, user, 000, K, output_path)
            results_df = pd.concat([results_row])
            results_df.to_csv(csv_file_path, index=False, mode='a', header=False)

    print(f"====================>>>> Result Generation Finished for LOSO Data==========>\n\n")
