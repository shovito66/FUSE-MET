import os
# -*- coding: utf-8 -*-
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

def movement_intensity(ax, ay, az):
    return sqrt(ax * ax + ay * ay + az * az)


def average_intensity(matrixA):
    sum = 0
    T = len(matrixA)
    for i in range(T):
        sum += movement_intensity(matrixA[i][0], matrixA[i][1], matrixA[i][2])
    return sum / T


def variance_intensity(matrixA, ai=None):
    sum = 0
    T = len(matrixA)
    if ai is None:
        ai = average_intensity(matrixA)
    for i in range(T):
        diff = ai - movement_intensity(matrixA[i][0], matrixA[i][1], matrixA[i][2])
        sum += diff * diff
    return sum / T


def normalized_SMA(matrixA):
    sum = 0
    T = len(matrixA)
    for i in range(T):
        sum += abs(matrixA[i][0]) + abs(matrixA[i][1]) + abs(matrixA[i][2])
    return sum / T


def export_model_wise_LOSO_to_csv(base_path, lambda_list, k_list):
    average_df = []
    for llambda in lambda_list:
        for K in k_list:
            file_to_open = f'model_results_LOSO_lambda_{llambda}_K_{K}_merged.csv'
            output_path = f'{base_path}/results_loso/'
            # read dataframe
            df = pd.read_csv(f'{output_path}/{file_to_open}')
            for classifier, group in df.groupby('Classifier'):
                csv_file_path = os.path.join(output_path,
                                             f'model_results_LOSO_lambda_{llambda}_K_{K}_model_{classifier}.csv')
                group.to_csv(csv_file_path, index=False)

                # Calculate the averages for each metric
                avg_accuracy = group['Accuracy'].mean()
                avg_precision = group['Precision'].mean()
                avg_recall = group['Recall'].mean()
                avg_f1 = group['F1'].mean()
                lambda_value = group['Lambda'].iloc[0]
                k_value = group['Cluster No (K)'].iloc[0]

                average_df.append({
                    'Lambda': lambda_value,
                    'Cluster No (K)': k_value,
                    'Classifier': classifier,
                    'Accuracy': avg_accuracy,
                    'Precision': avg_precision,
                    'Recall': avg_recall,
                    'F1': avg_f1
                })
    average_df = pd.DataFrame(average_df)
    average_df.to_csv(f'{base_path}/results_loso/model_results_LOSO_summary.csv', index=False)


'''
Figure Generation Code
2 type of figure 
1. Acc Merged, 
    a. Merged
    b. Separately
2. F1. Merged
    a. Merged
    b. Separately
'''


def plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='Accuracy'):
    """
    This function will export subplots for different K values in a single plot for the given metric (Accuracy, F1, Recall, Precision).

    :param df: The DataFrame containing the results
    :param output_dir: Directory to save the plots
    :param metric: The metric to plot ('Accuracy', 'F1', 'Recall', 'Precision')

    To plot Accuracy:    plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='Accuracy')
    To plot F1 Score:    plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='F1')
    To plot Recall:     plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='Recall')
    To plot Precision:    plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='Precision')
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    K_values = sorted(df['Cluster No (K)'].unique())
    plt.rcParams['axes.linewidth'] = 1.5 # Set the linewidth of the axes, darker/stornger border

    for i, k in enumerate(K_values):
        ax = axes[i]
        df_k = df[df['Cluster No (K)'] == k]

        # Dynamically plot the given metric vs lambda
        sns.lineplot(data=df_k, x='Lambda', y=metric, hue='Classifier', ax=ax, marker='o', markersize=5)

        # ax.set_title(f'Plot for K={k}')
        # ax.set_xlabel('Lambda')
        # ax.set_ylabel(f'{metric}')
        ax.text(0.5, 0.95, f'Cluster No, K={k}', transform=ax.transAxes, fontsize=13, verticalalignment='top',
                horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        ax.set_xlabel('Lambda', labelpad=-32, fontsize=13)
        ax.set_ylabel(f'{metric}', labelpad=-18, fontsize=13)
        ax.set_ylim(0, 1)

        ax.legend(title='Classifier')

    # plt.suptitle(f'{metric} vs Lambda for Different Classifiers and K Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplots to fit the title
    plt.savefig(f'{output_dir}/combined_result-{metric}.pdf')
    plt.close()


def plot_best_model_vs_lambda_with_baseline_shade(df, df_base, output_dir, colors, markers, metric='F1',
                                                  shade_factor=0.5):
    """
    Plot F1 or accuracy vs Lambda for the best model in each cluster and include the baseline models as a shortened shaded region (mean ± shade_factor * std).

    :param df: DataFrame containing the results of the models
    :param df_base: DataFrame containing the baseline results without Lambda
    :param output_dir: Directory to save the plots
    :param colors: List of colors to use for the best model plots
    :param markers: List of markers to use for the best model plots
    :param metric: Metric to use ('F1' or 'Accuracy')
    :param shade_factor: A multiplier to shorten the shaded region (e.g., 0.5 will shorten the standard deviation range by half)
    """
    K_values = sorted(df['Cluster No (K)'].unique())
    K_styles = dict(zip(K_values, zip(colors, markers)))

    # Determine the common Y-axis limits for all plots
    y_min = df[metric].min() - 0.05  # Adding some margin
    y_max = df[metric].max() + 0.05

    # Create a figure to plot all clusters in the same plot with different colors for each cluster
    plt.figure(figsize=(6.6, 3.3))
    plt.rcParams['axes.linewidth'] = 1.5  # Set the linewidth of the axes, darker/stornger border

    # Iterate through each cluster (K value)
    for idx, k in enumerate(K_values):
        # Subset the data for the current cluster (K value)
        df_k = df[df['Cluster No (K)'] == k]

        # Find the best model for the current cluster based on the metric (F1/Accuracy)
        best_model = df_k.loc[df_k[metric].idxmax()]['Classifier']
        best_model_data = df_k[df_k['Classifier'] == best_model]

        # Plot F1/accuracy vs Lambda for the best model
        plt.plot(best_model_data['Lambda'], best_model_data[metric], label=f'Best Model K={k} - {best_model}',
                 color=K_styles[k][0], marker=K_styles[k][1], linestyle='-', markersize=5)

        # For baseline models, calculate mean and std for the current cluster K
        df_baseline_k = df_base[df_base['Cluster No (K)'] == k]

        # Calculate the mean and std of the baseline models' metric (F1/Accuracy/Recall/Precision)
        baseline_mean = df_baseline_k[metric].mean()
        baseline_std = df_baseline_k[metric].std()
        shaded_region_bottom = baseline_mean - shade_factor * baseline_std
        shaded_region_top = baseline_mean + shade_factor * baseline_std
        # Plot a single horizontal shaded line representing baseline models' mean ± shade_factor * std
        plt.axhline(y=baseline_mean, color=K_styles[k][0], linestyle='--', label=f'Baseline Mean K={k}')
        plt.fill_between(best_model_data['Lambda'],
                         shaded_region_bottom, shaded_region_top,
                         color=K_styles[k][0], alpha=0.2,
                         #  label=f'Baseline ± {shade_factor} * Std K={k}'
                         )

    # Set title, labels, and grid
    # plt.title(f'{metric} vs Lambda for Best Models and Baseline (Shaded) for All Clusters', fontsize=12)
    # plt.xlabel('Lambda')
    # plt.ylabel(metric)
    plt.xlabel('Lambda', labelpad=-35, fontsize=13)
    plt.ylabel(metric, labelpad=-18, fontsize=13)
    plt.ylim(0, 1)
    # Set X-axis to start from 0
    # plt.xlim(left=-0.9)  # Set the lower limit of X-axis to 0
    plt.legend(fontsize=8, ncol=3)
    # plt.grid(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_model_vs_lambda_{metric}_with_baseline_shade.pdf')
    # plt.show()
    plt.close()


def plot_metric_vs_lambda_for_all_K_shaded(df, output_dir, metric='Accuracy'):
    """
    This function exports a single plot where different K values are shown on the same graph [metric vs Lambda for different K values in a single plot],
    with each K represented by a distinct color and line. It plots the mean performance across
    classifiers with a shaded region representing the standard deviation for each K value.

    :param df: The DataFrame containing the results
    :param output_dir: Directory to save the plots
    :param metric: The metric to plot ('Accuracy', 'F1', 'Recall', 'Precision')
    """
    plt.figure(figsize=(5, 3))

    # Define colors for each K value
    colors = sns.color_palette("tab10", len(df['Cluster No (K)'].unique()))

    K_values = sorted(df['Cluster No (K)'].unique())

    # Plot each K value
    for i, k in enumerate(K_values):
        df_k = df[df['Cluster No (K)'] == k]

        # Group by Lambda to compute the mean and standard deviation for each lambda across classifiers
        grouped = df_k.groupby('Lambda').agg(
            mean_metric=(metric, 'mean'),
            std_metric=(metric, 'std')
        ).reset_index()

        # Plot the mean line with shaded standard deviation for each K value
        plt.plot(grouped['Lambda'], grouped['mean_metric'], label=f'K={k}', color=colors[i], marker='o', linestyle='-',
                 markersize=5)
        plt.fill_between(grouped['Lambda'], grouped['mean_metric'] - grouped['std_metric'],
                         grouped['mean_metric'] + grouped['std_metric'],
                         color=colors[i], alpha=0.2)

    # Labels and Title
    plt.title(f'{metric} vs Lambda for Different Clusters', fontsize=10)
    plt.xlabel('Lambda')
    plt.ylabel(f'{metric}')
    plt.legend(title='Cluster No (K)')
    plt.ylim(0, 1)
    # plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_result-{metric}-mean-std-all_K.pdf')
    # plt.show()
    plt.close()


def plot_accuracy_vs_lambda_for_all_clusters(df, output_dir):
    '''
    it will export the accuracy subplots for different K in a single plot
    '''
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    K_values = sorted(df['Cluster No (K)'].unique())

    for i, k in enumerate(K_values):
        ax = axes[i]
        df_k = df[df['Cluster No (K)'] == k]
        # df_base_k = df_base[df_base['Cluster No (K)'] == k]

        sns.lineplot(data=df_k, x='Lambda', y='Accuracy', hue='Classifier', ax=ax, marker='o', markersize=5)
        # # Plotting baseline results as horizontal lines
        # for idx, row in df_base_k.iterrows():
        #     ax.axhline(y=row['Accuracy'], color='gray', linestyle='--', linewidth=1.5, label=f"{row['Classifier']} Baseline")

        ax.set_title(f'Plot for K={k}')
        ax.set_xlabel('Lambda')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend(title='Classifier')

        # To handle legend duplication issue
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))  # removes duplicated labels/handles
        # ax.legend(by_label.values(), by_label.keys(), title='Classifier')

    plt.suptitle('Accuracy vs Lambda for Different Classifiers and K Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplots to fit the title
    plt.savefig(f'{output_dir}/combined_result-accuracy.pdf')
    # plt.show()
    plt.close()


def plot_F1_vs_lambda_for_all_clusters(df, output_dir):
    '''
    it will export the F1 subplots for different K in a single plot
    '''

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    K_values = sorted(df['Cluster No (K)'].unique())

    for i, k in enumerate(K_values):
        ax = axes[i]
        df_k = df[df['Cluster No (K)'] == k]

        sns.lineplot(data=df_k, x='Lambda', y='F1', hue='Classifier', ax=ax, marker='o', markersize=5)
        ax.set_title(f'K={k}')
        ax.set_xlabel('Lambda')
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0, 1)
        ax.legend(title='Classifier')
        # ax.grid(True)

    plt.suptitle('F1-Score vs Lambda for Different Classifiers and Clusters (K Values)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplots to fit the title

    plt.savefig(f'{output_dir}/combined_result-F1.pdf')
    # plt.show()
    plt.close()


def plot_accuracy_vs_lambda_for_each_cluster(df, output_dir, colors, markers):
    '''
    it will export the accuracy subplots for different Ks in separate plots
    '''
    classifiers = df['Classifier'].unique()
    classifier_styles = dict(zip(classifiers, zip(colors, markers)))

    # Values for K
    K_values = sorted(df['Cluster No (K)'].unique())

    # Creating separate plots for each K value and saving them
    for k in K_values:
        plt.figure(figsize=(9, 4.5))
        df_k = df[df['Cluster No (K)'] == k]

        # Plot each classifier separately
        for classifier in classifiers:
            if classifier in df_k['Classifier'].values:
                subset = df_k[df_k['Classifier'] == classifier]
                plt.plot(subset['Lambda'], subset['Accuracy'], label=classifier,
                         #  color=classifier_styles[classifier][0],
                         marker=classifier_styles[classifier][1], linestyle='-', markersize=5)

        plt.title(f'Accuracy vs Lambda for K={k}')
        plt.xlabel('Lambda')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend(title='Classifier')
        plt.grid(True)
        # plt.show()
        # Save the figure
        plt.savefig(f'{output_dir}/combined_result-Acc-K_{k}.pdf')  # Save the plot as a pdf file
        plt.close()  # Close the plot to free up memory


def plot_F1_vs_lambda_for_each_cluster(df, output_dir, colors, markers):
    classifiers = df['Classifier'].unique()
    classifier_styles = dict(zip(classifiers, zip(colors, markers)))

    # Values for K
    K_values = sorted(df['Cluster No (K)'].unique())

    # Creating separate plots for each K value and saving them
    for k in K_values:
        plt.figure(figsize=(9, 4.5))
        df_k = df[df['Cluster No (K)'] == k]

        # Plot each classifier separately
        for classifier in classifiers:
            if classifier in df_k['Classifier'].values:
                subset = df_k[df_k['Classifier'] == classifier]
                plt.plot(subset['Lambda'], subset['F1'], label=classifier,
                         #  color=classifier_styles[classifier][0],
                         marker=classifier_styles[classifier][1], linestyle='-', markersize=8)

        plt.title(f'F1-Score vs Lambda for K={k}')
        plt.xlabel('Lambda')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        plt.legend(title='Classifier')
        plt.grid(True)
        # plt.show()
        # Save the figure
        plt.savefig(f'{output_dir}/combined_result-F1-K_{k}.pdf')
        plt.close()  # Close the plot to free up memory


def plot_accuracy_vs_lambda_for_each_model(df, df_base, output_dir, colors, markers):
    K_values = sorted(df['Cluster No (K)'].unique())
    K_styles = dict(zip(K_values, zip(colors, markers)))

    # Determine the common Y-axis limits for all plots
    y_min = df['Accuracy'].min() - 0.05  # Adding some margin
    y_max = df['Accuracy'].max() + 0.05

    # Unique classifiers
    classifiers = df['Classifier'].unique()
    import matplotlib.cm as cm
    baseline_colormap = cm.get_cmap('tab20', len(K_values))
    # Create separate plots for each classifier
    for classifier in classifiers:
        plt.figure(figsize=(9.5, 4.5))
        df_classifier = df[df['Classifier'] == classifier]
        df_baseline_classifier = df_base[df_base['Classifier'] == classifier]

        # Plot each K value separately
        for idx, k in enumerate(K_values):
            subset = df_classifier[df_classifier['Cluster No (K)'] == k]
            if not subset.empty:
                plt.plot(subset['Lambda'], subset['Accuracy'], label=f'K={k}',
                         color=K_styles[k][0],
                         # color=baseline_colormap(idx),
                         marker=K_styles[k][1], linestyle='-', markersize=8)

            # Add baseline as dashed line
            baseline_subset = df_baseline_classifier[df_baseline_classifier['Cluster No (K)'] == k]
            if not baseline_subset.empty:
                baseline_accuracy = baseline_subset['Accuracy'].values[0]
                plt.axhline(y=baseline_accuracy,
                            color=K_styles[k][0],
                            # color=baseline_colormap(idx),
                            linestyle='--', linewidth=1.5, label=f'Baseline K={k}')

        plt.title(f'Accuracy vs Lambda for {classifier}')
        plt.xlabel('Lambda')
        plt.ylabel('Accuracy')
        # plt.ylim(y_min, y_max)  # Set the same Y-axis limits for all plots
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{output_dir}/{classifier}_result-Accuracy.pdf')
        # plt.show()
        plt.close()  # Close the plot to free up memory


def plot_F1_vs_lambda_for_each_model(df, df_base, output_dir, colors, markers):
    # Define the classifier colors and markers manually
    K_values = sorted(df['Cluster No (K)'].unique())
    K_styles = dict(zip(K_values, zip(colors, markers)))

    y_min = df['F1'].min() - 0.05  # Adding some margin
    y_max = df['F1'].max() + 0.05
    # Unique classifiers
    classifiers = df['Classifier'].unique()

    # Create separate plots for each classifier
    for classifier in classifiers:
        plt.figure(figsize=(9.5, 4.5))
        df_classifier = df[df['Classifier'] == classifier]
        df_baseline_classifier = df_base[df_base['Classifier'] == classifier]

        # Plot each K value separately
        for k in K_values:
            subset = df_classifier[df_classifier['Cluster No (K)'] == k]
            if not subset.empty:
                plt.plot(subset['Lambda'], subset['F1'], label=f'K={k}',
                         color=K_styles[k][0],
                         marker=K_styles[k][1], linestyle='-', markersize=8)

            # Add baseline as dashed line
            baseline_subset = df_baseline_classifier[df_baseline_classifier['Cluster No (K)'] == k]
            if not baseline_subset.empty:
                baseline_accuracy = baseline_subset['F1'].values[0]
                plt.axhline(y=baseline_accuracy,
                            color=K_styles[k][0],
                            # color=baseline_colormap(idx),
                            linestyle='--', linewidth=1.5, label=f'Baseline K={k}')

        plt.title(f'F1-Score vs Lambda for {classifier}')
        plt.xlabel('Lambda')
        plt.ylabel('F1 Score')
        plt.legend(title='Cluster Size (K)')
        plt.grid(True)
        # plt.ylim(y_min, y_max)  # Set the same Y-axis limits for all plots
        plt.ylim(0, 1)
        plt.savefig(f'{output_dir}/{classifier}_result-F1.pdf')
        # plt.show()
        plt.close()  # Close the plot to free up memory


def plot_best_model_vs_lambda_for_each_cluster(df, df_base, output_dir, colors, markers, metric='F1'):
    """
    Plot F1 or accuracy vs Lambda for the best model in each cluster and include baseline models as horizontal lines.

    :param df: DataFrame containing the results of the models
    :param df_base: DataFrame containing the baseline results
    :param output_dir: Directory to save the plots
    :param colors: List of colors to use for the plots
    :param markers: List of markers to use for the plots
    :param metric: Metric to use ('F1' or 'Accuracy')
    """
    K_values = sorted(df['Cluster No (K)'].unique())
    K_styles = dict(zip(K_values, zip(colors, markers)))

    # Determine the common Y-axis limits for all plots
    y_min = df[metric].min() - 0.05  # Adding some margin
    y_max = df[metric].max() + 0.05

    # Create a figure with 4 subplots, one for each cluster (K value)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate through each cluster (K value)
    for idx, k in enumerate(K_values):
        ax = axes[idx]

        # Subset the data for the current cluster (K value)
        df_k = df[df['Cluster No (K)'] == k]

        # Find the best model for the current cluster based on the metric (F1/Accuracy)
        best_model = df_k.loc[df_k[metric].idxmax()]['Classifier']
        best_model_data = df_k[df_k['Classifier'] == best_model]

        # Plot F1/accuracy vs Lambda for the best model
        ax.plot(best_model_data['Lambda'], best_model_data[metric], label=f'**{best_model}',
                color=K_styles[k][0], marker=K_styles[k][1], linestyle='-', markersize=5)

        # Plot baseline models as horizontal lines
        df_baseline_k = df_base[df_base['Cluster No (K)'] == k]
        # for base_model in df_baseline_k['Classifier'].unique():
        #     baseline_metric = df_baseline_k[df_baseline_k['Classifier'] == base_model][metric].values[0]
        #     ax.axhline(y=baseline_metric, color='gray', linestyle='--', linewidth=1.5, label=f'Baseline: {base_model}')
        for base_model in df_baseline_k['Classifier'].unique():
            baseline_metric = df_baseline_k[df_baseline_k['Classifier'] == base_model][metric].values[0]
            base_color = baseline_colors.get(base_model, 'gray')  # Get baseline color or default to gray
            ax.axhline(y=baseline_metric, color=base_color, linestyle='--', linewidth=1.5,
                       label=f'{base_model}')

        # Set title, labels, and grid
        ax.set_title(f'Cluster K={k}')
        ax.set_xlabel('Lambda')
        ax.set_ylabel(metric)
        # ax.set_ylim(y_min, y_max)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)
        ax.grid(True)

    # Add a common title for the figure
    fig.suptitle(f'{metric} vs Lambda for Best Models and Baseline Models', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title

    # Save the figure
    plt.savefig(f'{output_dir}/best_model_vs_lambda_{metric}.pdf')
    # plt.show()  # Uncomment if you want to display the plot
    plt.close()


baseline_colors = {
    'RF': 'blue',
    '1NN': 'green',
    'SVM': 'red',
    'NN': 'purple',
    'XGB': 'orange',
    'DANets': 'cyan',
    'TabNet': 'magenta'
}


def export_all_figures_train_test_split(basepath, result_dir):
    result_file_path = f'{result_dir}/model_results.csv'
    baseline_path = f'{basepath}baseline-cluster-llm/results_train-test-split/model_results_baseline.csv'
    output_dir = f'{result_dir}/figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(result_file_path)
    df_base = pd.read_csv(baseline_path)

    df = df.drop(columns=['Unnamed: 0'])
    df_base = df_base.drop(columns=['Unnamed: 0'])

    # Define the classifier colors and markers manually
    # classifiers = ['RF', '1NN', 'SVM', 'NN', 'XGB', 'DANets', 'TabNet']
    colors = ['blue', 'green', 'red', 'orange', 'cyan', 'magenta', 'yellow', 'black']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    # plot_accuracy_vs_lambda_for_all_clusters(df, output_dir)
    plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='Accuracy')
    plot_accuracy_vs_lambda_for_each_cluster(df, output_dir, colors, markers)

    # plot_F1_vs_lambda_for_all_clusters(df, output_dir)
    plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='F1')
    plot_F1_vs_lambda_for_each_cluster(df, output_dir, colors, markers)

    plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='Recall')
    plot_metric_vs_lambda_for_all_clusters(df, output_dir, metric='Precision')

    plot_metric_vs_lambda_for_all_K_shaded(df, output_dir, metric='Accuracy')
    plot_metric_vs_lambda_for_all_K_shaded(df, output_dir, metric='F1')
    plot_metric_vs_lambda_for_all_K_shaded(df, output_dir, metric='Recall')
    plot_metric_vs_lambda_for_all_K_shaded(df, output_dir, metric='Precision')

    plot_accuracy_vs_lambda_for_each_model(df, df_base, output_dir, colors, markers)
    plot_F1_vs_lambda_for_each_model(df, df_base, output_dir, colors, markers)

    plot_best_model_vs_lambda_for_each_cluster(df, df_base, output_dir, colors, markers, metric='F1')
    plot_best_model_vs_lambda_for_each_cluster(df, df_base, output_dir, colors, markers, metric='Accuracy')
    plot_best_model_vs_lambda_for_each_cluster(df, df_base, output_dir, colors, markers, metric='Recall')
    plot_best_model_vs_lambda_for_each_cluster(df, df_base, output_dir, colors, markers, metric='Precision')

    plot_best_model_vs_lambda_with_baseline_shade(df, df_base, output_dir, colors, markers, metric='F1')
    plot_best_model_vs_lambda_with_baseline_shade(df, df_base, output_dir, colors, markers, metric='Accuracy')
    plot_best_model_vs_lambda_with_baseline_shade(df, df_base, output_dir, colors, markers, metric='Recall')
    plot_best_model_vs_lambda_with_baseline_shade(df, df_base, output_dir, colors, markers, metric='Precision')

    print("===> figure exported successfully! Directory: ", output_dir)


def save_label_counts(Ytrain, Ytest, train_test_split_path, K, llambda, output_file='label_counts.txt',
                      isbaseline=False):
    # Count unique labels in train and test
    unique_train_labels, train_counts = np.unique(Ytrain, return_counts=True)
    unique_test_labels, test_counts = np.unique(Ytest, return_counts=True)

    # Append the counts to the text file
    with open(f'{train_test_split_path}/{output_file}', 'a') as f:
        if isbaseline:
            f.write(f"====BASELINE llm/ChatGPT, K: {K} ===>>>>>\n")
        else:
            f.write(f"====lambda: {llambda}, K: {K} ===>>>>>\n")
        f.write(f" Shape of Ytrain: {Ytrain.shape}, Shape of Ytest: {Ytest.shape}\n")
        for label, count in zip(unique_train_labels, train_counts):
            f.write(f"Label {label}: {count}\n")

        f.write("\nTest Labels:\n")
        for label, count in zip(unique_test_labels, test_counts):
            f.write(f"Label {label}: {count}\n")

        f.write("<<<<<===========================================>>>>>\n\n")


def stratified_train_test_split(features, clustered_labels, train_ratio=0.2, random_state=0):
    """
    Split the features and labels into train and test sets in a stratified manner.
    The samples are first grouped by their cluster labels, and then one sample from each cluster is taken for the test set.
    The remaining samples are shuffled and split into train and test sets.
    :param features: The features to split
    :param clustered_labels: The one-hot encoded cluster labels
    :param train_ratio: The ratio of samples to keep in the train set
    :param random_state: The random seed for shuffling
    :return: The train and test sets for features and labels
    """

    # Convert the one-hot encoded labels to plain labels
    clustered_labels_plain = np.argmax(clustered_labels, axis=1)
    unique_clusters = np.unique(clustered_labels_plain)
    # print(f"Unique clusters: {unique_clusters}")
    Xtrain, Ytrain, Xtest, Ytest = [], [], [], []
    # Ytrain_llm, Ytest_llm = [], []

    for cluster in unique_clusters:
        # Get the indices of the samples in the current cluster
        indices = np.where(clustered_labels_plain == cluster)[0]

        if len(indices) > 1:
            # Multiple samples (at least 2), take one for train and one for test
            test_idx = indices[0]
            train_idx = indices[1]

            Xtrain.append(features[train_idx])
            Ytrain.append(clustered_labels[train_idx])
            # Ytrain_llm.append(baseline_clustered_labels[train_idx])

            Xtest.append(features[test_idx])
            Ytest.append(clustered_labels[test_idx])
            # Ytest_llm.append(baseline_clustered_labels[test_idx])

            if len(indices) == 2:
                print(
                    f"Cluster {cluster} has only two samples, indices: {indices}, Train: {train_idx}, Test: {test_idx}")
            #---Can comment the following part if not using perturbation for now for two samples in a cluster-------
            #------- not using perturbation for now for two samples in a cluster-------
            # if len(indices) == 2:
                # in case of only 2 samples in the cluster :  Add perturbed versions to train and test set in a reversed way
                perturbed_train_sample = features[test_idx] + np.random.normal(0, 1e-5, features[test_idx].shape)
                perturbed_test_sample = features[train_idx] + np.random.normal(0, 1e-5, features[train_idx].shape)

                Xtrain.append(perturbed_train_sample)
                Ytrain.append(clustered_labels[test_idx])

                Xtest.append(perturbed_test_sample)
                Ytest.append(clustered_labels[train_idx])

            features = np.delete(features, [train_idx, test_idx], axis=0)
            clustered_labels = np.delete(clustered_labels, [train_idx, test_idx], axis=0)
            clustered_labels_plain = np.delete(clustered_labels_plain, [train_idx, test_idx], axis=0)
            # baseline_clustered_labels = np.delete(baseline_clustered_labels, [train_idx, test_idx], axis=0)

        else:
            print(f"Cluster {cluster} has only one sample, length: {len(indices)}")
            # Only one sample in this cluster
            test_idx = indices[0]
            sample = features[test_idx]
            sample_label = clustered_labels[test_idx]
            # sample_label_llm = baseline_clustered_labels[test_idx]

            Xtest.append(sample)
            Ytest.append(sample_label)
            # Ytest_llm.append(sample_label_llm)

            # Optionally, add a slightly perturbed version to the train set
            perturbed_sample = sample + np.random.normal(0, 1e-5, sample.shape)
            print(f"Sample shape: {sample.shape}, Perturbed shape: {perturbed_sample.shape}")
            Xtrain.append(perturbed_sample)
            Ytrain.append(sample_label)
            # Ytrain_llm.append(sample_label_llm)

            features = np.delete(features, test_idx, axis=0)
            clustered_labels = np.delete(clustered_labels, test_idx, axis=0)
            clustered_labels_plain = np.delete(clustered_labels_plain, test_idx, axis=0)
            # baseline_clustered_labels = np.delete(baseline_clustered_labels, test_idx, axis=0)

            #check if the index is removed from the features and labels
            if test_idx in np.where(clustered_labels == cluster)[0]:
                print(f"Index {test_idx} is not removed from the cluster {cluster}")

    # Shuffle the remaining samples
    features, clustered_labels = shuffle(features, clustered_labels, random_state=random_state)

    pivot = int(len(features) * train_ratio)
    Xtrain.extend(features[:pivot])
    Ytrain.extend(clustered_labels[:pivot])
    # Ytrain_llm.extend(baseline_clustered_labels[:pivot])
    Xtest.extend(features[pivot:])
    Ytest.extend(clustered_labels[pivot:])
    # Ytest_llm.extend(baseline_clustered_labels[pivot:])

    # print("Train samples before np.array: ", Ytrain[:10])
    return np.array(Xtrain), np.array(Ytrain), np.array(Xtest), np.array(Ytest)
    # return Xtrain, Ytrain, Xtest, Ytest


# def generate_synthetic_data(Xtrain, Ytrain, random_state=0):
#     """
#     Generate synthetic data using SMOTE to balance the classes in the training set.
#     :param Xtrain: Training features
#     :param Ytrain: Training labels (one-hot encoded or plain labels)
#     :param random_state: Random seed for reproducibility
#     :return: Balanced Xtrain and Ytrain with synthetic samples
#     """
#     # Convert one-hot encoded labels to plain labels if needed
#     if len(Ytrain.shape) > 1 and Ytrain.shape[1] > 1:
#         Ytrain_plain = np.argmax(Ytrain, axis=1)
#     else:
#         Ytrain_plain = Ytrain
#
#     smote = SMOTE(random_state=random_state)
#     Xtrain_resampled, Ytrain_resampled = smote.fit_resample(Xtrain, Ytrain_plain)
#
#     # If the labels were one-hot encoded => return them in the same format
#     if len(Ytrain.shape) > 1 and Ytrain.shape[1] > 1:
#         num_classes = Ytrain.shape[1]
#         Ytrain_resampled_onehot = np.eye(num_classes)[Ytrain_resampled]
#     else:
#         Ytrain_resampled_onehot = Ytrain_resampled
#
#     return Xtrain_resampled, Ytrain_resampled_onehot




def generate_synthetic_data(Xtrain, Ytrain, random_state=0, noise_factor=1e-5):
    """
    Manually balance the classes by adding Gaussian noise to all the data points.
    For minority classes, augment the samples to match the majority class.
    :param Xtrain: Training features
    :param Ytrain: Training labels (one-hot encoded or plain labels)
    :param random_state: Random seed for reproducibility
    :param noise_factor: Standard deviation of Gaussian noise to add to the samples
    :return: Balanced and shuffled Xtrain and Ytrain with synthetic samples
    """
    np.random.seed(random_state)

    # Convert one-hot encoded labels to plain labels if needed
    if len(Ytrain.shape) > 1 and Ytrain.shape[1] > 1:
        Ytrain_plain = np.argmax(Ytrain, axis=1)
    else:
        Ytrain_plain = Ytrain

    unique_classes, class_counts = np.unique(Ytrain_plain, return_counts=True)
    max_class_size = max(class_counts)  # Majority class size
    Xtrain_augmented, Ytrain_augmented = [], []

    # Loop through each class and augment the data
    for cls, count in zip(unique_classes, class_counts):
        indices = np.where(Ytrain_plain == cls)[0]
        X_class = Xtrain[indices]
        Y_class = Ytrain_plain[indices]

        # Duplicate original data and add Gaussian noise to each sample
        X_class_noisy = X_class + np.random.normal(0, noise_factor, X_class.shape)

        # Add both the original and noisy data to the training set
        Xtrain_augmented.append(np.vstack([X_class, X_class_noisy]))
        Ytrain_augmented.append(np.hstack([Y_class, Y_class]))

        # If this class is smaller than the majority class, augment it further
        while len(Ytrain_augmented[-1]) < max_class_size:
            needed = max_class_size - len(Ytrain_augmented[-1])
            # Take some samples and add Gaussian noise
            X_augment = X_class[:needed] + np.random.normal(0, noise_factor, X_class[:needed].shape)
            Xtrain_augmented[-1] = np.vstack([Xtrain_augmented[-1], X_augment])
            Ytrain_augmented[-1] = np.hstack([Ytrain_augmented[-1], Y_class[:needed]])

    # Concatenate all the augmented data and shuffle the result
    Xtrain_augmented = np.vstack(Xtrain_augmented)
    Ytrain_augmented = np.hstack(Ytrain_augmented)

    # If your labels were one-hot encoded and you want to return them in the same format
    if len(Ytrain.shape) > 1 and Ytrain.shape[1] > 1:
        num_classes = Ytrain.shape[1]
        Ytrain_resampled_onehot = np.eye(num_classes)[Ytrain_augmented]
    else:
        Ytrain_resampled_onehot = Ytrain_augmented

    # Shuffle the data
    Xtrain_augmented, Ytrain_resampled_onehot = shuffle(Xtrain_augmented, Ytrain_resampled_onehot, random_state=random_state)

    return Xtrain_augmented, Ytrain_resampled_onehot

# Now you can use Xtrain_resampled and Ytrain_resampled_onehot for your model training

def rearrange_and_concat_data(X):
    """
    This function rearranges the input data to match the format expected by the ConvNet model.
    :param X: Input data of shape (num_samples, 89) where the first 59 columns are static features and the remaining 30 columns are sensor data.
    :return: Rearranged data of shape (num_samples, 5, 59+6) where the first 59 columns are repeated 5 times and the sensor data is rearranged into a 2D array.
    """

    # Extracting the first 59 columns (index 0 to 58)
    # static_features = X[:, :59]  # Shape (num_samples, 59)
    static_features = X[:, :26]  # Shape (num_samples, 59)
    # Repeating each feature 5 times to match the 5 timestamps
    repeated_static_features = np.repeat(static_features[:, np.newaxis, :], 5, axis=1)  # Shape (num_samples, 5, 59)

    # # Rearrange the sensor data as described previously
    # accelX = X[:, 59:64]  # Columns corresponding to accelX0, accelX1, ..., accelX4
    # accelY = X[:, 64:69]  # Columns corresponding to accelY0, accelY1, ..., accelY4
    # accelZ = X[:, 69:74]  # Columns corresponding to accelZ0, accelZ1, ..., accelZ4
    # gyroX = X[:, 74:79]  # Columns corresponding to gyroX0, gyroX1, ..., gyroX4
    # gyroY = X[:, 79:84]  # Columns corresponding to gyroY0, gyroY1, ..., gyroY4
    # gyroZ = X[:, 84:89]  # Columns corresponding to gyroZ0, gyroZ1, ..., gyroZ4

    # Rearrange the sensor data as described previously
    accelX = X[:, 26:31]  # Columns corresponding to accelX0, accelX1, ..., accelX4
    accelY = X[:, 31:36]  # Columns corresponding to accelY0, accelY1, ..., accelY4
    accelZ = X[:, 36:41]  # Columns corresponding to accelZ0, accelZ1, ..., accelZ4
    gyroX = X[:, 41:46]  # Columns corresponding to gyroX0, gyroX1, ..., gyroX4
    gyroY = X[:, 46:51]  # Columns corresponding to gyroY0, gyroY1, ..., gyroY4
    gyroZ = X[:, 51:56]  # Columns corresponding to gyroZ0, gyroZ1, ..., gyroZ4


    # Rearranging the sensor data into a 2D array
    rearranged_sensor_data = np.stack([accelX, accelY, accelZ, gyroX, gyroY, gyroZ],
                                      axis=2)  # Shape (num_samples, 5, 6)

    # Transpose to match the dimensions for concatenation
    rearranged_sensor_data = rearranged_sensor_data.transpose(0, 1, 2)  # Shape (num_samples, 5, 6)

    # Concatenate the repeated static features with the rearranged sensor data
    combined_data = np.concatenate([repeated_static_features, rearranged_sensor_data],
                                   axis=2)  # Shape (num_samples, 5, 59+6)

    return combined_data
