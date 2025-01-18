from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def substr_btwn(main_str, sub1, sub2):
    """
    This function will return the substring between two substrings from a string
    :param main_str: the main string
    :param sub1: first substring after which the result will begin
    :param sub2: second substring before which the result will end
    :return: the string between sub1 and sub2
    """
    i1 = main_str.find(sub1)
    i2 = main_str.find(sub2, i1 + len(sub1))
    if i1 < 0 or i2 < 0:
        return ''
    return main_str[i1 + len(sub1):i2]


def create_csv_from_summary(textfilepath, raw=False):
    """
    This function will create a csv file from the summary text file.
    :param textfilepath: textfilepath is the path to the summary text file
    :param raw: if we want to print the inverse label ratio in raw format. Default is False.
    When False, the inverse label ratio in terms of every 10 hours. When True, in terms of number of total samples.
    :return:
    """
    with open(textfilepath) as file:
        txt = file.read()
    csvtxt = ''
    for line in txt.strip().split('\n'):
        userid = substr_btwn(line, 'User: ', ' ;')
        n = substr_btwn(line, 'Info(n = ', ',')
        n_labeled_records = substr_btwn(line, 'n_labeled_records = ', ',')
        inverse_label_ratio = substr_btwn(line, '1 label for ', ' records,')
        n_unique_labels = substr_btwn(line, 'n_unique_labels = ', ')')

        if raw:
            if '.' in inverse_label_ratio:
                ilr = str(round(float(inverse_label_ratio), 0))
                ilr = ilr[:ilr.find('.')]
            else:
                ilr = inverse_label_ratio

            n_outcome = n
        else:
            ilr = str(round(int(n_labeled_records) * 36000.0 / int(n) / 2.0, 1))
            n_outcome = str(int(n) / 3600.0 * 2)
            n_outcome = n_outcome[:n_outcome.find('.')]

        csvtxt += '{},{},{},{},{}\n'.format(userid, n_outcome, n_labeled_records, ilr, n_unique_labels)

    print(csvtxt)


def plot_tsne(features, clustered_labels, llambda, K, export_path=None, ):
    """
    This function will plot the t-SNE visualization of the clustered labels
    :param features: the features to be visualized
    :param clustered_labels: the labels of the clusters
    :return:
    """
    # Perform t-SNE
    # tsne = TSNE(n_components=K, random_state=0)
    # tsne_result = tsne.fit_transform(features)
    #
    # # Plot t-SNE
    # plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clustered_labels, cmap='viridis')
    # plt.title('t-SNE Visualization of Clusters, lambda={}, K={}'.format(llambda, K))
    # plt.show()
    # plt.savefig(f'{export_path}/tsne_lambda_{llambda}_K_{K}.png')
    # plt.close()

    # # Count the samples per class
    # class_counts = np.bincount(clustered_labels)
    # # Plot the distribution
    # plt.bar(range(len(class_counts)), class_counts)
    # plt.xlabel('Class')
    # plt.ylabel('Number of Samples')
    # plt.title('Class Distribution')
    # plt.show()
    # plt.savefig(f'{export_path}/class_distribution_lambda_{llambda}_K_{K}.png')
    # plt.close()

    # pca = PCA(n_components=K)
    # pca.fit(features)
    #
    # # Plot explained variance ratio
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Cumulative Explained Variance')


    pca = PCA(n_components=K)
    pca_result = pca.fit_transform(features)
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.title('PCA viz K={}'.format(K))

    plt.savefig(f'{export_path}/pca_vis_lambda_{llambda}_K_{K}.png')
    plt.show()
