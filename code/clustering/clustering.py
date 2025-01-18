import os

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


def metdf2dict(metdf):
    """
    This function converts the MET dataframe to a dictionary
    :param metdf: The MET dataframe
    :return: A dictionary containing the MET values
    """
    met_dict = {}
    labels = metdf.word.to_numpy().astype(str)
    met_values = metdf.met_value.to_numpy().astype(float)
    for i in range(len(labels)):
        met_dict[labels[i]] = met_values[i]
    return met_dict


def plot_silhouette_figure(K, silhouette_scores, outputpath, method,
                           title=None, fig_size=(12, 6)):
    """
    This function will plot the silhouette scores for the given data.
    :param K: The range of K values
    :param silhouette_scores: The silhouette scores for the given K values
    :param outputpath: The path where the output image will be stored
    :return: None
    """

    if title is None:
        title = 'The Silhouette Score for optimal k'

    plt.figure(figsize=fig_size)
    plt.plot(K[1:], silhouette_scores, marker='o')  # K[1:] to align with silhouette_scores starting from k=2
    plt.title(title)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig(outputpath + f'silhouette_{method}.png')

    # Saving the K values and silhouette scores to a CSV file
    data = pd.DataFrame({
        'K': K[1:],
        'Silhouette Score': silhouette_scores
    })
    csv_filename = f'{outputpath}/silhouette_data_{method}.csv'
    data.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")


def plot_elbow_figure(K, distortions, outputpath, method,
                      title=None, fig_size=(12, 6)):
    """
    This function will plot the elbow method for the given data.
    :param K: The range of K values
    :param distortions: The distortion values for the given K values
    :param outputpath: The path where the output image will be stored
    :return: None
    """
    if title is None:
        title = 'The Elbow method showing the optimal k'

    plt.figure(figsize=fig_size)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outputpath + f'elbow_{method}.png')

    # Saving the K values and distortions to a CSV file
    data = pd.DataFrame({
        'K': K,
        'Distortion': distortions
    })
    csv_filename = f'{outputpath}/elbow_data_{method}.csv'
    data.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")


def elbow_method(all_labels, all_vecs, outputpath, method):
    """
    This function will plot the elbow method for the given data.
    :param all_labels:
    :param all_vecs:
    :param outputpath:
    :param method:
    :return:
    """
    n = len(all_labels)

    df = pd.DataFrame(all_vecs, columns=np.arange(all_vecs.shape[1]).tolist())

    distortions = []
    silhouette_scores = []
    K = range(1,
              n)  #======>>> This is the range of K values that will be tested,  1 to n-1, changed due to silhoute score
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        labels = kmeanModel.fit_predict(df)
        distortions.append(kmeanModel.inertia_)
        if k > 1:  # Silhouette score cannot be calculated for a single cluster
            # print(silhouette_score(df, labels))
            silhouette_scores.append(silhouette_score(df, labels))

    plot_elbow_figure(K, distortions, outputpath, method)
    plot_silhouette_figure(K, silhouette_scores, outputpath, method)

    # plt.figure(figsize=(12, 6))
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow method showing the optimal k')
    # plt.savefig(outputpath + f'elbow_{method}.png')


def save_clusters_to_textfile(clus, outputpath, method, K):
    # Save the clusters to a text file
    text_file_path = f'{outputpath}/clusters_{method}_K_{K}.txt'
    with open(text_file_path, 'w') as file:
        for cluster_index, cluster in enumerate(clus):
            file.write(f"Cluster {cluster_index + 1}:\n")
            for label in cluster:
                file.write(f"{label}\n")
            file.write("================\n\n")
    print(f"Clusters saved to {text_file_path}")


def print_clusters(clus, K):
    print('===Printing the clusters===')
    for i in range(K):
        for j in range(len(clus[i])):
            print(clus[i][j])
        print("==")


def create_and_plot_clusters(all_labels, all_vecs, outputpath, method, K=4):
    """
    This function will create the clusters and plot them.
    :param all_vecs:
    :param all_labels:
    :param outputpath:
    :param method:
    :param K:
    :return:
    """
    df = pd.DataFrame(all_vecs, columns=np.arange(all_vecs.shape[1]).tolist())
    n = len(all_labels)

    #Creating the clusters
    kmeanModel = KMeans(n_clusters=K)
    kmeanModel.fit(df)
    df.columns = df.columns.astype(str)
    k_means = kmeanModel.predict(df)

    #Plotting the clusters
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    axes.scatter(df['0'], np.zeros((n,)), c=k_means, cmap=plt.cm.Set1)
    axes.set_title('K_Means', fontsize=18)
    plt.savefig(outputpath + f'/kmeans_{method}_K_{K}.png')

    #Printing the clusters with the labels in them
    clus = [[] for i in range(K)]
    for i in range(len(all_labels)):
        clus[k_means[i]].append(all_labels[i])

    # print_clusters(clus, K)

    save_clusters_to_textfile(clus, outputpath, method, K)

    return clus


class MetAutoencoder(nn.Module):
    """
    Autoencoder to increase the dimension of MET values to match with BERT embeddings (768-dimensions)

    Args:
    input_dim: int, default=1, input dimension of MET values
    hidden_dim: int, default=786, hidden dimension of the autoencoder

    Returns:
    encoded: torch.tensor, encoded MET values
    """

    def __init__(self, input_dim=1, hidden_dim=768):
        super(MetAutoencoder, self).__init__()
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.to(self.device)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU())

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    # train method to train the autoencoder (using MPS to device)
    def train_model(self, X, epochs=100, lr=0.001):
        # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        device = self.device
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        X = torch.tensor(X, dtype=torch.float32).to(device)  # X ==> MET Vecrors

        # for epoch in range(epochs):
        # for i in range(0, len(X), batch_size):
        #     inputs = torch.tensor(X[i:i + batch_size], device=device).float()
        #     optimizer.zero_grad()
        #     encoded, decoded = self(inputs)
        #     loss = criterion(decoded, inputs)
        #     loss.backward()
        #     optimizer.step()
        # if epoch % 10 == 0:
        #     print(f'Epoch {epoch} Loss: {loss.item()}')
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            encoded, decoded = self(X)
            loss = criterion(decoded, X)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch} Loss: {loss.item()}')

        # Extract higher-dimensional MET representations
        self.eval()
        with torch.no_grad():
            tranformed_X = self.encoder(X).cpu().numpy()  # MET values are transformed to 786-dimensions

        return tranformed_X


def get_vectors_from_words_by_BERT(lowercase_labels, multi_word_handle='avg'):
    """
    This function will return the 768-d vector embeddings for all the words in the all_words list, considering the special cases.
    Special cases handled here -> multi-word labels, labels not found in the glove embedding
    :param all_words: all the labels convereted to lowercase and stored in a list
    :param embeddings_dict:
    :param multi_word_handle: 'avg' or 'concatenate' or 'default' (default is do nothing) to handle multi-word labels in the BERT embedding
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    n = len(lowercase_labels)
    all_vecs = []
    for i in range(n):
        words = lowercase_labels[i].strip().split()
        if len(words) > 1 and multi_word_handle == 'avg':
            '''
            If the label is a multi-word label, we will take the average of the embeddings of all the splitted words in the label.
            '''
            #print("Processing multi-word label: ", lowercase_labels[i])
            this_vecs = []
            for j in range(len(words)):
                input_ids = tokenizer.encode(words[j], return_tensors='pt')
                with torch.no_grad():
                    last_hidden_states = model(input_ids)[0]
                this_vecs.append(last_hidden_states.mean(dim=1).numpy())
            avg_vec = np.mean(np.array(this_vecs), axis=0)
            all_vecs.append(avg_vec)
        elif len(words) > 1 and multi_word_handle == 'concatenate':
            '''
            If the label is a multi-word label, we will concatenate the multiwords using '_' and take the embeddings of all the concatenated word.
            '''
            concatenated_word = '_'.join(words)  # words = ['taking', 'a', 'walk'] -> taking_a_walk
            input_ids = tokenizer.encode(concatenated_word, return_tensors='pt')
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]
            all_vecs.append(last_hidden_states.mean(dim=1).numpy())
        else:
            word = words[0]
            input_ids = tokenizer.encode(word, return_tensors='pt')
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]
            all_vecs.append(last_hidden_states.mean(dim=1).numpy())
    # return np.array(all_vecs)
    # return np.array([vec[0] for vec in all_vecs])
    return np.vstack(all_vecs)


def get_glove100_embedding_dict(basepath):
    """
    This function will return a dictionary containing 100-d vector embeddings for all the words available in
    the GLOVE file.
    :return:
    """
    embeddings_dict = {}
    with open(f"{basepath}/glove.6B.100d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def get_vectors_from_words(lowercase_labels, embeddings_dict):
    """
    This function will return the 100-d vector embeddings for all the words in the all_words list, considering the special cases.
    Special cases handled here -> multi-word labels, labels not found in the glove embedding
    :param all_words: all the labels convereted to lowercase and stored in a list
    :param embeddings_dict:
    :return:
    """
    n = len(lowercase_labels)
    all_vecs = []
    for i in range(n):
        words = lowercase_labels[i].strip().split()
        if len(words) > 1:
            #print("Processing multi-word label: ", lowercase_labels[i])
            this_vecs = []
            for j in range(len(words)):
                if embeddings_dict.get(words[j], None) is not None:
                    this_vecs.append(embeddings_dict[words[j]])
                else:
                    pass
                    #print(words[j], "not found in glove embedding")
            avg_vec = np.mean(np.array(this_vecs), axis=0)
            all_vecs.append(avg_vec)
        else:
            word = words[0]
            if embeddings_dict.get(word, None) is not None:
                all_vecs.append(embeddings_dict[word])
            else:
                '''
                    If the word is not found in the glove embedding, we can look up on the spacy database,
                     and if we don't find it even there, we try to find a similar word in the embedding
                     e.g. websurf is not found in the glove embedding, but web and surf are. So we manually make a way around for this word.
                '''
                #print("Single-word label", word, "not found in glove embedding")
                if word == 'websurf':
                    #print('Using "web surf" instead of "websurf"')
                    all_vecs.append((embeddings_dict['web'] + embeddings_dict['surf']) / 2)

    return np.array(all_vecs)


def create_directory_if_not_exists(base_path, output_folder):
    if not os.path.exists(base_path + output_folder):
        os.makedirs(base_path + output_folder)


def clusterMET(all_labels, metdf, outputpath):
    """
    This function performs a k-means clustering using the MET values only. See more clusterGlove, clusterCombined
    :param all_labels: All different activity labels in a numpy array, e.g. Shopping, Cooking, Dog Walk, etc.
    :param metdf: The MET values associated with the activity labels
    :return: None
    """
    met_dict = metdf2dict(metdf)
    all_vecs = [met_dict[label] for label in all_labels if met_dict.get(label, None)]
    all_vecs = np.array(all_vecs).reshape(-1, 1)  # column vector for the MET values, shape = (#Labels, 1)
    elbow_method(all_labels, all_vecs, outputpath, 'met')  #output default_exp

    for K in range(2, 6):
        output_path_for_clusters = f'/K_Mean={K}'
        create_directory_if_not_exists(outputpath, output_path_for_clusters)
        if K == 5:
            return create_and_plot_clusters(all_labels, all_vecs, outputpath + output_path_for_clusters, 'met', K)
        else:
            create_and_plot_clusters(all_labels, all_vecs, outputpath + output_path_for_clusters, 'met', K)


def clusterGlove(all_labels_original, basepath, outputpath):
    """
    This function will perform a k-means clustering using the GLOVE embeddings.
    :param all_labels_original: all the original labels in a numpy array. E.g. Shopping, Cooking, Dog Walk, etc.
    :param basepath: basepath is the path to the root directory containing the data
    :param outputpath: outputpath is the path to the directory where the output files will be stored
    :return: clusters containing the labels belonging to each cluster that was created by the clustering algorithm
    """
    embeddings_dict = get_glove100_embedding_dict(basepath)
    lowercase_labels = [label.lower() for label in all_labels_original]
    all_vecs = get_vectors_from_words(lowercase_labels, embeddings_dict)
    elbow_method(all_labels_original, all_vecs, outputpath, 'glove')
    # return create_and_plot_clusters(all_labels_original, all_vecs, outputpath, 'glove', 5)
    for K in range(2, 6):
        output_path_for_clusters = f'/K_Mean={K}'
        create_directory_if_not_exists(outputpath, output_path_for_clusters)
        if K == 5:
            return create_and_plot_clusters(all_labels_original, all_vecs, outputpath + output_path_for_clusters,
                                            'glove', K)
        else:
            create_and_plot_clusters(all_labels_original, all_vecs, outputpath + output_path_for_clusters, 'glove', K)


def clusterCombined(all_labels_original, metdf, basepath, outputpath, K=2, llambda=1, embedding='glove'):
    """
    This function will perform a k-means clustering using the GLOVE embeddings and the MET values.
    :param all_labels_original: all the original labels in a numpy array. E.g. Shopping, Cooking, Dog Walk, etc.
    :param metdf: The MET values associated with the activity labels
    :param basepath: basepath is the path to the root directory containing the data
    :param outputpath: outputpath is the path to the directory where the output files will be stored
    :param K: The number of clusters to be created
    :param llambda: The lambda value to be used for the combined embeddings
    :param embedding: The embedding to be used for the clustering. It can be 'glove' or 'bert'
    :return: clusters containing the labels belonging to each cluster that was created by the clustering algorithm
    """
    met_dict = metdf2dict(metdf)
    embeddings_dict = get_glove100_embedding_dict(basepath)
    lowercase_labels = [label.lower() for label in all_labels_original]
    if embedding == 'bert':
        embedded_vecs = get_vectors_from_words_by_BERT(
            lowercase_labels)  # default is nothing / multi_word_handle='avg'/ 'concatenate'
    else:
        embedded_vecs = get_vectors_from_words(lowercase_labels, embeddings_dict)

    met_vecs = np.array([met_dict[label] for label in all_labels_original if met_dict.get(label, None)]).reshape(-1, 1)

    # Normalize both MET and GLOVE vectors using Z-score normalization
    scaler = StandardScaler()
    embedded_vecs = scaler.fit_transform(embedded_vecs)
    met_vecs = scaler.fit_transform(met_vecs)

    all_vecs = np.concatenate(((1 - llambda) * embedded_vecs, llambda * met_vecs), axis=1)
    elbow_method(all_labels_original, all_vecs, outputpath, f'combined_lambda_is_{llambda}')

    cluster_list = []
    for K_no in range(2, 6):
        output_path_for_clusters = f'/K_Mean={K_no}'
        clus = create_and_plot_clusters(all_labels_original, all_vecs, outputpath + output_path_for_clusters,
                                        f'combined_lambda_is_{llambda}', K_no)

        print(f"cluster created for K:{K_no}, clust:{clus} ")
        cluster_list.append(clus)
    return cluster_list


def clusterCombined_with_autoencoder(all_labels_original, metdf, basepath, outputpath, K=2, llambda=1,
                                     embedding='glove', norm_before=True, add_embeddings=True):
    """
        This function will perform a k-means clustering using the GLOVE embeddings and the MET values.
        :param all_labels_original: all the original labels in a numpy array. E.g. Shopping, Cooking, Dog Walk, etc.
        :param metdf: The MET values associated with the activity labels
        :param basepath: basepath is the path to the root directory containing the data
        :param outputpath: outputpath is the path to the directory where the output files will be stored
        :param K: The number of clusters to be created
        :param llambda: The lambda value to be used for the combined embeddings
        :param embedding: The embedding to be used for the clustering. It can be 'glove' or 'bert'
        :param norm_before: Boolean value to determine whether to normalize the vectors before combining them
        :param add_embeddings: Boolean value to determine whether to add the embeddings or concatenate them
        :return: clusters containing the labels belonging to each cluster that was created by the clustering algorithm
        """

    met_model = MetAutoencoder()

    met_dict = metdf2dict(metdf)
    embeddings_dict = get_glove100_embedding_dict(basepath)
    # lowercase_labels = [label.lower() for label in all_labels_original]
    lowercase_labels = all_labels_original # we are not converting the labels to lowercase
    if embedding == 'bert':
        embedded_vecs = get_vectors_from_words_by_BERT(
            lowercase_labels)  # default is nothing / multi_word_handle='avg'/ 'concatenate'
    else:
        embedded_vecs = get_vectors_from_words(lowercase_labels, embeddings_dict)

    met_vecs = np.array([met_dict[label] for label in all_labels_original if met_dict.get(label, None)]).reshape(-1, 1)
    met_vecs = met_model.train_model(met_vecs, epochs=50, lr=0.001)  # MET values are transformed to 768-dimensions

    # Normalize both MET and GLOVE vectors using Z-score normalization
    scaler = StandardScaler()

    if norm_before:
        embedded_vecs = scaler.fit_transform(embedded_vecs)
        met_vecs = scaler.fit_transform(met_vecs)
        if add_embeddings:
            # Weighted combination of the vectors (element-wise addition)
            all_vecs = (1 - llambda) * embedded_vecs + llambda * met_vecs
        else:
            all_vecs = np.concatenate(((1 - llambda) * embedded_vecs, llambda * met_vecs), axis=1)
    else:
        if add_embeddings:
            # Weighted combination of the vectors (element-wise addition)
            all_vecs = (1 - llambda) * embedded_vecs + llambda * met_vecs
        else:
            all_vecs = np.concatenate(((1 - llambda) * embedded_vecs, llambda * met_vecs), axis=1)
        all_vecs = scaler.fit_transform(all_vecs)

    elbow_method(all_labels_original, all_vecs, outputpath, f'combined_lambda_is_{llambda}')

    cluster_list = []
    for K_no in range(2, 6):
        output_path_for_clusters = f'/K_Mean={K_no}'
        clus = create_and_plot_clusters(all_labels_original, all_vecs, outputpath + output_path_for_clusters,
                                        f'combined_lambda_is_{llambda}', K_no)

        print(f"cluster created for K:{K_no}, clust:{clus} ")
        cluster_list.append(clus)
    return cluster_list


def create_llm_clusters_for_same_dataset(basepath, clustered_labels, labels, K=2):
    base_clustering_path = f'{basepath}/baseline-cluster-llm'
    base_cluster_file_path = f'{base_clustering_path}/llm-cluster-K_{K}.csv'
    df_base = pd.read_csv(base_cluster_file_path)

    # Convert the clustered_labels into a DataFrame for easy manipulation
    clustered_labels_plain = np.argmax(clustered_labels, axis=1)
    df_all_users_data = pd.DataFrame({
        'activity_label': labels,
        'clustered_labels': clustered_labels_plain,
    })

    # Step 4: Create a mapping from the base cluster file
    label_map = df_base.set_index('Word')['Label'].to_dict()
    df_all_users_data['activity_label'] = df_all_users_data['activity_label'].str.strip()
    # Clean the keys in the label map dictionary as well
    # cleaned_label_map = {k.lower().strip(): v for k, v in label_map.items()}
    # Perform the mapping using the cleaned labels
    df_all_users_data['mapped_clustered_labels'] = df_all_users_data['activity_label'].map(
        label_map).fillna(-1).astype(int)

    encoder = OneHotEncoder(sparse=False, categories=[list(range(K))])
    one_hot_encoded_labels = encoder.fit_transform(df_all_users_data[['mapped_clustered_labels']]).astype(int)

    # Step 6: Extract both the original and the mapped baseline clustered labels
    return df_all_users_data['mapped_clustered_labels'].values, df_all_users_data['clustered_labels'].values, one_hot_encoded_labels


def create_clusters_from_llm(basepath, K=2):
    base_clustering_path = f'{basepath}/baseline-cluster-llm'

    base_cluster_file_path = f'{base_clustering_path}/llm-cluster-K_{K}.csv'
    df_base = pd.read_csv(base_cluster_file_path)

    df_all_users_data = pd.read_csv(
        f'{basepath}/NIH/clustered-input+labels+headers/clustered_input_lambda_1_K_2[HEADINGs].csv')  # choose any csv file, doesnot matter

    # Create a mapping from Word to Label in df2
    label_map = df_base.set_index('Word')['Label'].to_dict()

    # Use the mapping to update 'clustered_labels' in df1
    df_all_users_data['clustered_labels'] = df_all_users_data['activity_label'].map(label_map).fillna(
        df_all_users_data['clustered_labels'])

    cols = df_all_users_data.columns.tolist()
    start = cols.index('monday')
    end = cols.index('clustered_labels')

    # make the columns integer
    df_all_users_data['clustered_labels'] = df_all_users_data['clustered_labels'].astype(int)

    df_all_users_data = df_all_users_data.iloc[:, start:end + 1]

    df_all_users_data.to_csv(f'{base_clustering_path}/base_clustered_input_K_{K}.csv', index=False)
