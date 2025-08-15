import torch
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from textwrap import wrap
import collections


def get_dimensions(
    property="conscientiousness",
    filename="steerability/steering_df.csv",
):
    """
    Return positive and negative statements in the form of a list for a given dimension.
    Args:
    - property: str with the dimension to be extracted from the dataframe
    - filename: csv to load df

    returns two lists of statements
    """
    steering_df = pd.read_csv(filename)
    property_pos = steering_df[
        (steering_df["persona_dim"] == property)
        & (steering_df["direction"] == "positive")
    ]
    property_pos = property_pos.statement.to_list()
    property_neg = steering_df[
        (steering_df["persona_dim"] == property)
        & (steering_df["direction"] == "negative")
    ]
    property_neg = property_neg.statement.to_list()

    return property_pos, property_neg


def get_hidden_states(statements, layer=-1, tokenizer=None, model=None, CLS=True):
    """
    Extract hidden states for a given layer at an index token

    Args:
    - layer: number of layer to extract
    - statements: positive or negative statements for a given property
    - tokenizer: to preprocess text
    - model: to do the fwd pass

    Returns:
    list of embeddings for corresponding statements

    """

    torch.set_grad_enabled(False)
    last_layers_pos = list()

    for p in statements:
        stts = [{"role": "system", "content": p}]
        input_ids = tokenizer.apply_chat_template(
            stts,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        outputs = model(input_ids, output_hidden_states=True)
        last_layers_pos.append(outputs.hidden_states[layer])

    if CLS:
        last_layers_pos = [pos[:, -1, :] for pos in last_layers_pos]
        last_layers_pos = [pos.numpy() for pos in last_layers_pos]

    return last_layers_pos


def generate_PCA_directions(
    last_layers_pos,
    last_layers_neg,
    layer,
    dimension,
    dir_output,
    n_statements=300,
    model_name="LLAMA3-8B-Instruct",
):
    """
    Calculate PCA for positive and negative directions + Visualization

    Args:
    - last_layers_pos: list of embeddings for sentences in positive direction
    - last_layers_neg: list of embeddings for sentences in negative direction
    - layer: layer number where the embedding was extracted

    Returns:
    Xr : numpy array with PCs
    y : y labels for Xr
    """

    df_pos = pd.DataFrame(np.array(last_layers_pos).reshape(n_statements, -1))
    df_neg = pd.DataFrame(np.array(last_layers_neg).reshape(n_statements, -1))
    df_pos["y"] = 1
    df_neg["y"] = 0
    X_ = pd.concat([df_pos, df_neg])
    X = X_[X_.columns[:-1]]
    y = X_[X_.columns[-1]]
    # TODO n_components should be passed by parameters
    components = 4
    pca = PCA(n_components=components)
    X_r = pca.fit(X).transform(X)
    df_pcs = pd.DataFrame(X_r, columns=["PC{}".format(i) for i in range(components)])
    df_pcs["y"] = y.to_numpy()

    # seaborn boiler plate
    g = sns.PairGrid(df_pcs, hue="y", y_vars=["PC0"], x_vars=df_pcs.columns.values[:-1])
    g.map_diag(sns.histplot, multiple="stack", element="step")
    g.map_upper(sns.kdeplot, alpha=0.8)
    g.map_offdiag(sns.scatterplot, alpha=0.3)
    g.add_legend(loc="upper right")

    new_labels = ["positive {}".format(dimension), "negative {}".format(dimension)]
    for t, lab in zip(g._legend.texts, new_labels):
        t.set_text(lab)
    plt.suptitle(
        "Hidden states {} at Layer {} \n for dimension {}".format(
            model_name,
            layer,
            dimension,
        ),
    )
    plt.tight_layout()
    plt.savefig(
        "{}/fig_layer_{}_{}_dimension_{}.png".format(
            dir_output,
            model_name,
            layer,
            dimension,
        ),
    )

    return X_r, y


def __get_embedding_vectors(layers, layer_ix, dims):
    """
    Creates numpy arrays for both features and labels from embedding vectors.
    This function is useful to prep formats for sklearn classifiers

    Args:
    - layer_ix: layer index
    - layers: dictionary with all hidden states across dimensions
    - dims: dictionary with dimensions selected

    Returns:
    X: with each row with 4096 elements
    y: associated numeric label for each dimension
    """

    print("embeddings from layer {}".format(layer_ix))
    layer_ = list()
    for d in dims.keys():
        # probably there is a nicer way to get features + label from lists
        for i in range(len(layers[(layer_ix, d)])):
            layer_.extend(
                [np.append(layers[(layer_ix, d)][i].reshape(1, -1), np.array(dims[d]))],
            )

    X_ = pd.DataFrame(layer_)
    X = X_[X_.columns[:-1]]
    y = X_[X_.columns[-1]]

    return X, y


def get_scores_classifier_layers(layers, layer_idxs, dims):
    """
    Generate metrics for simple ETC across all layers

    Args:
    - layers: dictionary with hidden states from statements
    - layer_idxs: layer number to retrive from dictionary
    - dims: list of properties/dimensions from persona

    Returns:
        dictionary with metrics
    """

    scores_m = ["precision_weighted", "f1_weighted", "accuracy"]
    metrics = collections.defaultdict(list)
    for layer in layer_idxs:
        X, y = __get_embedding_vectors(layers, layer, dims)
        clf = ExtraTreesClassifier(n_estimators=20, random_state=0)
        for s in scores_m:
            scores = cross_val_score(clf, X, y, cv=10, scoring=s)
            metrics[layer].extend([scores.mean()])

    return metrics


def plot_cls_layers(
    metrics,
    layer_idxs,
    dims,
    dir_output="./output",
    model_name="LLAMA3-8B-Instruct",
):
    """
    Plot heatmap across layers for classifiers built at every layer.

    Args:
    - metrics: dataframe with metrics extracted from crossvalidation.
    - layer_idxs: layer number to retrive from dictionary
    - dims: list of properties/dimensions from persona
    - dir_output: folder to store generated figure
    - model_name: for legends and figure titles
    """

    metrics_df = pd.DataFrame(metrics)
    metrics_df.index = ["precision_weighted", "f1_weighted", "accuracy"]
    metrics_df.columns = ["layer {}".format(i) for i in layer_idxs]
    plt.title(
        "\n".join(
            wrap(
                "Hidden states from {} as predictors for {}".format(
                    model_name,
                    list(dims.keys()),
                ),
            ),
        ),
    )
    sns.heatmap(
        metrics_df,
        cmap=sns.color_palette("tab10", 3),
        linewidths=0.5,
        annot=True,
    )
    plt.tight_layout()
    plt.savefig("{}/fig_cls_metrics_{}.png".format(dir_output, model_name))
