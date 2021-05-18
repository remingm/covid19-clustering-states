"""
Experiments with dimensionality reduction and clustering time series to find similarities in US states' Covid-19 trajectories.
"""
import streamlit as st
from sklearn.preprocessing import minmax_scale, scale
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS
import numpy as np

from process_data import process_data


def user_interface():
    st.write("# Clustering States By Covid-19 Trends")

    cols = [
        "inIcuCurrently",
        "hospitalizedCurrently",
        "deathIncrease",
        "positiveIncrease",
        "percentPositive",
        "totalTestResultsIncrease",
        "Case Fatality Rate",
        "Infection Fatality Rate",
    ]
    cols.extend(
        [
            "retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "parks_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ]
    )
    cols = st.multiselect("Columns to Use", cols, default="hospitalizedCurrently")
    # do_scale = st.checkbox("zscore")
    # minmax = st.checkbox("minmax scale from 0-1", True)
    scale_method = st.selectbox(
        "Preprocessing method", ["Minmax scale from 0-1", "Zscore", "None"]
    )
    do_scale = False
    minmax = False
    if scale_method == "Zscore":
        do_scale = True
        minmax = False
    if scale_method == "Minmax scale from 0-1":
        do_scale = False
        minmax = True

    min_samples = st.slider("Minimum number of states per cluster", 1, 10, 2)

    return do_scale, minmax, cols, min_samples


def prepare_data(cols):
    states = pd.read_csv("states_daily.csv")["state"].unique()[:]
    states_data = {}
    for state in states:
        with st.spinner("Processing " + state):
            df = process_data(False, state)
            states_data[state] = df[cols].fillna(0)[-250:]

    return states, states_data


def reduce_dims(do_scale, minmax, states_data, states, cols):
    # PCA

    reduced = {}
    for state in states:
        if do_scale:
            X_valid_1D = scale(states_data[state])
        elif minmax:
            X_valid_1D = minmax_scale(states_data[state])
        else:
            X_valid_1D = states_data[state]

        pca = PCA(n_components=1)
        X_valid_1D = pca.fit_transform(X_valid_1D)
        reduced[state] = [i[0] for i in X_valid_1D]

    df = pd.DataFrame(reduced).interpolate()
    return df, reduced


def group_states(clustering, reduced, df):
    # Group states by cluster
    buckets = {}
    for l in clustering.labels_:
        buckets[l] = []
    for i, state in enumerate(reduced.keys()):
        group = clustering.labels_[i]
        buckets[group].append(state)
    for group in set(clustering.labels_):
        # Noisy samples and points which are not included in a leaf cluster of cluster_hierarchy_ are labeled as -1.
        if group == -1:
            continue
        st.subheader("Group {}".format(group))
        st.write(buckets[group])
        st.line_chart(df[buckets[group]])


def plot_hierarchy(clustering, key):
    import graphviz as graphviz

    edges = clustering.cluster_hierarchy_

    # Create a graphlib graph object
    graph = graphviz.Digraph()
    for edge in edges:
        s1, s2 = key[edge[0]], key[edge[1]]
        graph.edge(s1, s2)

    st.graphviz_chart(graph)


def run_tsne(states, states_data):
    tsne = TSNE()
    state = st.selectbox("state", states)
    st.write(states_data[state])
    X_valid_2D = tsne.fit_transform(states_data[state])
    X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], s=10)
    st.pyplot(fig)


def cluster_states():
    do_scale, minmax, cols, min_samples = user_interface()
    states, states_data = prepare_data(cols)

    df, reduced = reduce_dims(do_scale, minmax, states_data, states, cols)

    # Cluster
    clustering = OPTICS(min_samples=min_samples).fit(df.transpose())

    st.write("Resulting groups:", len(np.unique(clustering.labels_)) - 1)

    st.header("Groups")
    group_states(clustering, reduced, df)

    st.subheader("All States")
    st.line_chart(df)

    st.stop()


if __name__ == "__main__":
    cluster_states()
