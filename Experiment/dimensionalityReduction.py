import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from plot import create_bar_plot, create_2d_scatterplot, create_3d_scatterplot, create_2d_scatterplot_tester_colors


def calculate_PCA(df, number_components, graphs, colors):
    print("calculating PCA...")
    print(df)
    # pca = PCA(n_components=number_components)
    pca = PCA()
    pca_results = pca.fit_transform(df[df.columns].values)
    df['pca-one'] = pca_results[:, 0]
    df['pca-two'] = pca_results[:, 1]

    if number_components > 2:
        df['pca-three'] = pca_results[:, 2]

    print('Explained variation per principal component: {}'.format(
        pca.explained_variance_ratio_))
    print('Explained variation per principal component cumulative: {}'.format(
        pca.explained_variance_ratio_.cumsum()))

    if graphs == True:
        create_bar_plot(list(range(0, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_,
                        'Principle Components (ordered by highest variance to lowest)', 'Variance Ratio')
        if number_components == 2:
            create_2d_scatterplot_tester_colors(
                df,  "pca-one", "pca-two", colors, "PCA")
           # create_2d_scatterplot(df, "pca-one", "pca-two")
        if number_components == 3:
            create_3d_scatterplot(df, "pca-one", "pca-two",
                                  "pca-three", colors, "PCA")
    return pd.DataFrame(pca_results)


def calculate_TSNE(df, number_components, graphs, colors):
    print("calculating TSNE...")
    tsne = TSNE(n_components=number_components,
                init='random', perplexity=40, n_iter=5000, learning_rate=20)

    tsne_results = tsne.fit_transform(df.to_numpy())
    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]

    if number_components > 2:
        df['tsne-three'] = tsne_results[:, 2]

    if graphs == True:
        if number_components == 2:
            # create_2d_scatterplot(df, "tsne-one", "tsne-two", "TSNE")
            create_2d_scatterplot_tester_colors(
                df,  "tsne-one", "tsne-two", colors, "TSNE")
        if number_components == 3:
            create_3d_scatterplot(df, "tsne-one", "tsne-two",
                                  "tsne-three", colors, "t-SNE")

    return pd.DataFrame(tsne_results)
