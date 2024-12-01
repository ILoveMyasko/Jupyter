import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

class DataPreprocessing:
    
    # manage metadata
    def get_metadata(self, data):
        metadata = data.columns
        numerical_cols = data.select_dtypes(include = ["float64", "int64", "bool"]).columns.tolist()
        categorical_cols = data.select_dtypes(include = ["object"]).columns.tolist()        

        return metadata, numerical_cols, categorical_cols
    
    # function to filter missing data
    def filter_missing(self, data):
        sbn.displot(
            data = data.isna().melt(value_name="missing"),
            y = "variable",
            hue = "missing",
            multiple = "fill",
            aspect = 1.5
        )

        plt.show()

    # function to plot histogram of frequencies
    def hist_frequencies(self, data, numeric_cols, bins):
        # calculate the nrows and ncols for plots
        ncol_plots = 3
        nrow_plots = (len(numeric_cols) + ncol_plots - 1) // ncol_plots
        # create the subplots for specific row and column
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize = (16, 4 * nrow_plots))
        axs = axs.flatten()

        for i, col in enumerate(numeric_cols):
            sbn.histplot(data[col], color = "blue", bins = bins, ax = axs[i])
            axs[i].set_title("Histogram of frequencies for " + col)
            plt.xlabel(col)
            plt.ylabel("Frequencies")
        plt.tight_layout()
        plt.show()

    # function to plot correlation between numerical features
    def plot_correlation(self, data, cols):
        corr = data[cols].corr()
        plt.matshow(corr, cmap = "coolwarm")
        plt.xticks(range(len(cols)), cols, rotation = 90)
        plt.yticks(range(len(cols)), cols)

        # add the correlation values in each cell
        for (i, j), val in np.ndenumerate(corr):
            plt.text(j, i, f"{val:.1f}", ha='center', va='center', color='black')
        plt.title("Correlation Analysis")
        plt.colorbar()    
        plt.show()

    # function to get the frequencies of instances for each categorical variable
    def get_categorical_instances(self, data, categ_cols):
        for col in categ_cols:
            print("\n***** " + col + " ******")
            print(data[col].value_counts())

    # plot pie chart distribution of the categorical instances
    def plot_piechart(self, dataset, col):
        # count the #samples for each categogy
        results = dataset[col].value_counts()
        # calculate the relative frequencies
        total_samples = results.sum()
        rel_freq = results/total_samples
        sbn.set_style("whitegrid")
        plt.figure(figsize=(6,6))
        plt.pie(rel_freq.values.tolist(), labels = rel_freq.index.tolist(), autopct='%1.1f%%')
        plt.title("Relative frequency analysis by " + col)
        plt.show()

    # iteratively pie chart
    def iter_piechart(self, dataset, categ_cols):
        # calculate the nrows and ncols for plots
        ncol_plots = 2
        nrow_plots = (len(categ_cols) + ncol_plots - 1) // ncol_plots
        # create the subplots for specific row and column
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize = (16, 4 * nrow_plots))
        axs = axs.flatten()

        for i, col in enumerate(categ_cols):
            # count the #samples for each categogy
            results = dataset[col].value_counts()
            # calculate the relative frequencies
            total_samples = results.sum()
            rel_freq = results/total_samples
            sbn.set_style("whitegrid")    
            axs[i].pie(rel_freq.values.tolist(), labels = rel_freq.index.tolist(), autopct='%1.1f%%')
            axs[i].set_title("Relative frequency analysis by " + col)
        plt.tight_layout()
        plt.show()
                
    # probability distribution of the target variable
    def plot_target_distribution(self, data, target):
        plt.figure(figsize=[8,4])
        sbn.histplot(data[target], color='g', edgecolor="black", linewidth=2, bins=20)

        plt.title("Target Variable Distribution")
        plt.show()
