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
        # Ð’Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸Ðµ ÐºÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð° ÑÑ‚Ñ€Ð¾Ðº Ð¸ ÑÑ‚Ð¾Ð»Ð±Ñ†Ð¾Ð² Ð´Ð»Ñ Ð¿Ð¾Ð´Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²
        ncol_plots = 3
        nrow_plots = (len(numeric_cols) + ncol_plots - 1) // ncol_plots
        # Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¿Ð¾Ð´Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð² Ð´Ð»Ñ ÑÐ¾Ð¾Ñ‚Ð²ÐµÑ‚ÑÑ‚Ð²ÑƒÑŽÑ‰Ð¸Ñ… ÑÑ‚Ñ€Ð¾Ðº Ð¸ ÑÑ‚Ð¾Ð»Ð±Ñ†Ð¾Ð²
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
        axs = axs.flatten()

        # Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¿Ð°ÑÑ‚ÐµÐ»ÑŒÐ½Ð¾Ð¹ Ð¿Ð°Ð»Ð¸Ñ‚Ñ€Ñ‹
        pastel_colors = sbn.color_palette("muted", len(numeric_cols))

        for i, col in enumerate(numeric_cols):
            # ÐŸÐ¾ÑÑ‚Ñ€Ð¾ÐµÐ½Ð¸Ðµ Ð³Ð¸ÑÑ‚Ð¾Ð³Ñ€Ð°Ð¼Ð¼Ñ‹ Ñ Ð¸Ð½Ð´Ð¸Ð²Ð¸Ð´ÑƒÐ°Ð»ÑŒÐ½Ñ‹Ð¼ Ñ†Ð²ÐµÑ‚Ð¾Ð¼ Ð´Ð»Ñ ÐºÐ°Ð¶Ð´Ð¾Ð³Ð¾ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ°
            sbn.histplot(data[col], color=pastel_colors[i], bins=bins, ax=axs[i])
            axs[i].set_title(f"Histogram of frequencies for {col}")
            axs[i].set_xlabel(col)  # ÐŸÐµÑ€ÐµÐ½Ñ‘Ñ ÑƒÑÑ‚Ð°Ð½Ð¾Ð²ÐºÑƒ Ð¿Ð¾Ð´Ð¿Ð¸ÑÐ¸ Ð¾ÑÐ¸ X Ð² axs[i]
            axs[i].set_ylabel("Frequencies")  # Ð£ÑÑ‚Ð°Ð½Ð¾Ð²ÐºÐ° Ð¿Ð¾Ð´Ð¿Ð¸ÑÐ¸ Ð¾ÑÐ¸ Y

        # ÐšÐ¾Ñ€Ñ€ÐµÐºÑ‚Ð¸Ñ€ÑƒÐµÐ¼ Ñ€Ð°ÑÐ¿Ð¾Ð»Ð¾Ð¶ÐµÐ½Ð¸Ðµ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²
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

    # Ð¤ÑƒÐ½ÐºÑ†Ð¸Ñ Ð´Ð»Ñ Ð¾Ñ‚Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ñ Ð¿Ð¾Ð´Ð¿Ð¸ÑÐµÐ¹ Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð´Ð»Ñ Ð±Ð¾Ð»ÑŒÑˆÐ¸Ñ… ÑÐµÐºÑ‚Ð¾Ñ€Ð¾Ð²
    def autopct_large_only(self, pct, threshold=10):
        # ÐŸÐ¾ÐºÐ°Ð·Ñ‹Ð²Ð°Ñ‚ÑŒ Ð¿Ñ€Ð¾Ñ†ÐµÐ½Ñ‚ Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð´Ð»Ñ ÑÐµÐºÑ‚Ð¾Ñ€Ð¾Ð², ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ðµ Ð±Ð¾Ð»ÑŒÑˆÐµ Ð¿Ð¾Ñ€Ð¾Ð³Ð°
        return f'{pct:.1f}%' if pct > threshold else ''
    
    # Ð˜Ñ‚ÐµÑ€Ð°Ñ‚Ð¸Ð²Ð½Ñ‹Ð¹ Ð¼ÐµÑ‚Ð¾Ð´ Ð´Ð»Ñ Ð¿Ð¾ÑÑ‚Ñ€Ð¾ÐµÐ½Ð¸Ñ Ð¿Ð°Ð¹Ñ‡Ð°Ñ€Ñ‚Ð¾Ð²
    def iter_piechart(self, dataset, categ_cols, threshold=2):
        import matplotlib.pyplot as plt
        import seaborn as sbn
        import pandas as pd

        # Ð—Ð°ÐºÑ€Ñ‹Ñ‚Ð¸Ðµ Ñ„Ð¸Ð³ÑƒÑ€ Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð² ÐºÐ¾Ð½Ñ‚ÐµÐºÑÑ‚Ðµ pie chart
        plt.close('all')  # Ð—Ð°ÐºÑ€Ñ‹Ñ‚ÑŒ Ð²ÑÐµ Ð¿Ñ€ÐµÐ´Ñ‹Ð´ÑƒÑ‰Ð¸Ðµ Ñ„Ð¸Ð³ÑƒÑ€Ñ‹

        # Ð’Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸Ðµ ÐºÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð° ÑÑ‚Ñ€Ð¾Ðº Ð¸ ÑÑ‚Ð¾Ð»Ð±Ñ†Ð¾Ð² Ð´Ð»Ñ Ð¿Ð¾Ð´Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²
        ncol_plots = 2
        nrow_plots = (len(categ_cols) + ncol_plots - 1) // ncol_plots
        # Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¿Ð¾Ð´Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð² Ð´Ð»Ñ ÑÐ¾Ð¾Ñ‚Ð²ÐµÑ‚ÑÑ‚Ð²ÑƒÑŽÑ‰Ð¸Ñ… ÑÑ‚Ñ€Ð¾Ðº Ð¸ ÑÑ‚Ð¾Ð»Ð±Ñ†Ð¾Ð²
        fig, axs = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
        axs = axs.flatten()

        # Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¿Ð°ÑÑ‚ÐµÐ»ÑŒÐ½Ð¾Ð¹ Ð¿Ð°Ð»Ð¸Ñ‚Ñ€Ñ‹
        pastel_colors = sbn.color_palette("pastel")

        for i, col in enumerate(categ_cols):
            # ÐŸÐ¾Ð´ÑÑ‡Ñ‘Ñ‚ ÐºÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð° Ð¾Ð±Ñ€Ð°Ð·Ñ†Ð¾Ð² Ð´Ð»Ñ ÐºÐ°Ð¶Ð´Ð¾Ð¹ ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ð¸
            results = dataset[col].value_counts()
            # Ð’Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸Ðµ Ð¾Ñ‚Ð½Ð¾ÑÐ¸Ñ‚ÐµÐ»ÑŒÐ½Ñ‹Ñ… Ñ‡Ð°ÑÑ‚Ð¾Ñ‚
            total_samples = results.sum()
            rel_freq = results / total_samples

            sbn.set_style("whitegrid")
            
            # ÐŸÐ¾Ñ€Ð¾Ð³Ð¸ Ð´Ð»Ñ Ð¾Ñ‚Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ñ "Other" ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ð¸
            if threshold > 0:
                # ÐŸÐµÑ€ÐµÐ¼ÐµÑ‰Ð°ÐµÐ¼ ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ð¸ Ñ Ð½Ð¸Ð·ÐºÐ¸Ð¼ Ð¿Ñ€Ð¾Ñ†ÐµÐ½Ñ‚Ð¾Ð¼ Ð² "Other"
                rel_freq = rel_freq[rel_freq > threshold / 100]
                other_sum = 1 - rel_freq.sum()
                if other_sum > 0:
                    # Ð˜ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÐ¼ pd.concat Ð´Ð»Ñ Ð¾Ð±ÑŠÐµÐ´Ð¸Ð½ÐµÐ½Ð¸Ñ Series
                    rel_freq = pd.concat([rel_freq, pd.Series([other_sum], index=["Other"])])
            
            # ÐŸÐ¾ÐºÐ°Ð· Ð¿Ð¾Ð´Ð¿Ð¸ÑÐµÐ¹ Ñ‚Ð¾Ð»ÑŒÐºÐ¾ Ð´Ð»Ñ ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ð¹ Ñ Ñ‡Ð°ÑÑ‚Ð¾Ñ‚Ð¾Ð¹ Ð²Ñ‹ÑˆÐµ Ð¿Ð¾Ñ€Ð¾Ð³Ð°
            wedges, texts, autotexts = axs[i].pie(
                rel_freq.values.tolist(),
                labels=rel_freq.index.tolist(),
                autopct=lambda pct: self.autopct_large_only(pct, threshold),
                colors=pastel_colors[:len(rel_freq)],  # ÐŸÑ€Ð¸Ð¼ÐµÐ½ÐµÐ½Ð¸Ðµ Ð¿Ð°ÑÑ‚ÐµÐ»ÑŒÐ½Ñ‹Ñ… Ñ†Ð²ÐµÑ‚Ð¾Ð²
            )
            
            # Ð£ÑÑ‚Ð°Ð½Ð¾Ð²ÐºÐ° Ñ†Ð²ÐµÑ‚Ð¾Ð² Ð´Ð»Ñ ÑÐµÐºÑ‚Ð¾Ñ€Ð¾Ð²
            for wedge in wedges:
                wedge.set_edgecolor('black')  # Ð£ÑÑ‚Ð°Ð½Ð°Ð²Ð»Ð¸Ð²Ð°ÐµÐ¼ Ñ‡ÐµÑ€Ð½Ñ‹Ðµ ÐºÑ€Ð°Ñ Ð´Ð»Ñ ÑÐµÐºÑ‚Ð¾Ñ€Ð¾Ð²
            
            axs[i].set_title(f"Relative frequency analysis by {col}", fontsize=14)  # Ð£Ð²ÐµÐ»Ð¸Ñ‡Ð¸Ð²Ð°ÐµÐ¼ ÑˆÑ€Ð¸Ñ„Ñ‚ Ð·Ð°Ð³Ð¾Ð»Ð¾Ð²ÐºÐ°

            # Ð£ÑÑ‚Ð°Ð½Ð¾Ð²Ð¸Ð¼ ÑÑ‚Ð¸Ð»ÑŒ Ð´Ð»Ñ Ð¿Ð¾Ð´Ð¿Ð¸ÑÐµÐ¹
            for text in texts:
                text.set_fontsize(12)  # Ð£Ð²ÐµÐ»Ð¸Ñ‡Ð¸Ð²Ð°ÐµÐ¼ ÑˆÑ€Ð¸Ñ„Ñ‚ Ð¿Ð¾Ð´Ð¿Ð¸ÑÐµÐ¹ ÐºÐ°Ñ‚ÐµÐ³Ð¾Ñ€Ð¸Ð¹
            for autotext in autotexts:
                autotext.set_fontsize(8)  # Ð£Ð²ÐµÐ»Ð¸Ñ‡Ð¸Ð²Ð°ÐµÐ¼ ÑˆÑ€Ð¸Ñ„Ñ‚ Ð°Ð²Ñ‚Ð¾Ð¿Ð¾Ð´Ð¿Ð¸ÑÐµÐ¹ (Ð¿Ñ€Ð¾Ñ†ÐµÐ½Ñ‚Ð¾Ð²)

        # Ð£Ð´Ð°Ð»ÐµÐ½Ð¸Ðµ Ð»Ð¸ÑˆÐ½Ð¸Ñ… Ð¿Ð¾Ð´Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²
        for j in range(len(categ_cols), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()  # Ð”Ð»Ñ ÑƒÐ»ÑƒÑ‡ÑˆÐµÐ½Ð½Ð¾Ð³Ð¾ Ñ€Ð°Ð·Ð¼ÐµÑ‰ÐµÐ½Ð¸Ñ Ð³Ñ€Ð°Ñ„Ð¸ÐºÐ¾Ð²
        plt.show()

                
    # probability distribution of the target variable
    def plot_target_distribution(self, data, target):
        plt.figure(figsize=[8,4])
        sbn.histplot(data[target], color='g', edgecolor="black", linewidth=2, bins=20)

        plt.title("Target Variable Distribution")
        plt.show()