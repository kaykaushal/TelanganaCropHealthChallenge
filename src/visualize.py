import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, normalize=False, figsize=(10, 8), cmap='coolwarm', title="Confusion Matrix"):
    """
    Plots a confusion matrix with options for normalization.

    Parameters:
    -----------
    y_true : array-like
        Actual target labels.
    y_pred : array-like
        Predicted target labels.
    normalize : bool, optional (default=False)
        Whether to normalize the confusion matrix by row (class-wise proportions).
    figsize : tuple, optional (default=(10, 8))
        Size of the plot.
    cmap : str, optional (default='coolwarm')
        Colormap for the heatmap.
    title : str, optional (default="Confusion Matrix")
        Title for the plot.

    Returns:
    --------
    None
    """
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {i}" for i in range(len(cm))],
        columns=[f"Predicted {i}" for i in range(len(cm))]
    )

    # Plot the confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=True, fmt='.0f' if normalize else 'd', cmap=cmap, cbar=True)
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.tight_layout()
    plt.show()

def plot_numerical_distributions(df, target_col=None):
    """
    Plots histogram, violin plot, and line plot for each numerical column in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing numerical columns.
        
    target_col : str, optional
        Column to exclude from plotting (e.g., 'Target'). Default is None.
    """
    
    # Drop target column if provided
    if target_col and target_col in df.columns:
        cols = df.drop([target_col], axis=1).select_dtypes(include='number').columns
    else:
        cols = df.select_dtypes(include='number').columns
    
    # Loop over each numerical column
    for col in cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        
        # Histogram for distribution
        sns.histplot(df[col], kde=True, bins=30, ax=axes[0])
        axes[0].set_title(f'Histogram of {col}')

        # Violin plot for spread and distribution
        sns.violinplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'Violin Plot of {col}')

        # Line plot to observe value progression
        sns.lineplot(data=df[col], ax=axes[2], marker="o", color='r')
        axes[2].set_title(f'Line Plot of {col}')

        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()

