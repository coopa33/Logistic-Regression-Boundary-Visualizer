import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from shiny import render, reactive
from shiny.express import ui, input
from shared import df, extract_feature_names, drop_features, read_structure_data


### Render structured Dataframe ###
ui.input_selectize(
        "features_exclude", 
        "Select features to be excluded from subsequent transformation",
        choices = list(df.columns),
        multiple = True)

ui.input_selectize(
    "y_select",
    "Select dependent variable",
    choices = list(df.columns),
    multiple = True)

ui.input_selectize(
        "one_hot_select",
        "Select categorical variables to represent as one-hot",
        choices = list(df.columns),
        multiple = True)

ui.input_selectize(
        "feature_plot",
        "Select features to plot",
        choices = list(df.columns),
        multiple = True)

df_original = df

"""
Selection of features is paramount to the data cleaning functioning properly. Currently the user must chose the features to exclude, feature to use as y, and features to one-hot. If by the user choice, there remains features that need but are not selected (f.e. a categorical feature with strings, which needs to be one-hot encoded) then the resulting cleaned and normalized data will not render. Therefore choosing features correctly must be done before rendering the data and decision boundary plots. 
"""

@reactive.effect
def update_y_select_choices():
    first_val = input.features_exclude()
    if first_val is None:
        first_val = []
    y_choice = {feature: feature for feature in list(df.columns) if feature not in first_val}
    ui.update_selectize("y_select", choices = y_choice, server=True)


@reactive.effect
def update_one_hot_plot():
    first_val = input.features_exclude()
    second_val = input.y_select()
    if first_val is None:
        first_val = []
    if second_val is None:
        second_val = []
    onehot_choice = {feature: feature for feature in list(df.columns) if feature not in first_val and feature not in second_val}
    ui.update_selectize("one_hot_select", choices = onehot_choice, server=True)

@reactive.effect
def update_feature_selection():
    df = df_original
    features_to_drop = list(input.features_exclude())
    y = list(input.y_select())
    one_hot = list(input.one_hot_select())
    if not (features_to_drop == [] or y == [] or one_hot == []):
        try:
            df_cleaned = read_structure_data(df, y[0], features_to_drop, one_hot)
            feature_choice = {i: list(df_cleaned.columns)[i] for i in range(len(list(df_cleaned.columns)))}
            ui.update_selectize("feature_plot", choices = feature_choice)
        except:
            print("a") 
    

@render.data_frame
def exclude_features_df():
    df = df_original
    features_to_drop = list(input.features_exclude())
    if len(features_to_drop)<1:
        return render.DataGrid(df)
    else:
        df = drop_features(df, features_to_drop)
        return render.DataGrid(df)
    
@render.data_frame
def cleaned_df():
    df = df_original
    features_to_drop = list(input.features_exclude())
    y = list(input.y_select())
    one_hot = list(input.one_hot_select())
    if not (features_to_drop == [] or y == [] or one_hot == []):
        try:
            df_cleaned = read_structure_data(df, y[0], features_to_drop, one_hot)
            return render.DataGrid(df_cleaned)
        except:
            return render.DataGrid(df_original)

@render.plot
def plot_data():
    df = df_original
    features_to_drop = list(input.features_exclude())
    y = list(input.y_select())
    one_hot = list(input.one_hot_select())
    features_select = list(input.feature_plot())

    if not (features_to_drop == [] or y == [] or one_hot == [] or len(features_select ) < 2):
        df_cleaned = read_structure_data(df, y[0], features_to_drop, one_hot)
        feature_names = list(df_cleaned.columns)
        # Plot
        fig, ax = plt.subplots()

        x_ax = df_cleaned[feature_names[int(features_select[0])]]
        y_ax = df_cleaned[feature_names[int(features_select[1])]]
        ax.scatter(x_ax, y_ax, c = df_cleaned[y[0]])
        ax.set_title("Distribution of features")
        ax.set_xlabel(feature_names[int(features_select[0])])
        ax.set_ylabel(feature_names[int(features_select[1])])
        return fig








@render.text
def result():
    first = input.feature_transform_selection()
    second = input.y_select()
    third = input.feature_plot()
    if first and second and third:
        return f"You selected: {first}, and {second}, and {third}"
    elif first and second:
        return f"You selected: {first} and {second}, please choose the third option"
    elif first:
        return f"You selected: {first}, please choose a sub-option"
    else:
        return f"Please select your first option"
    
    
    
