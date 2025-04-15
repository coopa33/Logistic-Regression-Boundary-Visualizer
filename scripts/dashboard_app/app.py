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
        df_cleaned = read_structure_data(df, y[0], features_to_drop, one_hot)
        return render.DataGrid(df_cleaned)

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
    
    
    
