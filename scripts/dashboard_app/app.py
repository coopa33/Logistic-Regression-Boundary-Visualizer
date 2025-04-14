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
        "feature_plot",
        "Select features to plot",
        choices = list(df.columns),
        multiple = True)

@reactive.effect
def update_y_select_choices():
    first_val = input.features_exclude()
    if first_val is None:
        first_val = []
    y_choice = {feature: feature for feature in list(df.columns) if feature not in first_val}
    ui.update_selectize("y_select", choices = y_choice, server=True)

@reactive.effect
def update_feature_plot():
    first_val = input.features_exclude()
    second_val = input.y_select()
    if second_val is None:
        second_val = []
    plot_choice = {feature : feature for feature in list(df.columns) if (feature not in second_val) and (feature not in first_val)}
    ui.update_selectize("feature_plot", choices = plot_choice, server=True)


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
    
    
    
