import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from shiny.express import input, render, ui
from shared import X, y, df, w, b



ui.page_opts(title="Logistic Regression Decision Boundary Plotter")

ui.input_selectize(
        "feature_selector", 
        "select a feature below:",
        {"Pclass": "Passenger Class", 
         "Sex": "Gender",
         "Age": "Age",
         "SibSp": "Number of Siblings or spouses",
         "Parch": "No. of parents of children",
         },
        multiple=True)

@render.plot
def plot():
    if len(input.feature_selector()) == 2:

        sns.stripplot(data = df, x=input.feature_selector()[0], y=input.feature_selector()[1], hue="Survived")
    
