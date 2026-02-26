from shiny.express import input, render, ui
from matplotlib import pyplot as plt


ui.input_selectize(
    "var", "Select variable",
    choices=["x", "y"]
)


def plot(data,var):
    plt.hist(data[input.var()])
    plt.xlabel(input.var())
    plt.ylabel("count")


@render.plot
def hist():
    
    data = {
        'x': [1,1,2,3,3,3],
        'y' : [5,5,5,6,7,7]
    }
    plot(data,input.var())
    