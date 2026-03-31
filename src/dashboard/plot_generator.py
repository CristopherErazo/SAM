import matplotlib.pyplot as plt
from configurations import set_font_sizes, apply_general_styles
from configurations.plot_config import create_fig

apply_general_styles()
set_font_sizes(conf='tight')


def plot_panel(ax,time,data_panel,data_settings,title=None):
    for metric_name in data_panel.keys():
        metric_data = data_panel[metric_name]
        ax.plot(time,metric_data,**data_settings[metric_name])
    if title is not None:
        ax.set_title(title)
    ax.legend()
    # return ax
