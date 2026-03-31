from shiny.express import input, render, ui
from matplotlib import pyplot as plt
from configurations import load_data , make_data_paths, set_font_sizes, apply_general_styles, make_params_dict
from configurations.plot_config import create_fig 
from sam.dataset import generate_dual_task_batch
from dashboard.data_manager import get_scalar_measurements, get_config
from dashboard.plot_generator import plot_panel
import yaml
apply_general_styles()




params = get_config()
print(params)

Ks = [5,10,15,20,30][:2]
configs = ['dirichlet_full',
          'dirichlet_spiked',
          'zipf_spiked'][:2]


metrics = {
    'time': ['step','model_steps'],
    'loss' : ['loss'],
    'accuracy' : ['accuracy'],
    'kl' : ['kl_full_b','kl_full_u']
}

metrics_settings = {
    'loss' : {'label': 'Loss','color':'black'},
    'accuracy' : {'label': 'Accuracy','color':'black'},
    'kl_full_b' : {'label': r'$D_{KL}(P_\theta|\pi_b)$','color':'purple'},
    'kl_full_u' : {'label': r'$D_{KL}(P_\theta|\pi_u)$','color':'darkgreen'}
}
results = get_scalar_measurements(params, Ks,configs,metrics)
#results -> results[fix_trig][freq_trig][K][metric_name] = data[metric_name]
print('exit loading data')
print(results.keys())
# exit() 
# shiny run --reload --launch-browser scripts/dash_test.py



ui.input_selectize(
    id = "K", 
    label = "Select number of Triggers",
    choices = [str(k) for k in Ks],
    selected = str(Ks[0])
)

ui.input_selectize(
    id = "conf",
    label = "Select Configuration",
    choices = configs,
    selected = configs[0]
)


with ui.card(full_screen=True):
    ui.card_header("A card with a header")
    @render.plot
    def plot():
        fig , axes = create_fig(ncols=3,size='double',h=0.3)

        K = int(input.K())
        conf = input.conf()

        data = results[conf][K]

        time = data['time']['step'] 
        for i , panel_name in enumerate(['loss','accuracy','kl']):
            ax = axes[i]
            data_panel = data[panel_name]
            plot_panel(ax,time,data_panel,metrics_settings,title=panel_name.capitalize())
        
        return fig

    