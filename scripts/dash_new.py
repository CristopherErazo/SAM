from shiny.express import input, render, ui
from matplotlib import pyplot as plt
from configurations import load_data , make_data_paths, set_font_sizes, apply_general_styles, make_params_dict
from configurations.plot_config import create_fig 
from sam.dataset import generate_dual_task_batch
from dashboard.data_manager import get_scalar_measurements, get_config, get_new_data_scalars, get_new_data_matrix
from dashboard.plot_generator import plot_panel
import yaml
import pickle

apply_general_styles()



# shiny run --reload --launch-browser scripts/dash_test.py


# Load data for all configurations and Ks
paths = ['full', 'induction', 'bigram','full_trigg'] # full_trigg is not included because it is not fully trained yet
configs = ['dirichlet','spiked_dirichlet','spiked_zipf']
Ks = [10,15,20]
experiment_name = 'dual_task_new'

results_scalar = get_new_data_scalars(paths, Ks, configs)
# results_matrix = get_new_data_matrix(paths, Ks, configs, batch_id=0)


loss_configs = {
    'full': {
        'tags' : ['loss_total'],#,'loss_bigram','loss_ind'],
        'labels' : [r'$L^{(total)}|_{all}$',r'$L^{(bigram)}|_{\sim t}$',r'$L^{(ind)}| t$'],
        'lstyle' : ['-','--',':'],
        'color': 'black'
    },
    'bigram': {
        'tags' : ['loss_bigram'],
        'labels' : [r'$L^{(bigram)}| _{\sim trigg}$'],

        'lstyle' : ['-'],
        'color': 'blue'
    },
    'induction': {
        'tags' : ['loss_ind'],
        'labels' : [r'$L^{(ind)}|_{trigg}$'],
        'lstyle' : ['-'],
        'color': 'red'
    },
    'full_trigg': {
        'tags' : ['loss_ind'],
        'labels' : [r'$L^{(total)}|_{trigg}$'],
        'lstyle' : ['-'],
        'color': 'green'}
}

kl_configs = {
    'full': {
        'tags' : ['kl_b_total','kl_b_bigram'],
        'labels' : [r'$\pi_b | P_\theta^{(total)}$',r'$\pi_b | P_\theta^{(bigram)}$'],
        'lstyle' : ['-','--'],
        'color': 'black'
            },
    'bigram': {
        'tags' : ['kl_b_bigram'],
        'labels' : [r'$\pi_b | P_\theta^{(bigram)}$'],
        'lstyle' : ['-'],
        'color': 'blue'
    },
    'induction': {
        'tags' : [],
        'labels' : [],
        'lstyle' : [],
        'color': 'red'
    },
    'full_trigg': {
        'tags': [],
        'labels': [],
        'lstyle': [],
        'color': 'green'}
}


# Add page title and sidebar
ui.page_opts(title="Dual Task Dashboard", fillable=True)
with ui.sidebar():
    ui.input_selectize(id="bigram_config", label="Select Bigram Configuration", choices=configs, selected="dirichlet")
    # ui.input_selectize(id="path", label="Select Path", choices=["full", "induction", "bigram"], selected="full")
    ui.input_slider(id="K",label="Number of Triggers", min=10, max=20, value=15, pre="",step=5)
    # ui.input_slider(id="t_id",label="Time id for Attention", min=0, max=9, value=9, pre="",step=1)
    ui.input_checkbox_group("conf_to_plot", "Configs to Plot", paths, selected=["full"], inline=True)
    # ui.input_action_button("reset", "Reset filters")
    # input to select log scale or not
    ui.input_checkbox("log_scale", "Log scale for steps", value=False)




#---------------------------

with ui.layout_columns():#(col_widths=[1,1,1,1]):
    with ui.card(full_screen=True,fill=False):
        ui.card_header("Dual Task Metrics")
        @render.plot
        def plot1():
            conf = input.bigram_config()
            K = input.K()
            paths = input.conf_to_plot()
            fig , axes = create_fig(ncols=3,size='double',h=0.2)
            for i , path in enumerate(paths):
                time = results_scalar[conf][K][path]['step']    
                cfg = loss_configs[path]
                for i , tag in enumerate(cfg['tags']):
                    axes[0].plot(time, results_scalar[conf][K][path][tag],label=cfg['labels'][i],ls=cfg['lstyle'][i],color=cfg['color'])
                axes[0].legend()   
                axes[0].set_title('Losses')
                axes[0].set_xlabel('Steps')
                axes[1].plot(time, results_scalar[conf][K][path]['top1_accuracy'],label=path.capitalize(),ls='-',color=cfg['color'])
                axes[1].legend()
                axes[1].set_title('Top-1 Accuracy')
                axes[1].set_xlabel('Steps')

                cfg = kl_configs[path]
                for i , tag in enumerate(cfg['tags']):
                    axes[2].plot(time, results_scalar[conf][K][path][tag],label=cfg['labels'][i],ls=cfg['lstyle'][i],color=cfg['color'])
                axes[2].legend()
                axes[2].set_title('KL Divergence')
                axes[2].set_xlabel('Steps')
            
            if input.log_scale():
                axes[0].set_xscale('log')
                axes[0].set_xlim(1, time[-1])


            return fig
        ui.card_footer(f"""
                       Black: 'full' model train on all inputs.\n
                        Red: 'induction' model trained only on triggers.\n
                        Blue: 'bigram' model trained only on non-triggers.\n
                        Green: 'full' model trained only on triggers.
                       """)

####

# 'kl_b_total', 'kl_b_bigram',
# 'logit_std_total', 'logit_std_bigram', 'logit_std_induction',

