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


configs = ['dirichlet_full',
          'dirichlet_spiked',
          'zipf_spiked'][:2]

Ks = [5,10,15,20,30]
if 'zipf_spiked' in configs:
    Ks = [5,10,15]

metrics = {
    'time': ['step','model_steps'],
    'loss' : ['loss','loss_b_tot','loss_b_filt','loss_ind_tot','loss_ind_filt',
              'entropy_Pb', 'max_entropy', 'opt_loss_dual', 'opt_loss_ind', 'opt_loss_b'],
    'accuracy' : ['accuracy', 'accuracy_top3'],
    'kl' : ['kl_b_WF','kl_b_full','kl_full_u']
}

# loss_b_tot: (100,)
# loss_b_filt: (100,)
# loss_ind_tot: (100,)
# loss_ind_filt: 

metrics_settings = {
    'loss' : {'label': 'Loss','color':'black','lw':2},
    'loss_b_tot' : {'label': r'$L_{b}^{tot}$','color':'purple'},
    'loss_b_filt' : {'label': r'$L_{b}^{filt}$','color':'darkgreen'},
    'loss_ind_tot' : {'label': r'$L_{ind}^{tot}$','color':'orange'},
    'loss_ind_filt' : {'label': r'$L_{ind}^{filt}$','color':'red'},
    'entropy_Pb' : {'label': r'$H(\pi_b)$','color':'blue'},
    'max_entropy' : {'label': r'Max Entropy','color':'gray'},
    'opt_loss_dual' : {'label': r'Optimal $\Phi^{tot}$','color':'darkgreen'},
    'opt_loss_ind' : {'label': r'Optimal $\Phi^{ind}$','color':'orange'},
    'opt_loss_b' : {'label': r'Optimal $\Phi^{b}$','color':'darkblue'},
    'accuracy' : {'label': 'Top-1','color':'black'},
    'accuracy_top3' : {'label': 'Top-3','color':'orange'},
    'kl_b_full' : {'label': r'$D_{KL}(\pi_b|P_\theta)$','color':'purple'},
    'kl_full_u' : {'label': r'$D_{KL}(\pi_u|P_\theta)$','color':'darkgreen'},
    'kl_b_WF' : {'label': r'$D_{KL}(\pi_b|P_{WF})$','color':'orange'},
}
results = get_scalar_measurements(params, Ks,configs,metrics)
#results -> results[fix_trig][freq_trig][K][metric_name] = data[metric_name]
print('exit loading data')
print(results.keys())
# exit() 
# shiny run --reload --launch-browser scripts/dash_test.py

extra_losses = ['loss_ind_filt','loss_b_filt','loss_ind_tot','loss_b_tot']


ui.input_selectize(
    id = "K", 
    label = "Select number of Triggers",
    choices = [str(k) for k in Ks],
    selected = str(Ks[1])
)

ui.input_selectize(
    id = "conf",
    label = "Select Configuration",
    choices = configs,
    selected = configs[0]
)

ui.input_selectize(
    id = "losses",
    label = "Select Extra Losses to Plot",
    choices = extra_losses,
    selected = extra_losses[:0],
    multiple = True
)

ui.input_radio_buttons(
    id="yes_no_option",
    label="Plot reference losses",  # Label for the Yes/No option
    choices=["Yes", "No"],  # Options to display
    selected="No"  # Default selection
)

with ui.card(full_screen=True):
    ui.card_header("A card with a header")
    @render.plot
    def plot():
        fig , axes = create_fig(ncols=3,size='double',h=0.2 )

        K = int(input.K())
        conf = input.conf()
        extra_losses_to_plot = input.losses()
        button_option = input.yes_no_option().lower() == 'yes'

        data = results[conf][K]

        time = data['time']['step'] 
        for i , panel_name in enumerate(['loss','accuracy','kl']):
            ax = axes[i]
            data_panel = data[panel_name]
            if panel_name == 'loss':
                print('data_panel keys before filtering:', data_panel.keys())
                
                if button_option:
                    ax.axhline(data_panel['opt_loss_dual'], color='darkgreen', ls='--', label=r'$\Phi^{tot}$', lw=1)
                    ax.axhline(data_panel['opt_loss_ind'], color='orange', ls='--', label=r'$\Phi^{ind}$', lw=1)
                    ax.axhline(data_panel['opt_loss_b'], color='darkblue', ls='--', label=r'$\Phi^{b}$', lw=1)
                    ax.axhline(data_panel['entropy_Pb'], color='blue', ls='--', label=r'$H(\pi_b)$', lw=1)
                else:
                    ax.axhline(data_panel['entropy_Pb'], color='blue', ls='--', label=r'$H(\pi_b)$', lw=1)
                    ax.axhline(data_panel['max_entropy'], color='gray', ls='--', lw=1,label=r'$\log V$')

                data_panel = {k:v for k,v in data_panel.items() if k in extra_losses_to_plot or k=='loss'}
            
            
            plot_panel(ax,time,data_panel,metrics_settings,title=panel_name.capitalize())

            if panel_name == 'loss':
                ax.legend(loc=(-0.7, 0.1),fontsize=9)

            elif panel_name == 'accuracy':
                ax.legend(fontsize=9)
                ax.set_ylim(0,1)
            elif panel_name == 'kl':
                ax.legend(loc=(1.1,0.2),fontsize=9)
                ax.set_ylim(0,None)
        return fig

    