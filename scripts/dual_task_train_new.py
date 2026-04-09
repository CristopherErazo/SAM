import time
from omegaconf import OmegaConf
import pickle

import torch
import numpy as np

from sam.config import TrainerArgs
from sam.dataset_new import get_distributions , get_triggers , generate_dual_task_batch
from sam.dual_model_new import DualModel , initialize_model
from sam.evaluation_new import compute_entropies_and_dkl , optimal_pop_losses, Evaluator, IC_TopKAccuracy, KLMetric, LossMetric, LogitMeanMetric, LogitStdMetric, evaluate_model, get_attention_patterns
from sam.utils import reduced_batch

def main():
    # Initialize full config
    defaults = OmegaConf.structured(TrainerArgs())
    cli_config = OmegaConf.from_cli()

    cfg = OmegaConf.merge(defaults, cli_config)
    
    # Print the parameters
    print("Experiment Configuration:")
    print(OmegaConf.to_yaml(cfg))

    

    # Get all parameters in a flat dictionary and take out some parameters
    flat_dict = OmegaConf.to_container(cfg, resolve=True)
    vocab_size = cfg.model_args.vocab_size
    seq_len = cfg.model_args.seq_len
    opt_name = cfg.optim_args.opt.lower()
    test_size = cfg.data_args.test_size
    batch_size = cfg.data_args.batch_size
    K = cfg.data_args.K
    path = cfg.extra_args.path


    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Define distributions
    distributions = get_distributions(cfg.data_args, vocab_size, device=device)

    # Compute entropies and reference KL divergences and print
    dist_measures = compute_entropies_and_dkl(distributions['P_b'], distributions['P_u'])
    print("\n".join(f"{key}: {value:.4f}" for key, value in dist_measures.items()))

    # Initialize Model
    model = DualModel(cfg.model_args).to(device)
    model = initialize_model(model,path=path)

    # Define triggers
    trigger_set = get_triggers(cfg.data_args, distributions['P_t'])#.to(device)

    # Print trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    kwargs = {'lr': cfg.optim_args.lr,'weight_decay': cfg.optim_args.weight_decay}
    if opt_name == 'sgd':
        print("Using SGD optimizer")
        kwargs['momentum'] = cfg.optim_args.momentum
        optimizer = torch.optim.SGD(trainable_params, **kwargs)
    elif opt_name == 'adam':
        print("Using Adam optimizer")
        optimizer = torch.optim.Adam(trainable_params, **kwargs)
    elif opt_name == 'adamw':
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(trainable_params, **kwargs)
    else:
        raise ValueError("Invalid optimizer type. Options are 'SGD', 'adam', and 'adamW'.")

    test_batch = generate_dual_task_batch(test_size,
                                          seq_len,
                                          K,
                                          distributions,
                                          trigger_set=trigger_set)
                                        #   device=device)



    opt_losses = optimal_pop_losses(test_batch, P_b=distributions['P_b'])

    print("\n".join(f"{key}: {value:.4f}" for key, value in opt_losses.items()))



    metrics = [
        IC_TopKAccuracy(1),
        IC_TopKAccuracy(3),
        LossMetric(name = "loss_total",
                   logits_fn = lambda ctx: ctx.logits[ctx.all],
                   target_fn = lambda ctx: ctx.target[ctx.all],
                   rescale = True),
        LossMetric(name = "loss_bigram",
                   logits_fn = lambda ctx: ctx.logits_bigram[ctx.only_non_triggers],
                   target_fn = lambda ctx: ctx.target[ctx.only_non_triggers], 
                   rescale = False),
        LossMetric(name = "loss_ind",
                   logits_fn = lambda ctx: ctx.logits_induction[ctx.only_triggers],
                   target_fn = lambda ctx: ctx.target[ctx.only_triggers], 
                   rescale = False),    
        KLMetric(name="kl_b_total",
                P_fn=lambda ctx: ctx.P_b[ctx.input],
                Q_fn=lambda ctx: ctx.model_prob,
                ),
        KLMetric(name="kl_b_bigram",
                P_fn=lambda ctx: ctx.P_b[ctx.input],
                Q_fn=lambda ctx: ctx.model_prob_bigram,
                ),    
        LogitStdMetric(name="logit_std_total",
                        logits_fn=lambda ctx: ctx.logits[ctx.all]),
        LogitStdMetric(name="logit_std_bigram",
                        logits_fn=lambda ctx: ctx.logits_bigram[ctx.only_non_triggers]),
        LogitStdMetric(name="logit_std_induction",
                        logits_fn=lambda ctx: ctx.logits_induction[ctx.only_triggers]),

                    
        ]




    evaluator = Evaluator(metrics)

    results_log = {m.name: [] for m in metrics}
    results_log["step"] = []

    matrix_log = {"step": [],"attn1": [], "attn2": []}
    

    # Training loop parameters
    total_steps = cfg.extra_args.total_steps
    nprints = cfg.extra_args.n_prints
    nprints_model = cfg.extra_args.n_prints_model
    print_scale = cfg.extra_args.print_scale

    
    if print_scale == 'log':
        print_total_steps = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints).astype(int))
        print_total_steps_model = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints_model).astype(int))
    elif print_scale == 'linear':
        print_total_steps = np.linspace(0, total_steps-1, num=nprints).astype(int)
        print_total_steps_model = np.linspace(0, total_steps-1, num=nprints_model).astype(int)

    t0 = time.time()
    print(f"Step/{total_steps}\t" + "\t".join(m.name for m in metrics))
    for step in range(total_steps):
        # Evaluations and logging of scalars
        if step in print_total_steps:
            res = evaluator.evaluate(model, test_batch, loss_fn, distributions['P_b'], distributions['P_u'])
            results_log["step"].append(step)
            for k, v in res.items():
                results_log[k].append(v)
            print(f"{step}\t" + "\t".join(f"{v:.4f}" for v in res.values()))
        # Evaluations and logging of attention patterns
        if step in print_total_steps_model:
            print(f'Saving attention patterns at step {step}')
            att_patterns = get_attention_patterns(model, test_batch, path, device, n_test = 5)
            matrix_log["step"].append(step)
            for k, v in att_patterns.items():
                matrix_log[k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)

        
        batch = generate_dual_task_batch(batch_size,
                                          seq_len,
                                          K,
                                          distributions,
                                          trigger_set=trigger_set)
                                        #   device=device)

        loss = evaluate_model(model,batch,loss_fn,path,device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t1 = time.time()
    print(f"Training completed in {t1-t0:.2f} seconds = {((t1-t0)/60):.2f} minutes")

    results_log = {**results_log, **opt_losses, **dist_measures}

    for key in results_log:
        results_log[key] = np.array(results_log[key])
        print(f"{key}: {results_log[key].shape}")
    for key in matrix_log:
        matrix_log[key] = np.array(matrix_log[key])
        print(f"{key}: {matrix_log[key].shape}")
    
    results_log = {**results_log, "exp_config": flat_dict}
    test_batch = reduced_batch(test_batch, n_test=5)
    matrix_log = {**matrix_log, "test_batch": test_batch}

    # Save results as a pickle file
   
    with open(f"./data/{cfg.extra_args.experiment_name}/{cfg.extra_args.file_name}_scalar.pkl", "wb") as f:
        pickle.dump(results_log, f)
    with open(f"./data/{cfg.extra_args.experiment_name}/{cfg.extra_args.file_name}_matrix.pkl", "wb") as f:
        pickle.dump(matrix_log, f)
    

if __name__ == "__main__":
    main()