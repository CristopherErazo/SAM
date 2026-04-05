import time
from omegaconf import OmegaConf

import torch
import numpy as np

from sam.config import TrainerArgs
from sam.dataset_new import get_distributions , get_triggers , generate_dual_task_batch, generate_dual_task_batch_optimized
from sam.dual_model_new import DualModel , initialize_model
from sam.evaluation_new import compute_entropies_and_dkl , optimal_pop_losses, Evaluator, IC_TopKAccuracy, KLMetric, LossMetric

from sam.dataset import generate_dual_task_batch_fast

from line_profiler import profile

@profile

def main():
    # Initialize full config
    defaults = OmegaConf.structured(TrainerArgs())
    cli_config = OmegaConf.from_cli()

    cfg = OmegaConf.merge(defaults, cli_config)
    
    # Print the parameters
    print("Experiment Configuration:")
    print(OmegaConf.to_yaml(cfg))

    

    # Get all parameters in a flat dictionary
    flat_dict = OmegaConf.to_container(cfg, resolve=True)
    print("Flat dictionary of parameters:")
    print(flat_dict)

    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Define distributions
    distributions = get_distributions(cfg.data_args, cfg.model_args.vocab_size, device=device)

    # Compute entropies and reference KL divergences and print
    dist_measures = compute_entropies_and_dkl(distributions['P_b'], distributions['P_u'])
    print("\n".join(f"{key}: {value:.4f}" for key, value in dist_measures.items()))

    # Initialize Model
    model = DualModel(cfg.model_args).to(device)
    model = initialize_model(model)

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
    opt_name = cfg.optim_args.opt.lower()
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

    test_batch = generate_dual_task_batch_optimized(cfg.data_args.test_size,
                                          cfg.model_args.seq_len,
                                          cfg.data_args.K,
                                          distributions,
                                          trigger_set=trigger_set,
                                          device=device)



    opt_losses = optimal_pop_losses(test_batch, P_b=distributions['P_b'])

    print("\n".join(f"{key}: {value:.4f}" for key, value in opt_losses.items()))



    metrics = [
        IC_TopKAccuracy(1),
        IC_TopKAccuracy(3),
        KLMetric(),
        LossMetric()
        ]




    evaluator = Evaluator(metrics)

    results_log = {m.name: [] for m in metrics}
    results_log["step"] = []

    # Training loop parameters
    total_steps = cfg.extra_args.total_steps
    nprints = cfg.extra_args.n_prints
    nprints_model = cfg.extra_args.n_prints_model

    
    if cfg.extra_args.print_scale == 'log':
        print_total_steps = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints).astype(int))
        print_total_steps_model = np.unique(np.logspace(-0.01, np.log10(total_steps-1), num=nprints_model).astype(int))
    elif cfg.extra_args.print_scale == 'linear':
        print_total_steps = np.linspace(0, total_steps-1, num=nprints).astype(int)
        print_total_steps_model = np.linspace(0, total_steps-1, num=nprints_model).astype(int)

    t0 = time.time()
    for step in range(total_steps):

        if step in print_total_steps:
            res = evaluator.evaluate(model, test_batch, loss_fn, distributions['P_b'], distributions['P_u'])

            results_log["step"].append(step)
            for k, v in res.items():
                results_log[k].append(v)
            print(f"Step {step}/{total_steps}: " + ", ".join(f"{k}: {v:.4f}" for k, v in res.items()))
        
        batch = generate_dual_task_batch_optimized(cfg.data_args.batch_size,
                                          cfg.model_args.seq_len,
                                          cfg.data_args.K,
                                          distributions,
                                          trigger_set=trigger_set,
                                         device=device)
        # Evaluate model on the dual task
        sequence = batch['sequence'].to(device) # shape (batch_size, seq_len + 1)
        input = sequence[:, :-1] # shape (batch_size, seq_len)
        target = sequence[:, 1:] # shape (batch_size, seq_len)
        mask = batch['mask'].to(device) # shape (batch_size, seq_len, seq_len)

        logits = model(input, mask, path='full') # shape (batch_size, seq_len, vocab_size)

        # Compute loss and update model
        loss = loss_fn(logits.reshape(-1, cfg.model_args.vocab_size), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t1 = time.time()
    print(f"Training completed in {t1-t0:.2f} seconds = {((t1-t0)/60):.2f} minutes")

    for key in results_log:
        results_log[key] = np.array(results_log[key])
        print(f"{key}: {results_log[key].shape}")

if __name__ == "__main__":
    main()