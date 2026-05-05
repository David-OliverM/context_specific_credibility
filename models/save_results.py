import json
import os

import os
import json
import torch

def save_experiment_metrics(
    *,
    test_pred,
    test_target,
    test_credibility,     # shape: [N, num_modalities]
    metrics,              # dict of torchmetrics
    cred_ind,
    cred_ind_corr_samples,
    experiment_name,
    exp_setup,
    group_tag,
    noise_setting,
    test_noise,
    run_id,
    num_classes,
    save_dir,
    filename="results.json"
):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # ---------- Compute classification metrics ----------
    metric_results = {}
    for name, metric in metrics.items():
        metric_results[name.lower()] = (
            metric.cpu()(test_pred, test_target).item()
        )

    # ---------- Compute credibility ----------
    global_cred = test_credibility.mean(dim=0)
    global_cred_ind = (cred_ind.sum()/cred_ind.shape[0]).item()

    classwise_cred = {}
    classwise_cred_ind = {}
    for cls in range(num_classes):
        idx = (test_target == cls).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        classwise_cred[str(cls)] = (
            test_credibility[idx].mean(dim=0).tolist()
        )
        cred_ind_by_cls = cred_ind[idx]
        classwise_cred_ind[str(cls)] = (
            (cred_ind_by_cls.sum()/cred_ind_by_cls.shape[0]).item()
        )

    # ---------- Final payload ----------
    results = {
        "exp_setup": exp_setup,
        "experiment": experiment_name,
        "group": group_tag,
        "train_noise": noise_setting,
        "test_noise": test_noise,
        "run_id": run_id,
        "metrics": metric_results,
        "credibility": {
            "global": global_cred.tolist(),
            "per_class": classwise_cred
        },
        "cred_ind":{
            "global": global_cred_ind,
            "per_cls": classwise_cred_ind,
            "corrupt_samples_per_cls": cred_ind_corr_samples,
        }
    }

    # ---------- Append or overwrite ----------
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(results)

    with open(filepath, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"✅ Saved experiment results → {filepath}")





def save_metric_json(
        metric_name,        # e.g. "Accuracy", "AUROC"
        group_tag,
        experiment_name,    # e.g. "CSPN"
        noise_setting,      # integer 0-3
        run_id,             
        multimodal,
        unimodal1,
        unimodal2,
        file="scores_noise_on_one.json"
    ):
    filename = os.path.join("/path/to/dir", file)
    print(f"Saving results to {os.path.abspath(filename)}")
    # Load existing JSON or create new
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Ensure metric exists
    if metric_name not in results:
        results[metric_name] = {}

    noise_key = f"noise_{noise_setting}"

    # Ensure noise setting exists
    if noise_key not in results[metric_name]:
        results[metric_name][noise_key] = {}
    
    #Ensure group_tag exists
    if group_tag not in results[metric_name][noise_key]:
        results[metric_name][noise_key][group_tag] = {}

    # Ensure experiment exists
    if experiment_name not in results[metric_name][noise_key][group_tag]:
        results[metric_name][noise_key][group_tag][experiment_name] = {}

    if run_id not in results[metric_name][noise_key][group_tag][experiment_name]:
        results[metric_name][noise_key][group_tag][experiment_name][run_id] = {
        "multimodal": multimodal,
        "unimodal1": unimodal1,
        "unimodal2": unimodal2
    }

    # Save back to file
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved → {metric_name} / {noise_key} / {group_tag} / {experiment_name} / run={run_id}")



def save_modality_wise_scores_json(
        metric_name,        # e.g. "Accuracy", "AUROC"
        group_tag,
        experiment_name,    # e.g. "CSPN"
        noise_setting,      # integer 0-3
        run_id,     
        corrupted_modality,        
        multimodal = None,
        unimodal1 = None,
        unimodal2 = None,
        credibility = None,
        file="scores_modality_wise.json"
    ):
    filename = os.path.join("/path/to/dir", file)
    print(f"Saving results to {os.path.abspath(filename)}")
    # Load existing JSON or create new
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # Ensure metric exists
    if metric_name not in results:
        results[metric_name] = {}

    noise_key = f"noise_{noise_setting}"

    # Ensure noise setting exists
    if noise_key not in results[metric_name]:
        results[metric_name][noise_key] = {}
    
    #Ensure group_tag exists
    if group_tag not in results[metric_name][noise_key]:
        results[metric_name][noise_key][group_tag] = {}

    # Ensure experiment exists
    if experiment_name not in results[metric_name][noise_key][group_tag]:
        results[metric_name][noise_key][group_tag][experiment_name] = {}

    if str(run_id) not in results[metric_name][noise_key][group_tag][experiment_name]:
        results[metric_name][noise_key][group_tag][experiment_name][str(run_id)] = {}
    
    if metric_name == 'credibility':
        if corrupted_modality not in results[metric_name][noise_key][group_tag][experiment_name][str(run_id)]:
            results[metric_name][noise_key][group_tag][experiment_name][str(run_id)][corrupted_modality] = {
            "credibility": credibility
        }
    else:
        if corrupted_modality not in results[metric_name][noise_key][group_tag][experiment_name][str(run_id)]:
            results[metric_name][noise_key][group_tag][experiment_name][str(run_id)][corrupted_modality] = {
            "multimodal": multimodal,
            "unimodal1": unimodal1,
            "unimodal2": unimodal2
        }

    # Save back to file
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved → {metric_name} / {noise_key} / {group_tag} / {experiment_name} / run={run_id}")





def save_corruption_wise_scores_json(
    metric_name,        # e.g. "Accuracy", "AUROC"
    group_tag,
    experiment_name,    # e.g. "CSPN"
    noise_setting,      # integer 0-3
    run_id,     
    corrupted_modality,        
    multimodal = None,
    unimodal1 = None,
    unimodal2 = None,
    credibility = None,
    file="scores_modality_wise.json"):

    filename = os.path.join("/path/to/dir", file)
    print(f"Saving results to {os.path.abspath(filename)}")
    # Load existing JSON or create new
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results = json.load(f)
    else:
        results = {}

    # if 'credibility' not in results:
    #     results['credibility'] = {noise_key: {group_tag : {experiment_name: {str(run_id): {corrupted_modality: credibility}}}}}

    # if corrupted_modality not in results['credibility'][noise_key][group_tag][experiment_name][str(run_id)]:
    #     results['credibility'][noise_key][group_tag][experiment_name][str(run_id)][corrupted_modality] = credibility

    # Ensure metric exists
    if metric_name not in results:
        results[metric_name] = {}
    

    noise_key = f"noise_{noise_setting}"

    # Ensure noise setting exists
    if noise_key not in results[metric_name]:
        results[metric_name][noise_key] = {}
    
    #Ensure group_tag exists
    if group_tag not in results[metric_name][noise_key]:
        results[metric_name][noise_key][group_tag] = {}

    # Ensure experiment exists
    if experiment_name not in results[metric_name][noise_key][group_tag]:
        results[metric_name][noise_key][group_tag][experiment_name] = {}

    if str(run_id) not in results[metric_name][noise_key][group_tag][experiment_name]:
        results[metric_name][noise_key][group_tag][experiment_name][str(run_id)] = {}
    
    if metric_name == 'credibility':
        if corrupted_modality not in results[metric_name][noise_key][group_tag][experiment_name][str(run_id)]:
            results[metric_name][noise_key][group_tag][experiment_name][str(run_id)][corrupted_modality] = {
            "credibility": credibility
        }
    else:
        if corrupted_modality not in results[metric_name][noise_key][group_tag][experiment_name][str(run_id)]:
            results[metric_name][noise_key][group_tag][experiment_name][str(run_id)][corrupted_modality] = {
            "multimodal": multimodal,
            "unimodal1": unimodal1,
            "unimodal2": unimodal2
        }

    # Save back to file
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved → {metric_name} / {noise_key} / {group_tag} / {experiment_name} / run={run_id}")
