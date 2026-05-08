import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json


import torch
def make_model_diagrams(outputs, labels, n_bins=10, prefix = ''):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    # softmaxes = torch.nn.functional.softmax(outputs, 1)
    softmaxes = outputs
    confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)
    
    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
    
    plt.figure(figsize=(8, 8))
    gap = np.array(bin_scores - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    # bin_corrects = np.nan_to_num(np.array([bin_correct.numpy()  for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')

    ece = _calculate_ece(softmaxes, labels)

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    plt.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    plt.title("Reliability Diagram", size=22)
    plt.ylabel("Accuracy",  size=18)
    plt.xlabel("Confidence",  size=18)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(f'{prefix}_reliability_diagram.png')
    plt.close()  # was plt.show() — blocks the entire pipeline on any host with an interactive
                 # matplotlib backend. plt.savefig() above is what actually persists the figure.
    return ece


def _calculate_ece(logits, labels, n_bins=10):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # softmaxes = F.softmax(logits, dim=1)
    softmaxes = logits
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()





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
    filename = os.path.join("/home/pxt220000/Projects/CS_Credibility", file)
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
