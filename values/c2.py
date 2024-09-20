#C2: Uncertainty Measure

### AU Measure: aleatoric uncertainty ~ data-related
### EU Measure: epistemic uncertainty ~ model-related
### PU Measure: predictive uncertainty ~ AU + EU

## deterministic model 
# PU Measure: 1 - maximum softmax response(MSR)

## bayesian model
# PU Measure: predictive entropy
# EU Measure: MI(Y,w|x)
# AU Measure: expected entropy

## TTA - random augmentation variable T
# PU Measure: predictive entropy
# EU Measure: MI(Y,T|x)
# AU Measure: expected entropy

## SSN
# PU Measure: predictive entropy
# EU Measure: expected entropy
# AU Measure: MI(Y,Z|x)
import numpy as np


def sigmoid_to_softmax(sig_pred):
    if len(sig_pred.shape) != 2:
        raise ValueError("Array shape is not appropriate. The dimension must be 2")
    else:
        return np.stack([sig_pred, 1 - sig_pred], 0)


## deterministic model 
# PU Measure: 1 - maximum softmax response(MSR)

def calculate_one_minus_msr(softmax_pred: np.array):
    max_softmax = softmax_pred[0].max(axis=0)
    pred_entropy = 1 - max_softmax
    return pred_entropy


## bayesian model
# PU Measure: predictive entropy
# EU Measure: MI(Y,w|x)
# AU Measure: expected entropy

def calculate_uncertainty(softmax_preds: np.array, ssn: bool = False):
    uncertainty_dict = {}
    mean_softmax = np.mean(softmax_preds, 0)
    pred_entropy = np.zeros(softmax_preds.shape[2:])
    for y in range(mean_softmax.shape[0]):
        pred_entropy_class = mean_softmax[y] * np.log(mean_softmax[y])
        nan_pos = np.isnan(pred_entropy_class)
        pred_entropy[~nan_pos] += pred_entropy_class[~nan_pos]
    pred_entropy *= -1
    expected_entropy = np.zeros((softmax_preds.shape[0], *softmax_preds.shape[2:]))
    for pred in range(softmax_preds.shape[0]):
        entropy = np.zeros(softmax_preds.shape[2:])
        for y in range(softmax_preds.shape[1]):
            entropy_class = softmax_preds[pred, y] * np.log(softmax_preds[pred, y])
            nan_pos = np.isnan(entropy_class)
            entropy[~nan_pos] += entropy_class[~nan_pos]
        entropy *= -1
        expected_entropy[pred] = entropy
    expected_entropy = np.mean(expected_entropy, 0)
    mutual_information = pred_entropy - expected_entropy
    uncertainty_dict["pred_entropy"] = pred_entropy
    if not ssn:
        uncertainty_dict["aleatoric_uncertainty"] = expected_entropy
        uncertainty_dict["epistemic_uncertainty"] = mutual_information
    else:
        print("mutual information is aleatoric unc")
        uncertainty_dict["aleatoric_uncertainty"] = mutual_information
        uncertainty_dict["epistemic_uncertainty"] = expected_entropy
    # value["softmax_pred"] = np.mean(value["softmax_pred"], axis=0)
    return uncertainty_dict