import numpy as np
import torch
from scipy.stats import mstats
import jitsdp.constants as const


def loss(classifier, dataloader, criterion):
    with torch.no_grad():
        classifier.eval()
        loss = 0
        for inputs, targets in dataloader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = classifier(inputs.float())
            batch_loss = criterion(outputs.squeeze(), targets.float())
            loss += batch_loss.item()

        return loss / len(dataloader)


def __recalls(targets, predictions):
    confusion_matrix, _, _ = np.histogram2d(
        targets, predictions, bins=const.N_CLASSES)
    recalls = np.diag(confusion_matrix) / \
        ((np.sum(confusion_matrix, axis=1)) + 1e-12)
    return recalls

def gmean_recalls(targets, predictions):
    recalls = __recalls(targets, predictions)
    return mstats.gmean(recalls), recalls

def classifier_gmean_recalls(classifier, dataloader):
    if torch.cuda.is_available():
        classifier = classifier.cuda()

    with torch.no_grad():
        classifier.eval()
        recalls = np.zeros((const.N_CLASSES))
        for inputs, targets in dataloader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = classifier(inputs.float())
            predictions = torch.round(outputs).int()
            predictions = predictions.view(predictions.shape[0])
            recalls += __recalls(targets.detach().cpu().numpy(), predictions.detach().cpu().numpy())
        recalls = recalls / len(dataloader)
        return mstats.gmean(recalls), recalls


def classifier_gmean(classifier, dataloader):
    gmean, _ = classifier_gmean_recalls(classifier, dataloader)
    return gmean
