from image_manipulation import Editions
import numpy as np


def get_means_and_confidences(model_dict, validation_edition=None):
    values = {}
    for test_edition in [Editions.LU, Editions.ZHANG, Editions.SIKU, Editions.ZHU, Editions.SHANGHAI]:
        test_list = model_dict[test_edition]
        values[test_edition] = {}
        values[test_edition]["mean"] = np.mean(test_list)
        values[test_edition]["ci_length"] = 1.96 * np.std(test_list)/np.sqrt(len(test_list))
        values[test_edition]["std"] = np.std(test_list)
    return values