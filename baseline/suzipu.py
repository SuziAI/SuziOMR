import cv2
import json
import numpy as np
import os

class_to_annotation = {
    "pitch": {0: "HE", 1: "SI", 2: "YI", 3: "SHANG", 4: "GOU", 5: "CHE", 6: "GONG", 7: "FAN", 8: "LIU", 9: "WU",
              10: "GAO_WU"},
    "secondary": {0: None, 1: "DA_DUN", 2: "XIAO_ZHU", 3: "DING_ZHU", 4: "DA_ZHU", 5: "ZHE", 6: "YE"}
}

annotation_to_class = {
    "pitch": {"HE": 0, "SI": 1, "YI": 2, "SHANG": 3, "GOU": 4, "CHE": 5, "GONG": 6, "FAN": 7, "LIU": 8, "WU": 9,
              "GAO_WU": 10},
    "secondary": {None: 0, "DA_DUN": 1, "XIAO_ZHU": 2, "DING_ZHU": 3, "DA_ZHU": 4, "ZHE": 5, "YE": 6}
}

PITCH_DICT = {0: "HE", 1: "SI", 2: "YI", 3: "SHANG", 4: "GOU", 5: "CHE", 6: "GONG", 7: "FAN", 8: "LIU", 9: "WU",
              10: "GAO_WU"}
SECONDARY_DICT = {0: None, 1: "DA_DUN", 2: "XIAO_ZHU", 3: "DING_ZHU", 4: "DA_ZHU", 5: "ZHE", 6: "YE"}

class_to_annotation_dict = {}
idx = 0
for secondary in SECONDARY_DICT.keys():
    for pitch in PITCH_DICT.keys():
        class_to_annotation_dict[idx] = f"{PITCH_DICT[pitch]}, {SECONDARY_DICT[secondary]}"
        idx += 1

class_to_subclasses_dict = {}
idx = 0
for secondary in SECONDARY_DICT.keys():
    for pitch in PITCH_DICT.keys():
        class_to_subclasses_dict[idx] = (pitch, secondary)
        idx += 1


def _class_to_annotation(class_idx):
    return class_to_annotation_dict[class_idx]


def _annotation_to_class(annotation):
    for key in class_to_annotation_dict.keys():
        if class_to_annotation_dict[key] == annotation:
            return key
    raise Exception("Invalid annotation", annotation)


def properties_to_class(pitch, secondary):
    return _annotation_to_class(f"{pitch}, {secondary}")


def class_to_subclasses(class_idx):
    return class_to_subclasses_dict[class_idx]


def individual_labels_to_class(pitch, secondary):
    return properties_to_class(PITCH_DICT[int(pitch)], SECONDARY_DICT[int(secondary)])


## this function takes the path_to_folder (i.e., the folder where the dataset.json is in)
## and returns a list of the dataset entries. Each entry consists of the keys:
##     "image_path":
##     "type": The type of the box (in our case, this is 'Music' only)
##     "annotation": The annotation string
##     "image": The image as uint8 array representation
##     "is_simple": This is True if the notation is "simple notation" as opposed to "composite notation"
def open_suzipu_dataset(path_to_folder):
    path_to_json = os.path.join(path_to_folder, "dataset.json")
    with open(path_to_json) as file:
        dataset_json = json.load(file)
        output_list = []

        for idx in range(len(dataset_json)):
            if dataset_json[idx]["notation_type"] != "Suzipu" or dataset_json[idx]["annotation"] is None or \
                    dataset_json[idx]["annotation"]["pitch"] is None or dataset_json[idx]["annotation"][
                "pitch"] == "None":
                continue
            temp_dict = {}
            temp_dict["image"] = cv2.imread(os.path.join(path_to_folder, dataset_json[idx]["image_path"]),
                                            cv2.IMREAD_GRAYSCALE)
            temp_dict["is_simple"] = True if dataset_json[idx]["annotation"]["secondary"] == None else False
            temp_dict["annotation"] = {}
            temp_dict["annotation"]["pitch"] = annotation_to_class["pitch"][dataset_json[idx]["annotation"]["pitch"]]
            temp_dict["annotation"]["secondary"] = annotation_to_class["secondary"][
                dataset_json[idx]["annotation"]["secondary"]]
            temp_dict["edition"] = os.path.basename(dataset_json[idx]["image_path"]).split("_")[0]
            temp_dict["image_id"] = "_".join(dataset_json[idx]["image_path"].split("_")[:-1])
            temp_dict["image_path"] = os.path.basename(dataset_json[idx]["image_path"])
            output_list.append(temp_dict)
    return output_list


def split_training_validation_by_class_suzipu(dataset, percentage):
    sorted_by_label = {}
    for entry in dataset:
        class_id = individual_labels_to_class(entry["annotation"]["pitch"], entry["annotation"]["secondary"])
        if class_id in sorted_by_label:
            sorted_by_label[class_id].append(entry)
        else:
            sorted_by_label[class_id] = [entry]

    split_by_label = {}
    for label in sorted_by_label.keys():
        indices = [i for i in range(len(sorted_by_label[label]))]
        np.random.shuffle(indices)
        split = int(len(sorted_by_label[label]) * percentage)
        split_by_label[label] = [[sorted_by_label[label][idx] for idx in indices[:split]], [sorted_by_label[label][idx] for idx in indices[split:]]]
    train = [entry[0] for entry in split_by_label.values()]
    validation = [entry[1] for entry in split_by_label.values()]

    train_return = []
    for t in train:
        train_return += t

    validation_return = []
    for v in validation:
        validation_return += v

    return train_return, validation_return


def split_training_validation_by_label_suzipu(dataset, label_type, percentage):
    sorted_by_label = {}
    label = "pitch" if "pitch" in label_type else "secondary"
    for entry in dataset:
        class_id = str(entry["annotation"][label])
        if class_id in sorted_by_label:
            sorted_by_label[class_id].append(entry)
        else:
            sorted_by_label[class_id] = [entry]

    split_by_label = {}
    for label in sorted_by_label.keys():
        indices = [i for i in range(len(sorted_by_label[label]))]
        np.random.shuffle(indices)
        split = int(len(sorted_by_label[label]) * percentage)
        split_by_label[label] = [[sorted_by_label[label][idx] for idx in indices[:split]], [sorted_by_label[label][idx] for idx in indices[split:]]]
    train = [entry[0] for entry in split_by_label.values()]
    validation = [entry[1] for entry in split_by_label.values()]

    train_return = []
    for t in train:
        train_return += t

    validation_return = []
    for v in validation:
        validation_return += v

    return train_return, validation_return