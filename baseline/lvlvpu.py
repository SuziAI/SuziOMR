import cv2
import dataclasses
import json
import numpy as np
import os


@dataclasses.dataclass
class ExtendedLvlv:
    HUANGZHONG: str = "HUANGZHONG"
    DALV: str = "DALV"
    TAICU: str = "TAICU"
    JIAZHONG: str = "JIAZHONG"
    GUXIAN: str = "GUXIAN"
    ZHONGLV: str = "ZHONGLV"
    RUIBIN: str = "RUIBIN"
    LINZHONG: str = "LINZHONG"
    YIZE: str = "YIZE"
    NANLV: str = "NANLV"
    WUYI: str = "WUYI"
    YINGZHONG: str = "YINGZHONG"
    HUANGZHONG_QING: str = "HUANGZHONG_QING"
    DALV_QING: str = "DALV_QING"
    TAICU_QING: str = "TAICU_QING"
    JIAZHONG_QING: str = "JIAZHONG_QING"
    ZHE_ZI: str = "ZHE_ZI"

    @classmethod
    def to_name(cls, lvlv):
        return {
            cls.HUANGZHONG: "黃",
            cls.DALV: "大",
            cls.TAICU: "太",
            cls.JIAZHONG: "夾",
            cls.GUXIAN: "姑",
            cls.ZHONGLV: "仲",
            cls.RUIBIN: "蕤",
            cls.LINZHONG: "林",
            cls.YIZE: "夷",
            cls.NANLV: "南",
            cls.WUYI: "無",
            cls.YINGZHONG: "應",
            cls.HUANGZHONG_QING: "清黃",
            cls.DALV_QING: "清大",
            cls.TAICU_QING: "清太",
            cls.JIAZHONG_QING: "清夹",
            cls.ZHE_ZI: "字折",
        }[lvlv]

    @classmethod
    def from_name(cls, lvlv):
        return {
            "黃": cls.HUANGZHONG,
            "大": cls.DALV,
            "太": cls.TAICU,
            "夾": cls.JIAZHONG,
            "姑": cls.GUXIAN,
            "仲": cls.ZHONGLV,
            "蕤": cls.RUIBIN,
            "林": cls.LINZHONG,
            "夷": cls.YIZE,
            "南": cls.NANLV,
            "無": cls.WUYI,
            "應": cls.YINGZHONG,
            "清黃": cls.HUANGZHONG_QING,
            "清大": cls.DALV_QING,
            "清太": cls.TAICU_QING,
            "清夹": cls.JIAZHONG_QING,
            "字折": cls.ZHE_ZI,
        }[lvlv]

    @classmethod
    def class_to_name(cls, idx):
        return ExtendedLvlv.to_name(dataclasses.astuple(ExtendedLvlv())[idx])

    @classmethod
    def from_class(cls, idx):
        return dataclasses.astuple(ExtendedLvlv())[idx]

    @classmethod
    def to_class(cls, extended_lvlv):
        try:
            return dataclasses.astuple(ExtendedLvlv()).index(extended_lvlv)
        except ValueError:
            return 7  # LINZHONG

    @classmethod
    def name_to_class(cls, name):
        return ExtendedLvlv.to_class(ExtendedLvlv.from_name(name))


## this function takes the path_to_folder (i.e., the folder where the dataset.json is in)
## and returns a list of the dataset entries. Each entry consists of the keys:
##     "image_path":
##     "type": The type of the box (in our case, this is 'Music' only)
##     "annotation": The annotation string
##     "image": The image as uint8 array representation
##     "is_simple": This is True if the notation is "simple notation" as opposed to "composite notation"
def open_lvlvpu_dataset(path_to_folder):
    path_to_json = os.path.join(path_to_folder, "dataset.json")
    with open(path_to_json) as file:
        dataset_json = json.load(file)
        output_list = []

        for idx in range(len(dataset_json)):
            if dataset_json[idx]["notation_type"] != "Lvlvpu" or dataset_json[idx]["annotation"]["pitch"] == "None" or \
                    dataset_json[idx]["annotation"]["pitch"] is None:
                continue

            temp_dict = {}
            temp_dict["image"] = cv2.imread(os.path.join(path_to_folder, dataset_json[idx]["image_path"]),
                                            cv2.IMREAD_GRAYSCALE)
            temp_dict["annotation"] = ExtendedLvlv.to_class(dataset_json[idx]["annotation"]["pitch"])
            temp_dict["edition"] = os.path.basename(dataset_json[idx]["image_path"]).split("_")[0]
            temp_dict["image_id"] = "_".join(dataset_json[idx]["image_path"].split("_")[:-1])
            temp_dict["image_path"] = os.path.basename(dataset_json[idx]["image_path"])
            output_list.append(temp_dict)
    return output_list


def split_training_validation_by_class_lvlvpu(dataset, percentage):
    sorted_by_label = {}
    for entry in dataset:
        class_id = ExtendedLvlv.to_class(entry["annotation"])
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