import configparser


config = configparser.ConfigParser()

config["random"] = {"seed": "7", "shuffle": "True"}

config["images"] = {
    "views": "3",
    "planes": "500",
    "cells": "500",
    "standardize": "False",
    "path": "/afs/cern.ch/work/r/rradev/public/vgg_cvn/data",
}

config["dataset"] = {
    "uniform": "False",
    "path": "/afs/cern.ch/user/r/rradev/vgg_cvn/dataset",
    "partition_prefix": "/partition",
    "labels_prefix": "/labels",
}

config["test"] = {
    "output_path": "/output",
    "output_prefix": "/test_info",
    "cut_nue": "0.7",
    "cut_numu": "0.5",
    "cut_nutau": "0.7",
    "cut_nc": "0.7",
    "batch_size": "32",
    "fraction": "0.1",
}

config["train"] = {
    "resume": "False",
    "class_weights_prefix": "/class_weights",
    "batch_size": "32",
    "fraction": "0.8",
    "weighted_loss_function": "False",
}

config["validation"] = {"batch_size": "32", "fraction": "0.1"}


config["model"] = {
    "outputs": 7,
}

config_path = "config.ini"
with open(config_path, "w") as configfile:
    print(f"Config written to: {config_path}")
    config.write(configfile)
