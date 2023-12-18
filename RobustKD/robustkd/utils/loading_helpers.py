import torch
import os

import numpy as np
import random

import json


def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def wordify(string):
    word = string.replace('_', ' ')
    return word


def make_descriptor_sentence(descriptor):
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith(
            'typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"



def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor


def load_gpt_descriptions(filename, classes_to_load=None, category_name_inclusion='prepend',
                          apply_descriptor_modification=True, before_text="", between_text=', ', after_text=""):
    gpt_descriptions_unordered = load_json(filename)
    unmodify_dict = {}

    if classes_to_load is not None:
        gpt_descriptions = {c: gpt_descriptions_unordered[c] for c in classes_to_load}
    else:
        gpt_descriptions = gpt_descriptions_unordered
    if category_name_inclusion is not None:
        if classes_to_load is not None:
            keys_to_remove = [k for k in gpt_descriptions.keys() if k not in classes_to_load]
            for k in keys_to_remove:
                print(f"Skipping descriptions for \"{k}\", not in classes to load")
                gpt_descriptions.pop(k)

        for i, (k, v) in enumerate(gpt_descriptions.items()):
            if len(v) == 0:
                v = ['']

            word_to_add = wordify(k)

            if category_name_inclusion == 'append':
                build_descriptor_string = lambda \
                        item: f"{modify_descriptor(item, apply_descriptor_modification)}{between_text}{word_to_add}"
            elif category_name_inclusion == 'prepend':
                build_descriptor_string = lambda \
                        item: f"{before_text}{word_to_add}{between_text}{modify_descriptor(item, apply_descriptor_modification)}{after_text}"
            else:
                build_descriptor_string = lambda item: modify_descriptor(item, apply_descriptor_modification)

            unmodify_dict[k] = {build_descriptor_string(item): item for item in v}

            gpt_descriptions[k] = [build_descriptor_string(item) for item in v]

            # print an example the first time
            if i == 0:  # verbose and
                print(f"\nExample description for class {k}: \"{gpt_descriptions[k][0]}\"\n")
    return gpt_descriptions
