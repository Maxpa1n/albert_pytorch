# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os
import torch
import random
import tqdm

random.seed(45)

from callback.progressbar import ProgressBar
from .utiles_multiple_choice import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def balance_truncate_seq_pair(tokens_a, tokens_b, tokens_c=[], max_length=512):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        len_list = [len(tokens_a), len(tokens_b), len(tokens_c)]
        if total_length <= max_length:
            break
        else:
            if len_list.index(max(len_list)) == 0:
                tokens_a.pop()
            elif len_list.index(max(len_list)) == 1:
                tokens_b.pop()
            else:
                tokens_c.pop()


def multiple_convert_examples_to_features(examples, tokenizer,
                                          max_length=512,
                                          task=None,
                                          label_list=None,
                                          output_mode=None,
                                          pad_on_left=False,
                                          pad_token=0,
                                          pad_token_segment_id=0,
                                          mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = multiple_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = multiple_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        context_tokens = tokenizer.tokenize(example.contexts)
        start_ending_tokens = tokenizer.tokenize(example.question)
        choices_features = []
        input_len = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = tokenizer.tokenize(ending)

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"

            balance_truncate_seq_pair(context_tokens_choice, start_ending_tokens, ending_tokens, max_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + start_ending_tokens + ending_tokens + ["[SEP]"]

            token_type_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (
                    len(start_ending_tokens + ending_tokens) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            input_len.append(len(input_ids))
            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                                max_length)
            choices_features.append((tokens, input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (text_token, input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("text_token: {}".format(' '.join(map(str, text_token))))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
                input_len=max(input_len)
            )
        )

    return features


class OpmrcProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "oqmrc_trainingset.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "validationset.json")), "dev")

    def get_test_examples(self, data_dir):
        """?"""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "oqmrc_testa.json")), "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [0, 1, 2]

    def get_anwser(self, anwser, anwser_list):
        for i, v in enumerate(anwser_list):
            if anwser == v:
                return i
            else:
                pass
        return None

    def _create_examples(self, lines, example_type):
        '''
            Creates examples for data
        '''
        datapbar = ProgressBar(n_total=len(lines), desc='create examples')
        examples = []
        for i, line in tqdm.tqdm(enumerate(lines)):
            id = line['query_id']
            context = line['passage']
            query = line['query']
            alternatives = line['alternatives'].split('|')
            random.shuffle(alternatives)
            if example_type == 'test':
                answer = None
            else:
                answer = self.get_anwser(line['answer'], alternatives)  # test 没有这项
            example = InputExample(example_id=id, question=query, contexts=context, endings=alternatives,
                                   label=answer)
            examples.append(example)
            # pbar(step=i)
        return examples


multiple_tasks_num_labels = {
    "opmrc": 3,
}

multiple_output_modes = {
    "opmrc": "classification",
}

multiple_processors = {
    "opmrc": OpmrcProcessor
}
