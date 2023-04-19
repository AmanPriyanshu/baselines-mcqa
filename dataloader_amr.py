import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
from transformers.file_utils import PaddingStrategy
import datasets

import torch
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class MultipleChoiceDataset(Dataset):
    """
    PyTorch multiple choice dataset class
    """

    features: List[Dict]

    def __init__(
        self,
        data_args,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        
        dataset = datasets.load_from_disk(data_args.data_set_path)

        tokenizer_name = re.sub('[^a-z]+', ' ', tokenizer.name_or_path).title().replace(' ', '')
        cached_features_file = os.path.join(
            '.cache',
            task,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer_name,
                str(max_seq_length),
                task,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        if not os.path.exists(os.path.join('.cache', task)):
            if not os.path.exists('.cache'):
                os.mkdir('.cache')
            os.mkdir(os.path.join('.cache', task))

        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {task}")
                if mode == Split.dev:
                    examples = dataset['validation']
                elif mode == Split.test:
                    examples = dataset['test']
                elif mode == Split.train:
                    examples = dataset['train']
                
                self.features = convert_examples_to_features(
                    examples,
                    max_seq_length,
                    tokenizer,
                )
                logger.info("Training examples: %s", len(self.features))

                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Dict:
        return self.features[i]


def convert_examples_to_features(
    examples: datasets.Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[Dict]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []

        for ending_idx, ending in enumerate(example['endings']):
            context = example['context']
            if example.get('question') is None:
                inputs = [tokenizer.bos_token] \
                    + context.split() \
                    + [tokenizer.eos_token] \
                    + [tokenizer.eos_token] \
                    + ending.split() \
                    + [tokenizer.eos_token] \

            else:
                question = example['question']

                inputs = [tokenizer.bos_token] \
                    + context.split() \
                    + [tokenizer.eos_token] \
                    + [tokenizer.eos_token] \
                    + question.split() \
                    + ending.split() \
                    + [tokenizer.eos_token] \

            if len(inputs) > max_length:
                logger.error("Input too long: implementation does not support truncate.")

            choices_inputs.append(inputs)

        input_ids = amr_batch_encode(tokenizer, choices_inputs)

        attention_mask = []
        for i in input_ids:
            attention_mask.append([1] * len(i))

        label = example['label']

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,

            }
        )
    return features    

def amr_batch_encode(tokenizer, input_lst):
    res = []
    for itm_lst in input_lst:
        res.append(
            get_ids(tokenizer, itm_lst)
        )

    return res

def get_ids(tokenizer, tokens):
    token_ids = [tokenizer.encoder.get(b, tokenizer.unk_token_id) for b in tokens]
    return token_ids


@dataclass
class CustomDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        
        f = []
        for i in features:
            for id, _ in enumerate(i['input_ids']):
                f.append({
                    'input_ids': i['input_ids'][id],
                    'attention_mask' : i['attention_mask'][id],
                    'label':i['label']
                })

        batch = self.tokenizer.pad(
            f,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        input_len = len(batch['input_ids'][0])

        batch['input_ids'] = torch.reshape(batch['input_ids'] , (batch_size, num_choices, input_len))
        batch['attention_mask'] = torch.reshape(batch['attention_mask'] , (batch_size, num_choices, input_len))

        indices = []
        for idx, v in enumerate(batch['label']):
            if (idx % num_choices) != 0:
                indices.append(idx)

        batch['label'] = th_delete(batch['label'], indices)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        return batch   

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]
