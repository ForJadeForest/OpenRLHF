import numbers
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openrlhf.utils.utils import convert_token_to_id
from .utils import zero_pad_sequences
from ..trainer.ppo_utils.data_processor import DATA_PROCESSOR_MAP

class ProcessRewardDataset(Dataset):
    """
    Dataset for process reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        processor: Callable,
        max_length: int,
        strategy,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        self.tokenizer = processor.tokenizer
        self.data_processor = DATA_PROCESSOR_MAP[type(processor)](processor)
        # chat_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.label_key = getattr(self.strategy.args, "label_key", None)
        self.placeholder_token = getattr(self.strategy.args, "placeholder_token", None)
        self.reward_tokens = getattr(self.strategy.args, "reward_tokens", None)

        self.placeholder_token_id = convert_token_to_id(self.placeholder_token, self.tokenizer)

        # Store the processed data in class attributes
        self.inputs = dataset[self.input_key]
        self.labels = dataset[self.label_key]

    def __len__(self):
        length = len(self.inputs)
        return length

    def __getitem__(self, idx):
        inputs = self.data_processor(
            self.inputs[idx],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = inputs.pop("input_ids")[0]
        label_values = self.labels[idx]
        assert isinstance(label_values, list), "labels should be a list of strings or numbers"
        if isinstance(label_values[0], str):
            label_tokens = []
            for label in label_values:
                assert (
                    self.reward_tokens is None or label in self.reward_tokens
                ), f"label should be in reward tokens {self.reward_tokens}, got {label}"
                label_tokens.append(convert_token_to_id(label, self.tokenizer))

            labels = torch.full_like(input_ids, -100)
            labels[input_ids == self.placeholder_token_id] = torch.tensor(label_tokens, dtype=input_ids.dtype)
        else:
            assert isinstance(label_values[0], numbers.Number), "labels should be a list of strings or numbers"
            labels = torch.full_like(input_ids, -100, dtype=torch.float)
            labels[input_ids == self.placeholder_token_id] = torch.tensor(label_values, dtype=torch.float)
        attention_mask = inputs.pop("attention_mask")[0]
        visual_inputs = inputs
        return (
            input_ids,
            attention_mask,
            labels,
            visual_inputs
        )

    def collate_fn(self, item_list):
        input_ids = []
        input_masks = []
        label_ids = []
        visual_inputs = []
        for input_id, input_mask, label_id, visual_input in item_list:
            input_ids.append(input_id)
            input_masks.append(input_mask)
            label_ids.append(label_id)
            visual_inputs.append(visual_input)

        padding_side = "right"
        input_ids = zero_pad_sequences(input_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        input_masks = zero_pad_sequences(input_masks, side=padding_side)
        label_ids = zero_pad_sequences(label_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        visual_inputs = self.data_processor.make_input_batch(visual_inputs)
        return input_ids, input_masks, label_ids, visual_inputs

    def packing_collate_fn(self, item_list):
        raise NotImplementedError("Packing collate function is not implemented yet")
        input_ids = []
        input_att_masks = []
        input_seq_lens = []
        label_ids = []
        index = 1
        for input_id, input_mask, label_id in item_list:
            input_ids.append(input_id.flatten())
            input_att_masks.append(torch.full_like(input_id.flatten(), index))
            input_seq_lens.append(len(input_id.flatten()))

            label_ids.append(label_id.flatten())
            index += 1

        packed_input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(input_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = input_seq_lens
        packed_label_ids = torch.cat(label_ids, dim=0).unsqueeze(0)

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, packed_label_ids
