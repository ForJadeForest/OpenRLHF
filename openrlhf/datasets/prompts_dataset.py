from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input") -> str:
    prompt = data[input_key]
    if input_template:
        prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        processor,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.processor = processor

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        answer_key = getattr(self.strategy.args, "answer_key", None)
        if answer_key is not None:
            self.answers = []
        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key)
            if answer_key is not None:
                answer = data[answer_key]
                self.answers.append(answer)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        if self.answers is not None:
            return self.prompts[idx], self.answers[idx]
        else:
            return self.prompts[idx]
