import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            status_code = response.status_code
            if status_code != 200:
                error_info = f"Request error, status code: {status_code}, response: {response.text}"
                logger.info(error_info)
                response.raise_for_status()
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards"):
    return remote_rm_fn(api_url, queries, score_key)


if __name__ == "__main__":
    # test utils
    url = "http://127.0.0.1:5000/get_reward"
    query="""<|im_start|>system
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|im_end|>
<|im_start|>user
How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?<|im_end|>
<|im_start|>assistant
<think> Since $1+1=2$ </think><answer>$2$</answer><｜end▁of▁sentence｜>"""
    score = remote_rm_fn(url, [query])
    print(score)
