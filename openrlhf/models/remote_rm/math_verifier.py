from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
import torch
import logging
from datetime import datetime
import argparse
import uvicorn

# 引入相关依赖
from latex2sympy2_extended import NormalizationConfig
from openrlhf.models.remote_rm.math_verifier import LatexExtractionConfig, parse, verify

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class RewardMode:
    STANDARD = "standard"
    R1 = "r1"

class RewardRequest(BaseModel):
    query: List[str]
    answers: Optional[List[str]] = None

def standard_reward(queries: List[str], answers: List[str]) -> List[float]:
    rewards = []
    for idx, (query, answer) in enumerate(zip(queries, answers)):
        try:
            model_answer = query.split('assistant')[-1].strip()
            logger.info(f"Prediction: {query.split('assistant')[-1].strip()}\n\nAnswer: {answer}")
            
            # 解析配置优化
            extraction_config = LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    malformed_operators=False,
                    nits=False,
                    basic_latex=True,
                    equations=True,
                    boxed=True,
                    units=False
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=True
            )
            
            # 标准答案解析
            gold_parsed = parse(
                answer,
                extraction_config=[extraction_config],
                extraction_mode="first_match"
            )
            
            # 模型回答解析
            answer_parsed = parse(
                model_answer,
                extraction_config=[extraction_config],
                extraction_mode="first_match"
            )
            
            # 验证逻辑增强
            if not gold_parsed:
                logger.warning(f"Gold answer parse failed: {answer}")
                rewards.append(0.0)
            else:
                reward = float(verify(answer_parsed, gold_parsed))
                rewards.append(reward)
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            rewards.append(0.0)
            
    return rewards

def r1_reward(queries: List[str], answers: List[str]) -> List[float]:
    """
    R1 mode reward function that combines format check and answer accuracy:
    """
    rewards = []
    for idx, (query, answer) in enumerate(zip(queries, answers)):
        query = query.split("assistant\n")[-1]
        logger.info(f"QUERY: {query}")
        logger.info(f"ANSWER: {answer}")
        try:
            # 1. 检查完整格式
            format_pattern = r'^<think>.*?</think><answer>.*?</answer>$'
            if not re.match(format_pattern, query, re.DOTALL):
                logger.warning(f"Invalid format in query {idx}")
                rewards.append(-1.0)
                continue
            
            # 2. 提取<answer>标签中的内容
            answer_tag_content = re.findall(r'<answer>(.*?)</answer>', query, re.DOTALL)
            model_answer = answer_tag_content[-1].strip()

            # 3. 验证答案正确性
            extraction_config = LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    malformed_operators=False,
                    nits=False,
                    basic_latex=True,
                    equations=True,
                    boxed=True,
                    units=False
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=True
            )
            
            gold_parsed = parse(
                answer,
                extraction_config=[extraction_config],
                extraction_mode="first_match"
            )
            
            answer_parsed = parse(
                model_answer,
                extraction_config=[extraction_config],
                extraction_mode="first_match"
            )
            logger.info(f"Gold answer parse: {gold_parsed}")
            logger.info(f"Answer parsed: {answer_parsed}")
            # 计算最终得分
            if not gold_parsed:
                logger.info(f"Gold answer parse failed: {answer}")
                reward = -0.5
            else:
                is_correct = verify(answer_parsed, gold_parsed)
                reward = 1.0 if is_correct else -0.5
            
            rewards.append(reward)
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            rewards.append(0.0)
            
    return rewards

def get_reward_registry(mode: str):
    if mode == RewardMode.STANDARD:
        return {
            "accuracy": standard_reward,
        }
    elif mode == RewardMode.R1:
        return {
            "r1": r1_reward,
        }
    else:
        raise ValueError(f"Unknown reward mode: {mode}")

# 全局变量用于存储reward函数注册表
reward_funcs_registry = {}

@app.post("/get_rm_score")
async def get_reward_score(request: RewardRequest):
    logger.debug("Received reward calculation request")
    logger.debug(f"Request data: {request}")

    if not request.answers:
        logger.error("Request missing answers")
        raise HTTPException(status_code=400, detail="Answers are required")

    if len(request.query) != len(request.answers):
        logger.error(f"Query length ({len(request.query)}) doesn't match answers length ({len(request.answers)})")
        raise HTTPException(status_code=400, detail="Number of queries and answers must match")

    reward_types = list(reward_funcs_registry.keys())
    logger.info(f"Processing reward types: {reward_types}")

    # 验证请求的reward类型是否有效
    for r_type in reward_types:
        if r_type not in reward_funcs_registry:
            logger.error(f"Invalid reward type requested: {r_type}")
            raise HTTPException(status_code=400, detail=f"Invalid reward type: {r_type}")

    # 计算每个reward函数的得分
    all_rewards = []
    for r_type in reward_types:
        reward_fn = reward_funcs_registry[r_type]
        rewards = reward_fn(request.query, request.answers)
        all_rewards.append(rewards)

    # 将所有reward相加（按样本）
    total_rewards = [sum(rewards) for rewards in zip(*all_rewards)]
    logger.info(f"Final rewards: {total_rewards}")
    return {"rewards": total_rewards}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reward Model Server')
    parser.add_argument('--mode', type=str, default=RewardMode.STANDARD,
                      choices=[RewardMode.STANDARD, RewardMode.R1],
                      help='Reward calculation mode')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                      help='Server host')
    parser.add_argument('--port', type=int, default=8000,
                      help='Server port')
    
    args = parser.parse_args()
    
    # 根据模式设置reward函数
    reward_funcs_registry = get_reward_registry(args.mode)
    
    logger.info(f"Starting reward model server in {args.mode} mode")
    uvicorn.run(app, host=args.host, port=args.port)