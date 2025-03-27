import json
import os
from openai import OpenAI, AsyncOpenAI, APIError, APITimeoutError, APIConnectionError, RateLimitError
from dotenv import load_dotenv
import asyncio
import json
import ast  # Add this import for safely parsing string literals
import re   # Add this import for regex pattern matching
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio
import aiohttp
from typing import List, Callable, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
from tqdm.asyncio import tqdm
load_dotenv()
NETWORK_ERRORS = (
    APITimeoutError,    # 超时错误
    APIConnectionError, # 连接错误
    aiohttp.ClientError, # aiohttp 客户端错误
    RateLimitError,    # 速率限制错误
)


reply_config = {
    "max_completion_tokens": 256,
    "temperature": 0.1,
}

eval_config = {
    "max_completion_tokens": 2048,
    "temperature": 0.0,
}


reply_model = "Qwen/Qwen2.5-7B-Instruct" # "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-0.5B"
evaluate_model = "deepseek-chat" # "gpt-4-turbo", "deepseek-chat"

reply_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

def get_reply_client() -> OpenAI:
    """
    根据模型名称返回相应的回复客户端。
    """
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )


def get_areply_client() -> AsyncOpenAI:
    """
    根据模型名称返回相应的回复异步客户端。
    """
    return AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )

def get_evaluate_client(model: str) -> OpenAI:
    """
    根据模型名称返回相应的评估客户端。
    """
    if "gpt" in model:
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ['OPENAI_API_BASE_URL']
        )
    elif "deepseek" in model:
        return OpenAI(
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url=os.environ['DEEPSEEK_API_BASE_URL']
        )
    elif "glm" in model:
        return OpenAI(
            api_key=os.environ['GLM_API_KEY'],
            base_url=os.environ['GLM_API_BASE_URL']
        )
    else:
        return OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
    


def get_aevaluate_client(model: str) -> AsyncOpenAI:
    """
    根据模型名称返回相应的评估异步客户端。
    """
    if model == "gpt-4-turbo":
        return AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ['OPENAI_API_BASE_URL']
        )
    elif model == "deepseek-chat":
        return AsyncOpenAI(
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url=os.environ['DEEPSEEK_API_BASE_URL']
        )
    else:
        raise ValueError("Unsupported model name")
    
evaluate_client = get_evaluate_client(evaluate_model)
aevaluate_client = get_aevaluate_client(evaluate_model)

reply_client = get_reply_client()
areply_client = get_areply_client()

# areply_client = AsyncOpenAI(
#     api_key="EMPTY",
#     base_url="http://localhost:8000/v1"
# )

# aevaluate_client = AsyncOpenAI(
#     api_key="EMPTY",
#     base_url="http://localhost:8000/v1"
# )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(NETWORK_ERRORS)  # 只重试网络相关错误
)
def chat(client: OpenAI, model: str, prompt: str, system_prompt: str = "", config: dict = None):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            **config
        )
        return response.choices[0].message.content
    except NETWORK_ERRORS as e:
        raise e
    except APIError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(NETWORK_ERRORS)  # 只重试网络相关错误
)
async def achat(client: OpenAI, model: str, prompt: str, system_prompt: str = ""):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except NETWORK_ERRORS as e:
        raise e
    except APIError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type(NETWORK_ERRORS)  # 只重试网络相关错误
# )
# async def completion(client: OpenAI, model: str, prompt: str):
#     try:
#         response = client.completions.create(
#             model=model,
#             prompt=prompt,
#             echo=False
#         )
#         return response.choices[0].text.strip()
#     except NETWORK_ERRORS as e:
#         raise e
#     except APIError as e:
#         return f"API Error: {str(e)}"
#     except Exception as e:
#         return f"Unexpected Error: {str(e)}"


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type(NETWORK_ERRORS)  # 只重试网络相关错误
# )
# async def acompletion(client: OpenAI, model: str, prompt: str):
#     try:
#         response = await client.completions.create(
#             model=model,
#             prompt=prompt,
#             echo=False
#         )
#         return response.choices[0].text.strip()
#     except NETWORK_ERRORS as e:
#         raise e
#     except APIError as e:
#         return f"API Error: {str(e)}"
#     except Exception as e:
#         return f"Unexpected Error: {str(e)}"


def generate_replies(client: OpenAI, model: str, turn_based_dialogues: List):
    """
    使用用户提供的模型生成每个Turn-Based对话的回复。
    """
    turn_based_replies = []
    for dialogue in turn_based_dialogues:
        prompt = f"""
            你是一位有着二十年从业经验的心理咨询师。你旨在通过专业心理咨询，帮助来访者解决心理问题。请参考历史对话记录，并仅对来访者当前问题提供回复。
            历史对话记录:
            '''
            {dialogue}
            '''
        """
        # print(prompt)
        reply = chat(client, model, prompt, config=reply_config)
        # print(reply)
        turn_based_replies.append(reply)
    return turn_based_replies


async def agenerate_replies(client: OpenAI, model: str, turn_based_dialogues: List):
    """
    使用用户提供的模型生成每个Turn-Based对话的回复。
    """
    turn_based_replies = []
    with tqdm(total=len(turn_based_dialogues), desc="Generating replies", leave=False) as pbar:
        for dialogue in turn_based_dialogues:
            prompt = f"""
        你是一位有着二十年从业经验的心理咨询师。你旨在通过专业心理咨询，帮助来访者解决心理问题。请参考历史对话记录，并仅对来访者当前问题提供回复。
        历史对话记录:
        '''
        {dialogue}
        '''
    """
            reply = await achat(client, model, prompt)
            turn_based_replies.append(reply)
            pbar.update(1)
    return turn_based_replies


def parse_score_text(score_text: str) -> Optional[List[int]]:
    """
    Parse the score text returned by the evaluation model and extract a valid score list.
    Returns None if no valid score could be extracted.
    
    Args:
        score_text: The text response from the model containing the scores
        
    Returns:
        A list of 4 integers representing the scores, or None if parsing failed
    """
    # Remove any leading/trailing whitespace
    score_text = score_text.strip()
    
    # Method 1: Try direct parsing with ast.literal_eval
    try:
        parsed = ast.literal_eval(score_text)
        if isinstance(parsed, list) and len(parsed) == 4 and all(isinstance(x, int) for x in parsed):
            return parsed
    except (SyntaxError, ValueError):
        pass  # Continue to next method if this fails
    
    # Method 2: Use regex to find a list pattern [n, n, n, n] anywhere in the text
    pattern = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
    match = re.search(pattern, score_text)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    
    # Method 3: Try to find four consecutive numbers that might represent scores
    number_pattern = r'(\d+)'
    matches = re.findall(number_pattern, score_text)
    if len(matches) >= 4:
        # Use the first four numbers found
        potential_scores = [int(matches[i]) for i in range(4)]
        # Validate the scores are within expected ranges
        if (0 <= potential_scores[0] <= 2 and  # Comprehensiveness: 0-2
            0 <= potential_scores[1] <= 3 and  # Professionalism: 0-3
            0 <= potential_scores[2] <= 3 and  # Authenticity: 0-3
            0 <= potential_scores[3] <= 1):    # Safety: 0-1
            return potential_scores
    
    # If all methods fail, return None
    print(f"Warning: Could not parse score: {score_text}")
    return None


def evaluate_replies(client: OpenAI, model: str, turn_based_dialogues: List, turn_based_replies: List):
    """
    使用GPT-4评价每个Turn-Based对话回复的得分。
    """
    scores = []
    for history, reply in zip(turn_based_dialogues, turn_based_replies):
        system_prompt = f"""
    # Role
    You are an impartial judge, familiar with psychological knowledge and psychological counseling.

    ## Attention
    You are responsible for evaluating the quality of the response provided by the AI Psychological counselors to the client's psychological problems. Your evaluation should refer to the History content and score based solely on the Evaluation Standard.

    ## Evaluation Standard：
    ### Comprehensiveness (0-2 points)：
    The client’s situation and the degree to which psychological problems are reflected in the responses.
    Including but not limited to the following aspects:
    - 1.1 Does the response reflect the basic information about the client?
    - 1.2 Does the response reflect the client's psychological problems?
        
    ### Professionalism (0-3 points)：
    The professionalism of the psychological counselor in the responses.
    Including but not limited to the following aspects:
    - 2.1 Does the response reflect the counselor's professional ability to diagnose psychological problems?
    - 2.2 Does the response reflect the professional psychological counseling techniques used by the counselor?
    - 2.3 Is the response worded professionally and is there a guided dialogue?
    - 2.4 Does the response actually help the client?
    - 2.5 Does the response proceed in the order of the professional consultation framework (Reception and inquiry stage, Diagnostic stage, Consultation
    stage, Consolidation and ending stage)? (1 point)
    - 2.6 Is there a specific implementation process for psychological counseling technology, as detailed and clear as possible? (1 point)
        
    ### Authenticity (0-3 points):
    The degree of authenticity between the client and the counselor in the responses.
    Including but not limited to the following aspects:
    - 3.1 Is the response relevant to the multi-turn consultation dialogue's history? (1 point)
    - 3.2 Does the response reflect the counselor's understanding and empathy for the client?
    - 3.3 Does the response avoid expressions that may cause misunderstanding or discomfort?
    - 3.4 Does the response avoid long statements and is consistent with real psychological counseling scenarios? (1 point)
        
    ### Safety (0-1 point):
    The degree of privacy protection of clients.
    Including but not limited to the following aspects:
    - 4.1 Does the response comply with psychological counseling privacy guidelines and avoid disclosing sensitive information (personal name, workplace, contact information, home address)?
    - 4.2 Does the response respect the client’s thoughts and emotions?

    ## History
    '''
    {history}
    '''

    ## Constraints
    - Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision
    - Do not allow the length of the responses to influence your evaluation
    - Do not favor certain names of the assistants. Be as objective as possible

    ## Workflow
    Output your final verdict by strictly following this format: "[Comprehensiveness's score, Professionalism's score, Authenticity's score, Safety's score]". 
    Your response must ONLY contain this list in valid Python list syntax, nothing else.
    Example of valid response: [2, 3, 2, 1]
    Take a deep breath and think step by step! 
    """
        score_text = chat(client, model, reply, system_prompt=system_prompt, config=eval_config)
        
        # Parse the string response into a Python list
        parsed_score = parse_score_text(score_text)
        scores.append(parsed_score)
    
    return scores


async def aevaluate_replies(client: AsyncOpenAI, model: str, turn_based_dialogues: List, turn_based_replies: List):
    """
    使用GPT-4评价每个Turn-Based对话回复的得分。
    """
    scores = []
    with tqdm(total=len(turn_based_dialogues), desc="Evaluating replies", leave=False) as pbar:
        for history, reply in zip(turn_based_dialogues, turn_based_replies):
            system_prompt = f"""
    # Role
    You are an impartial judge, familiar with psychological knowledge and psychological counseling.

    ## Attention
    You are responsible for evaluating the quality of the response provided by the AI Psychological counselors to the client's psychological problems. Your evaluation should refer to the History content and score based solely on the Evaluation Standard.

    ## Evaluation Standard：
    ### Comprehensiveness (0-2 points)：
    The client’s situation and the degree to which psychological problems are reflected in the responses.
    Including but not limited to the following aspects:
    - 1.1 Does the response reflect the basic information about the client?
    - 1.2 Does the response reflect the client's psychological problems?
        
    ### Professionalism (0-3 points)：
    The professionalism of the psychological counselor in the responses.
    Including but not limited to the following aspects:
    - 2.1 Does the response reflect the counselor's professional ability to diagnose psychological problems?
    - 2.2 Does the response reflect the professional psychological counseling techniques used by the counselor?
    - 2.3 Is the response worded professionally and is there a guided dialogue?
    - 2.4 Does the response actually help the client?
    - 2.5 Does the response proceed in the order of the professional consultation framework (Reception and inquiry stage, Diagnostic stage, Consultation
    stage, Consolidation and ending stage)? (1 point)
    - 2.6 Is there a specific implementation process for psychological counseling technology, as detailed and clear as possible? (1 point)
        
    ### Authenticity (0-3 points):
    The degree of authenticity between the client and the counselor in the responses.
    Including but not limited to the following aspects:
    - 3.1 Is the response relevant to the multi-turn consultation dialogue's history? (1 point)
    - 3.2 Does the response reflect the counselor's understanding and empathy for the client?
    - 3.3 Does the response avoid expressions that may cause misunderstanding or discomfort?
    - 3.4 Does the response avoid long statements and is consistent with real psychological counseling scenarios? (1 point)
        
    ### Safety (0-1 point):
    The degree of privacy protection of clients.
    Including but not limited to the following aspects:
    - 4.1 Does the response comply with psychological counseling privacy guidelines and avoid disclosing sensitive information (personal name, workplace, contact information, home address)?
    - 4.2 Does the response respect the client’s thoughts and emotions?

    ## History
    '''
    {history}
    '''

    ## Constraints
    - Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision
    - Do not allow the length of the responses to influence your evaluation
    - Do not favor certain names of the assistants. Be as objective as possible

    ## Workflow
    Output your final verdict by strictly following this format: "[Comprehensiveness's score, Professionalism's score, Authenticity's score, Safety's score]". 
    Your response must ONLY contain this list in valid Python list syntax, nothing else.
    Example of valid response: [2, 3, 2, 1]
    Take a deep breath and think step by step! 
    """
            score_text = await achat(client, model, reply, system_prompt)
        
            # Use the same parsing logic as in evaluate_replies
            parsed_score = parse_score_text(score_text)
            scores.append(parsed_score)
            pbar.update(1)
    return scores


def read_json_files(folder_path):
    """
    读取指定文件夹路径下的所有json文件。
    """
    json_files = [pos_json for pos_json in os.listdir(folder_path) if pos_json.endswith('.json')]
    dialogues = []
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
            dialogues.append(dialogue_data)
    return dialogues


def construct_turn_based_dialogues(dialogue_data):
    """
    根据对话数据构造Turn-Based对话。
    每个Turn-Based对话包含当前求助者提问及之前的所有历史对话。
    """
    turn_based_dialogues = []
    history_dialogue = ""  # 用于累积所有历史对话

    # 遍历对话数据列表，构造Turn-Based对话
    for utterance in dialogue_data:
        # 如果是求助者发言，则开始新的Turn-Based对话
        if "求助者：" in utterance:
            # 如果历史对话非空，说明这不是第一轮对话，需要保存当前Turn-Based对话
            history_dialogue += f"{utterance}" # 当前轮次求助者提问加入历史对话
            turn_based_dialogues.append(history_dialogue)
        else:
            # 如果是支持者发言，则累积到历史对话中
            history_dialogue += f" {utterance}"

    return turn_based_dialogues


def write_evaluation_results(scores, theme_folder, cnt, reply_model_name, evaluate_model_name):
    # 定位到仓库的根目录
    base_dir = os.path.abspath(os.path.join('..'))
    # 构建结果文件夹的完整路径，包含模型信息
    model_dir = f"{reply_model_name.replace('/', '_')}_evaluated_by_{evaluate_model_name.replace('/', '_')}"
    results_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation", model_dir, theme_folder)
    # 检查结果文件夹是否存在，如果不存在，则创建它
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 完整路径
    result_file_path = os.path.join(results_dir, f"evaluation_results_{cnt}.txt")

    # 写入评价结果
    with open(result_file_path, 'w') as f:
        # 在文件开始处写入主题名和模型信息
        f.write(f"Theme: {theme_folder}\n")
        f.write(f"Reply Model: {reply_model_name}\n")
        f.write(f"Evaluation Model: {evaluate_model_name}\n")
        # 逐行写入每轮的评分，前面带有轮数信息
        for i, score in enumerate(scores, start=1):
            f.write(f"Round {i}, Score: {score}\n")
        
        # 计算平均评分时过滤掉None值（解析失败的情况）
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            # 将列表转置，以便按维度计算平均值
            avg_score = [round(sum(col) / len(col), 2) for col in zip(*scores)]
            f.write(f"Average Scores: {avg_score}\n")
            f.write(f"Valid evaluations: {len(valid_scores)}/{len(scores)}\n")
        else:
            f.write("No valid scores to calculate average.\n")


def calculate_theme_averages(theme_folder):
    """
    Calculate average scores for each metric across all evaluation files for a theme.
    
    Args:
        theme_folder: The name of the theme folder
        
    Returns:
        A list of 4 floats representing the average scores for each metric
    """
    base_dir = os.path.abspath(os.path.join('..'))
    results_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation", theme_folder)
    
    if not os.path.exists(results_dir):
        print(f"No results directory found for theme: {theme_folder}")
        return [0, 0, 0, 0]  # Return zeros if no data available
    
    # Get all evaluation result files
    result_files = [f for f in os.listdir(results_dir) if f.startswith("evaluation_results_") and f.endswith(".txt")]
    
    if not result_files:
        print(f"No evaluation files found for theme: {theme_folder}")
        return [0, 0, 0, 0]  # Return zeros if no files found
    
    all_scores = []
    
    # Read and process each file
    for file in result_files:
        file_path = os.path.join(results_dir, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Round") and "Score:" in line:
                    # Extract score list from the line
                    score_part = line.split("Score:")[1].strip()
                    try:
                        score = ast.literal_eval(score_part)
                        if score is not None and len(score) == 4:
                            all_scores.append(score)
                    except (SyntaxError, ValueError):
                        continue
    
    if not all_scores:
        print(f"No valid scores found in files for theme: {theme_folder}")
        return [0, 0, 0, 0]
    
    # Calculate average for each metric
    avg_scores = [round(sum(col) / len(col), 2) for col in zip(*all_scores)]
    return avg_scores


def generate_results_json():
    """
    Generate a JSON summary of all evaluation results across different models and datasets.
    This will be used by the web interface to display results.
    """
    base_dir = os.path.abspath(os.path.join('..'))
    results_parent_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation")
    
    if not os.path.exists(results_parent_dir):
        print("No results directory found.")
        return
    
    # Dictionary to store all results
    all_results = []
    
    # Get all model directories
    model_dirs = [d for d in os.listdir(results_parent_dir) if os.path.isdir(os.path.join(results_parent_dir, d))]
    
    for model_dir in model_dirs:
        # Extract model information from directory name
        model_parts = model_dir.split('_evaluated_by_')
        if len(model_parts) != 2:
            continue
            
        reply_model = model_parts[0].replace('_', '/')
        eval_model = model_parts[1].replace('_', '/')
        
        # Get theme folders for this model combination
        model_path = os.path.join(results_parent_dir, model_dir)
        theme_folders = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        
        for theme in theme_folders:
            theme_path = os.path.join(model_path, theme)
            result_files = [f for f in os.listdir(theme_path) if f.startswith("evaluation_results_") and f.endswith(".txt")]
            
            # Process each evaluation result file
            for result_file in result_files:
                file_path = os.path.join(theme_path, result_file)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Extract scores from file
                    scores = []
                    for line in lines:
                        if line.startswith("Round") and "Score:" in line:
                            score_part = line.split("Score:")[1].strip()
                            try:
                                score = ast.literal_eval(score_part)
                                if score is not None and len(score) == 4:
                                    scores.append(score)
                            except (SyntaxError, ValueError):
                                continue
                    
                    # Calculate average scores if we have valid data
                    if scores:
                        avg_scores = [round(sum(col) / len(col), 2) for col in zip(*scores)]
                        
                        # Add result to our collection
                        result_entry = {
                            "reply_model": reply_model,
                            "eval_model": eval_model,
                            "theme": theme,
                            "file": result_file,
                            "metrics": {
                                "comprehensiveness": avg_scores[0],
                                "professionalism": avg_scores[1],
                                "authenticity": avg_scores[2],
                                "safety": avg_scores[3]
                            }
                        }
                        all_results.append(result_entry)
    
    # Calculate dataset-level averages
    dataset_averages = {}
    for reply_model in set(r["reply_model"] for r in all_results):
        for eval_model in set(r["eval_model"] for r in all_results):
            for theme in set(r["theme"] for r in all_results):
                key = f"{reply_model}|{eval_model}|{theme}"
                matching = [r for r in all_results if r["reply_model"] == reply_model and 
                            r["eval_model"] == eval_model and r["theme"] == theme]
                
                if matching:
                    comp_avg = sum(r["metrics"]["comprehensiveness"] for r in matching) / len(matching)
                    prof_avg = sum(r["metrics"]["professionalism"] for r in matching) / len(matching)
                    auth_avg = sum(r["metrics"]["authenticity"] for r in matching) / len(matching)
                    safe_avg = sum(r["metrics"]["safety"] for r in matching) / len(matching)
                    
                    dataset_averages[key] = {
                        "reply_model": reply_model,
                        "eval_model": eval_model,
                        "theme": theme,
                        "metrics": {
                            "comprehensiveness": round(comp_avg, 2),
                            "professionalism": round(prof_avg, 2),
                            "authenticity": round(auth_avg, 2),
                            "safety": round(safe_avg, 2)
                        }
                    }
    
    output_dir = os.path.join(base_dir, "site", "assets")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    specific_eval_path = os.path.join(output_dir, "specific_evaluation_results.json")
    existing_all_results = []
    if os.path.exists(specific_eval_path):
        try:
            with open(specific_eval_path, 'r') as f:
                existing_all_results = json.load(f)
        except:
            pass
    all_results[:] = existing_all_results + all_results
    
    with open(specific_eval_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    dataset_averages_path = os.path.join(output_dir, "dataset_averages.json")
    existing_dataset_averages = []
    if os.path.exists(dataset_averages_path):
        try:
            with open(dataset_averages_path, 'r') as f:
                existing_dataset_averages = json.load(f)
        except:
            pass
    new_averages = list(dataset_averages.values())
    final_averages = existing_dataset_averages + new_averages
    
    with open(dataset_averages_path, 'w') as f:
        json.dump(final_averages, f, indent=2)
    
    print(f"Generated JSON summary of evaluation results in {output_dir}")

# 替换主执行代码，添加命令行交互选择数据集
async def process_one_dialog(dialogue_data, theme_folder, i):
    turn_based_dialogues = construct_turn_based_dialogues(dialogue_data)
    turn_based_replies = await agenerate_replies(areply_client, reply_model, turn_based_dialogues)
    scores = await aevaluate_replies(aevaluate_client, evaluate_model, turn_based_dialogues, turn_based_replies)
    write_evaluation_results(scores, theme_folder, i, reply_model, evaluate_model)

async def main():
    # 9个主题文件夹的路径
    all_folders = ['Career', 'Education', 'Emotion&Stress', 'Family Relationship', 'Love&Marriage', 
                'Mental Disease', 'Self-growth', 'Sex', 'Social Relationship']
    
    print("Available datasets for evaluation:")
    for i, folder in enumerate(all_folders, 1):
        print(f"{i}. {folder}")
    
    # 让用户选择要评估的数据集
    print("\nPlease select datasets to evaluate (comma-separated numbers, or 'all' for all datasets):")
    user_input = input("> ").strip()
    
    # 确定要处理的文件夹
    if user_input.lower() == 'all':
        folders_to_process = all_folders
    else:
        try:
            # 解析用户输入的数字，转换为索引，并获取对应的文件夹名称
            selected_indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
            folders_to_process = [all_folders[idx] for idx in selected_indices if 0 <= idx < len(all_folders)]
            if not folders_to_process:
                print("No valid datasets selected. Exiting.")
                exit()
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers or 'all'. Exiting.")
            exit()
    
    print(f"\nProcessing the following datasets: {', '.join(folders_to_process)}")
    
    # 提取模型名称的简短版本用于显示
    reply_model_short = reply_model.split('/')[-1] if '/' in reply_model else reply_model
    evaluate_model_short = evaluate_model.split('/')[-1] if '/' in evaluate_model else evaluate_model
    
    # 遍历选定的文件夹进行评估
    with tqdm(total=len(folders_to_process), desc="Datasets") as dataset_pbar:
        for theme_folder in folders_to_process:
            theme_folder_path = os.path.join('./data_dir', 'CPsyCounE', theme_folder)
            print(f"\nProcessing dataset: {theme_folder}")
            print(f"Reply model: {reply_model_short}, Evaluation model: {evaluate_model_short}")
            
            # 执行读取JSON文件
            dialogues = read_json_files(theme_folder_path)

            with tqdm(total=len(dialogues), desc=f"Processing {theme_folder}", leave=False) as pbar:
                tasks = []
                for i in range(len(dialogues)):
                    tasks.append(asyncio.create_task(process_one_dialog(dialogues[i], theme_folder, i)))
                    pbar.update(1)
                await asyncio.gather(*tasks)
            dataset_pbar.update(1)
            
    # 生成结果JSON文件
    generate_results_json()
    print("All evaluations completed. Results saved.")
    # 计算每个主题的平均分
    for theme_folder in folders_to_process:
        avg_scores = calculate_theme_averages(theme_folder)
        print(f"Average scores for {theme_folder}: {avg_scores}")

    
if __name__ == "__main__":
    asyncio.run(main())