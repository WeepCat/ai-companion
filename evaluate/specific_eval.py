import json
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()


local_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

remote_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ['OPENAI_API_BASE_URL']
)

# 评估: 生成的时候需要区分 chat 模型和 completion 模型，但实际中使用的都是 chat 模型
# 现在应该考虑的是用 Qwen2.5-0.5B-Instruct 和 Llama3.2-0.5B-Instruct 两个模型

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


def generate_replies(turn_based_dialogues, model_reply):
    """
    使用用户提供的模型生成每个Turn-Based对话的回复。
    """
    turn_based_replies = []
    for dialogue in turn_based_dialogues:
        reply = model_reply(dialogue)
        turn_based_replies.append(reply)
    return turn_based_replies


def evaluate_replies(turn_based_dialogues, turn_based_replies, evaluate_reply):
    """
    使用GPT-4评价每个Turn-Based对话回复的得分。
    """
    scores = []
    for dialogue, reply in zip(turn_based_dialogues, turn_based_replies):
        score = evaluate_reply(dialogue, reply)
        scores.append(score)
    return scores



def write_evaluation_results(scores, theme_folder, cnt):
    # 定位到仓库的根目录
    base_dir = os.path.abspath(os.path.join('..'))
    # 构建结果文件夹的完整路径
    results_dir = os.path.join(base_dir, "Results_Turn_Based_Dialogue_Evaluation", theme_folder)
    # 检查结果文件夹是否存在，如果不存在，则创建它
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 完整路径
    result_file_path = os.path.join(results_dir, f"evaluation_results_{cnt}.txt")

    # 写入评价结果
    with open(result_file_path, 'w') as f:
        # 在文件开始处写入主题名
        f.write(f"{theme_folder}\n")
        # 逐行写入每轮的评分，前面带有轮数信息
        for i, score in enumerate(scores, start=1):
            f.write(f"Round {i}, Score: {score}\n")
        # 计算并写入平均评分
        avg_score = [round(sum(col) / len(col), 2) for col in zip(*scores)]
        f.write(f"Average Scores: {avg_score}\n")



# BUG
# 需用户完善模型的回复生成，并返回reply
# 示例模型回复函数(需要替换为实际使用的模型)
def model_reply(history):
    # 这里应该是模型生成回复的代码
    user_message = f"""
    你是一位有着二十年从业经验的心理咨询师。你旨在通过专业心理咨询，帮助来访者解决心理问题。请参考历史对话记录，并仅对来访者当前问题提供回复。
    历史对话记录:
    '''
    {history}
    '''
    """
    messages = [{"role": "user", "content": user_message}]
    response = local_client.completions.create(
        model = 'gpt-4-turbo',
        messages = messages,
        temperature = 0.0,
    )
    reply = response.choices[0].message["content"]
    # 返回生成的回复
    return reply 


# 返回内容需按照规定格式返回评分, 例如"[2,2,3,3]" (已在prompt中限定)
def evaluate_reply(history, reply):
    # GPT-4评价模型
    
    system_message = f"""
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

    Take a deep breath and think step by step! 
    """
   
    messages = [
        {"role": "system", "content":system_message},
        {"role": "user", "content": reply},
]
    response = remote_client.chat.completions.create(
        model = 'gpt-4-turbo',
        messages = messages,
        temperature = 0.0,
    )
    
    # 返回评价得分
    return response.choices[0].message["content"]



# # 主题：'Career', 'Education', 'Emotion&Stress', 'Family Relationship', 'Love&Marriage', 'Mental Disease', 'Self-growth', 'Sex', 'Social Relationship'
# # 主题文件夹的路径
# theme_folder = '' # 填入主题文件夹的名称    
# theme_folder_path = os.path.abspath(os.path.join('..', 'CPsyCounE', theme_folder))
# # 执行读取JSON文件
# dialogues = read_json_files(theme_folder_path)

# for i in range(len(dialogues)):
#     cnt = i
#     dialogue_data = dialogues[i]
#     # 构造Turn-Based对话
#     turn_based_dialogues = construct_turn_based_dialogues(dialogue_data)
#     # 生成回复
#     turn_based_replies = generate_replies(turn_based_dialogues, model_reply)
#     # 评价得分
#     scores = evaluate_replies(turn_based_dialogues, turn_based_replies, evaluate_reply)
#     # 写入评价结果
#     write_evaluation_results(scores, theme_folder, cnt)



# 9个主题文件夹的路径
folders = ['Career', 'Education', 'Emotion&Stress', 'Family Relationship', 'Love&Marriage', 'Mental Disease', 'Self-growth', 'Sex', 'Social Relationship']
for theme_folder in folders:
    theme_folder_path = os.path.join('./data_dir', 'CPsyCounE', theme_folder)
    # 执行读取JSON文件
    dialogues = read_json_files(theme_folder_path)

    for i in range(len(dialogues)):
        cnt = i
        dialogue_data = dialogues[i]
        # 构造Turn-Based对话
        turn_based_dialogues = construct_turn_based_dialogues(dialogue_data)
        # 生成回复
        turn_based_replies = generate_replies(turn_based_dialogues, model_reply)
        # 评价得分
        scores = evaluate_replies(turn_based_dialogues, turn_based_replies, evaluate_reply)
        # 写入评价结果
        write_evaluation_results(scores, theme_folder, cnt)