import os

src_dir = os.path.dirname(os.path.abspath(__file__))                # config
base_dir = os.path.dirname(src_dir) 
# base_dir = os.path.dirname(src_dir)                                 # base
model_repo = 'ajupyter/EmoLLM_aiwei'

# model
model_dir = os.path.join(base_dir, 'model')                         # model
embedding_path = os.path.join(model_dir, 'embedding_model')         # embedding
embedding_model_name = 'BAAI/bge-small-zh-v1.5'
rerank_path = os.path.join(model_dir, 'rerank_model')  	        	  # embedding
rerank_model_name = 'BAAI/bge-reranker-large'

llm_path = os.path.join(model_dir, 'pythia-14m')                    # llm

# data
data_dir = os.path.join(base_dir, 'data')                           # data
knowledge_json_path = os.path.join(data_dir, 'knowledge.json')      # json
knowledge_pkl_path = os.path.join(data_dir, 'knowledge.pkl')        # pkl
doc_dir = os.path.join(data_dir, 'txt')   
qa_dir = os.path.join(data_dir, 'json')   
cloud_vector_db_dir = os.path.join(base_dir, 'EmoLLMRAGTXT')

# log
log_dir = os.path.join(base_dir, 'log')                             # log
log_path = os.path.join(log_dir, 'log.log')                         # file

# txt embedding 切分参数     
chunk_size=1000
chunk_overlap=100

# vector DB
vector_db_dir = os.path.join(cloud_vector_db_dir, 'vector_db')

# RAG related
# select num: 代表rerank 之后选取多少个 documents 进入 LLM
# retrieval num： 代表从 vector db 中检索多少 documents。（retrieval num 应该大于等于 select num）
select_num = 3
retrieval_num = 3

# LLM key
glm_key = ''

# prompt
# prompt_template = """
# 	你是一个拥有丰富心理学知识的温柔邻家温柔大姐姐艾薇，我有一些心理问题，请你用专业的知识和温柔、可爱、俏皮、的口吻帮我解决，回复中可以穿插一些可爱的Emoji表情符号或者文本符号。\n

# 	根据下面检索回来的信息，回答问题。
# 	{content}
# 	问题：{query}
# """

prompt_template = """
	你是心理健康助手WeepCat, 由WeepCat打造, 是一个研究过无数具有心理健康问题的病人与心理健康医生对话的心理专家, 在心理方面拥有广博的知识储备和丰富的研究咨询经验。你旨在通过专业心理咨询, 协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术, 一步步帮助来访者解决心理问题。\n

	根据下面检索回来的信息，回答问题。
	{content}
	问题：{query}
"""