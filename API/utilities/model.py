from langchain.llms.together import Together
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from dotenv import dotenv_values


import os
os.environ['TRANSFORMERS_CACHE'] = './cache'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './cache'

env = dotenv_values()


def get_retriever(persist_directory="victorstore/chroma_db", k: int=2, score_threshold: float=0.55, lambda_mult: float = 0.25,  embeddings_model_name='BAAI/bge-base-en-v1.5'):
    """
    Function to get a retriever with specified parameters.

    Parameters:
    - persist_directory (str): Directory to persist the Chroma database.
    - embeddings_model_name (str): Name of the Hugging Face embeddings model.

    Returns:
    - Retriever object with specified search parameters.
    """
    # Load database
    db = Chroma(persist_directory=persist_directory, embedding_function=HuggingFaceEmbeddings(model_name=embeddings_model_name))

    # Get the retriever
    return db.as_retriever(search_type="similarity_score_threshold",
                           search_kwargs={
                               'k': k,
                               'score_threshold': score_threshold
                            #    'lambda_mult': lambda_mult
                           })



def __get_string_prompt(instruction, system_prompt):
    """
    Internal function to create a string prompt template.

    Parameters:
    - instruction (str): Instruction to be included in the prompt.
    - system_prompt (str): System prompt to be included in the prompt.

    Returns:
    - Formatted string prompt template.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def get_template_prompt():
    """
    Function to get a template prompt for the conversation.

    Returns:
    - PromptTemplate object with a predefined system prompt and instruction.
    """
    sys_prompt = """You are a reliable and respectful assistant designed to provide helpful responses based on the given context.\
    Answer questions succinctly, ensuring that your responses are directly related to the information provided.\
    and not have any text after the answer is done. Do not mention in the answer, that is taken from the given context 

    If a question appears nonsensical or lacks factual coherence, kindly explain the issue"""
    instruction = """CHAT HISTORY:
    {chat_history}
    Question: {question}"""
    llama_2_prompt = PromptTemplate(
        template=__get_string_prompt(instruction, sys_prompt),
        input_variables=["chat_history", "question"], 
        validate_template=False
    )

    return llama_2_prompt


def get_model():
    """
    Function to get the Llama 2 chat model with specified parameters.

    Returns:
    - ConversationalRetrievalChain object with the Llama 2 chat model, retriever, and other configurations.
    """
    # Load the model 13b
    Llama_2_chat_13b = Together(
        model="togethercomputer/llama-2-13b-chat",
        temperature=0.1,
        max_tokens=256,
        top_k=50,
        top_p=0.9,
        together_api_key=${{secrets.CSRF_GITHUB_JENKINS_TOKE}}
    )
    chat_history_13b = ConversationBufferWindowMemory(k=6, memory_key='chat_history')
    Llama2chat_13b_chain_Window = ConversationalRetrievalChain.from_llm(Llama_2_chat_13b,
                                                                       retriever=get_retriever(),
                                                                       condense_question_prompt=get_template_prompt(),
                                                                       memory=chat_history_13b,
                                                                       get_chat_history=lambda h: h)
    return Llama2chat_13b_chain_Window