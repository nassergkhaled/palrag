from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

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


def get_template_prompt(qa=False):
    """
    Function to get a template prompt for the conversation.

    Returns:
    - PromptTemplate object with a predefined system prompt and instruction.
    """
    sys_prompt = """You are a reliable and respectful assistant designed to provide helpful responses based on the given context.\
    Answer questions succinctly, ensuring that your responses are directly related to the information provided.\
    and not have any text after the answer is done. Do not mention in the answer, that is taken from the given context 

    If a question appears nonsensical or lacks factual coherence, kindly explain the issue"""
    
    if not qa:
        instruction = """CHAT HISTORY:\n\n{chat_history} \nQuestion: {question}"""
        input_variables = ["chat_history", "question"]
    else:
        instruction = """Question: {question}"""
        input_variables = ["question"]
        
    llama_2_prompt = PromptTemplate(
        template=__get_string_prompt(instruction, sys_prompt),
        input_variables=input_variables,
        validate_template=False
    )
    
    return llama_2_prompt
        

def get_retriever(persist_directory: str, k: int, score_threshold: float, lambda_mult: float = 0.25, embeddings_model_name: str ='BAAI/bge-base-en-v1.5'):
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
                               'score_threshold': score_threshold,
                               'lambda_mult': lambda_mult
                           })

