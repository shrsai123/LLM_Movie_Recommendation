import torch
import logging
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from src.retriever import load_retriever

logger = logging.getLogger(__name__)

def build_chain(retriever,config):
    logger.info("Building conversational chain")
    model_id = config["model"]["llm"]
    hf_pipeline = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto")
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    template_prefix="""You are an expert movie recommender. For user queries about actors/directors/genres:
1. Suggest exactly 3 SPECIFIC movies with YEAR and LEAD ACTORS
2. Include 1 to 3-sentence descriptions
3. Explain WHY they match the request
4. NEVER suggest irrelevant movies


If the user asks for details about any movie or about the cast, provide the following:
1. The movie's full description, including the plot.
2. Information on the lead actors and their roles.
3. Special details about the movie like notable achievements, awards, or critical reception.

Example good response:
"Here are great Russell Crowe movies:
- Gladiator (2000): A former Roman general seeks revenge on the corrupt emperor who murdered his family and sentenced him to slavery. Features Crowe's iconic performance.
- A Beautiful Mind (2001): A Beautiful Mind is a 2001 American biographical drama film about the mathematician John Nash, a Nobel Laureate in Economics, played by Russell Crowe. Crowe won an Oscar for this role.
Why recommended? All showcase Crowe's range in historical dramas and character-driven stories."

IMPORTANT:  If the user asks for "all movies," "every movie," or something similar, DO NOT try to list every movie.
Instead, suggest a few popular movies from different genres.
Explain that listing all movies is not possible.
Do not add " or \h1 at the end of the response

Context: {context}"""
    user_info = """This is what we know about the user, and you can use this information to better tune your research:
Age: {age}
Gender: {gender}"""
    chat_history_part = """Chat History:
{chat_history}"""
    template_suffix= """Question: {question}
Your response:"""
    user_info=user_info.format(age=18, gender='male')
    COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ chat_history_part + "\n\n" + template_suffix
    PROMPT=PromptTemplate(template=COMBINED_PROMPT, input_variables=["context", "age", "gender", "chat_history", "question"])
    chain_type_kwargs = {"prompt": PROMPT, "memory":ConversationBufferMemory(memory_key="chat_history", input_key="question")}
    qa = RetrievalQA.from_chain_type(llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs=chain_type_kwargs)
    logger.info("Chain built successfully")
    return qa