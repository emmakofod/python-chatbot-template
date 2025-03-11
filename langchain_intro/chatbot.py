import dotenv
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI  # Use ChatMistralAI for Mistral models
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
import os
dotenv.load_dotenv()
print(os.getenv("HF_TOKEN")) #testing if token is loaded 

# Define your prompt template for patient reviews
review_template_str = """Your job is to use patient
reviews to answer questions about their experience at
a hospital. Use the following context to answer questions.
Be as detailed as possible, but don't make up any information
that's not from the context. If you don't know an answer, say
you don't know.

{context}
"""

# Define the system message template
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

# Define the human message template
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)

# Define the messages to be passed into the model
messages = [review_system_prompt, review_human_prompt]

# Set up the chat prompt template with the context and question
review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

# Initialize Mistral API-based model (ChatMistralAI)
chat_model = ChatMistralAI(
    model="mistral-small",  # You can change to "mistral-medium" or "mistral-large"
    temperature=0.7,
    hf_token=os.getenv("HF_TOKEN"),
)

# Chain the prompt template with the Mistral model
review_chain = review_prompt_template | chat_model

# Example input to run the chain
context = "This is the patient's review: The hospital staff was friendly, but the wait time was long, like 46 minutes and thats wayyyyyy too long."
question = "What was the experience with the hospital staff?"

# Get the response from the model
response = review_chain.invoke({"context": context, "question": question})
formatted_response = response.content.split("\n")[-1]

print(formatted_response)

