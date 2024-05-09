import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceEndpoint

st.title('🦜🔗 Yudi\'s Advise Generator Via LLM')

# HUGGINGFACEHUB_API_TOKEN = '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'
HUGGINGFACEHUB_API_TOKEN = st.sidebar.text_input('HuggingFace API KEY', value='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', type='password')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

your_endpoint_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
llm = HuggingFaceEndpoint(
    endpoint_url=f"{your_endpoint_url}",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

def generate_response_with_huggingface(user_instruction, text_to_respond):
  template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Generate an response in the same language."),
    ("human", "Here is the text to respond: {text_to_respond}"),
    ("ai", "add the following context {user_instruction}")
  ])

  chain = template | llm | StrOutputParser()
  response = chain.invoke (
      {
          "text_to_respond": text_to_respond,
          "user_instruction": user_instruction
      }
  )
  
  st.info(response)

with st.form('my_form'):
    user_persona_default = f"""A wise advisor who often give a short but concise answer."""
    user_instruction = st.text_area('Persona:', user_persona_default)
    default_question = f"""What are the three key pieces of advice for learning how to be a great leader?"""
    text_to_respond = st.text_area('Enter Question:', default_question)
    submitted = st.form_submit_button('Submit')

    if not HUGGINGFACEHUB_API_TOKEN.startswith('hf'):
        st.warning('Please enter your HuggingFace API key!', icon='⚠')
    if submitted and HUGGINGFACEHUB_API_TOKEN.startswith('hf'):        
        generate_response_with_huggingface(user_instruction, text_to_respond)