import streamlit as st # Streamlit library used for styling and frontend
from langchain.prompts import PromptTemplate # PromptTemplate class used to generate prompts
from langchain.llms import CTransformers # CTransformers class used to generate responses from LLma 2 model

## Function to get response from LLma 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    ### LLama2 model
    llm = CTransformers(model='Blog Generation LLM App/llama-2-7b-chat.ggmlv3.q8_0.bin', model_type='llama', config={'max_new_tokens': 256, 'temperature': 0.01}) 

    ## Prompt Template
    template = """
    Write a blog for a {blog_style} job profile on the topic "{input_text}" 
    within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)

    # Generate the response from LLama2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response

# Streamlit app setup
st.set_page_config(page_title="Generate Blogs", page_icon="🤖", layout="centered", initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖")

# Input for the blog topic
input_text = st.text_input("Enter the Blog Topic")

# Columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("Number of Words")  # Input for number of words
with col2:
    blog_style = st.selectbox("Writing the blog for", 
                              ('Researchers', 'Data Scientists', 'Common People'), index=0)  # Blog style dropdown

# Generate blog button
submit = st.button("Generate Blog")

# Final response display
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
