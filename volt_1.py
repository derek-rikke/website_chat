import streamlit as st
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import OpenAIEmbeddings
import qdrant_client
import json
import numpy as np

st.set_page_config(page_title=None,
                   page_icon=None,
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

def get_vector_store():
    client = qdrant_client.QdrantClient(st.secrets["QDRANT_HOST"], api_key=st.secrets["QDRANT_API_KEY"])
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(client, collection_name=st.secrets["QDRANT_COLLECTION_NAME"], embeddings=embeddings)
    return vector_store

def create_crc_llm(vector_store):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever()
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    st.session_state.crc = crc
    return crc

def add_flair(crc_with_source, history):
    # Define prefix that explains the prompt
    prefix = """ 
    You are a large language model that receives an answer instead of a query, from another LLM.
    The answer that you recieve is the ansewr to a question that was asked by the user about the
    lore and history of the fictional land called The Frontierverse. 
    You're job is to modify the answer so the user feels like they are having a conversation with
    the character Volt. Volt is seasoned adventurer in the Frontierverse and has gained an immense
    amount of knowledge through his travels. He has a dry wit, but loves to talk about the amazing
    lore of the Frontierverse.
    a conversation 
    """
    # load examples
    examples = [
    {
        "query": "The protagonist of our story in Modern Genesis City, Night Ape is a vigilante and detective, patrolling areas where the Heroes League and police don’t have jurisdiction, such as the Docks and DeGen Alley.",
        "answer": """
        Ah, gather 'round and lend an ear, my neon-washed friend, for I'm about to unravel the saga of an enigmatic soul who haunts the shadow-laden corridors of Modern Genesis City. Enter Night Ape, not your garden-variety hero but a rogue vigilante and a sleuth with a penchant for the places where law dares not tread. This solitary sentinel patrols the forgotten fringes—yes, the Docks and the infamous DeGen Alley—realms abandoned by the vainglorious Heroes League and the overmatched boys in blue. Here, amidst the neon glow and the perpetual twilight, Night Ape reigns supreme, a dark knight uncloaked, solving mysteries that whisper through the damp, electric air."""
    },
    {
        "query": "The head of LD Industries, Alexander Yowleus Duma, or Lord Duma as he calls himself, is one of Genesis City's most prominent businessmen. A public figure who doesn't make a point of hiding his discontent with the Heroes League and the world created under the heroes' tutelage.",
        "answer": """
        Ah, let me paint you a picture of a titan amongst mere mortals in the glimmering high-rises of Genesis City. Behold Alexander Yowleus Duma, the grand poobah of LD Industries, though he fancies himself as Lord Duma, a title as grandiose as his towering ambitions. This captain of industry strides through the city like a colossus, a veritable fixture in the glossy social scene, his gaze heavy with a disdain that’s as public as the neon signs illuminating our urban sprawl. Make no mistake, this luminary is no fan of the Heroes League—oh no! He openly scoffs at the so-called utopia forged under their mighty hands. A master of the boardroom, a skeptic of the superpowered, Lord Duma stands as a beacon of defiance in a world policed by capes and cowls."""
    }
    ]
    example_format = """Human: {query}\nAI: {answer}"""
    example_template = PromptTemplate(input_variables=["query", "answer"], template=example_format)
    suffix = """\n\nHuman: {query}\nAI:"""
    # Construct few-shot prompt template
    prompt_template = FewShotPromptTemplate(examples=examples,
                                            example_prompt=example_template,
                                            input_variables=["query"],
                                            prefix=prefix,
                                            suffix=suffix,
                                            example_separator="\n")
    # Create new llm instance
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    # Create chain with llm and prompt template
    chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)
    # Run chain on query
    result = chain.invoke({"query": crc_with_source, "chat_history": history})
    return result["text"]

def process_user_message(user_message):
    with st.spinner("Thinking..."):
        crc_response = st.session_state.crc.run({'question': user_message, 'chat_history': st.session_state.history})
        final_response = add_flair(crc_response, st.session_state.history)
        st.session_state.history.append((user_message, final_response))  # Append to history in session state

def display_last_response():
    if st.session_state.history:
        last_message, last_response = st.session_state.history[-1]
        st.markdown("**Volt:**")
        st.write(last_response)

def display_history():
    with st.sidebar:
        st.subheader("Session History")
        for idx, (message, response) in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Conversation {idx + 1}"):
                st.markdown("**You:**")
                st.write(message)
                st.markdown("**Volt:**")
                st.write(response)

def main():
    st.title('Ask Volt about Frontierverse Lore')
    st.header("What knowledge do you desire??? 💬")
    st.markdown("""
    <style>
    .small-font {
        font-size:16px !important;
        font-weight: normal;
    }
    </style>
    <div class='small-font'>
        Non Monotonic Moms Inc.
    </div>
    """, unsafe_allow_html=True)


    # Initialize necessary components and state if not already done
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vector_store()

    if 'crc' not in st.session_state:
        st.session_state.crc = create_crc_llm(st.session_state.vector_store)

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Text input for user message
    user_message = st.text_input('You:', key='user_input_text', placeholder='Type your message here...')
    st.caption("Press Enter to submit your question. Remember to clear the text box for new questions.")

    # Handle user message input
    if user_message and (user_message != st.session_state.get('last_message', '')):
        st.session_state.last_message = user_message  # Save the last message to session state
        process_user_message(user_message)

    # Display the last response just below the text input
    display_last_response()

    # Display the entire conversation history in the sidebar
    display_history()

if __name__ == '__main__':
    main()