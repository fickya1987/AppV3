import streamlit as st
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import DuckDuckGoSearchRun, YouTubeSearchTool, PubmedQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain import LLMMathChain
from youtubesearchpython import VideosSearch

# Load environment variables
load_dotenv()

# Ensure OpenAI API Key is loaded
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    st.error("Please set the OpenAI API Key in your .env file")

# Initialize OpenAI LLM
llm = OpenAI(temperature=0.7)

def create_research_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                introduction TEXT,
                quant_facts TEXT,
                publications TEXT,
                books TEXT,
                ytlinks TEXT,
                prev_ai_research TEXT
            )
        """)

def insert_research(user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research))

def read_research_table():
    with sqlite3.connect('MASTER.db') as conn:
        query = "SELECT * FROM Research"
        df = pd.read_sql_query(query, conn)
    return df

def generate_research(userInput):
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()
    pubmed = PubmedQueryRun()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        Tool(name="Wikipedia Research", func=wiki.run, description="Search Wikipedia"),
        Tool(name="Internet Search", func=DDGsearch.run, description="Search the web"),
        Tool(name="YouTube Search", func=YTsearch.run, description="Find YouTube videos"),
        Tool(name="Math Solver", func=llm_math_chain.run, description="Solve mathematical problems"),
        Tool(name="PubMed Research", func=pubmed.run, description="Search medical research")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

    with st.expander("Generated Research", expanded=True):
        st.subheader("User Input:")
        st.write(userInput)

        st.subheader("Introduction:")
        with st.spinner("Generating..."):
            intro = agent(f'Write an academic introduction about {userInput}')
            st.write(intro['output'])

        st.subheader("Quantitative Facts:")
        with st.spinner("Fetching statistics..."):
            quantFacts = agent(f'Generate 3-5 statistical facts about {userInput}')
            st.write(quantFacts['output'])

        st.subheader("Recent Publications:")
        with st.spinner("Searching for papers..."):
            papers = agent(f'Find 2-3 recent academic papers related to {userInput}')
            st.write(papers['output'])

        st.subheader("Recommended Books:")
        with st.spinner("Searching books..."):
            books = agent(f'List 5 recommended books on {userInput}')
            st.write(books['output'])

        st.subheader("YouTube Links:")
        with st.spinner("Fetching videos..."):
            search = VideosSearch(userInput, limit=5)
            ytlinks = "\n".join([f"{i+1}. {vid['title']} - [Watch](https://www.youtube.com/watch?v={vid['id']})" for i, vid in enumerate(search.result()['result'])])
            st.write(ytlinks)

        insert_research(userInput, intro['output'], quantFacts['output'], papers['output'], books['output'], ytlinks, "")

        embedding_function = OpenAIEmbeddings()
        vectordb = Chroma.from_texts([userInput, intro['output'], quantFacts['output'], papers['output'], books['output'], ytlinks], embedding_function, persist_directory="./chroma_db")
        vectordb.persist()
        st.session_state.embeddings_db = vectordb

def main():
    st.set_page_config(page_title="Research Bot")
    create_research_db()

    embedding_function = OpenAIEmbeddings()
    if os.path.exists("./chroma_db"):
        st.session_state.embeddings_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    st.header("GPT-4 LangChain Research Bot")
    deploy_tab, prev_tab = st.tabs(["Generate Research","Previous Research"])

    with deploy_tab:
        userInput = st.text_area(label="Enter your research topic")
        if st.button("Generate Report") and userInput:
            generate_research(userInput)

    with prev_tab:
        df = read_research_table()
        if not df.empty:
            st.dataframe(df)
            selected_input = st.selectbox(label="Select Previous Input", options=df["user_input"])
            if st.button("Show Research"):
                selected_df = df[df["user_input"] == selected_input].reset_index(drop=True)
                st.write(selected_df.iloc[0])

if __name__ == '__main__':
    main()
