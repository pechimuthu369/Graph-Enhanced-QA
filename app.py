import streamlit as st
from langchain.graphs import Neo4jGraph
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import JSONLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import tempfile
import json

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    neo4j_url = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    return neo4j_url, neo4j_username, neo4j_password, openai_api_key

def connect_to_neo4j(url, username, password):
    """Establish connection to Neo4j database."""
    try:
        graph = Neo4jGraph(url=url, username=username, password=password)
        st.success("Connected to Neo4j database")
        return graph
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

def process_json_file(file, graph, llm):
    """Process uploaded JSON file and add data to Neo4j graph."""
    json_data = json.loads(file.read())
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmp_file:
        json.dump(json_data, tmp_file)
        tmp_file_path = tmp_file.name

    loader = JSONLoader(file_path=tmp_file_path, jq_schema='.', text_content=False)
    documents = loader.load()

    allowed_nodes = ["Person", "Journal", "Paper", "Wikipage"]
    allowed_relationships = ["is_author_of", "was_published_in", "mention_concept", "mention_paper", "cited"]

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=False,
        relationship_properties=False
    )
    graph_documents = transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, include_source=True)

def setup_qa_chain(llm, graph):
    """Set up the question-answering chain."""
    schema = graph.get_schema
    template = """
    Cypher Example:
    MATCH (j2:Journal)
    WHERE j2.id <> "Journal of Artificial Intelligence Research"
    Task: Generate a Cypher statement to query the graph database.
    Instructions: Use only relationship types and properties provided in schema.
    Do not use other relationship types or properties that are not provided.
    schema: {schema}
    Note: Do not include explanations or apologies in your answers.
    Do not answer questions that ask anything other than creating Cypher statements.
    Do not include any text other than generated Cypher statements.
    Question: {question}
    """
    question_prompt = PromptTemplate(template=template, input_variables=["schema", "question"])
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=question_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )

def main():
    st.set_page_config(layout="wide", page_title="Graph-Enhanced QA")
    st.title("Graph-Enhanced Question Answering")

    neo4j_url, neo4j_username, neo4j_password, openai_api_key = load_environment_variables()

    if not all([neo4j_url, neo4j_username, neo4j_password, openai_api_key]):
        st.error("Missing required environment variables. Please check your .env file.")
        return

    os.environ['OPENAI_API_KEY'] = openai_api_key

    llm = ChatOpenAI(model_name="gpt-4")
    embeddings = OpenAIEmbeddings()

    graph = connect_to_neo4j(neo4j_url, neo4j_username, neo4j_password)
    if not graph:
        return

    st.subheader("Upload dataset to Graph Database")
    uploaded_file = st.file_uploader("Select a JSON file", type="json")
    overwrite = st.checkbox("Overwrite existing data")

    if uploaded_file and st.button("Process file"):
        with st.spinner("Processing the JSON file..."):
            try:
                if overwrite:
                    graph.query("MATCH (n) DETACH DELETE n;")
                process_json_file(uploaded_file, graph, llm)
                st.success(f"{uploaded_file.name} has been processed and added to the graph database.")
            except Exception as e:
                st.error(f"Error processing JSON file: {e}")

    qa = setup_qa_chain(llm, graph)

    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        with st.spinner("Generating answer..."):
            try:
                res = qa.invoke({"query": question})
                st.write("\n**Answer:**\n" + res['result']) 
            except Exception as e:
                st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()