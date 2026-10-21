

# Load .env from the project root (adjust if needed).

import neo4j
import ollama
import os
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)
from neo4j import GraphDatabase
from .db import create_driver

# db=Neo4jConnection()
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

def traverse_graph(driver, start_node_name, max_hops=2):
    # Validate the hop limit before inserting it into the query
    if not isinstance(max_hops, int) or not 1 <= max_hops <= 5:
        raise ValueError("max_hops must be an integer between 1 and 5")

    query = f"""
    MATCH path = (start)-[*1..{max_hops}]-(connected)
    Where start.name = $node
    UNWIND relationships(path) AS r
    WITH DISTINCT r
    RETURN
        startNode(r).name AS start_node,
        labels(startNode(r)) AS start_labels,
        type(r) AS relation,
        labels(endNode(r)) AS relation_end_labels,
        endNode(r).name AS relation_end_node
    LIMIT 100
    """

    result, _, _ = driver.execute_query(
        query,
        node=start_node_name,
    )

    return result


def find_relevent_nodes(driver, question, top_k=3):
    question+=" I am  Vikram Singh"
    question_embedding = model.encode(question).tolist()

    # 1. Retrieve the top-k semantically relevant Intern nodes
    results, _, _ = driver.execute_query(
        """
        MATCH (emp:Intern)
        SEARCH emp IN (
            VECTOR INDEX intern_embedding_index
            FOR $queryEmbedding
            LIMIT $top_k
        ) SCORE AS similarityScore
        RETURN emp.name AS name,
               emp.description AS description,
               similarityScore
        """,
        queryEmbedding=question_embedding,
        top_k=top_k,
    )

    return results
    
def fetch_nodes_context(set_of_nodes, driver):
    nodes_context = []
    
    for node_name in set_of_nodes:
        node_results, _, _ = driver.execute_query(
            """
            MATCH (n)
            WHERE n.name = $nodeName
            RETURN labels(n) AS labels,
                    properties(n) AS details
            LIMIT 10
            """,
            nodeName=node_name,
        )

        for node_record in node_results:
            details = dict(node_record["details"] or {})
            details.pop("embedding", None)

            nodes_context.append(
                f"Labels: {node_record['labels']}, Properties: {details}"
            )

    # 4. Build the final context string
    return nodes_context

def graph_answer(driver, question, top_k=3):
    question+=" I am  Vikram Singh"
    relavant_nodes_context = find_relevent_nodes(driver, question, top_k)
    relation_context = []
    set_of_nodes = set()
    
    for record in relavant_nodes_context:
        start_name = record["name"]

        if not start_name:
            continue

        set_of_nodes.add(start_name)

        relations = traverse_graph(driver, start_name, max_hops=2)

        for relation in relations:
            data = relation.data()

            start = data.get("start_node")
            end = data.get("relation_end_node")
            relation_type = data.get("relation")
            end_labels = data.get("relation_end_labels") or []

            if start:
                set_of_nodes.add(start)
            if end:
                set_of_nodes.add(end)

            relation_context.append(
                f"{start} -[{relation_type}]-> {end} "
                f"(end labels: {end_labels})"
            )

    nodes_context = fetch_nodes_context(set_of_nodes, driver)
    context = f"""
    I am  Vikram Singh
    Nodes context:
    {chr(10).join(nodes_context)}

    Relationship context:
    {chr(10).join(relation_context)}
    """

    return context

    # 3. Retrieve node properties without embeddings
  
# def retrieve(question, top_k=5):
#     # Generate query embedding locally
#     query_embedding = model.encode(question)

#     # Retrieve nearest nodes using the existing vector index
    
#     results = db.read('''
#         MATCH (emp:Employee)
#           SEARCH emp IN (
#             VECTOR INDEX employee_embedding_index
#             FOR $queryEmbedding
#             LIMIT 5
#           ) SCORE AS similarityScore
#         RETURN emp.name AS name, emp.description AS description, similarityScore
#         ''', params={
#             "top_k": top_k,
#             "queryEmbedding": query_embedding
#         })
    
#     return results


# if __name__ == "__main__":
    
#     try:
#         results = retrieve("Who is Siddharth?")
#         print(type(results))
#         print(results.__dict__)
#         # for item in results:
#         #     print(item)
#     finally:
#         db.close()

# with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
#     driver.verify_connectivity()
# def find_relevent_nodes(driver,question, top_k=3):
#     question_embedding = model.encode(question).toList()
    
    # results, _, _ = driver.execute_query(
    #     """
    #     MATCH (i:Intern)
    #     RETURN
    #         i.name AS name,
    #         i.description AS description;
    #     """
    # )
    # for record in results:
    #     data=record.data()
    #     about=data.get("name")+" "+data.get("description")
    #     embedding = model.encode(about)
    #     driver.execute_query("""
    #     MATCH (i:Intern)
    #     WHERE i.name = $name
    #     SET i.embedding = $embedding
    #     """,embedding=embedding,name=data.get("name"))
#     results, _, _ = driver.execute_query('''
#         MATCH (emp:Intern)
#             SEARCH emp IN (
#             VECTOR INDEX intern_embedding_index
#             FOR $queryEmbedding
#             LIMIT $topK
#             ) SCORE AS similarityScore
#         RETURN emp.name AS name, emp.description AS description, similarityScore
#         ''', queryEmbedding=question_embedding,topK=top_k)
#     # # print(f'Employees whose description and name relate to `{question}`:')
#     relation_context=""
#     set_of_nodes=set()
#     for record in results:
#         relations=traverse_graph(driver,record.data().get("name"),2)
#         for relation in relations:
#             data=relation.data()
#             set_of_nodes.add(data.get("relation_end_node"))
#             relation_context+=data.get("start_node")+" "+data.get("relation")+" "+data.get("relation_end_node")+" "+data.get("relation_end_labels")[0]+"\n"
#     nodes_context=""
#     for node in set_of_nodes:
#         node_data, _, _ = driver.execute_query('''
#             MATCH (n {name:$nodeName})
#             RETURN apoc.map.removeKey(properties(n), 'embedding') AS details;
#             ''', nodeName=node)
#         nodes_context+=str(node_data)+"\n"
#     context="""
#      nodes_context:
#      {nodes_context}
#      relation_context:
#      {relation_context}
#     """  
#     return context