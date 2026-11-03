
from neo4j import GraphDatabase
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def create_driver():
    uri = NEO4J_URI
    username = NEO4J_USER
    password = NEO4J_PASSWORD

    if not all([uri, username, password]):
        raise ValueError("Missing Neo4j credentials in .env")

    driver = GraphDatabase.driver(
        uri,
        auth=(username, password)
    )

    try:
        driver.verify_connectivity()
        return driver
    except Exception:
        driver.close()
        raise


def close_driver(driver):
    if driver is not None:
        driver.close()
