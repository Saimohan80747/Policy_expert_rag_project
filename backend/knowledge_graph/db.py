
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def create_driver():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

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
