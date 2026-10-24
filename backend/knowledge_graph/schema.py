"""Neo4j constraints and indexes. Safe to run repeatedly."""

STATEMENTS = [
    "CREATE CONSTRAINT entity_uid IF NOT EXISTS FOR (n:Entity) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT alias_aid IF NOT EXISTS FOR (a:Alias) REQUIRE a.aid IS UNIQUE",
    "CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX alias_norm IF NOT EXISTS FOR (a:Alias) ON (a.norm)",
    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]",
    "CREATE CONSTRAINT sourcedoc_sid IF NOT EXISTS FOR (d:SourceDoc) REQUIRE d.sid IS UNIQUE",
    "CREATE CONSTRAINT chunk_cid IF NOT EXISTS FOR (c:Chunk) REQUIRE c.cid IS UNIQUE",
    "CREATE INDEX sourcedoc_doc_id IF NOT EXISTS FOR (d:SourceDoc) ON (d.doc_id)",
    "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
    "CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX alias_ft IF NOT EXISTS FOR (a:Alias) ON EACH [a.alias]",
]


def ensure_schema(run):
    for statement in STATEMENTS:
        run(statement)
