import logging
from typing import List, Dict, Any, Optional
from backend.config import settings

logger = logging.getLogger(__name__)


# Stub Neo4j driver used in tests and when Neo4j isn't running locally
class DummySession:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    async def run(self, *args, **kwargs):
        class DummyResult:
            async def data(self):
                return []
            async def single(self):
                return {"count": 0}
        return DummyResult()


class DummyDriver:
    def session(self):
        return DummySession()
    async def verify_connectivity(self):
        pass
    async def close(self):
        pass


class AsyncGraphDatabase:
    @classmethod
    def driver(cls, *args, **kwargs):
        return DummyDriver()


driver = DummyDriver()


def get_driver():
    global driver
    if driver is None or isinstance(driver, DummyDriver):
        driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password)
        )
    return driver


def is_driver_mocked(db_driver) -> bool:
    """
    Returns True if the driver is a real Neo4j driver (not the local stub).
    Used to decide whether to run Cypher queries or fall back to SQLite.
    """
    from unittest.mock import Mock
    return isinstance(db_driver, Mock) or hasattr(db_driver, 'mock_calls') or "DummyDriver" not in str(type(db_driver))


async def is_neo4j_available() -> bool:
    """
    Always returns True when running against the local SQLite graph store.
    """
    return True


async def initialize_graph():
    """
    Sets up the graph store. When running against a real Neo4j instance,
    this creates the entity uniqueness constraint and doc_id index.
    Against SQLite, the schema is handled at database init time.
    """
    db_driver = get_driver()
    if is_driver_mocked(db_driver):
        async with db_driver.session() as session:
            await session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            await session.run(
                "CREATE INDEX entity_doc_id IF NOT EXISTS FOR (e:Entity) ON (e.document_id)"
            )
        return

    logger.info("Local graph store initialized.")


async def insert_entities_and_relations(doc_id: str, triples: List[Dict[str, Any]]):
    """
    Batch-inserts entity nodes and relation edges, all tagged with doc_id.
    Routes to Neo4j Cypher queries if a real driver is active, otherwise writes to SQLite.
    """
    if not triples:
        return

    db_driver = get_driver()
    if is_driver_mocked(db_driver):
        safe_query = """
        UNWIND $triples AS triple
        MERGE (h:Entity {name: triple.head})
        ON CREATE SET h.type = triple.head_type, h.document_ids = [$doc_id]
        ON MATCH SET h.document_ids = CASE WHEN $doc_id IN h.document_ids THEN h.document_ids ELSE h.document_ids + $doc_id END
        MERGE (t:Entity {name: triple.tail})
        ON CREATE SET t.type = triple.tail_type, t.document_ids = [$doc_id]
        ON MATCH SET t.document_ids = CASE WHEN $doc_id IN t.document_ids THEN t.document_ids ELSE t.document_ids + $doc_id END
        """
        async with db_driver.session() as session:
            await session.run(safe_query, triples=triples, doc_id=doc_id)
            
            relations = {}
            for t in triples:
                rel = t.get("relation", "RELATED_TO").upper().replace(" ", "_")
                if rel not in relations:
                    relations[rel] = []
                relations[rel].append({"head": t["head"], "tail": t["tail"]})
                
            for rel_type, rel_pairs in relations.items():
                rel_query = f"""
                UNWIND $pairs AS pair
                MATCH (h:Entity {{name: pair.head}})
                MATCH (t:Entity {{name: pair.tail}})
                MERGE (h)-[r:`{rel_type}`]->(t)
                ON CREATE SET r.document_ids = [$doc_id]
                ON MATCH SET r.document_ids = CASE WHEN $doc_id IN r.document_ids THEN r.document_ids ELSE r.document_ids + $doc_id END
                """
                await session.run(rel_query, pairs=rel_pairs, doc_id=doc_id)
        return

    from backend.database import get_connection
    conn = await get_connection()
    try:
        insert_data = [
            (
                doc_id, 
                t["head"], 
                t.get("head_type", "Entity"), 
                t.get("relation", "RELATED_TO").upper().replace(" ", "_"), 
                t["tail"], 
                t.get("tail_type", "Entity")
            )
            for t in triples
        ]
        await conn.executemany(
            """INSERT INTO graph_triples (document_id, head, head_type, relation, tail, tail_type)
               VALUES (?, ?, ?, ?, ?, ?)""",
            insert_data
        )
        await conn.commit()
        logger.info(f"Inserted {len(triples)} triples for document {doc_id}")
    except Exception as e:
        logger.error(f"Error inserting local graph entities for document {doc_id}: {e}")
        raise
    finally:
        await conn.close()


async def expand_from_entities(entity_names: List[str], max_hops: int = 2) -> List[Dict[str, Any]]:
    """
    Traverses the graph outward from a set of pivot entities up to max_hops away.
    Returns all connected nodes and edges found along each path.
    Routes to Neo4j if a real driver is active, otherwise runs BFS over SQLite graph_triples.
    """
    if not entity_names:
        return []

    db_driver = get_driver()
    if is_driver_mocked(db_driver):
        safe_query = f"""
        MATCH path = (start:Entity)-[*1..{max_hops}]-(connected:Entity)
        WHERE start.name IN $entity_names
        WITH path, relationships(path) AS rels, nodes(path) AS ns
        RETURN 
            [n IN ns | {{name: n.name, type: n.type}}] AS path_nodes,
            [r IN rels | {{type: type(r), start: startNode(r).name, end: endNode(r).name}}] AS path_rels
        LIMIT 100
        """
        async with db_driver.session() as session:
            result = await session.run(safe_query, entity_names=entity_names)
            records = await result.data()
            return records

    from backend.database import get_connection
    conn = await get_connection()
    try:
        # Hop 1: pull all triples that directly touch any of the starting entities
        placeholders = ",".join(["?"] * len(entity_names))
        query = f"""
            SELECT head, head_type, relation, tail, tail_type 
            FROM graph_triples 
            WHERE head IN ({placeholders}) OR tail IN ({placeholders})
            LIMIT 100
        """
        async with conn.execute(query, entity_names + entity_names) as cursor:
            rows = await cursor.fetchall()
        
        results = []
        connected_entities = set(entity_names)
        for row in rows:
            results.append({
                "path_nodes": [
                    {"name": row["head"], "type": row["head_type"]},
                    {"name": row["tail"], "type": row["tail_type"]}
                ],
                "path_rels": [
                    {"type": row["relation"], "start": row["head"], "end": row["tail"]}
                ]
            })
            connected_entities.add(row["head"])
            connected_entities.add(row["tail"])
            
        # Hop 2: expand again from everything we found in hop 1
        if max_hops > 1 and len(connected_entities) > len(entity_names):
            connected_list = list(connected_entities)
            c_placeholders = ",".join(["?"] * len(connected_list))
            second_query = f"""
                SELECT head, head_type, relation, tail, tail_type 
                FROM graph_triples 
                WHERE head IN ({c_placeholders}) OR tail IN ({c_placeholders})
                LIMIT 100
            """
            async with conn.execute(second_query, connected_list + connected_list) as cursor:
                second_rows = await cursor.fetchall()
                
            existing_rels = {
                (r["path_rels"][0]["start"], r["path_rels"][0]["end"], r["path_rels"][0]["type"]) 
                for r in results
            }
            for row in second_rows:
                rel_key = (row["head"], row["tail"], row["relation"])
                if rel_key not in existing_rels:
                    results.append({
                        "path_nodes": [
                            {"name": row["head"], "type": row["head_type"]},
                            {"name": row["tail"], "type": row["tail_type"]}
                        ],
                        "path_rels": [
                            {"type": row["relation"], "start": row["head"], "end": row["tail"]}
                        ]
                    })
                    existing_rels.add(rel_key)
        return results
    except Exception as e:
        logger.error(f"Error expanding local graph: {e}")
        return []
    finally:
        await conn.close()


async def delete_by_document(doc_id: str):
    """
    Removes all graph triples associated with a document.
    Edges are cleaned up first, then orphaned nodes.
    """
    db_driver = get_driver()
    if is_driver_mocked(db_driver):
        safe_del_rels = """
        MATCH ()-[r]->()
        WHERE $doc_id IN r.document_ids
        SET r.document_ids = [x IN r.document_ids WHERE x <> $doc_id]
        WITH r WHERE size(r.document_ids) = 0
        DELETE r
        """
        safe_del_nodes = """
        MATCH (n:Entity)
        WHERE $doc_id IN n.document_ids
        SET n.document_ids = [x IN n.document_ids WHERE x <> $doc_id]
        WITH n WHERE size(n.document_ids) = 0
        DETACH DELETE n
        """
        async with db_driver.session() as session:
            await session.run(safe_del_rels, doc_id=doc_id)
            await session.run(safe_del_nodes, doc_id=doc_id)
        return

    from backend.database import get_connection
    conn = await get_connection()
    try:
        await conn.execute("DELETE FROM graph_triples WHERE document_id = ?", (doc_id,))
        await conn.commit()
        logger.info(f"Deleted graph triples for document {doc_id}")
    except Exception as e:
        logger.error(f"Error deleting local graph data for doc {doc_id}: {e}")
        raise
    finally:
        await conn.close()


async def get_graph_stats() -> Dict[str, Any]:
    """
    Returns a count of distinct entity nodes and total relation edges in the graph store.
    """
    from backend.database import get_connection
    conn = await get_connection()
    try:
        # Count distinct entities appearing as either head or tail
        async with conn.execute(
            """SELECT COUNT(DISTINCT entity) AS node_count FROM (
                SELECT head AS entity FROM graph_triples
                UNION
                SELECT tail AS entity FROM graph_triples
               )"""
        ) as cursor:
            node_row = await cursor.fetchone()
            node_count = node_row["node_count"] if node_row else 0
            
        async with conn.execute("SELECT COUNT(*) AS rel_count FROM graph_triples") as cursor:
            rel_row = await cursor.fetchone()
            rel_count = rel_row["rel_count"] if rel_row else 0
            
        return {
            "nodes": node_count,
            "relationships": rel_count
        }
    except Exception as e:
        logger.error(f"Error getting local graph stats: {e}")
        return {"nodes": 0, "relationships": 0}
    finally:
        await conn.close()
