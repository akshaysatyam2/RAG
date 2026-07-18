import logging
from typing import List, Dict, Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from backend.config import settings

logger = logging.getLogger(__name__)

# Initialize Neo4j driver
driver: Optional[AsyncDriver] = None

def get_driver() -> AsyncDriver:
    global driver
    if driver is None:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password)
        )
    return driver


async def initialize_graph():
    """
    Create constraints and indexes.
    """
    db_driver = get_driver()
    async with db_driver.session() as session:
        try:
            # Create constraint on Entity names to be unique
            await session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            # Create index on document_id for efficient cleanup and filtering
            await session.run(
                "CREATE INDEX entity_doc_id IF NOT EXISTS FOR (e:Entity) ON (e.document_id)"
            )
            logger.info("Successfully initialized Neo4j constraints and indexes.")
        except Exception as e:
            logger.error(f"Error initializing Neo4j graph: {e}")
            raise


async def insert_entities_and_relations(doc_id: str, triples: List[Dict[str, Any]]):
    """
    Batch insert nodes+edges tagged with doc_id.
    triples format: [{"head": "E1", "head_type": "T1", "relation": "R", "tail": "E2", "tail_type": "T2"}]
    """
    if not triples:
        return

    query = """
    UNWIND $triples AS triple
    
    // Merge Head
    MERGE (h:Entity {name: triple.head})
    ON CREATE SET h.type = triple.head_type, h.document_ids = [$doc_id]
    ON MATCH SET h.document_ids = 
        CASE 
            WHEN NOT $doc_id IN h.document_ids THEN h.document_ids + $doc_id 
            ELSE h.document_ids 
        END
        
    // Merge Tail
    MERGE (t:Entity {name: triple.tail})
    ON CREATE SET t.type = triple.tail_type, t.document_ids = [$doc_id]
    ON MATCH SET t.document_ids = 
        CASE 
            WHEN NOT $doc_id IN t.document_ids THEN t.document_ids + $doc_id 
            ELSE t.document_ids 
        END
        
    // Merge Relationship
    WITH h, t, triple, $doc_id AS doc_id
    CALL apoc.merge.relationship(h, triple.relation, {}, {}, t, {}) YIELD rel
    
    // Add document_id to relationship
    SET rel.document_ids = 
        CASE 
            WHEN rel.document_ids IS NULL THEN [doc_id]
            WHEN NOT doc_id IN rel.document_ids THEN rel.document_ids + doc_id 
            ELSE rel.document_ids 
        END
    """
    
    # We use APOC for dynamic relationship merge, which is standard in GraphRAG.
    # Alternatively, we could group by relation type, but APOC is cleaner.
    # If APOC is not available, we have to group by relation in Python and run multiple queries.
    # We will do a generic CYPHER without APOC just to be safe and compatible:
    
    safe_query = """
    UNWIND $triples AS triple
    
    MERGE (h:Entity {name: triple.head})
    ON CREATE SET h.type = triple.head_type, h.document_ids = [$doc_id]
    ON MATCH SET h.document_ids = 
        CASE WHEN $doc_id IN h.document_ids THEN h.document_ids ELSE h.document_ids + $doc_id END
        
    MERGE (t:Entity {name: triple.tail})
    ON CREATE SET t.type = triple.tail_type, t.document_ids = [$doc_id]
    ON MATCH SET t.document_ids = 
        CASE WHEN $doc_id IN t.document_ids THEN t.document_ids ELSE t.document_ids + $doc_id END
    """
    
    db_driver = get_driver()
    async with db_driver.session() as session:
        try:
            # First insert nodes
            await session.run(safe_query, triples=triples, doc_id=doc_id)
            
            # Then insert relations grouped by type to avoid APOC requirement
            # Group by relation
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
                ON MATCH SET r.document_ids = 
                    CASE WHEN $doc_id IN r.document_ids THEN r.document_ids ELSE r.document_ids + $doc_id END
                """
                await session.run(rel_query, pairs=rel_pairs, doc_id=doc_id)
                
        except Exception as e:
            logger.error(f"Error inserting entities for document {doc_id}: {e}")
            raise


async def expand_from_entities(entity_names: List[str], max_hops: int = 2) -> List[Dict[str, Any]]:
    """
    Graph traversal from pivot entities.
    Returns connected subgraphs/paths.
    """
    if not entity_names:
        return []

    # Safe variable length path match up to max_hops
    query = """
    MATCH (start:Entity)
    WHERE start.name IN $entity_names
    CALL apoc.path.subgraphAll(start, {
        maxLevel: $max_hops,
        relationshipFilter: ">"
    })
    YIELD nodes, relationships
    RETURN nodes, relationships
    """
    
    # Non-APOC alternative for finding connected entities
    safe_query = f"""
    MATCH path = (start:Entity)-[*1..{max_hops}]-(connected:Entity)
    WHERE start.name IN $entity_names
    WITH path, relationships(path) AS rels, nodes(path) AS ns
    RETURN 
        [n IN ns | {{name: n.name, type: n.type}}] AS path_nodes,
        [r IN rels | {{type: type(r), start: startNode(r).name, end: endNode(r).name}}] AS path_rels
    LIMIT 100
    """

    db_driver = get_driver()
    results = []
    async with db_driver.session() as session:
        try:
            result = await session.run(safe_query, entity_names=entity_names)
            records = await result.data()
            for record in records:
                results.append(record)
            return results
        except Exception as e:
            logger.error(f"Error expanding from entities: {e}")
            return []


async def delete_by_document(doc_id: str):
    """
    Cascade remove all nodes/edges for a document.
    """
    query_rels = """
    MATCH ()-[r]->()
    WHERE $doc_id IN r.document_ids
    WITH r, 
         [x IN r.document_ids WHERE x <> $doc_id] AS new_docs
    CALL {
        WITH r, new_docs
        WITH r, new_docs WHERE size(new_docs) = 0
        DELETE r
        RETURN count(r) AS deletedRels
        UNION
        WITH r, new_docs
        WITH r, new_docs WHERE size(new_docs) > 0
        SET r.document_ids = new_docs
        RETURN 0 AS deletedRels
    }
    RETURN true
    """
    
    query_nodes = """
    MATCH (n:Entity)
    WHERE $doc_id IN n.document_ids
    WITH n, 
         [x IN n.document_ids WHERE x <> $doc_id] AS new_docs
    CALL {
        WITH n, new_docs
        WITH n, new_docs WHERE size(new_docs) = 0
        DETACH DELETE n
        RETURN count(n) AS deletedNodes
        UNION
        WITH n, new_docs
        WITH n, new_docs WHERE size(new_docs) > 0
        SET n.document_ids = new_docs
        RETURN 0 AS deletedNodes
    }
    RETURN true
    """
    
    # Cypher 4.x/5.x compatible simplified version without nested CALL
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

    db_driver = get_driver()
    async with db_driver.session() as session:
        try:
            await session.run(safe_del_rels, doc_id=doc_id)
            await session.run(safe_del_nodes, doc_id=doc_id)
            logger.info(f"Deleted graph components for document {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting graph data for doc {doc_id}: {e}")
            raise


async def get_graph_stats() -> Dict[str, Any]:
    """
    Get node/edge counts.
    """
    db_driver = get_driver()
    async with db_driver.session() as session:
        try:
            node_res = await session.run("MATCH (n) RETURN count(n) AS count")
            node_count = (await node_res.single())["count"]
            
            rel_res = await session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            rel_count = (await rel_res.single())["count"]
            
            return {
                "nodes": node_count,
                "relationships": rel_count
            }
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"nodes": 0, "relationships": 0}
