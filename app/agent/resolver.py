import logging
import re
from typing import Optional, List
from pydantic import BaseModel
from app.core.database import neo4j_driver

logger = logging.getLogger(__name__)

class PlantIdentity(BaseModel):
    local_name: str
    scientific_name: Optional[str] = None
    family: Optional[str] = None
    synonyms: List[str] = []
    confidence: float = 1.0
    sources: List[str] = ["Neo4j Graph DB"]

# Static fallback map for 100% correctness of common Indonesian herbs
STATIC_HERB_MAP = {
    "temulawak": {
        "local_name": "Temulawak",
        "scientific_name": "Curcuma xanthorrhiza Roxb.",
        "family": "Zingiberaceae",
        "synonyms": ["Koneng Gede", "Temu Labak"],
        "confidence": 1.0,
        "sources": ["Static Verification Map"]
    },
    "kunyit": {
        "local_name": "Kunyit",
        "scientific_name": "Curcuma longa L.",
        "family": "Zingiberaceae",
        "synonyms": ["Kunir", "Koneng", "Kunyir"],
        "confidence": 1.0,
        "sources": ["Static Verification Map"]
    },
    "jahe": {
        "local_name": "Jahe",
        "scientific_name": "Zingiber officinale Roscoe",
        "family": "Zingiberaceae",
        "synonyms": ["Hali", "Alia", "Lia"],
        "confidence": 1.0,
        "sources": ["Static Verification Map"]
    },
    "kencur": {
        "local_name": "Kencur",
        "scientific_name": "Kaempferia galanga L.",
        "family": "Zingiberaceae",
        "synonyms": ["Cikur", "Kencor"],
        "confidence": 1.0,
        "sources": ["Static Verification Map"]
    },
    "lengkuas": {
        "local_name": "Lengkuas",
        "scientific_name": "Alpinia galanga (L.) Willd.",
        "family": "Zingiberaceae",
        "synonyms": ["Laos", "Laja"],
        "confidence": 1.0,
        "sources": ["Static Verification Map"]
    }
}

def resolve_plant_identity(query: str) -> Optional[PlantIdentity]:
    """
    Ekstrak nama tanaman dari query dan cari identitas kanonik di Neo4j/Static Map.
    """
    if not query:
        return None

    query_lower = query.lower()

    # 1. Check static map first for ultra-reliability
    for key, data in STATIC_HERB_MAP.items():
        if key in query_lower:
            logger.info(f"Resolved plant identity from static map: {key}")
            return PlantIdentity(**data)

    # 2. Query Neo4j
    try:
        # Search by commonName or latinName
        cypher = """
        MATCH (h:Herb)
        WHERE toLower(h.commonName) CONTAINS toLower($q)
           OR toLower(h.latinName) CONTAINS toLower($q)
        OPTIONAL MATCH (h)-[:BELONGS_TO]->(f:Family)
        RETURN h.commonName AS commonName, h.latinName AS latinName,
               f.name AS family, h.localNames AS localNames
        LIMIT 1
        """
        # Try to find a specific word in the query
        words = [w.strip() for w in re.findall(r'\b\w{4,}\b', query_lower) if w.strip()]
        for word in words:
            if word in ("untuk", "sakit", "obat", "yang", "pada", "bagaimana", "cara", "resep"):
                continue
            records, _, _ = neo4j_driver.execute_query(cypher, q=word)
            if records:
                rec = records[0]
                return PlantIdentity(
                    local_name=rec.get("commonName") or word.capitalize(),
                    scientific_name=rec.get("latinName"),
                    family=rec.get("family"),
                    synonyms=rec.get("localNames") or [],
                    confidence=0.9,
                    sources=["Neo4j Graph DB"]
                )
    except Exception as e:
        logger.error(f"Error in resolve_plant_identity: {e}")

    return None
