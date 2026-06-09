import logging
from typing import Optional, List
from pydantic import BaseModel
from app.agent.plant_identity import resolve_canonical_plant_identity

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
    """Backward-compatible wrapper around the canonical resolver."""
    if not query:
        return None
    identity = resolve_canonical_plant_identity(query)
    if identity.resolution_method in {"not_found", "ambiguous"}:
        return None
    return PlantIdentity(
        local_name=identity.canonical_local_name or identity.extracted_local_name or "",
        scientific_name=identity.scientific_name,
        family=identity.family,
        synonyms=identity.synonyms,
        confidence=identity.confidence,
        sources=identity.evidence_sources or [identity.resolution_method],
    )
