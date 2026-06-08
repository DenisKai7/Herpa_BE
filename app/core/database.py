"""
Database Clients - Supabase (PostgreSQL/pgvector) & Neo4j (Graph DB).
Menyediakan koneksi singleton untuk dipakai seluruh modul backend.
"""

import logging
from typing import Any

try:
    from supabase import create_client, Client
except Exception as supabase_import_error:
    class _UnavailableTable:
        def __getattr__(self, _name: str):
            return self
        def __call__(self, *args: Any, **kwargs: Any):
            return self
        def execute(self):
            raise RuntimeError("Supabase client unavailable") from supabase_import_error

    class _UnavailableSupabase:
        def table(self, *args: Any, **kwargs: Any):
            return _UnavailableTable()

    Client = Any
    def create_client(*args: Any, **kwargs: Any):
        return _UnavailableSupabase()
from neo4j import GraphDatabase, Driver
from app.core.config import settings

logger = logging.getLogger(__name__)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SUPABASE CLIENT  (PostgreSQL + pgvector + Auth)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
logger.info("Supabase client initialized successfully.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# NEO4J DRIVER  (Graph Database)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
neo4j_driver: Driver = GraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
)


def get_neo4j_session():
    """Membuat Neo4j session baru. Gunakan dengan `with` statement."""
    return neo4j_driver.session()


def verify_neo4j_connection() -> bool:
    """Memeriksa koneksi ke Neo4j."""
    try:
        neo4j_driver.verify_connectivity()
        logger.info("Neo4j connection verified successfully.")
        return True
    except Exception as e:
        logger.warning(f"Neo4j connection failed: {e}")
        return False


def close_connections():
    """Menutup semua koneksi database saat shutdown."""
    try:
        neo4j_driver.close()
        logger.info("Neo4j driver closed.")
    except Exception as e:
        logger.error(f"Error closing Neo4j driver: {e}")

