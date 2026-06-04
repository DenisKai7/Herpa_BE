"""
Agentic AI Self-Learning Module - Skripsi Jofanza Denis Aldida.
Fungsi otomatisasi penyuntikan pengetahuan baru ke Neo4j & Supabase 
saat pengguna memberikan input koreksi data fitokimia baru.
"""

import logging
import re
from app.core.database import neo4j_driver, supabase
from app.core.embedding import embed_text

logger = logging.getLogger(__name__)

async def agent_learn_new_compound(compound_name: str, herb_name: str, description: str = "Diinput secara otomatis oleh Agentic AI Self-Learning.") -> bool:
    """
    Menyimpan data hubungan fitokimia baru hasil koreksi/input user ke Neo4j & Supabase secara dinamis.
    """
    try:
        comp_clean = compound_name.strip().title()
        herb_clean = herb_name.strip().title()
        
        logger.info(f"[Agentic Learning] Memulai proses belajar mandiri entitas: {comp_clean} -> {herb_clean}")

        # 1. SUNTIK DATA KE NEO4J GRAPH DATABASE
        cypher_write = """
        MERGE (h:Herb {name: $herb})
        MERGE (c:Compound {name: $compound})
        MERGE (h)-[r:HAS_COMPOUND]->(c)
        ON CREATE SET c.description = $desc, r.created_by = "Agentic_Self_Learning"
        RETURN h, c
        """
        await neo4j_driver.execute_query(
            cypher_write,
            parameters_={"herb": herb_clean, "compound": comp_clean, "desc": description}
        )
        logger.info("[Agentic Learning] Sukses menyuntikkan relasi baru ke grafik Neo4j.")

        # 2. SUNTIK DATA KE SUPABASE VECTOR STORAGE (Agar materi edukasinya ikut pintar)
        materi_edukasi = f"Senyawa aktif {comp_clean} merupakan kandungan fito-farmaka utama yang ditemukan pada tanaman herbal {herb_clean}. {description}"
        embedding_vector = embed_text(materi_edukasi)

        # Masukkan ke tabel edukasi Supabase via RPC atau insert langsung
        supabase.table("education_materials").insert({
            "title": f"Materi Otomatis: {comp_clean}",
            "content": materi_edukasi,
            "embedding": embedding_vector
        }).execute()
        
        logger.info("[Agentic Learning] Sukses menyuntikkan representasi vektor baru ke Supabase pgvector.")
        return True

    except Exception as e:
        logger.error(f"[Agentic Learning] Proses belajar mandiri gagal: {e}", exc_info=True)
        return False

def check_for_learning_trigger(user_message: str, ai_response: str):
    """
    Fungsi interseptor untuk memeriksa apakah user sedang memberikan ilmu baru/koreksi.
    Contoh pemicu: "itu gambar xanthorrhizol tanaman temulawak"
    """
    # Regex untuk menangkap pola pengenalan: "itu gambar [senyawa] tanaman [herbal]"
    pattern = r'(?:itu|ini)\s+gambar\s+([\w\s-]+)\s+(?:dari|tanaman|herbal)\s+([\w\s-]+)'
    match = re.search(pattern, user_message.lower())
    
    if match:
        compound = match.group(1).strip()
        herb = match.group(2).strip()
        return compound, herb
    return None