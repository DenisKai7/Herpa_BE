-- ═══════════════════════════════════════════════════════════
-- SUPABASE SCHEMA - Enterprise Medical AI Backend
-- Jalankan di Supabase SQL Editor secara berurutan.
-- ═══════════════════════════════════════════════════════════

-- ────────────────────────────────────
-- 1. ENABLE EXTENSIONS
-- ────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector untuk embedding search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ────────────────────────────────────
-- 2. TABLE: profiles (RBAC user profile)
-- Terhubung dengan Supabase Auth via trigger
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT,
    username TEXT UNIQUE,
    nama TEXT,
    role TEXT DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    instansi TEXT,
    provinsi TEXT,
    kota TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trigger: auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, username, nama, instansi, provinsi, kota)
    VALUES (
        NEW.id,
        NEW.email,
        NEW.raw_user_meta_data->>'username',
        NEW.raw_user_meta_data->>'nama',
        NEW.raw_user_meta_data->>'instansi',
        NEW.raw_user_meta_data->>'provinsi',
        NEW.raw_user_meta_data->>'kota'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ────────────────────────────────────
-- 3. TABLE: chats (Chat sessions)
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS chats (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    title TEXT DEFAULT 'Chat Baru',
    is_pinned BOOLEAN DEFAULT false,
    is_public BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chats_is_public ON chats(is_public) WHERE is_public = true;

-- ────────────────────────────────────
-- 4. TABLE: messages (Chat messages)
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'ai')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);

-- ────────────────────────────────────
-- 5. TABLE: plants (Tanaman obat + embedding)
-- Tabel utama ensiklopedia tanaman
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS plants (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    nama TEXT NOT NULL,
    nama_latin TEXT,
    famili TEXT,
    deskripsi TEXT,
    khasiat TEXT,
    cara_penggunaan TEXT,
    kandungan_kimia TEXT,
    kategori TEXT,
    embedding vector(768),  -- multilingual-e5-base dimension
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_plants_embedding ON plants
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ────────────────────────────────────
-- 6. TABLE: encyclopedia (Entri ensiklopedia umum)
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS encyclopedia (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    nama TEXT NOT NULL,
    nama_latin TEXT,
    kategori TEXT,
    deskripsi TEXT,
    sumber TEXT,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_encyclopedia_embedding ON encyclopedia
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ────────────────────────────────────
-- 7. TABLE: education_materials (Materi edukasi)
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS education_materials (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    topik TEXT NOT NULL,
    konten TEXT NOT NULL,
    kategori TEXT,  -- kimia, farmasi, biologi, etc.
    tingkat TEXT,   -- dasar, menengah, lanjut
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_education_embedding ON education_materials
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ────────────────────────────────────
-- 8. RPC FUNCTIONS (pgvector similarity search)
-- Dipanggil oleh retriever.py via supabase.rpc()
-- ────────────────────────────────────

-- Match Plants (untuk konsultasi)
CREATE OR REPLACE FUNCTION match_plants(
    query_embedding vector(768),
    match_count int DEFAULT 5,
    match_threshold float DEFAULT 0.5
)
RETURNS TABLE (
    id uuid,
    nama text,
    nama_latin text,
    deskripsi text,
    khasiat text,
    cara_penggunaan text,
    kandungan_kimia text,
    kategori text,
    similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id, p.nama, p.nama_latin, p.deskripsi,
        p.khasiat, p.cara_penggunaan, p.kandungan_kimia,
        p.kategori,
        (1 - (p.embedding <=> query_embedding))::float AS similarity
    FROM plants p
    WHERE 1 - (p.embedding <=> query_embedding) > match_threshold
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Match Encyclopedia (untuk ensiklopedia)
CREATE OR REPLACE FUNCTION match_encyclopedia(
    query_embedding vector(768),
    match_count int DEFAULT 5,
    match_threshold float DEFAULT 0.5
)
RETURNS TABLE (
    id uuid,
    nama text,
    nama_latin text,
    kategori text,
    deskripsi text,
    sumber text,
    similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id, e.nama, e.nama_latin, e.kategori,
        e.deskripsi, e.sumber,
        (1 - (e.embedding <=> query_embedding))::float AS similarity
    FROM encyclopedia e
    WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Match Education Materials (untuk edukasi)
CREATE OR REPLACE FUNCTION match_education(
    query_embedding vector(768),
    match_count int DEFAULT 5,
    match_threshold float DEFAULT 0.5
)
RETURNS TABLE (
    id uuid,
    topik text,
    konten text,
    kategori text,
    tingkat text,
    similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        em.id, em.topik, em.konten, em.kategori,
        em.tingkat,
        (1 - (em.embedding <=> query_embedding))::float AS similarity
    FROM education_materials em
    WHERE 1 - (em.embedding <=> query_embedding) > match_threshold
    ORDER BY em.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ────────────────────────────────────
-- 9. ROW LEVEL SECURITY (RLS)
-- ────────────────────────────────────
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Profiles: user hanya bisa lihat profilnya sendiri
CREATE POLICY "Users can view own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

-- Chats: user hanya bisa akses chatnya sendiri
CREATE POLICY "Users can view own chats" ON chats
    FOR SELECT USING (auth.uid() = user_id OR is_public = true);
CREATE POLICY "Users can create own chats" ON chats
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own chats" ON chats
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete own chats" ON chats
    FOR DELETE USING (auth.uid() = user_id);

-- Messages: akses berdasarkan kepemilikan chat
CREATE POLICY "Users can view messages in own chats" ON messages
    FOR SELECT USING (
        chat_id IN (SELECT id FROM chats WHERE user_id = auth.uid() OR is_public = true)
    );
CREATE POLICY "Users can insert messages in own chats" ON messages
    FOR INSERT WITH CHECK (
        chat_id IN (SELECT id FROM chats WHERE user_id = auth.uid())
    );

-- Service role key (backend) bypass RLS, jadi backend bisa akses semua data.
