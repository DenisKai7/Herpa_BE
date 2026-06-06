-- ═══════════════════════════════════════════════════════════
-- SUPABASE SCHEMA FOR GAMIFIED CHEMISTRY QUIZ ENGINE
-- ═══════════════════════════════════════════════════════════

-- ────────────────────────────────────
-- 1. TABLE: quiz_categories
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS quiz_categories (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ────────────────────────────────────
-- 2. TABLE: quiz_questions
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS quiz_questions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    category_id UUID REFERENCES quiz_categories(id) ON DELETE CASCADE NOT NULL,
    question_text TEXT NOT NULL,
    question_type TEXT NOT NULL DEFAULT 'multiple_choice' CHECK (question_type IN ('multiple_choice')),
    options JSONB NOT NULL, -- Format: [{"label": "A", "text": "value"}, {"label": "B", "text": "value"}, ...]
    correct_answer TEXT NOT NULL, -- Format: 'A', 'B', 'C', or 'D'
    explanation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quiz_questions_category ON quiz_questions(category_id);

-- ────────────────────────────────────
-- 3. TABLE: user_quiz_attempts
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_quiz_attempts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) ON DELETE CASCADE NOT NULL,
    category_id UUID REFERENCES quiz_categories(id) ON DELETE CASCADE NOT NULL,
    score FLOAT NOT NULL,
    total_questions INT NOT NULL,
    correct_answers INT NOT NULL,
    wrong_answers INT NOT NULL,
    duration INT NOT NULL, -- in seconds
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_attempts_user ON user_quiz_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_attempts_category ON user_quiz_attempts(category_id);

-- ────────────────────────────────────
-- 4. TABLE: user_quiz_answers
-- ────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_quiz_answers (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    attempt_id UUID REFERENCES user_quiz_attempts(id) ON DELETE CASCADE NOT NULL,
    question_id UUID REFERENCES quiz_questions(id) ON DELETE CASCADE NOT NULL,
    user_choice TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_answers_attempt ON user_quiz_answers(attempt_id);

-- ────────────────────────────────────
-- 5. ROW LEVEL SECURITY (RLS) FOR NEW TABLES
-- ────────────────────────────────────
ALTER TABLE quiz_categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE quiz_questions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_quiz_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_quiz_answers ENABLE ROW LEVEL SECURITY;

-- Select policies (accessible by authenticated users)
CREATE POLICY "Allow public select categories" ON quiz_categories FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow public select questions" ON quiz_questions FOR SELECT TO authenticated USING (true);

-- User specific policies for attempts and answers
CREATE POLICY "Users can insert own attempts" ON user_quiz_attempts FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can view own attempts" ON user_quiz_attempts FOR SELECT TO authenticated USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own answers" ON user_quiz_answers FOR INSERT TO authenticated WITH CHECK (
    EXISTS (
        SELECT 1 FROM user_quiz_attempts
        WHERE user_quiz_attempts.id = attempt_id AND user_quiz_attempts.user_id = auth.uid()
    )
);
CREATE POLICY "Users can view own answers" ON user_quiz_answers FOR SELECT TO authenticated USING (
    EXISTS (
        SELECT 1 FROM user_quiz_attempts
        WHERE user_quiz_attempts.id = attempt_id AND user_quiz_attempts.user_id = auth.uid()
    )
);

-- ────────────────────────────────────
-- 6. SEED DATA: quiz_categories (17 Modules)
-- ────────────────────────────────────
INSERT INTO quiz_categories (id, name, description) VALUES
('c0000001-0000-0000-0000-000000000001', 'Struktur Atom & SPU', 'Teori atom, konfigurasi elektron, bilangan kuantum, Sistem Periodik Unsur.'),
('c0000002-0000-0000-0000-000000000002', 'Ikatan Kimia', 'Ikatan ion, kovalen, logam, bentuk geometri molekul.'),
('c0000003-0000-0000-0000-000000000003', 'Stoikiometri', 'Tata nama senyawa, persamaan reaksi, hukum dasar kimia, konsep mol.'),
('c0000004-0000-0000-0000-000000000004', 'Larutan & Reaksi Redoks', 'Larutan elektrolit/nonelektrolit dan konsep reaksi reduksi-oksidasi.'),
('c0000005-0000-0000-0000-000000000005', 'Termokimia', 'Reaksi eksoterm/endoterm dan perubahan entalpi (Delta H).'),
('c0000006-0000-0000-0000-000000000006', 'Laju Reaksi', 'Faktor-faktor penentu laju reaksi dan orde reaksi.'),
('c0000007-0000-0000-0000-000000000007', 'Kesetimbangan Kimia', 'Pergeseran kesetimbangan dan tetapan kesetimbangan (K).'),
('c0000008-0000-0000-0000-000000000008', 'Asam Basa & Larutan Penyangga', 'Teori asam basa, pH, serta sifat larutan buffer.'),
('c0000009-0000-0000-0000-000000000009', 'Hidrolisis Garam & Kelarutan', 'Reaksi hidrolisis dan hasil kali kelarutan (Ksp).'),
('c0000010-0000-0000-0000-000000000010', 'Sifat Koligatif Larutan', 'Penurunan tekanan uap, kenaikan titik didih, penurunan titik beku, osmosis.'),
('c0000011-0000-0000-0000-000000000011', 'Sistem Koloid', 'Jenis, sifat, dan pembuatan koloid.'),
('c0000012-0000-0000-0000-000000000012', 'Redoks & Elektrokimia', 'Penyetaraan reaksi redoks, sel volta, sel elektrolisis, korosi.'),
('c0000013-0000-0000-0000-000000000013', 'Kimia Unsur', 'Kelimpahan, sifat, dan pembuatan unsur-unsur di alam.'),
('c0000014-0000-0000-0000-000000000014', 'Senyawa Karbon', 'Gugus fungsi, isomer, reaksi senyawa karbon (alkohol, eter, aldehid, keton, asam karboksilat, ester).'),
('c0000015-0000-0000-0000-000000000015', 'Benzena & Turunannya', 'Struktur, tata nama, kegunaan.'),
('c0000016-0000-0000-0000-000000000016', 'Makromolekul', 'Polimer, karbohidrat, protein, lemak, asam amino.'),
('c0000017-0000-0000-0000-000000000017', 'Radioaktivitas', 'Sifat inti atom dan waktu paruh.')
ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description;

-- ────────────────────────────────────
-- 7. SEED DATA: quiz_questions (Sample Questions for each category)
-- ────────────────────────────────────
INSERT INTO quiz_questions (category_id, question_text, options, correct_answer, explanation) VALUES
-- 1. Struktur Atom & SPU
('c0000001-0000-0000-0000-000000000001', 'Berapakah jumlah elektron maksimum pada kulit M?', '[{"label": "A", "text": "2"}, {"label": "B", "text": "8"}, {"label": "C", "text": "18"}, {"label": "D", "text": "32"}]', 'C', 'Jumlah elektron maksimum pada kulit ke-n dihitung menggunakan rumus 2n^2. Untuk kulit M (n=3), jumlah elektron maksimum adalah 2(3)^2 = 18.'),
('c0000001-0000-0000-0000-000000000001', 'Siapakah ilmuwan yang menemukan elektron melalui percobaan tabung sinar katode?', '[{"label": "A", "text": "J.J. Thomson"}, {"label": "B", "text": "Ernest Rutherford"}, {"label": "C", "text": "John Dalton"}, {"label": "D", "text": "Niels Bohr"}]', 'A', 'J.J. Thomson menemukan elektron pada tahun 1897 melalui eksperimen sinar katode dan mengajukan model atom kismis.'),

-- 2. Ikatan Kimia
('c0000002-0000-0000-0000-000000000002', 'Senyawa manakah di bawah ini yang berikatan kovalen polar?', '[{"label": "A", "text": "NaCl"}, {"label": "B", "text": "H2O"}, {"label": "C", "text": "CH4"}, {"label": "D", "text": "O2"}]', 'B', 'H2O memiliki perbedaan elektronegativitas yang besar antara H dan O, serta bentuk molekul asimetris (bengkok), sehingga bersifat kovalen polar.'),

-- 3. Stoikiometri
('c0000003-0000-0000-0000-000000000003', 'Berapakah massa dari 2 mol molekul air (H2O)? (Ar H=1, O=16)', '[{"label": "A", "text": "18 gram"}, {"label": "B", "text": "36 gram"}, {"label": "C", "text": "9 gram"}, {"label": "D", "text": "54 gram"}]', 'B', 'Mr H2O = (2x1) + 16 = 18. Massa = mol x Mr = 2 x 18 = 36 gram.'),

-- 4. Larutan & Reaksi Redoks
('c0000004-0000-0000-0000-000000000004', 'Di antara larutan berikut, manakah yang merupakan elektrolit kuat?', '[{"label": "A", "text": "CH3COOH"}, {"label": "B", "text": "HCl"}, {"label": "C", "text": "C6H12O6"}, {"label": "D", "text": "NH4OH"}]', 'B', 'HCl adalah asam kuat yang terionisasi sempurna dalam larutan air, menjadikannya elektrolit kuat.'),

-- 5. Termokimia
('c0000005-0000-0000-0000-000000000005', 'Reaksi yang membebaskan kalor dari sistem ke lingkungan disebut reaksi...', '[{"label": "A", "text": "Eksoterm"}, {"label": "B", "text": "Endoterm"}, {"label": "C", "text": "Isoterm"}, {"label": "D", "text": "Adiadatik"}]', 'A', 'Reaksi eksoterm adalah reaksi yang menghasilkan atau melepas kalor ke lingkungan, di mana perubahan entalpinya bernilai negatif (Delta H < 0).'),

-- 6. Laju Reaksi
('c0000006-0000-0000-0000-000000000006', 'Faktor manakah yang TIDAK mempercepat laju reaksi?', '[{"label": "A", "text": "Meningkatkan suhu"}, {"label": "B", "text": "Menambah katalis"}, {"label": "C", "text": "Memperkecil luas permukaan sentuh"}, {"label": "D", "text": "Meningkatkan konsentrasi reaktan"}]', 'C', 'Memperkecil luas permukaan (misal mengubah serbuk menjadi kepingan) akan mengurangi frekuensi tumbukan sehingga memperlambat laju reaksi.'),

-- 7. Kesetimbangan Kimia
('c0000007-0000-0000-0000-000000000007', 'Berdasarkan asas Le Chatelier, jika konsentrasi produk reaksi dikurangi, maka kesetimbangan akan bergeser ke arah...', '[{"label": "A", "text": "Kiri (reaktan)"}, {"label": "B", "text": "Kanan (produk)"}, {"label": "C", "text": "Tetap tidak berubah"}, {"label": "D", "text": "Tidak dapat diprediksi"}]', 'B', 'Jika konsentrasi suatu zat dikurangi, sistem kesetimbangan akan bergeser ke arah zat tersebut untuk menggantikan jumlah yang berkurang. Maka ia bergeser ke arah produk (kanan).'),

-- 8. Asam Basa & Larutan Penyangga
('c0000008-0000-0000-0000-000000000008', 'Campuran larutan manakah yang membentuk larutan penyangga asam?', '[{"label": "A", "text": "HCl dan NaCl"}, {"label": "B", "text": "CH3COOH dan CH3COONa"}, {"label": "C", "text": "NaOH dan NaCl"}, {"label": "D", "text": "NH3 dan NH4Cl"}]', 'B', 'Larutan penyangga asam terbentuk dari campuran asam lemah (CH3COOH) dan basa konjugasinya (CH3COO- dari CH3COONa).'),

-- 9. Hidrolisis Garam & Kelarutan
('c0000009-0000-0000-0000-000000000009', 'Garam manakah di bawah ini yang mengalami hidrolisis total di dalam air?', '[{"label": "A", "text": "NH4CN"}, {"label": "B", "text": "NaCl"}, {"label": "C", "text": "CH3COONa"}, {"label": "D", "text": "NH4Cl"}]', 'A', 'NH4CN terbentuk dari basa lemah (NH3) dan asam lemah (HCN), sehingga kedua kation dan anionnya mengalami hidrolisis (hidrolisis total).'),

-- 10. Sifat Koligatif Larutan
('c0000010-0000-0000-0000-000000000010', 'Manakah di antara sifat berikut yang bukan termasuk sifat koligatif larutan?', '[{"label": "A", "text": "Penurunan tekanan uap"}, {"label": "B", "text": "Kenaikan titik didih"}, {"label": "C", "text": "Tekanan osmotik"}, {"label": "D", "text": "Kenaikan titik beku"}]', 'D', 'Sifat koligatif larutan meliputi penurunan tekanan uap, kenaikan titik didih, penurunan titik beku (bukan kenaikan), dan tekanan osmotik.'),

-- 11. Sistem Koloid
('c0000011-0000-0000-0000-000000000011', 'Susu merupakan contoh dari sistem koloid jenis...', '[{"label": "A", "text": "Aerosol"}, {"label": "B", "text": "Emulsi"}, {"label": "C", "text": "Busa"}, {"label": "D", "text": "Sol"}]', 'B', 'Susu adalah emulsi cair, di mana fase terdispersinya berupa zat cair (lemak susu) dan medium pendispersinya juga zat cair (air).'),

-- 12. Redoks & Elektrokimia
('c0000012-0000-0000-0000-000000000012', 'Pada sel Volta, elektrode tempat terjadinya reaksi oksidasi disebut...', '[{"label": "A", "text": "Anode"}, {"label": "B", "text": "Katode"}, {"label": "C", "text": "Jembatan garam"}, {"label": "D", "text": "Elektrolit"}]', 'A', 'Pada sel Volta maupun sel elektrolisis, oksidasi selalu terjadi pada anode (ingat jembatan keledai AnOx - Anode Oksidasi, dan RedKat - Reduksi Katode).'),

-- 13. Kimia Unsur
('c0000013-0000-0000-0000-000000000013', 'Gas mulia yang paling banyak terdapat di atmosfer bumi adalah...', '[{"label": "A", "text": "Helium"}, {"label": "B", "text": "Neon"}, {"label": "C", "text": "Argon"}, {"label": "D", "text": "Kripton"}]', 'C', 'Argon (Ar) adalah gas mulia yang paling melimpah di atmosfer bumi, menyusun sekitar 0,93% udara kering.'),

-- 14. Senyawa Karbon
('c0000014-0000-0000-0000-000000000014', 'Senyawa dengan gugus fungsi -O- tergolong ke dalam kelompok...', '[{"label": "A", "text": "Alkohol"}, {"label": "B", "text": "Eter"}, {"label": "C", "text": "Aldehid"}, {"label": "D", "text": "Ester"}]', 'B', 'Gugus fungsi eter adalah etoksi (-O-) dengan rumus umum R-O-R''.'),

-- 15. Benzena & Turunannya
('c0000015-0000-0000-0000-000000000015', 'Senyawa turunan benzena yang digunakan sebagai bahan pengawet makanan adalah...', '[{"label": "A", "text": "Toluena"}, {"label": "B", "text": "Asam benzoat"}, {"label": "C", "text": "Fenol"}, {"label": "D", "text": "Anilina"}]', 'B', 'Asam benzoat (atau garam natriumnya, natrium benzoat) adalah zat turunan benzena yang umum digunakan sebagai bahan pengawet makanan.'),

-- 16. Makromolekul
('c0000016-0000-0000-0000-000000000016', 'Polimer alami yang terbentuk dari monomer-monomer asam amino adalah...', '[{"label": "A", "text": "Selulosa"}, {"label": "B", "text": "Karet alam"}, {"label": "C", "text": "Protein"}, {"label": "D", "text": "Amilum"}]', 'C', 'Protein adalah kopolimer alami yang tersusun dari monomer asam amino yang dihubungkan oleh ikatan peptida.'),

-- 17. Radioaktivitas
('c0000017-0000-0000-0000-000000000017', 'Waktu yang diperlukan oleh zat radioaktif untuk meluruh hingga massanya tinggal setengah dari massa semula disebut...', '[{"label": "A", "text": "Waktu paruh"}, {"label": "B", "text": "Konstanta peluruhan"}, {"label": "C", "text": "Waktu rata-rata"}, {"label": "D", "text": "Aktivitas radiasi"}]', 'A', 'Waktu paruh (t_1/2) adalah waktu yang dibutuhkan oleh inti zat radioaktif untuk meluruh hingga tersisa setengah (50%) dari jumlah atau aktivitas awalnya.')
ON CONFLICT DO NOTHING;
