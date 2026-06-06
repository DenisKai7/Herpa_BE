"""
Quiz API - Endpoints untuk sistem kuis gamifikasi kimia kelas SMA.

Fitur:
- GET /categories: Mengambil daftar kategori/topik kurikulum kimia beserta metadata statistik user.
- GET /questions: Mengambil pool pertanyaan kuis acak untuk sesi kuis.
- POST /submit: Mengirimkan jawaban user, menilai skor, menyimpan riwayat attempt/jawaban,
                 serta menghitung analisis performa dan rekomendasi belajar.

Keamanan:
- Rute-rute ini dilindungi oleh otentikasi JWT dan terbatas hanya untuk role 'pelajar'.
"""

import logging
import random
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.database import supabase
from app.core.dependencies import verify_pelajar
from app.models.schemas import (
    QuizCategoryResponse,
    QuizQuestionResponse,
    QuizSubmitRequest,
    QuizSubmitResponse,
    AnswerVerificationResult,
    QuizPerformanceRecommendation,
    QuizOptionSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ═══════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════

@router.get("/categories", response_model=list[QuizCategoryResponse], summary="Ambil kategori kuis kimia")
async def get_quiz_categories(
    request: Request,
    user_id: str = Depends(verify_pelajar),
) -> list[QuizCategoryResponse]:
    """
    Mengambil seluruh daftar 17 kategori kuis kurikulum kimia SMA.
    Melakukan query tambahan pada tabel user_quiz_attempts untuk menghitung
    high_score, completion_rate, dan total_attempts untuk user yang login.
    """
    # Strict role verification layer
    if not hasattr(request.state, "user") or request.state.user.role != "pelajar":
        raise HTTPException(
            status_code=403,
            detail="Akses ditolak. Fitur kuis hanya tersedia untuk akun Pelajar.",
        )

    try:
        # 1. Fetch categories
        cat_res = supabase.table("quiz_categories").select("*").order("name").execute()
        categories = cat_res.data or []

        # 2. Fetch all attempts for this user
        attempts_res = (
            supabase.table("user_quiz_attempts")
            .select("category_id, score")
            .eq("user_id", user_id)
            .execute()
        )
        attempts = attempts_res.data or []

        # Group attempts by category_id
        attempts_by_cat: dict[str, list[dict[str, Any]]] = {}
        for att in attempts:
            cat_id = att["category_id"]
            if cat_id not in attempts_by_cat:
                attempts_by_cat[cat_id] = []
            attempts_by_cat[cat_id].append(att)

        # 3. Assemble response objects
        result = []
        for cat in categories:
            cat_id = cat["id"]
            cat_attempts = attempts_by_cat.get(cat_id, [])
            total_attempts = len(cat_attempts)

            high_score = None
            completion_rate = None

            if total_attempts > 0:
                high_score = max(att["score"] for att in cat_attempts)
                # Completion rate = % of attempts with score >= 70.0 (Passing grade)
                passed_attempts = sum(1 for att in cat_attempts if att["score"] >= 70.0)
                completion_rate = (passed_attempts / total_attempts) * 100.0
            else:
                high_score = 0.0
                completion_rate = 0.0

            result.append(
                QuizCategoryResponse(
                    id=cat_id,
                    name=cat["name"],
                    description=cat.get("description"),
                    high_score=high_score,
                    completion_rate=completion_rate,
                    total_attempts=total_attempts,
                )
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch quiz categories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Gagal mengambil kategori kuis.",
        )


@router.get("/questions", response_model=list[QuizQuestionResponse], summary="Ambil soal kuis acak")
async def get_quiz_questions(
    request: Request,
    category_id: str,
    limit: int = 10,
    user_id: str = Depends(verify_pelajar),
) -> list[QuizQuestionResponse]:
    """
    Mengambil kumpulan soal kuis kimia acak untuk topik tertentu (category_id).
    Membatasi jumlah soal sesuai limit (default: 10).
    Jawaban benar disembunyikan dari response demi integritas (tidak bisa dicuri).
    """
    # Strict role verification layer
    if not hasattr(request.state, "user") or request.state.user.role != "pelajar":
        raise HTTPException(
            status_code=403,
            detail="Akses ditolak. Fitur kuis hanya tersedia untuk akun Pelajar.",
        )

    try:
        # Fetch questions
        q_res = (
            supabase.table("quiz_questions")
            .select("id, category_id, question_text, question_type, options")
            .eq("category_id", category_id)
            .execute()
        )
        questions = q_res.data or []

        if not questions:
            raise HTTPException(
                status_code=404,
                detail="Soal tidak ditemukan untuk kategori kuis ini.",
            )

        # Randomize pool in Python
        random.shuffle(questions)
        selected_questions = questions[:limit]

        return [
            QuizQuestionResponse(
                id=q["id"],
                category_id=q["category_id"],
                question_text=q["question_text"],
                question_type=q["question_type"],
                options=[
                    QuizOptionSchema(label=opt["label"], text=opt["text"])
                    for opt in q["options"]
                ],
            )
            for q in selected_questions
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch quiz questions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Gagal mengambil butir soal kuis.",
        )


@router.post("/submit", response_model=QuizSubmitResponse, summary="Submit jawaban kuis")
async def submit_quiz_answers(
    request: Request,
    payload: QuizSubmitRequest,
    user_id: str = Depends(verify_pelajar),
) -> QuizSubmitResponse:
    """
    Menerima lembar jawaban kuis dari client.
    1. Mencocokkan jawaban dengan kunci di database.
    2. Menghitung skor persentase.
    3. Menyimpan data attempt utama ke user_quiz_attempts.
    4. Menyimpan data jawaban detail per-soal ke user_quiz_answers.
    5. Melakukan analisis performa berdasarkan kelemahan konsep (keyword-matching).
    """
    # Strict role verification layer
    if not hasattr(request.state, "user") or request.state.user.role != "pelajar":
        raise HTTPException(
            status_code=403,
            detail="Akses ditolak. Fitur kuis hanya tersedia untuk akun Pelajar.",
        )

    try:
        # Fetch answer keys
        q_res = (
            supabase.table("quiz_questions")
            .select("id, question_text, correct_answer, explanation")
            .eq("category_id", payload.category_id)
            .execute()
        )
        db_questions = {q["id"]: q for q in q_res.data or []}

        if not db_questions:
            raise HTTPException(
                status_code=400,
                detail="Kategori kuis tidak valid atau tidak memiliki butir soal.",
            )

        correct_answers = 0
        wrong_answers = 0
        results: list[AnswerVerificationResult] = []
        wrong_question_texts = []

        # Evaluate answers
        for ans in payload.answers:
            q_id = ans.question_id
            if q_id not in db_questions:
                continue

            q_data = db_questions[q_id]
            correct_choice = q_data["correct_answer"].strip().upper()
            user_choice = ans.choice.strip().upper()

            is_correct = (user_choice == correct_choice)
            if is_correct:
                correct_answers += 1
            else:
                wrong_answers += 1
                wrong_question_texts.append(q_data["question_text"])

            results.append(
                AnswerVerificationResult(
                    question_id=q_id,
                    user_choice=user_choice,
                    correct_choice=correct_choice,
                    is_correct=is_correct,
                    explanation=q_data.get("explanation"),
                )
            )

        total_questions = correct_answers + wrong_answers
        if total_questions == 0:
            raise HTTPException(
                status_code=400,
                detail="Tidak ada jawaban valid yang diserahkan untuk dinilai.",
            )

        score = (correct_answers / total_questions) * 100.0

        # Save main attempt
        attempt_data = {
            "user_id": user_id,
            "category_id": payload.category_id,
            "score": score,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "wrong_answers": wrong_answers,
            "duration": payload.duration,
        }
        attempt_res = (
            supabase.table("user_quiz_attempts")
            .insert(attempt_data)
            .execute()
        )
        if not attempt_res.data:
            raise HTTPException(
                status_code=500,
                detail="Gagal menyimpan data histori pengerjaan kuis.",
            )
        attempt_id = attempt_res.data[0]["id"]

        # Save answers granularity
        answers_data = [
            {
                "attempt_id": attempt_id,
                "question_id": r.question_id,
                "user_choice": r.user_choice,
                "is_correct": r.is_correct,
            }
            for r in results
        ]
        supabase.table("user_quiz_answers").insert(answers_data).execute()

        # ─── PERFORMANCE ANALYSIS ALGORITHM ───
        weaknesses_set = set()
        recommendations_set = set()

        # Sub-concepts mapping by keyword matching in incorrect question texts
        SUB_CONCEPTS_MAP = [
            ("kuantum", "Konfigurasi Elektron & Bilangan Kuantum", "Pelajari kembali aturan Aufbau, larangan Pauli, dan aturan Hund untuk menentukan konfigurasi elektron."),
            ("konfigurasi", "Konfigurasi Elektron & Bilangan Kuantum", "Pelajari kembali aturan Aufbau, larangan Pauli, dan aturan Hund untuk menentukan konfigurasi elektron."),
            ("kulit", "Konfigurasi Elektron & Bilangan Kuantum", "Pelajari kembali aturan Aufbau, larangan Pauli, dan aturan Hund untuk menentukan konfigurasi elektron."),
            ("atom", "Teori Atom & Perkembangannya", "Pelajari kembali eksperimen sinar katode Thomson, hamburan Rutherford, dan tingkat energi Bohr."),
            ("kovalen", "Jenis Ikatan & Geometri Molekul", "Pahami perbedaan sifat senyawa ion dan kovalen serta pengaruh bentuk molekul terhadap kepolaran."),
            ("ion", "Jenis Ikatan & Geometri Molekul", "Pahami perbedaan sifat senyawa ion dan kovalen serta pengaruh bentuk molekul terhadap kepolaran."),
            ("polar", "Jenis Ikatan & Geometri Molekul", "Pahami perbedaan sifat senyawa ion dan kovalen serta pengaruh bentuk molekul terhadap kepolaran."),
            ("geometri", "Jenis Ikatan & Geometri Molekul", "Pahami perbedaan sifat senyawa ion dan kovalen serta pengaruh bentuk molekul terhadap kepolaran."),
            ("mol", "Hukum Dasar & Perhitungan Stoikiometri", "Latih kembali perhitungan konsep mol dan hubungan massa reaktan-produk dalam persamaan reaksi."),
            ("stoikiometri", "Hukum Dasar & Perhitungan Stoikiometri", "Latih kembali perhitungan konsep mol dan hubungan massa reaktan-produk dalam persamaan reaksi."),
            ("massa", "Hukum Dasar & Perhitungan Stoikiometri", "Latih kembali perhitungan konsep mol dan hubungan massa reaktan-produk dalam persamaan reaksi."),
            ("elektrolit", "Reaksi Redoks & Sifat Larutan Elektrolit", "Pelajari cara membedakan larutan elektrolit/nonelektrolit serta aturan penentuan bilangan oksidasi (biloks)."),
            ("redoks", "Reaksi Redoks & Sifat Larutan Elektrolit", "Pelajari cara membedakan larutan elektrolit/nonelektrolit serta aturan penentuan bilangan oksidasi (biloks)."),
            ("reduksi", "Reaksi Redoks & Sifat Larutan Elektrolit", "Pelajari cara membedakan larutan elektrolit/nonelektrolit serta aturan penentuan bilangan oksidasi (biloks)."),
            ("oksidasi", "Reaksi Redoks & Sifat Larutan Elektrolit", "Pelajari cara membedakan larutan elektrolit/nonelektrolit serta aturan penentuan bilangan oksidasi (biloks)."),
            ("eksoterm", "Termokimia & Perubahan Entalpi", "Pelajari diagram tingkat energi reaksi eksoterm/endoterm dan perhitungan entalpi menggunakan hukum Hess."),
            ("endoterm", "Termokimia & Perubahan Entalpi", "Pelajari diagram tingkat energi reaksi eksoterm/endoterm dan perhitungan entalpi menggunakan hukum Hess."),
            ("entalpi", "Termokimia & Perubahan Entalpi", "Pelajari diagram tingkat energi reaksi eksoterm/endoterm dan perhitungan entalpi menggunakan hukum Hess."),
            ("laju", "Kinetika Kimia & Faktor Laju Reaksi", "Pahami teori tumbukan dan pengaruh suhu, konsentrasi, luas permukaan, serta peran katalis dalam laju reaksi."),
            ("katalis", "Kinetika Kimia & Faktor Laju Reaksi", "Pahami teori tumbukan dan pengaruh suhu, konsentrasi, luas permukaan, serta peran katalis dalam laju reaksi."),
            ("orde", "Kinetika Kimia & Faktor Laju Reaksi", "Pahami teori tumbukan dan pengaruh suhu, konsentrasi, luas permukaan, serta peran katalis dalam laju reaksi."),
            ("kesetimbangan", "Pergeseran & Tetapan Kesetimbangan", "Pahami pengaruh tekanan, volume, suhu, dan konsentrasi terhadap arah pergeseran kesetimbangan (Asas Le Chatelier)."),
            ("geser", "Pergeseran & Tetapan Kesetimbangan", "Pahami pengaruh tekanan, volume, suhu, dan konsentrasi terhadap arah pergeseran kesetimbangan (Asas Le Chatelier)."),
            ("tetapan", "Pergeseran & Tetapan Kesetimbangan", "Pahami pengaruh tekanan, volume, suhu, dan konsentrasi terhadap arah pergeseran kesetimbangan (Asas Le Chatelier)."),
            ("asam", "pH Asam-Basa & Larutan Buffer", "Pelajari rumus perhitungan pH asam/basa kuat-lemah serta cara kerja komponen penyangga dalam mempertahankan pH."),
            ("basa", "pH Asam-Basa & Larutan Buffer", "Pelajari rumus perhitungan pH asam/basa kuat-lemah serta cara kerja komponen penyangga dalam mempertahankan pH."),
            ("ph", "pH Asam-Basa & Larutan Buffer", "Pelajari rumus perhitungan pH asam/basa kuat-lemah serta cara kerja komponen penyangga dalam mempertahankan pH."),
            ("buffer", "pH Asam-Basa & Larutan Buffer", "Pelajari rumus perhitungan pH asam/basa kuat-lemah serta cara kerja komponen penyangga dalam mempertahankan pH."),
            ("penyangga", "pH Asam-Basa & Larutan Buffer", "Pelajari rumus perhitungan pH asam/basa kuat-lemah serta cara kerja komponen penyangga dalam mempertahankan pH."),
            ("hidrolisis", "Hidrolisis Garam & Kelarutan", "Pelajari jenis garam yang terhidrolisis sebagian/total dan hubungan Ksp dengan kelarutan senyawa sukar larut."),
            ("ksp", "Hidrolisis Garam & Kelarutan", "Pelajari jenis garam yang terhidrolisis sebagian/total dan hubungan Ksp dengan kelarutan senyawa sukar larut."),
            ("kelarutan", "Hidrolisis Garam & Kelarutan", "Pelajari jenis garam yang terhidrolisis sebagian/total dan hubungan Ksp dengan kelarutan senyawa sukar larut."),
            ("garam", "Hidrolisis Garam & Kelarutan", "Pelajari jenis garam yang terhidrolisis sebagian/total dan hubungan Ksp dengan kelarutan senyawa sukar larut."),
            ("koligatif", "Sifat Koligatif Larutan", "Pahami rumus penurunan titik beku/kenaikan titik didih untuk larutan elektrolit (faktor van't Hoff) dan nonelektrolit."),
            ("titik beku", "Sifat Koligatif Larutan", "Pahami rumus penurunan titik beku/kenaikan titik didih untuk larutan elektrolit (faktor van't Hoff) dan nonelektrolit."),
            ("titik didih", "Sifat Koligatif Larutan", "Pahami rumus penurunan titik beku/kenaikan titik didih untuk larutan elektrolit (faktor van't Hoff) dan nonelektrolit."),
            ("osmosis", "Sifat Koligatif Larutan", "Pahami rumus penurunan titik beku/kenaikan titik didih untuk larutan elektrolit (faktor van't Hoff) dan nonelektrolit."),
            ("koloid", "Sistem Koloid & Efek Tyndall", "Pelajari klasifikasi sistem koloid berdasarkan fase terdispersi-medium pendispersi serta sifat khas seperti efek Tyndall."),
            ("susu", "Sistem Koloid & Efek Tyndall", "Pelajari klasifikasi sistem koloid berdasarkan fase terdispersi-medium pendispersi serta sifat khas seperti efek Tyndall."),
            ("volta", "Elektrokimia & Sel Volta/Elektrolisis", "Pelajari arah aliran elektron, penentuan potensial sel (E0 sel), serta reaksi di katode/anode pada sel elektrolisis."),
            ("elektrolisis", "Elektrokimia & Sel Volta/Elektrolisis", "Pelajari arah aliran elektron, penentuan potensial sel (E0 sel), serta reaksi di katode/anode pada sel elektrolisis."),
            ("korosi", "Elektrokimia & Sel Volta/Elektrolisis", "Pelajari arah aliran elektron, penentuan potensial sel (E0 sel), serta reaksi di katode/anode pada sel elektrolisis."),
            ("gas mulia", "Kelimpahan & Kimia Unsur", "Pelajari sifat fisik/kimia golongan gas mulia, halogen, alkali, alkali tanah, dan unsur periode ketiga."),
            ("unsur", "Kelimpahan & Kimia Unsur", "Pelajari sifat fisik/kimia golongan gas mulia, halogen, alkali, alkali tanah, dan unsur periode ketiga."),
            ("gugus fungsi", "Gugus Fungsi & Isomer Senyawa Karbon", "Hafalkan tata nama IUPAC dan isomer struktur/ruang senyawa alkohol, eter, aldehid, keton, asam karboksilat, dan ester."),
            ("alkohol", "Gugus Fungsi & Isomer Senyawa Karbon", "Hafalkan tata nama IUPAC dan isomer struktur/ruang senyawa alkohol, eter, aldehid, keton, asam karboksilat, dan ester."),
            ("eter", "Gugus Fungsi & Isomer Senyawa Karbon", "Hafalkan tata nama IUPAC dan isomer struktur/ruang senyawa alkohol, eter, aldehid, keton, asam karboksilat, dan ester."),
            ("karbon", "Gugus Fungsi & Isomer Senyawa Karbon", "Hafalkan tata nama IUPAC dan isomer struktur/ruang senyawa alkohol, eter, aldehid, keton, asam karboksilat, dan ester."),
            ("isomer", "Gugus Fungsi & Isomer Senyawa Karbon", "Hafalkan tata nama IUPAC dan isomer struktur/ruang senyawa alkohol, eter, aldehid, keton, asam karboksilat, dan ester."),
            ("benzena", "Benzena & Senyawa Turunannya", "Pelajari reaksi substitusi benzena (nitrasi, sulfonasi, halogenasi) serta kegunaan zat turunannya di kehidupan sehari-hari."),
            ("benzoat", "Benzena & Senyawa Turunannya", "Pelajari reaksi substitusi benzena (nitrasi, sulfonasi, halogenasi) serta kegunaan zat turunannya di kehidupan sehari-hari."),
            ("fenol", "Benzena & Senyawa Turunannya", "Pelajari reaksi substitusi benzena (nitrasi, sulfonasi, halogenasi) serta kegunaan zat turunannya di kehidupan sehari-hari."),
            ("protein", "Struktur Makromolekul & Biopolimer", "Pelajari penggolongan polimer adisi/kondensasi serta struktur dan uji identifikasi karbohidrat, protein, dan lemak."),
            ("karbohidrat", "Struktur Makromolekul & Biopolimer", "Pelajari penggolongan polimer adisi/kondensasi serta struktur dan uji identifikasi karbohidrat, protein, dan lemak."),
            ("polimer", "Struktur Makromolekul & Biopolimer", "Pelajari penggolongan polimer adisi/kondensasi serta struktur dan uji identifikasi karbohidrat, protein, dan lemak."),
            ("waktu paruh", "Radioaktivitas & Waktu Paruh Inti", "Pelajari jenis-jenis sinar radioaktif (alfa, beta, gama) dan rumus laju peluruhan menggunakan konsep waktu paruh."),
            ("radioaktif", "Radioaktivitas & Waktu Paruh Inti", "Pelajari jenis-jenis sinar radioaktif (alfa, beta, gama) dan rumus laju peluruhan menggunakan konsep waktu paruh."),
            ("peluruhan", "Radioaktivitas & Waktu Paruh Inti", "Pelajari jenis-jenis sinar radioaktif (alfa, beta, gama) dan rumus laju peluruhan menggunakan konsep waktu paruh."),
        ]

        for text in wrong_question_texts:
            text_lower = text.lower()
            matched = False
            for key, sub, rec in SUB_CONCEPTS_MAP:
                if key in text_lower:
                    weaknesses_set.add(sub)
                    recommendations_set.add(rec)
                    matched = True
            if not matched:
                # Fallback to category name if no sub-concept keywords match
                cat_name_res = (
                    supabase.table("quiz_categories")
                    .select("name")
                    .eq("id", payload.category_id)
                    .execute()
                )
                cat_name = (
                    cat_name_res.data[0]["name"]
                    if cat_name_res.data
                    else "Topik Kuis"
                )
                weaknesses_set.add(f"Konsep dasar pada {cat_name}")
                recommendations_set.add(
                    f"Tinjau kembali bab {cat_name} pada buku catatan atau modul pembelajaran Anda."
                )

        # Performance evaluation based on score
        if score >= 80.0:
            status_eval = "Sangat Baik"
            if not weaknesses_set:
                recommendations_set.add(
                    "Luar biasa! Pertahankan pemahaman Anda dan coba tantang diri Anda di kategori lain."
                )
        elif score >= 60.0:
            status_eval = "Cukup Baik"
        else:
            status_eval = "Perlu Belajar Lagi"

        analysis = QuizPerformanceRecommendation(
            status_eval=status_eval,
            weaknesses=list(weaknesses_set),
            recommendations=list(recommendations_set),
        )

        return QuizSubmitResponse(
            attempt_id=attempt_id,
            score=score,
            total_questions=total_questions,
            correct_answers=correct_answers,
            wrong_answers=wrong_answers,
            duration=payload.duration,
            results=results,
            analysis=analysis,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit and evaluate quiz: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Gagal memproses penilaian kuis.",
        )
