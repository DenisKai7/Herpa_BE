"""
Quiz Schemas - Pydantic models untuk validasi output kuis interaktif.

Schema ini digunakan untuk:
1. Mendefinisikan tool parameter schema di OpenAI Tool-Calling.
2. Memvalidasi response LLM sebelum dikirim ke frontend.
3. Memastikan konsistensi format kuis di seluruh pipeline.
"""

from pydantic import BaseModel, Field


class QuizOption(BaseModel):
    """Satu opsi jawaban dalam soal kuis pilihan ganda."""

    label: str = Field(description="Huruf opsi, misal: 'A', 'B', 'C', 'D'")
    text: str = Field(description="Isi teks dari opsi jawaban tersebut")


class QuizQuestion(BaseModel):
    """Satu soal kuis lengkap dengan opsi dan pembahasan."""

    id_soal: str = Field(description="ID unik untuk soal, misal: 'Q-01'")
    tingkat_kesulitan: str = Field(
        description="Tingkat kesulitan: Mudah, Menengah, atau HOTS"
    )
    pertanyaan: str = Field(description="Teks pertanyaan kuis")
    opsi_jawaban: list[QuizOption] = Field(
        min_length=4,
        max_length=4,
        description="Tepat 4 opsi jawaban pilihan ganda (A, B, C, D)"
    )
    jawaban_benar: str = Field(
        description="Label jawaban yang benar, misal: 'A'"
    )
    pembahasan: list[str] = Field(
        description=(
            "Penjelasan langkah demi langkah mengapa jawaban tersebut benar. "
            "Format: array of strings, setiap elemen satu langkah."
        )
    )
    penjelasan_salah: str = Field(
        description="Penjelasan singkat mengapa pilihan jawaban lainnya salah atau jebakan yang sering terjadi"
    )


class PerformanceAnalysis(BaseModel):
    """Analisis performa hasil pengerjaan kuis."""

    sorotan: list[str] = Field(
        description="Daftar poin sorotan mengenai konsep materi yang dikuasai pengguna berdasarkan topik kuis"
    )
    area_fokus: list[str] = Field(
        description="Daftar poin area fokus atau kelemahan konsep materi yang perlu dipelajari lebih lanjut"
    )


class QuizResponse(BaseModel):
    """Response lengkap kuis interaktif yang di-generate oleh AI."""

    topik: str = Field(description="Topik utama dari kuis ini")
    daftar_soal: list[QuizQuestion] = Field(
        description="Daftar pertanyaan kuis yang di-generate"
    )
    analisis_performa: PerformanceAnalysis = Field(
        description="Analisis performa hasil pengerjaan kuis secara akademis"
    )
