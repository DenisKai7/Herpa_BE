import json
# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure app is in path
sys.path.append(os.getcwd())

# Set dummy env vars conditionally for loading app modules if not present
if "SUPABASE_URL" not in os.environ:
    os.environ["SUPABASE_URL"] = "https://example.supabase.co"
if "SUPABASE_SERVICE_KEY" not in os.environ:
    os.environ["SUPABASE_SERVICE_KEY"] = "dummykey"
if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = "dummyhftoken"

from app.agent.quiz_generator import (
    _extract_jumlah_soal,
    _clean_topic,
    QUIZ_GENERATOR_VERSION,
    MAX_QUIZ_QUESTIONS,
    MIN_QUIZ_QUESTIONS,
    DEFAULT_QUIZ_QUESTIONS,
    generate_interactive_quiz_tool,
    create_quiz_blueprint,
    detect_duplicate_questions
)

# 1. Parsing Jumlah Soal Tests
def test_extract_jumlah_soal():
    assert _extract_jumlah_soal("buat kuis 12 soal tentang jeruk nipis") == 12
    assert _extract_jumlah_soal("20 pertanyaan tentang farmasi") == 20
    assert _extract_jumlah_soal("buat soal tentang kimia") == DEFAULT_QUIZ_QUESTIONS
    assert _extract_jumlah_soal("buat 100 soal") == MAX_QUIZ_QUESTIONS

# 2. Runtime Version Test
def test_runtime_version():
    assert QUIZ_GENERATOR_VERSION == "2.0.0"
    assert MAX_QUIZ_QUESTIONS == 50

# 3. Blueprint Creation and Option Balancing Test
def test_create_quiz_blueprint():
    count = 12
    blueprint = create_quiz_blueprint("jeruk nipis", count, "tanaman_obat")
    assert blueprint.topik == "jeruk nipis"
    assert len(blueprint.items) == count
    
    # Check balance of labels A, B, C, D (max difference of 1)
    labels = [item.correct_label for item in blueprint.items]
    counts = {l: labels.count(l) for l in ["A", "B", "C", "D"]}
    max_val = max(counts.values())
    min_val = min(counts.values())
    assert (max_val - min_val) <= 1
    
    # Check that difficulties have at least 3 distinct values
    diffs = set(item.tingkat_kesulitan for item in blueprint.items)
    assert len(diffs) >= 3

# 4. Duplicate Detection Tests
def test_detect_duplicate_questions():
    questions = [
        {"pertanyaan": "Metabolit sekunder pada jeruk nipis adalah flavonoid.", "id_soal": "Q-01"},
        {"pertanyaan": "Metabolit sekunder pada jeruk nipis adalah flavonoid.", "id_soal": "Q-02"}, # Exact duplicate
        {"pertanyaan": "Apa kandungan utama dari jeruk nipis?", "id_soal": "Q-03"},
    ]
    dups = detect_duplicate_questions(questions)
    assert 1 in dups # Q-02 at index 1 is a duplicate

# 5. API Failure Fallback Test
@pytest.mark.asyncio
@patch("app.agent.quiz_generator.embed_text")
@patch("app.agent.quiz_generator._client")
async def test_api_failure_fallback(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 384
    # Simulate API timeout or exception
    mock_client.chat.completions.create.side_effect = Exception("API Timeout")
    
    # We call generate_interactive_quiz_tool
    res = await generate_interactive_quiz_tool("jeruk nipis", 5, "Pelajar")
    
    assert res is not None
    assert len(res["questions"]) == 5
    assert res["generation_metadata"]["fallback_used"] is True
    # Make sure correct answer positions are balanced/randomized and not all "A"
    corrects = [q["correct_answer"] for q in res["questions"]]
    # The synthetic fallback uses blueprint correct labels or random choice
    # which distributes answers A-D. Let's verify that correct answers are not all "A"
    assert not all(c == "A" for c in corrects)

# 6. Partial Result Repair and Duplicate Repair Test
@pytest.mark.asyncio
@patch("app.agent.quiz_generator.embed_text")
@patch("app.agent.quiz_generator._client")
async def test_partial_result_and_duplicate_repair(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 384
    # Mock LLM to return only 7 questions for a request of 12
    # The first call returns 7 questions, then complete_missing_questions is called
    # which makes a repair call to generate the remaining 5 questions.
    # We mock both calls in sequence.
    
    # Batch 1 response (returns 7 questions, but index 0 and 1 are duplicate)
    # So we have 6 unique questions, meaning it will require a repair of 6 questions.
    batch1_questions = [
        {
            "id_soal": f"Q-{i+1:02d}",
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Pertanyaan unik nomor {i+1} tentang jeruk nipis?",
            "opsi_jawaban": [
                {"label": "A", "text": "Opsi A"},
                {"label": "B", "text": "Opsi B"},
                {"label": "C", "text": "Opsi C"},
                {"label": "D", "text": "Opsi D"}
            ],
            "jawaban_benar": "A",
            "pembahasan": ["Langkah 1"],
            "penjelasan_salah": "Salah."
        } for i in range(7)
    ]
    # Inject a duplicate in the list: make Q-02's question text same as Q-01
    batch1_questions[1]["pertanyaan"] = batch1_questions[0]["pertanyaan"]
    
    # Repair response: returns 6 questions to complete the 12
    repair_questions = [
        {
            "id_soal": f"Q-{i+8:02d}",
            "tingkat_kesulitan": "Menengah",
            "pertanyaan": f"Pertanyaan perbaikan nomor {i+8} tentang jeruk nipis?",
            "opsi_jawaban": [
                {"label": "A", "text": "Opsi A"},
                {"label": "B", "text": "Opsi B"},
                {"label": "C", "text": "Opsi C"},
                {"label": "D", "text": "Opsi D"}
            ],
            "jawaban_benar": "A",
            "pembahasan": ["Langkah 1"],
            "penjelasan_salah": "Salah."
        } for i in range(6)
    ]
    
    mock_res1 = MagicMock()
    mock_res1.choices = [MagicMock(message=MagicMock(tool_calls=[
        MagicMock(function=MagicMock(arguments=json.dumps({"daftar_soal": batch1_questions})))
    ]))]
    
    mock_res2 = MagicMock()
    mock_res2.choices = [MagicMock(message=MagicMock(tool_calls=[
        MagicMock(function=MagicMock(arguments=json.dumps({"daftar_soal": repair_questions})))
    ]))]
    
    mock_client.chat.completions.create.side_effect = [mock_res1, mock_res2, mock_res2]
    
    res = await generate_interactive_quiz_tool("jeruk nipis", 12, "Pelajar")
    
    # Result must contain exactly 12 questions
    assert len(res["questions"]) == 12
    # Ensure there are no duplicate question texts
    texts = [q["question"] for q in res["questions"]]
    assert len(set(texts)) == 12

# 7. Five Times Prompt Variance Test
@pytest.mark.asyncio
@patch("app.agent.quiz_generator.embed_text")
@patch("app.agent.quiz_generator._client")
async def test_prompt_variance_five_times(mock_client, mock_embed):
    mock_embed.return_value = [0.1] * 384
    # Force API failure to trigger the rich procedural fallback which uses the blueprint and randomizer
    mock_client.chat.completions.create.side_effect = Exception("API Offline")
    
    results = []
    for _ in range(5):
        res = await generate_interactive_quiz_tool("buat kuis 12 soal tentang kandungan jeruk nipis", 12, "Pelajar")
        results.append(res)
        
    # Kriteria verification
    for idx, res in enumerate(results):
        # 1. Tepat 12 soal
        assert len(res["questions"]) == 12
        
        # 2. Tidak ada exact duplicate pertanyaan dalam satu kuis
        q_texts = [q["question"] for q in res["questions"]]
        assert len(set(q_texts)) == 12
        
        # 3. Posisi jawaban benar tidak semuanya sama
        corrects = [q["correct_answer"] for q in res["questions"]]
        assert len(set(corrects)) > 1 # must be distributed
        
        # 4. Minimal tiga tingkat kesulitan
        diffs = [q["tingkat_kesulitan"] for q in res["questions"]]
        assert len(set(diffs)) >= 3
        
        # 5. Minimal empat jenis soal
        types = [q.get("jenis_soal") for q in res["questions"]]
        assert len(set(types)) >= 4
        
        # 6. Semua pertanyaan relevan dengan jeruk nipis
        for q in res["questions"]:
            assert "jeruk nipis" in q["question"].lower() or "jeruk nipis" in q["explanation"].lower() or "jeruk nipis" in q.get("subtopik", "").lower()

    # 7. Tidak ada hasil kuis yang identik 100%
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            # Check that correct labels list or questions list order is different
            labels_i = [q["correct_answer"] for q in results[i]["questions"]]
            labels_j = [q["correct_answer"] for q in results[j]["questions"]]
            questions_i = [q["question"] for q in results[i]["questions"]]
            questions_j = [q["question"] for q in results[j]["questions"]]
            
            # They should not be identical in both questions and answers
            assert not (labels_i == labels_j and questions_i == questions_j)
