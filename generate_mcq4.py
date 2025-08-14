# generate_mcq3.py (patched, stable)

import fitz
import re as _re
import os
import json
import time
import csv
import random
from openai import OpenAI
from dotenv import load_dotenv
from ast_inference import detect_misconception_nodes


# ==== Setup ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in your .env")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==== Config ====
BOOKS = [
    {"name": "Java_How_to_Program", "path": "PDFs/1_Java_How to Program 11th Early Objects.pdf", "start": 100, "end": 200},
    {"name": "Head_First_Java", "path": "PDFs/2_Head_First_Java_Second_Edition.pdf", "start": 200, "end": 300},
    {"name": "Think_Java", "path": "PDFs/10_Think Java- How to Think Like a Computer Scientist.pdf", "start": 100, "end": 200},
    {"name": "Starting_Out_with_Java", "path": "PDFs/8_Starting Out with Java- From Control Structures through Objects.pdf", "start": 800, "end": 900},
    {"name": "Java_Programming", "path": "PDFs/7_Java Programming.pdf", "start": 500, "end": 600},
]
concept_json = "concept_inventory2.json"
concept_csv = "concept_inventory2.csv"
mcq_csv = "mcq_inventory2.csv"

# ==== CSV loaders ====
def load_distractor_csv(csv_path):
    db = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Expect columns: topic, option
        if "topic" not in reader.fieldnames or "option" not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} must contain 'topic' and 'option' columns")
        for row in reader:
            topic = (row.get('topic') or '').strip()
            option = (row.get('option') or '').strip()
            if not topic or not option:
                continue
            db.setdefault(topic, []).append(option)
    return db

def load_fallback_distractors(path):
    # Same format as load_distractor_csv
    return load_distractor_csv(path)


# ==== Text processing ====
def extract_pdf_text(path, start=0, end=50):
    doc = fitz.open(path)
    text = ""
    for i in range(start, min(end, len(doc))):
        text += doc[i].get_text()
    return text

def filter_text_blocks(text):
    lines = text.split('\n')
    filtered = []
    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue

        if _re.search(r'[{}();<>]', line) or line.endswith(';'):
            filtered.append(line); continue
        if line.lower().startswith(("figure", "table")):
            filtered.append(line); continue
        if any(keyword in line.lower() for keyword in ["note", "shows", "example"]):
            filtered.append(line); continue
        filtered.append(line)
    return '\n'.join(filtered)

def split_text(text, chunk_size=3000):
    lines = text.split('\n')
    chunks, current = [], ""
    for line in lines:
        if len(current) + len(line) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
            current = ""
        current += line + '\n'
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ==== Prompts ====
def build_concept_prompt(text):
    return f'''
You are a CS education expert. Analyze the following Java textbook content and extract core CS1 concepts grouped by topic.

Return JSON only, like:
[
  {{"topic": "Control Flow", "concept": "while vs do-while loop"}},
  {{"topic": "Data Types", "concept": "Integer vs floating-point division"}}
]

Text:
\"\"\"{text}\"\"\"
'''.strip()

def build_stem_prompt(topic, concept):
    return f'''
You are an AI tutor. Given the topic and concept below, generate:
- a single-choice question stem
- the correct answer
- a Bloom's level (Remember / Understand / Apply)
- a short explanation

Topic: {topic}
Concept: {concept}

Return only JSON:
{{
  "question": "...",
  "correct_answer": "...",
  "bloom_level": "...",
  "explanation": "..."
}}
'''.strip()

def build_distractor_prompt(question, correct, candidate_list, structure_errors=None):
    candidate_str = '\n'.join([f"- {c}" for c in candidate_list])

    structure_hint = ""
    if structure_errors:
        structure_hint = "\nLikely misconceptions in these structures:\n"
        for e in structure_errors:
            subtree = e.get("subtree", "<code>")
            etype = e.get("type", "unknown")
            structure_hint += f"- {subtree} (type: {etype})\n"

    return f'''
You are a Java tutor. Select 3 plausible but incorrect distractor options from the list below for the given question.

Question: {question}
Correct answer: {correct}
{structure_hint}
Candidates:
{candidate_str}

Rules:
- Do not repeat the correct answer
- Prioritize options that reflect typical student misconceptions
- Match the topic and code semantics

Return JSON list of 3 strings.
'''.strip()

def build_free_distractor_prompt(question, correct, topic, concept, structure_errors=None):
    structure_hint = ""
    if structure_errors:
        structure_hint = "\nLikely misconceptions in these structures:\n"
        for e in structure_errors:
            subtree = e.get("subtree", "<code>")
            etype = e.get("type", "unknown")
            structure_hint += f"- {subtree} (type: {etype})\n"

    return f'''
You are a Java tutor. For the following question, generate 3 plausible but incorrect distractors.

Topic: {topic}
Concept: {concept}
Question: {question}
Correct answer: {correct}
{structure_hint}

Rules:
- Make them realistic misconceptions for CS1 students
- Same style/length as the correct answer
- Must be incorrect but tempting
- Do not duplicate the correct answer

Return JSON list of 3 strings.
'''.strip()


# ==== Robust JSON helpers ====
def _strip_code_fences(s: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    return _re.sub(r"```(?:json)?\s*|```", "", s).strip()

def parse_json_safe(text, expect_list_or_obj="either"):
    """
    expect_list_or_obj: 'list' / 'obj' / 'either'
    - Strips code fences
    - Extracts first JSON object or array
    - Returns {} or [] if not found
    """
    if not text:
        return [] if expect_list_or_obj != "obj" else {}
    s = _strip_code_fences(text)

    obj_match = _re.search(r"\{.*?\}", s, flags=_re.S)
    lst_match = _re.search(r"\[.*?\]", s, flags=_re.S)

    candidates = []
    if expect_list_or_obj in ("either", "obj") and obj_match:
        candidates.append(obj_match.group(0))
    if expect_list_or_obj in ("either", "list") and lst_match:
        candidates.append(lst_match.group(0))

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    # fallback: slice from first bracket/brace to last
    try:
        start = s.find('[') if '[' in s else s.find('{')
        end = s.rfind(']') if ']' in s else s.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
    except Exception:
        pass

    return [] if expect_list_or_obj != "obj" else {}

# ==== GPT Call ====
def call_gpt(prompt, expect="text"):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        content = resp.choices[0].message.content
        if expect == "json_list":
            return parse_json_safe(content, expect_list_or_obj="list")
        elif expect == "json_obj":
            return parse_json_safe(content, expect_list_or_obj="obj")
        else:
            return content
    except Exception as e:
        print("❌ OpenAI API error:", e)
        return [] if expect.startswith("json") else ""

# ==== Output ====
def save_as_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_as_csv(data, filename):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["topic", "concept"])
        writer.writeheader()
        for item in data:
            writer.writerow(item)

def save_mcq_csv(data, filename):
    with open(filename, "w", newline='', encoding='utf-8') as f:
        fieldnames = ["topic", "concept", "question", "correct_answer", "bloom_level", "explanation",
                      "option_A", "option_B", "option_C", "option_D"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            row = {
                "topic": item["topic"],
                "concept": item["concept"],
                "question": item["question"],
                "correct_answer": item["correct_answer"],  # label A/B/C/D
                "bloom_level": item["bloom_level"],
                "explanation": item["explanation"],
                **{f"option_{k}": v for k, v in item["options"].items()}
            }
            writer.writerow(row)


# ==== Main ====
def main():
    # ==== Load distractors safely (moved from global) ====
    try:
        misconception_db = load_distractor_csv("distractor_output/cleaned_again2_classified_distractors.csv")
        print(f"✅ Loaded main distractor DB: {sum(len(v) for v in misconception_db.values())} items")
    except Exception as e:
        print("⚠️ Failed to load main distractor CSV:", e)
        misconception_db = {}

    try:
        fallback_db = load_fallback_distractors("fallback_distractors.csv")
        print(f"✅ Loaded fallback distractor DB: {sum(len(v) for v in fallback_db.values())} items")
    except Exception as e:
        print("⚠️ Failed to load fallback distractor CSV:", e)
        fallback_db = {}

    all_concepts = []
    chunk_text_map = {}
    seen_concepts = set()

    # ==== Concept extraction ====
    for book in BOOKS:
        print(f"\n📘 {book['name']}")
        try:
            raw = extract_pdf_text(book["path"], book["start"], book["end"])
        except Exception as e:
            print(f"⚠️ Failed to open PDF {book['path']}: {e}")
            continue

        filtered = filter_text_blocks(raw)
        chunks = split_text(filtered)
        print(f"   ➜ {len(chunks)} chunks to analyze")

        for idx, chunk in enumerate(chunks, 1):
            prompt = build_concept_prompt(chunk)
            result = call_gpt(prompt, expect="json_list")
            concepts = result if isinstance(result, list) else []

            added = 0
            for c in concepts:
                topic = c.get("topic")
                concept = c.get("concept")
                if not topic or not concept:
                    continue
                key = (topic.strip(), concept.strip())
                if key in seen_concepts:
                    continue
                seen_concepts.add(key)
                chunk_text_map[concept.strip()] = chunk
                all_concepts.append({"topic": topic.strip(), "concept": concept.strip()})
                added += 1

            print(f"   - chunk {idx}/{len(chunks)} -> +{added} concepts, total {len(all_concepts)}")
            if len(all_concepts) >= 50:
                break
            time.sleep(1.2)
        if len(all_concepts) >= 50:
            break

    save_as_json(all_concepts, concept_json)
    save_as_csv(all_concepts, concept_csv)
    print(f"\n🧾 Saved concepts to {concept_json} / {concept_csv} (total {len(all_concepts)})")

    # ==== MCQ generation ====
    print("\n🎯 Generating MCQs...")
    all_mcq = []

    for i, item in enumerate(all_concepts, 1):
        print(f"\n[{i}/{len(all_concepts)}] {item['topic']} :: {item['concept']}")
        stem_prompt = build_stem_prompt(item["topic"], item["concept"])
        try:
            stem_raw = call_gpt(stem_prompt, expect="text")
            partial = parse_json_safe(stem_raw, expect_list_or_obj="obj")

            # Validate required fields
            for k in ("question", "correct_answer", "bloom_level", "explanation"):
                if k not in partial or not partial[k]:
                    raise ValueError(f"Missing field in stem JSON: {k}")

            question = str(partial["question"]).strip()
            correct = str(partial["correct_answer"]).strip()

            # Structure analysis on the whole chunk (safe)
            chunk = chunk_text_map.get(item["concept"], "")
            structure_errors = []
            if chunk and isinstance(chunk, str) and len(chunk) > 0:
                try:
                    structure_errors = detect_misconception_nodes(chunk)
                except Exception as e:
                    print("   ⚠️ structure analysis failed:", e)

            # Candidate distractors
            candidate_pool = misconception_db.get(item["topic"], [])
            if len(candidate_pool) < 5:
                fallback_pool = fallback_db.get(item["topic"], [])
                combined_pool = list(dict.fromkeys(candidate_pool + fallback_pool))
            else:
                combined_pool = candidate_pool

            if combined_pool:
                distractor_prompt = build_distractor_prompt(question, correct, combined_pool[:20], structure_errors)
                distractors = call_gpt(distractor_prompt, expect="json_list")
            else:
                free_prompt = build_free_distractor_prompt(question, correct, item["topic"], item["concept"], structure_errors)
                distractors = call_gpt(free_prompt, expect="json_list")

            if not isinstance(distractors, list):
                distractors = []
            # If still <3, top up with free generation
            if len(distractors) < 3:
                need = 3 - len(distractors)
                extra_prompt = build_free_distractor_prompt(question, correct, item["topic"], item["concept"], structure_errors)
                extra = call_gpt(extra_prompt, expect="json_list")
                if isinstance(extra, list):
                    for d in extra:
                        if d and d not in distractors and d != correct:
                            distractors.append(d)
                            if len(distractors) >= 3:
                                break
            distractors = [d for d in distractors if d and d != correct][:3]

            # Build options (ensure 4 options)
            options = [correct] + distractors
            while len(options) < 4:
                options.append(f"None of the above {len(options)}")  # placeholder fallback
            options = options[:4]
            random.shuffle(options)

            labels = ["A", "B", "C", "D"]
            option_dict = dict(zip(labels, options))
            correct_label = next((k for k, v in option_dict.items() if v == correct), None)
            if not correct_label:
                # If correct was replaced by shuffle/placeholder, force it into A
                option_dict["A"] = correct
                correct_label = "A"

            mcq = {
                "topic": item["topic"],
                "concept": item["concept"],
                "question": question,
                "correct_answer": correct_label,  # label
                "bloom_level": str(partial["bloom_level"]).strip(),
                "explanation": str(partial["explanation"]).strip(),
                "options": option_dict
            }
            all_mcq.append(mcq)
            print(f"   ✅ MCQ ok (correct: {correct_label})")

        except Exception as e:
            print(f"   ⚠️ MCQ generation failed: {e}")
        time.sleep(1.2)

    save_mcq_csv(all_mcq, mcq_csv)
    print(f"\n✅ Generated {len(all_mcq)} questions -> {mcq_csv}")


if __name__ == "__main__":
    main()
