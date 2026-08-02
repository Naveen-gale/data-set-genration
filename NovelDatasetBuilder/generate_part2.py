import sys
import logging
from pathlib import Path
import json
import random

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from src.extract_pdf import extract_pdf
from src.detect_sections import detect_sections
from src.chunk_builder import build_chunks
from src.llm_generator import LLMGenerator
from src.config import INPUT_PDF, OUTPUT_DIR, PROMPTS_DIR

EXISTING_DATASET = OUTPUT_DIR / "final_dataset.jsonl"
NEW_DATASET = OUTPUT_DIR / "working_dataset_part2.jsonl"
PART2_PROMPT_FILE = PROMPTS_DIR / "part2_generation_prompt.txt"

# 1. Load existing pairs
existing_instructions = set()
existing_responses = set()

for file_path in [EXISTING_DATASET, NEW_DATASET]:
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        existing_instructions.add(data.get('instruction', '').strip().lower())
                        existing_responses.add(data.get('response', '').strip().lower())
                    except json.JSONDecodeError:
                        pass

logger.info(f"Loaded {len(existing_instructions)} existing instructions to avoid.")

# 2. Extract PDF and build chunks
novel_text = extract_pdf(INPUT_PDF)
sections = detect_sections(novel_text)
chunks = build_chunks(sections)

# 3. Create a custom LLMGenerator
class Part2Generator(LLMGenerator):
    def __init__(self, output_path, prompt_template):
        # We set resume=False so we don't skip chunks based on part 1's checkpoint
        super().__init__(output_path=output_path, prompt_template=prompt_template, resume=False)
        self.global_seen_instructions = existing_instructions
        self.global_seen_responses = existing_responses

    def _build_prompt(self, chunk, num_pairs, existing):
        avoid = ""
        if self.global_seen_instructions:
            sample = random.sample(list(self.global_seen_instructions), min(15, len(self.global_seen_instructions)))
            avoid = (
                "\n\nALREADY GENERATED IN PREVIOUS PASS (DO NOT REPEAT OR REPHRASE THESE TOPICS):\n"
                + "\n".join("- " + s[:100] for s in sample)
            )
        return self.prompt_template.format(
            num_pairs=num_pairs,
            chunk_text=chunk.text,
            source_section=chunk.source_section,
            section_type=chunk.section_type,
        ) + avoid

    def _generate_batch(self, chunk, num_pairs, batch_idx, seen):
        # Call the original method to get raw valid pairs
        pairs = super()._generate_batch(chunk, num_pairs, batch_idx, seen)
        
        # Now filter them against the global set
        filtered_pairs = []
        for p in pairs:
            inst = p["instruction"].strip().lower()
            resp = p["response"].strip().lower()
            if inst in self.global_seen_instructions or resp in self.global_seen_responses:
                logger.info(f"Dropped duplicate pair (global check): {inst[:50]}...")
                continue
            
            self.global_seen_instructions.add(inst)
            self.global_seen_responses.add(resp)
            filtered_pairs.append(p)
            
        return filtered_pairs

new_prompt_text = """You are an expert academic dataset engineer and university-level literature professor.

Your task is to generate exactly {num_pairs} UNIQUE, high-quality instruction-response pairs for a GPT fine-tuning dataset, based ONLY on the novel passage provided below.

IMPORTANT: This is PART 2 of the dataset generation. Basic questions about the plot, main characters, and central themes have ALREADY been generated.
YOU MUST generate entirely NEW questions covering MISSING topics, nuanced details, and different question styles.

════════════════════════════════════════════════════════════
NOVEL PASSAGE:
────────────────────────────────────────────────────────────
{chunk_text}
════════════════════════════════════════════════════════════

SECTION CONTEXT:
- Source: {source_section}
- Type: {section_type}

════════════════════════════════════════════════════════════
CRITICAL RULES (NEVER VIOLATE):
════════════════════════════════════════════════════════════

1. BASE EVERYTHING ONLY ON THE PASSAGE ABOVE. Do NOT hallucinate characters, events, quotes, or scenes. Skip any example that is not fully supported by the PDF passage.
2. NEVER repeat an instruction or response that would be considered standard or obvious. Dig deep into specific quotes, imagery, minor themes, symbolism, vocabulary, and scene analysis.
3. Keep generating complex, high-quality examples useful for exam preparation.
4. Maintain balanced answer lengths: ~20% short (20-40 words), ~50% medium (60-120 words), ~30% long (150-250 words).
5. Ensure factually correct, exam-oriented, natural English, clear and specific responses.
6. The instruction field must contain ONLY the user's question. The response field must contain ONLY the answer.
7. NEVER include "Student:", "Teacher:", "A:", "Q:" inside the instruction or response.

════════════════════════════════════════════════════════════
INSTRUCTION TYPES TO GENERATE (distribute evenly across all {num_pairs} pairs):
════════════════════════════════════════════════════════════
Maintain balanced question types:
- Characters, Themes, Plot, Scene analysis
- Literary devices, Symbolism, Imagery
- Quotes, Vocabulary
- 1-mark, 2-mark, 5-mark, 10-mark, Essay
- MCQ, True/False, Fill in the blanks, Flashcards
- Simple English explanations
- Compare characters
- Why/How questions

════════════════════════════════════════════════════════════
OUTPUT FORMAT (MANDATORY — STRICTLY FOLLOW):
════════════════════════════════════════════════════════════

Return a valid JSON array containing exactly {num_pairs} objects.
Each object must have exactly two keys: "instruction" and "response".

[
  {{"instruction": "...", "response": "..."}},
  {{"instruction": "...", "response": "..."}}
]

RULES FOR OUTPUT:
- Output ONLY the JSON array. No markdown. No explanation.
- Every string value must be properly JSON-escaped.
- No trailing commas. No comments inside JSON.
"""

PART2_PROMPT_FILE.write_text(new_prompt_text, encoding='utf-8')

logger.info("Starting Part 2 Dataset Generation...")
generator = Part2Generator(output_path=NEW_DATASET, prompt_template=new_prompt_text)
total_written = generator.process_chunks(chunks)
logger.info(f"Finished. Generated {total_written} pairs in working_dataset_part2.jsonl")
