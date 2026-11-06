# Human-review merge: validation report

Inputs (unmodified): `reports/audit/review_package/review_checklist.csv`, `reports/audit/review_package/dialogue_review_20(1).xlsx`.  Output: `reports/audit/review_package/review_checklist_human_completed.csv`.

## Checks
- both files read; 20 unique IDs D01-D20 in each; no duplicates, no missing/extra IDs
- 13 experimental columns compared cell-by-cell: **0 discrepancies**; headers identical
- experimental columns also match the logged dev episodes (decision, final, utility, base_id, split)
- 7 H_ columns copied verbatim for 20 dialogues; round-trip check of merged file passed
- dialogues with clarification: 13; without: 7

## Distributions of human entries (verbatim)
- H_question_relevance(1-5 or Y/N): {'': 7, '5': 6, '2': 2, '4': 1, '3': 2, 'Y': 1, '1': 1}
- H_reply_appropriateness: {'': 7, 'Y': 12, 'N': 1}
- H_audio_asr_consistency: {'N': 3, 'Y': 16, 'unsure': 1}
- H_final_action_correct: {'unsure': 2, 'Y': 14, 'N': 4}
- H_overall_valid(Y/N/unsure): {'unsure': 3, 'Y': 17}
- H_reviewed_by_human: {'YES': 20}

## Items for human attention (nothing was altered)
- D02: H_notes may be truncated (ends 'ce was already 94.5%')
- D03: H_notes may be truncated (ends 'g the requested call')
- D04: H_notes may be truncated (ends 'ce was already 99.4%')
- D05: H_notes may be truncated (ends 'confidence was 87.2%')
- D11: H_notes may be truncated (ends 'that was appropriate')
- D12: H_notes may be truncated (ends 'sufficiently certain')
- D16: H_notes may be truncated (ends 'eatedly across cases')
- D17: overall valid=Y but final action correct=N (may be intended: dialogue valid, agent wrong)
- D17: H_notes may be truncated (ends ' and F10 were raised')
- D18: overall valid=Y but final action correct=N (may be intended: dialogue valid, agent wrong)
- D20: overall valid=Y but final action correct=N (may be intended: dialogue valid, agent wrong)

## Information
- readable: csv 20 rows x 20 cols; xlsx sheets ['Dialogue Review', 'How to Review'], 20 rows x 20 cols
- experimental columns re-checked against data/phase5/dev.v1/episodes.jsonl (read-only)
- D01: contains 'unsure' (OC=unsure, FC=unsure, AC=N); kept verbatim
- D03: contains 'unsure' (OC=Y, FC=unsure, AC=N); kept verbatim
- D04: contains 'unsure' (OC=Y, FC=Y, AC=unsure); kept verbatim
- D08: contains 'unsure' (OC=unsure, FC=N, AC=N); kept verbatim
- D19: contains 'unsure' (OC=unsure, FC=Y, AC=Y); kept verbatim
