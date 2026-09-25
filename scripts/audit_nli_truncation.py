"""Replay actual ModernBERT pair tokenization without loading model weights."""
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pair_evaluation import load_rows, evaluate_pairs
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/ModernBERT-large-nli"), local_files_only=True)
    pairs = load_rows(ROOT / "data/ragtruth/processed/doc_claim_pairs_50.jsonl")
    runs = {"old": ROOT / "outputs/nli_results_50-39082.jsonl",
            "512": ROOT / "outputs/modernbert-dev-512-39205/nli_pairs_50.jsonl",
            "2048": ROOT / "outputs/modernbert-all-2048-39156/nli_pairs_50.jsonl"}
    rows = {k: load_rows(v) for k, v in runs.items()}
    for value in rows.values():
        evaluate_pairs(value, pairs)
    indexed = {k: {r['pair_id']: r for r in v} for k, v in rows.items()}
    audit = []
    for pair in pairs:
        document, claim = pair['document'], pair['claim']
        full = tokenizer(document, claim, truncation=False)['input_ids']
        record = {'pair_id': pair['pair_id'], 'gold': pair['gold_label'], 'tokens': len(full)}
        for limit in (512, 2048):
            enc = tokenizer(document, claim, truncation=True, max_length=limit, return_offsets_mapping=True)
            offsets = enc['offset_mapping']; sequences = enc.sequence_ids()
            doc_end = max((end for (start, end), seq in zip(offsets, sequences) if seq == 0), default=0)
            claim_end = max((end for (start, end), seq in zip(offsets, sequences) if seq == 1), default=0)
            record[str(limit)] = {'truncated': len(full) > limit, 'input_tokens': len(enc['input_ids']),
                                  'document_end': doc_end, 'claim_end': claim_end,
                                  'pred': indexed[str(limit)][pair['pair_id']]['pred_label']}
        record['old_pred'] = indexed['old'][pair['pair_id']]['pred_label']
        audit.append(record)
    groups = {}
    for label, subset in [('all', audit), ('le512', [r for r in audit if r['tokens'] <= 512]),
                           ('gt512', [r for r in audit if r['tokens'] > 512]),
                           ('gt2048', [r for r in audit if r['tokens'] > 2048])]:
        groups[label] = {'count': len(subset), 'gold_counts': dict(Counter(r['gold'] for r in subset)),
                         'errors': {k: sum((r['old_pred'] if k == 'old' else r[k]['pred']) != r['gold'] for r in subset) for k in runs}}
    changed = []
    source = {r['pair_id']: r for r in pairs}
    for r in audit:
        if r['512']['pred'] != r['2048']['pred']:
            p = source[r['pair_id']]
            changed.append({**r, 'claim': p['claim'], 'document': p['document'],
                            'added_document': p['document'][r['512']['document_end']:r['2048']['document_end']],
                            'scores': {k: indexed[k][r['pair_id']]['nli_scores'] for k in ('512', '2048')}})
    lengths = sorted(r['tokens'] for r in audit)
    summary = {'tokenizer': 'local ModernBERT-large-nli', 'truncation_strategy': tokenizer.truncation_side + ' / longest_first',
               'lengths': {'min': min(lengths), 'median': lengths[len(lengths)//2], 'max': max(lengths)},
               'groups': groups, 'changed': changed, 'pairs': audit}
    output = ROOT / 'outputs/modernbert_truncation_audit.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k,v in summary.items() if k != 'pairs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
