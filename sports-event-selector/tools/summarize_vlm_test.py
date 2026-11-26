"""Create a readable audit of a completed selector_vlm evaluation."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', required=True)
p.add_argument('--dataset', default='data/tenniset_selector_vlm')
args = p.parse_args()
run, dataset = Path(args.run), Path(args.dataset)
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
predictions = read(run/'predictions.json')
metrics = read(run/'metrics.json')
examples = [e for e in read(dataset/'examples.json') if e['split']=='test']
if metrics.get('partial_run') or len(predictions) != len(examples):
    raise ValueError('Wait for the entire test run to finish')
durations = [r['seconds'] for r in predictions.values()]
usage = {k: sum((r.get('usage') or {}).get(k) or 0 for r in predictions.values())
         for k in ('prompt_token_count','candidates_token_count','thoughts_token_count','total_token_count')}
summary = {'requests':len(predictions), 'total_request_seconds':sum(durations),
           'mean_request_seconds':mean(durations), 'median_request_seconds':median(durations),
           'usage':usage, 'note':'Request durations exclude dataset preparation; cached requests count once.'}
(run/'usage-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
rows=[]
for e in examples:
    result=predictions[e['id']]
    rows.append({'id':e['id'],'predicted_center_seconds':e['center_seconds'],
        'window_start_seconds':e['frames'][0]['timestamp'],'window_end_seconds':e['frames'][-1]['timestamp'],
        'target_id':e['target_id'],
        'original_annotation_times':json.dumps(e['original_timestamps']),
        'expected_event':json.dumps(e['output']), 'predicted_event':json.dumps(result['answer']),
        'seconds':result['seconds']})
with (run/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# Trained VLM test results','',f"Model endpoint: `{metrics['model']}`",'',
       f"Completed {len(examples)} windows; {metrics['labeled_windows']} labeled windows scored, "
       f"{metrics['unscored_windows']} unmatched windows retained for review.",'',
       '| Metric | Result |','|---|---:|',
       f"| Exact event accuracy | {metrics['exact_event_accuracy']:.1%} |"]
for field,value in metrics['field_accuracy'].items():
    lines.append(f'| {field} exact accuracy | {value:.1%} |')
lines += [f"| Distinct exactly correct events | {metrics['distinct_exact_events']} / {metrics['reference_events']} |",
          f"| End-to-end exact event recall | {metrics['end_to_end_exact_event_recall']:.1%} |",
          f"| Invalid or missing outputs | {metrics['invalid_or_missing_outputs']} |",'',
          f"Mean request duration: {mean(durations):.2f} seconds. Median: {median(durations):.2f} seconds.",
          f"Input tokens: {usage['prompt_token_count']:,}. Output tokens: {usage['candidates_token_count']:,}.",'',
          'Accuracy uses exact field strings after case/whitespace normalization. Synonyms are not merged. '
          'The 31 labeled windows map to 17 distinct target events; repeated windows do not inflate '
          'event recall. This pilot reuses the V010 match and is not an independent test.', '',
          'Inspect comparison.csv for original annotation timestamps, expected events and every response. '
          'Unmatched windows are unscored, not established negatives. Check visually before concluding '
          'that a model output or source label is wrong. The run does not estimate dollar cost.']
(run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
