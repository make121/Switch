"""Summarize physical-push recovery attempts (not a deployment threshold fit)."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    attempts = [r for r in rows if r['event'] == 'attempt_result']
    starts = [r for r in rows if r['event'] == 'attempt_start']
    finished = {(r['fall_event_id'], r['attempt_id']) for r in attempts}
    return {
        'file': str(path),
        'physical_pushes': sum(r['event'] == 'fall_push' for r in rows),
        'fall_events': len({r['fall_event_id'] for r in starts}),
        'completed_attempts': len(attempts),
        'unfinished_attempts': sum((r['fall_event_id'], r['attempt_id']) not in finished for r in starts),
        'returned_to_task': sum(r['result'] == 'success' for r in attempts),
        'hard_resets': sum(r['event'] == 'hard_reset' for r in rows),
        'failures_by_stage_and_reason': dict(Counter(
            str(r.get('failure_stage', 'unknown')) + ':' + str(r.get('reason'))
            for r in attempts if r['result'] != 'success')),
        'results_by_recovery_skill': {
            str(sid): dict(Counter(r['result'] for r in attempts if r['entry']['skill_id'] == sid))
            for sid in sorted({r['entry']['skill_id'] for r in attempts})
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.events), indent=2, ensure_ascii=False))
