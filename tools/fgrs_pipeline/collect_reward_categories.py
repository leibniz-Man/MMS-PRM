import json
from collections import Counter
path = '/Users/kaka/Desktop/paper_projects/CoS-main/tools/fgrs_pipeline/output/dpo_step_judgments.jsonl'
out_path = '/Users/kaka/Desktop/paper_projects/CoS-main/tools/fgrs_pipeline/output/reward_categories_summary.txt'
by_id_path = '/Users/kaka/Desktop/paper_projects/CoS-main/tools/fgrs_pipeline/output/reward_categories_by_id.jsonl'

good = Counter()
bad = Counter()
by_id = []
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        obj = json.loads(line)
        gid = obj.get('id')
        good_cats = []
        bad_cats = []
        for r in obj.get('chosen_step_reviews', []):
            cats = r.get('reward_categories', [])
            if isinstance(cats, list):
                good.update(cats)
                good_cats.extend(cats)
        for r in obj.get('rejected_step_reviews', []):
            cats = r.get('reward_categories', [])
            if isinstance(cats, list):
                bad.update(cats)
                bad_cats.extend(cats)
        by_id.append({
            'id': gid,
            'good_reward_categories': good_cats,
            'bad_reward_categories': bad_cats,
        })

with open(out_path, 'w', encoding='utf-8') as w:
    w.write('Good reward_categories (counts):\n')
    for k, v in sorted(good.items(), key=lambda x: (-x[1], x[0])):
        w.write(f'{k}: {v}\n')
    w.write('\nBad reward_categories (counts):\n')
    for k, v in sorted(bad.items(), key=lambda x: (-x[1], x[0])):
        w.write(f'{k}: {v}\n')
    w.write('\nGood unique:\n')
    w.write(', '.join(sorted(good.keys())) + '\n')
    w.write('\nBad unique:\n')
    w.write(', '.join(sorted(bad.keys())) + '\n')

with open(by_id_path, 'w', encoding='utf-8') as wj:
    for item in by_id:
        wj.write(json.dumps(item, ensure_ascii=False) + '\n')

print('Wrote summary to:', out_path)
print('Wrote per-id JSONL to:', by_id_path)