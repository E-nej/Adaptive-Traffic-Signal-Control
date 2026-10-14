"""Copy worker runs that failed acceptance but had a learned checkpoint, as table candidates.

    python configs/v6/admit_candidates.py outputs/train/v6_wk_maribor_maxpressure

Writes outputs/train/v6c_wk_.../s*/ with the run's config and best weights and the
hierarchy meta switched from 'expert alone' to 'worker active'.  The originals (and their
accept.json) stay as they are; a table that uses the copies measures every level with and
without the worker (hierarchy.table_alone) and keeps the worker only where it helps.
"""

import glob
import json
import os
import shutil
import sys


def main(src_root):
    name = os.path.basename(os.path.normpath(src_root))
    dst_root = os.path.join(os.path.dirname(os.path.normpath(src_root)),
                            name.replace('v6_wk_', 'v6c_wk_', 1))
    for src in sorted(glob.glob(os.path.join(src_root, 's*'))):
        with open(os.path.join(src, 'accept.json')) as f:
            acc = json.load(f)
        if 'learned' not in acc:
            print(f'{src}: no learned checkpoint ({acc.get("reason")}), skipped')
            continue
        dst = os.path.join(dst_root, os.path.basename(src))
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, 'best_*')) + [os.path.join(src, 'config.toml'),
                                                         os.path.join(src, 'accept.json')]:
            shutil.copy2(f, dst)
        meta_path = os.path.join(dst, 'best_hierarchy.json')
        with open(meta_path) as f:
            meta = json.load(f)
        meta['agree'] = False
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        print(f'{src} -> {dst} (acceptance {acc["change"]:+.1%}, p={acc["p"]:.3f}; now a candidate)')


if __name__ == '__main__':
    for root in sys.argv[1:]:
        main(root)
