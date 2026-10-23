import os, sys, numpy as np, pandas as pd
ROOT='/home/admin0/Co3SOP/Baseline'; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+'/tools'); os.chdir(ROOT)
from projects.mmdet3d_plugin_carla_v2v.datasets.occ_beam_dataset import OccBeamDataset
out=sys.argv[1]
for split in ('val','train'):
    ds=OccBeamDataset(split,'gt','target')
    rows=[]
    for k in ds.infos:
        r=ds._beam_by_key[k]
        d=dict(sample_id=k[0], link=k[1], tx=int(r.optimal_tx_beam_idx), rx=int(r.optimal_rx_beam_idx))
        for c in ('is_los','source_run','tx_id'):
            if hasattr(r,c): d[c]=getattr(r,c)
        rows.append(d)
    pd.DataFrame(rows).to_csv(f'{out}/meta_{split}.csv',index=False)
    print(split,len(rows),'fields',type(r), getattr(r,'_fields',None) and r._fields[:200])
