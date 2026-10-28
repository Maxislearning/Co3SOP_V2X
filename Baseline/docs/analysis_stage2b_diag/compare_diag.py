import pandas as pd, numpy as np
D='/home/admin0/Co3SOP/Baseline/work_dirs/loro_cv/stage2b_diag/data/'
old=pd.read_csv(D+'target_diag_rows.csv'); new=pd.read_csv(D+'target_diag_rows_loro_seed5.csv')
def prep(d):
    d=d[d.gt_vox>0].copy()
    b=d.gt_bearing.abs()
    d['bin']=np.where(b<30,'0-30',np.where(b<=150,'30-150(侧)','150-180'))
    d['dbin']=pd.cut(d.gt_dist,[0,10,20,30,60],labels=['<10','10-20','20-30','30+'])
    d['pred_on_ego']=d.pred_dist<3
    d['lr']=d.tx_panel.isin([1,3])|d.rx_panel.isin([1,3])
    return d
old,new=prep(old),prep(new)
def summ(d,by):
    g=d.groupby(by,observed=True)
    return pd.DataFrame({'n':g.size(),'IoU':g.iou3d.mean(),'近目标质量':g.mass_near_frac.mean(),
        '方位误差中位':g.bearing_err_deg.median(),'预测落在本车3m内':g.pred_on_ego.mean()}).round(3)
for name,sel in [('seed5（未见，干净）',lambda d:d.run=='seed5'),('legacy+seed1-4（新Stage2B训练过）',lambda d:d.run!='seed5')]:
    print('\n=====',name)
    for by in [None,'bin','dbin','link']:
        o,n_=old[sel(old)],new[sel(new)]
        if by is None:
            o=o.assign(all='all'); n_=n_.assign(all='all'); by='all'
        print(f'-- by {by}'); print(pd.concat({'旧':summ(o,by),'新':summ(n_,by)},axis=1).to_string())
# 按 run 的 IoU
print('\n-- IoU by run'); print(pd.DataFrame({'旧':old.groupby('run').iou3d.mean(),'新':new.groupby('run').iou3d.mean()}).round(3).to_string())
