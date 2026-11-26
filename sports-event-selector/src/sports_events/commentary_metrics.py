"""Reference-based commentary metrics, with explicit temporal and lexical limits."""
from collections import Counter
import re
import numpy as np


def tokens(text):
    return re.findall(r"[a-z0-9]+",text.lower())


def rouge_l(a,b):
    a,b=tokens(a),tokens(b)
    if not a or not b:return 0.
    row=[0]*(len(b)+1)
    for x in a:
        new=[0]
        for j,y in enumerate(b):new.append(row[j]+1 if x==y else max(row[j+1],new[-1]))
        row=new
    return 2*row[-1]/(len(a)+len(b))


def semantic_metrics(predictions,references,vectors,tolerance=3.):
    """Soft best-match scores, not factual accuracy or one-to-one event matching."""
    scores=np.zeros((len(predictions),len(references)))
    for i,p in enumerate(predictions):
        for j,r in enumerate(references):
            if p['clip_id']!=r['clip_id'] or abs(p['seconds']-r['seconds'])>tolerance:continue
            a,b=np.asarray(vectors[p['text']]),np.asarray(vectors[r['text']])
            denom=np.linalg.norm(a)*np.linalg.norm(b)
            scores[i,j]=max(0.,float(np.dot(a,b)/denom)) if denom else 0.
    recall=float(scores.max(axis=0).mean()) if len(predictions) and len(references) else 0.
    precision=float(scores.max(axis=1).mean()) if len(predictions) and len(references) else 0.
    counts=Counter((p['clip_id'],' '.join(tokens(p['text']))) for p in predictions)
    return {'semantic_soft_recall':recall,'semantic_soft_precision':precision,
            'semantic_soft_f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'exact_duplicate_fraction':sum(n-1 for n in counts.values())/len(predictions) if predictions else 0.,
            'temporal_tolerance_seconds':tolerance,'reference_events':len(references),'predicted_segments':len(predictions)}
