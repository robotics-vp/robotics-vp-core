import numpy as np

def cosine_similarity(a, b, eps=1e-8):
    na = a / (np.linalg.norm(a) + eps)
    nb = b / (np.linalg.norm(b) + eps)
    return float(np.clip(np.dot(na, nb), -1.0, 1.0))

def novelty_score(emb, ref_embs):
    if len(ref_embs) == 0: return 1.0
    sims = [cosine_similarity(emb, r) for r in ref_embs]
    # novelty = 1 - max similarity
    return 1.0 - max(sims)
