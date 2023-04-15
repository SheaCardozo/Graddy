def CrossEntropyLoss(pred, true, eps=1e-8):
  return -sum([sum([tt * (pt + eps).log() for pt, tt in zip(p, t)]) for p, t in zip(pred, true)]) / len(pred)