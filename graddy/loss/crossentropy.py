def CrossEntropyLoss(pred, true):
  return -sum([sum([tt * pt.log() for pt, tt in zip(p, t)]) for p, t in zip(pred, true)]) / len(pred)