def MSELoss(pred, true):
  return -sum([sum([(tt - pt) ** 2 for pt, tt in zip(p, t)]) for p, t in zip(pred, true)]) / len(pred)