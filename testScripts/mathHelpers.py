import math





#given two tuples (mean,variance), output tuple of mean,var after multiplication
def multiplyIndVar(a, b):
  # m1, v1 = float(a[0]), float(a[1]) 
  # m2, v2 = float(b[0]), float(b[1])
  m1, v1 = a  
  m2, v2 = b 
  m3 = m1 * m2
  v3_partial = math.sqrt((v1/m1)**2 + (v2/m2)**2)
  return m3,  m3 * v3_partial

#tuple with int or float values that should be rounded and cast to string
#roundToTuple - a tuple that specifies how many places each element should be rounded to 
def tupleToStringsRounding(tup, roundToTuple=(2,2)):
  if len(tup) == 2:
    return str(round(tup[0],roundToTuple[0])), str(round(tup[1],roundToTuple[1]))
