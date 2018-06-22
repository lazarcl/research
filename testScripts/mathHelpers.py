import math





#given two tuples (mean,variance), output tuple of mean,var after multiplication
def multiplyIndVar(a, b):
  m1, v1 = a  
  m2, v2 = b 
  m3 = m1 * m2
  v3_partial = (v1/m1) + (v2/m2)
  return m3,  m3 * v3_partial


def divIndVar(a,b):
  m1, v1 = a  
  m2, v2 = b 
  m3 = m1 * m2
  v3_partial = (v1/m1) + (v2/m2)
  return m3,  m3**2 * v3_partial

def addIndVar(a,b):
  m1, v1 = a  
  m2, v2 = b 
  return m1 + m2,  v1 + v2

def subIndVar(a,b):
  m1, v1 = a  
  m2, v2 = b 
  return m1 - m2,  v1 + v2

def multIndVarAndConst(a, const):
	return a[0] * const, a[1] * const**2



#tuple with int or float values that should be rounded and cast to string
#roundToTuple - a tuple that specifies how many places each element should be rounded to 
def tupleToRoundedStrings(tup, roundToTuple=(2,2)):
  if len(tup) == 2:
    return str(round(tup[0],roundToTuple[0])), str(round(tup[1],roundToTuple[1]))


#convert the variance to a percentage of the mean
def varToPercent(meanVarTuple):
	whole, part = meanVarTuple
	return (whole, 100*(math.sqrt(part)/whole))


