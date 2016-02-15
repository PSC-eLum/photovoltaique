import pandas as pd
import numpy as np
class data:
	def __init__(self,X,y,colsX,I):
		self.base = pd.DataFrame(X,columns=colsX)
		self.y    = y
		self.colsI = I[:,0]
	
	def toPred(self):
		copy = self.base.copy(deep=True)
		return  copy.drop(self.colsI,axis=1)

	def toArray(self):
		return np.array(self.toPred()).astype(np.float)
		
