"""
This file is cereated to clean the data:
	take out reapeated rows 
	normalize and scale data.
"""

import pandas as pd

def saveExcel(path, df):
	df.to_excel(path, index=False)
	
	
	



def readExcel(path,sheet_name, skiprows=range(0), skipfooter=0):
	data = pd.read_excel(path,sheet_name = sheet_name, skiprows = skiprows, skipfooter =skipfooter)
	
	return data
