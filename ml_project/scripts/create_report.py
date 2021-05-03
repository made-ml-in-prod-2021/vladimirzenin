import pandas as pd
from pandas_profiling import ProfileReport


def create_report():
	data = pd.read_csv('../data/external/heart.csv')
	profile = ProfileReport(data)
	profile.to_file('../reports/EDA.html')
	

def main():
	create_report()

	
if __name__ == "__main__":
	main()