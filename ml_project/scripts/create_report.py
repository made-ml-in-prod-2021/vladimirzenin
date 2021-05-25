import pandas as pd
from pandas_profiling import ProfileReport
from typing import NoReturn


def create_report() -> NoReturn:
	data = pd.read_csv('../data/external/heart.csv')
	profile = ProfileReport(data)
	profile.to_file('../reports/EDA.html')
	

def main() -> NoReturn:
	create_report()

	
if __name__ == "__main__":
	main()