import pandas as pd

class DataProcessing():

    def load_data(self,path):
        # Looked at the data (for the rawData) and it is clean as is
    
        df = pd.read_excel(path)
        print(df.head())
        df = df.to_numpy()
        return df
        