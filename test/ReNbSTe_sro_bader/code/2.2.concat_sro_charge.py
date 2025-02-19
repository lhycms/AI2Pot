import os
import pandas as pd


if __name__ == "__main__":
    dir_path: str = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSTe_sro_bader/output"
    csv1_path: str = "total_41_75.csv"
    csv2_path: str = "total_41_16.csv"
    csv3_path: str = "total_75_16.csv"
    csv4_path: str = "total_75_41.csv"
    
    df1 = pd.read_csv(os.path.join(dir_path, csv1_path))
    df2 = pd.read_csv(os.path.join(dir_path, csv2_path))
    df3 = pd.read_csv(os.path.join(dir_path, csv3_path))
    df4 = pd.read_csv(os.path.join(dir_path, csv4_path))
    
    df_combined = pd.concat([df1, df2, df3, df4], axis=1)
    df_combined.to_csv(os.path.join(dir_path, "toatl.csv"))