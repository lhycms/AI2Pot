import os
import pandas as pd


if __name__ == "__main__":
    dir_path = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSTe_sro_bader/output"
    system_1 = "mc_2_6"
    system_2 = "mc_4_32"
    system_3 = "mc_4_335"
    system_4 = "mc_5_202"
    system_5 = "mc_5_423"
    system_6 = "mc_5_491"
    output_dir = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSTe_sro_bader/output"
    
    df1 = pd.read_csv(os.path.join(dir_path, system_1, "41_75.csv"))
    df2 = pd.read_csv(os.path.join(dir_path, system_2, "41_75.csv"))
    df3 = pd.read_csv(os.path.join(dir_path, system_3, "41_75.csv"))
    df4 = pd.read_csv(os.path.join(dir_path, system_4, "41_75.csv"))
    df5 = pd.read_csv(os.path.join(dir_path, system_5, "41_75.csv"))
    df6 = pd.read_csv(os.path.join(dir_path, system_6, "41_75.csv"))
    df_combined = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    print(df_combined)
    df_combined.to_csv(os.path.join(output_dir, "total_41_75.csv"))
    