import os
import pandas as pd


if __name__ == "__main__":
    dir_path = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSSe/output"
    system_1 = "mc_1_0"
    system_2 = "mc_2_543"
    system_3 = "mc_2_715"
    system_4 = "mc_2_904"
    system_5 = "mc_2_1027"
    system_6 = "mc_3_28"
    output_dir = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSSe/output"
    
    df1 = pd.read_csv(os.path.join(dir_path, system_1, "75_41.csv"))
    df2 = pd.read_csv(os.path.join(dir_path, system_2, "75_41.csv"))
    df3 = pd.read_csv(os.path.join(dir_path, system_3, "75_41.csv"))
    df4 = pd.read_csv(os.path.join(dir_path, system_4, "75_41.csv"))
    df5 = pd.read_csv(os.path.join(dir_path, system_5, "75_41.csv"))
    df6 = pd.read_csv(os.path.join(dir_path, system_6, "75_41.csv"))
    df_combined = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    print(df_combined)
    df_combined.to_csv(os.path.join(output_dir, "total_75_41.csv"))
    