import pandas as pd


def filter_csv(input_file, output_file, known_values):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 根据已知列表中的值筛选行
    filtered_df = df[df["SUBJECT_ID"].isin(known_values)]

    # 保存筛选结果为新的CSV文件
    filtered_df.to_csv(output_file, index=False)


# 输入文件路径
input_file = "/home/rongqin/codes/icu-benchmarks/Datasets/mimic4/icustay_desc.csv"

# 输出文件路径
output_file = "/home/rongqin/codes/icu-benchmarks/Datasets/mimic4/icustay_desc_hf.csv"

# 已知列表中的值
df = pd.read_csv('hf_patients_subjectID_4.csv')
known_values = df['filepath'].tolist()  # 用您自己的值替换这个示例列表

# 调用函数进行筛选和保存
filter_csv(input_file, output_file, known_values)
