def calculate_saps2_score(age, heart_rate, systolic_bp, temperature, paO2_FiO2_ratio, urine_output,
                          blood_urea_nitrogen, white_blood_cell_count, potassium_level, sodium_level,
                          bicarbonate_level, bilirubin_level, glasgow_coma_scale, chronic_diseases,
                          admission_type, mechanical_ventilation, ph_level):
    score = 0

    # 年龄分数
    if age < 40:
        score += 0
    elif 40 <= age <= 59:
        score += 7
    elif 60 <= age <= 69:
        score += 12
    elif 70 <= age <= 74:
        score += 15
    elif age >= 75:
        score += 16

    # 心率
    if heart_rate < 40:
        score += 11
    elif 40 <= heart_rate <= 69:
        score += 2
    elif 70 <= heart_rate <= 119:
        score += 0
    elif 120 <= heart_rate <= 159:
        score += 4
    elif heart_rate >= 160:
        score += 7

    # 收缩压
    if systolic_bp < 70:
        score += 13
    elif 70 <= systolic_bp <= 99:
        score += 5
    elif 100 <= systolic_bp <= 199:
        score += 0
    elif systolic_bp >= 200:
        score += 2

    # 体温
    if temperature < 39:
        score += 0
    else:
        score += 3

    # paO2/FiO2比率
    if paO2_FiO2_ratio < 100:
        score += 11
    elif 100 <= paO2_FiO2_ratio <= 199:
        score += 9

    # 尿量
    if urine_output < 500:
        score += 11
    elif 500 <= urine_output <= 999:
        score += 4

    # 血尿素氮
    if blood_urea_nitrogen >= 28:
        score += 10

    # 白细胞计数
    if white_blood_cell_count < 1:
        score += 12
    elif 20 <= white_blood_cell_count <= 39.9:
        score += 3
    elif white_blood_cell_count >= 40:
        score += 3

    # 钾
    if potassium_level < 3:
        score += 3
    elif potassium_level > 5:
        score += 3

    # 钠
    if sodium_level < 125:
        score += 5

    # 碳酸氢盐
    if bicarbonate_level < 15:
        score += 6

    # 总胆红素
    if bilirubin_level >= 6:
        score += 9

    # 格拉斯哥昏迷指数
    if glasgow_coma_scale < 6:
        score += 26
    elif 6 <= glasgow_coma_scale <= 8:
        score += 13
    elif 9 <= glasgow_coma_scale <= 10:
        score += 7
    elif 11 <= glasgow_coma_scale <= 13:
        score += 5
    elif 14 <= glasgow_coma_scale <= 15:
        score += 0

    # 慢性疾病
    score += chronic_diseases

    # 入院类型
    if admission_type == 'unplanned':
        score += 8

    # 是否使用机械通气
    if mechanical_ventilation:
        score += 9

    # 血pH值
    if ph_level < 7.25:
        score += 6

    return score


# saps2_score = calculate_saps2_score(age=65, heart_rate=110, systolic_bp=100, temperature=37.0, paO2_FiO2_ratio=300,
#                                     urine_output=800, blood_urea_nitrogen=30, white_blood_cell_count=10,
#                                     potassium_level=4, sodium_level=135, bicarbonate_level=24, bilirubin_level=1.2,
#                                     Glasgow_coma_scale=15, chronic_diseases=0, admission_type='unplanned')
# print(f"SAPS II Score: {saps2_score}")
