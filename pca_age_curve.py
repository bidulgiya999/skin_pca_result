import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. 데이터 로드 및 전처리
df = pd.read_csv(r'C:\Users\User\Desktop\swin_transformer\pca_real_result\raw_grades.csv')
features = ['chin_sagging', 'forehead_pigmentation', 'forehead_wrinkle', 
            'glabellus_wrinkle', 'l_cheek_pore', 'lip_dryness', 'r_cheek_pore']

# 2. PCA 모델 학습 (기준 만들기)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_scaled)

# 데이터프레임에 점수(PC1: 노화도) 추가
df['Skin_Aging_Score'] = pca_scores[:, 0]  # PC1을 노화 지수로 사용

# 3. 연령대별 랭킹 기준표 생성 (Percentile 계산)
# 10대, 20대, 30대... 그룹핑
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 19, 29, 39, 49, 59, 100], 
                         labels=['10s', '20s', '30s', '40s', '50s', '60s+'])

# 각 연령대별로 분위수(Quantile) 계산 함수
def get_rank_table(df):
    rank_table = df.groupby('Age_Group')['Skin_Aging_Score'].describe(percentiles=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
    # 점수가 낮을수록(Low) 동안이므로, 낮은 점수가 상위 퍼센트입니다.
    return rank_table[['min', '1%', '10%', '30%', '50%', '70%', '90%', 'max']]

rank_reference = get_rank_table(df)
print("=== 연령대별 피부 나이 기준표 (점수가 낮을수록 좋음) ===")
print(rank_reference)

# 기준표를 CSV로 저장
output_dir = r'C:\Users\User\Desktop\swin_transformer\pca_machine_learning'
rank_reference.to_csv(f'{output_dir}\\age_rank_reference.csv', encoding='utf-8-sig')
print(f"\n✓ 연령대별 기준표가 저장되었습니다: {output_dir}\\age_rank_reference.csv")

# ---------------------------------------------------------
# [시뮬레이션] 새로운 고객이 들어왔을 때
# ---------------------------------------------------------
def predict_skin_rank(user_age, user_features):
    # 1. 사용자 특징을 기존 스케일로 변환 및 PCA 점수 계산
    user_scaled = scaler.transform([user_features])
    user_score = pca.transform(user_scaled)[0, 0] # PC1 점수
    
    # 2. 연령대 확인
    if user_age < 20: age_group = '10s'
    elif user_age < 30: age_group = '20s'
    elif user_age < 40: age_group = '30s'
    elif user_age < 50: age_group = '40s'
    elif user_age < 60: age_group = '50s'
    else: age_group = '60s+'
    
    # 3. 해당 연령대에서 내 위치(백분위) 계산
    group_data = df[df['Age_Group'] == age_group]['Skin_Aging_Score']
    percentile = (group_data < user_score).mean() * 100 # 내 점수보다 낮은 사람이 몇 %인가?
    
    # 내 점수가 낮을수록(주름이 없을수록) 상위권이므로, 100에서 뺍니다.
    top_percent = percentile 
    
    return top_percent, user_score

# 테스트: 35세, 피부 상태가 꽤 안 좋은(주름 많은) 가상 유저
# 특징: [턱처짐(1), 이마색소(2), 이마주름(4-심함), 미간(3), 왼볼모공(2), 입술(2), 오른볼모공(2)]
my_rank, my_score = predict_skin_rank(35, [1, 2, 4, 3, 2, 2, 2])

print(f"\n[진단 결과]")
print(f"고객님의 피부 노화 점수는 {my_score:.2f}점 입니다.")
print(f"30대 평균 대비, 고객님은 '하위 {100 - my_rank:.1f}%' (상위 {my_rank:.1f}%)에 해당합니다.")

# 진단 결과를 텍스트 파일로 저장
with open(f'{output_dir}\\diagnosis_result.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("피부 노화 진단 결과\n")
    f.write("="*60 + "\n\n")
    f.write(f"고객님의 피부 노화 점수: {my_score:.2f}점\n")
    f.write(f"30대 평균 대비 위치: 하위 {100 - my_rank:.1f}% (상위 {my_rank:.1f}%)\n")
    f.write("\n" + "="*60 + "\n")
print(f"\n✓ 진단 결과가 저장되었습니다: {output_dir}\\diagnosis_result.txt")