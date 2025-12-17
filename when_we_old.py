# 우리가 언제 피부상태가 달라지는 구간을 알 수 있음
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows: 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리
# ---------------------------------------------------------
# 어제 그 CSV 파일을 그대로 사용합니다.
df = pd.read_csv(r'C:\Users\User\Desktop\swin_transformer\pca_real_result\raw_grades.csv')

# 분석할 7가지 핵심 피부 지표
features = ['chin_sagging', 'forehead_pigmentation', 'forehead_wrinkle', 
            'glabellus_wrinkle', 'l_cheek_pore', 'lip_dryness', 'r_cheek_pore']

# ---------------------------------------------------------
# 2. PCA를 이용한 '피부 노화 종합 점수(Aging Score)' 산출
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA(n_components=1)
# PC1 점수 추출 (이 점수가 높을수록 피부가 '늙었다'는 뜻)
df['Aging_Score'] = pca.fit_transform(X_scaled)

# ---------------------------------------------------------
# 3. 노화 '변곡점(Turning Point)' 찾기
# ---------------------------------------------------------
# 나이별 평균 점수 계산
age_trend = df.groupby('Age')['Aging_Score'].mean().sort_index()

# 데이터를 부드럽게 만듦 (노이즈 제거) -> 미분(변화율 계산)
smooth_trend = age_trend.rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill')
aging_velocity = smooth_trend.diff()  # 1년마다 노화가 얼마나 진행되는지(속도)

# 출력 디렉토리 설정
output_dir = r'C:\Users\User\Desktop\swin_transformer\pca_machine_learning'

# 변화 속도가 평균보다 1.5배 이상 빠른 '급변 구간' 찾기 (scipy 라이브러리 활용)
peaks, _ = find_peaks(aging_velocity, height=aging_velocity.mean() * 1.5, distance=3)
critical_ages = age_trend.index[peaks]

print("=== 🚨 데이터가 발견한 '피부 노화 관리 골든타임' ===")
results = []
for age in critical_ages:
    if age > 20: # 20세 이상 성인 데이터만 유의미하다고 판단
        message = f"👉 {age}세: 노화가 급가속되는 시기 (집중 관리 필요)"
        print(message)
        results.append(message)

# 골든타임 결과를 텍스트 파일로 저장
with open(f'{output_dir}\\golden_time_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("피부 노화 관리 골든타임 분석 결과\n")
    f.write("="*60 + "\n\n")
    for result in results:
        f.write(result + "\n")
    f.write("\n" + "="*60 + "\n")
print(f"\n✓ 골든타임 분석 결과가 저장되었습니다: {output_dir}\\golden_time_analysis.txt")

# ---------------------------------------------------------
# 4. 시각화 (보고서용)
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(smooth_trend.index, smooth_trend, label='피부 노화 곡선', color='black')
plt.bar(aging_velocity.index, aging_velocity, color='skyblue', alpha=0.7, label='노화 속도')
plt.scatter(critical_ages, smooth_trend.loc[critical_ages], color='red', s=100, zorder=5, label='급변 구간')

plt.title('나이별 피부 노화 속도와 골든타임 분석')
plt.xlabel('나이')
plt.ylabel('노화 점수 / 속도')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}\\skin_turning_points.png', dpi=300, bbox_inches='tight')
print(f"\n✓ 그래프가 저장되었습니다: {output_dir}\\skin_turning_points.png")