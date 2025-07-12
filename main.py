import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

st.title("미세중력렌즈 시뮬레이션")

# --- 설정값 ---
st.sidebar.header("시뮬레이션 설정")
orbital_period = st.sidebar.slider("행성 공전 주기 (프레임 수)", 200, 1000, 400, key="period_slider")
# 행성 질량비 범위 조정: 더 현실적인 값으로 (목성급: 0.001, 지구급: 0.000003)
planet_mass_ratio = st.sidebar.slider("행성 질량비 (중심별=1)", 0.000001, 0.002, 0.0001, format="%.7f", key="mass_ratio_slider") # 목성 0.001
observer_angle_deg = st.sidebar.slider("관찰자 시선 방향 (도)", 0, 360, 90, key="observer_angle_slider")
st.sidebar.markdown("---")
st.sidebar.info("💡 **팁:** 'Play' 버튼을 누르거나 아래 슬라이더를 움직여 애니메이션을 제어하세요.")

# --- 시뮬레이션 상수 ---
R_STAR = 0.5  # 중심별 시각적 반지름 (단위)
R_ORBIT = 5.0  # 행성 궤도 반지름 (단위)
R_OBSERVER_DIST = 10.0 # 관찰자의 중심별로부터의 거리 (시각화용)

# --- 미세중력렌즈 광도 계산 함수 (재수정: 단일 피크 및 현실적 스케일) ---
def calculate_magnification(planet_pos, observer_pos, star_pos, planet_mass_ratio):
    magnification = 1.0 # 기본 광도

    # 1. 관찰자-중심별 시선 벡터
    vec_obs_to_star = np.array(star_pos) - np.array(observer_pos)
    unit_obs_to_star = vec_obs_to_star / np.linalg.norm(vec_obs_to_star)

    # 2. 행성 위치에서 관찰자-중심별 시선까지의 수직 거리 (impact parameter)
    vec_obs_to_planet = np.array(planet_pos) - np.array(observer_pos)
    
    # Check if planet is roughly "between" observer and star on the line of sight
    # The dot product of vec_obs_to_planet and unit_obs_to_star should be positive
    # and less than distance_obs_to_star (or slightly beyond)
    
    # Calculate impact parameter 'b' (perpendicular distance)
    # Using cross product for 2D is a bit hacky, but equivalent to perp_dist
    # Let's stick to proj_length method as it's cleaner
    proj_on_los_length = np.dot(vec_obs_to_planet, unit_obs_to_star)
    
    # Closest point on the LOS from planet
    closest_point_on_los = np.array(observer_pos) + proj_on_los_length * unit_obs_to_star
    perpendicular_dist = np.linalg.norm(np.array(planet_pos) - closest_point_on_los)
    
    # 3. 아인슈타인 반경 프록시 (시각적 효과 조절)
    # 이 값을 조절하여 렌즈 효과의 폭과 강도를 미세 조정할 수 있습니다.
    # 목성 질량비(0.001)일 때 0.005 정도가 적절한 시작점.
    einstein_radius_visual_proxy = 0.005 * np.sqrt(planet_mass_ratio / 0.001)
    
    if einstein_radius_visual_proxy <= 0: return 1.0

    u = perpendicular_dist / einstein_radius_visual_proxy # 정규화된 충격 매개변수

    # 4. 단일 피크를 위한 정렬 조건 강화
    # 행성이 관찰자와 중심별을 잇는 "직선 상"에 매우 가깝게 위치해야 함
    # 즉, observer_pos, planet_pos, star_pos가 거의 일직선상에 있어야 함
    
    # observer-planet 벡터와 observer-star 벡터 사이의 각도 계산
    # 이 각도가 0에 가까워야 함 (직선 정렬)
    
    # Clip dot product to avoid floating point errors leading to acos(>1) or acos(<-1)
    dot_product = np.dot(unit_obs_to_star, vec_obs_to_planet / np.linalg.norm(vec_obs_to_planet))
    alignment_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # 허용할 최대 각도 편차 (라디안) - 이 값을 줄일수록 피크는 더 날카로워지고, 두 번의 피크를 방지
    # 0.01 라디안 = 약 0.57도
    angular_tolerance_rad = np.radians(0.5) 

    # 또한, 행성이 중심별 '앞' (관찰자 기준)을 지나가야 함
    # 즉, 행성의 투영 위치가 중심별 투영 위치보다 관찰자에게 가까워야 함
    # (star_pos - observer_pos) 벡터와 (planet_pos - observer_pos) 벡터의 길이가
    # 적절한 관계를 가져야 함.
    
    # Simplified check for "betweenness" along the line of sight (not just proximity to line)
    # If the planet is roughly *between* observer and star in the projection onto the LOS
    # The projected length `proj_on_los_length` should be positive (planet is not behind observer)
    # And less than (approx) the distance from observer to star
    
    dist_obs_to_star = np.linalg.norm(vec_obs_to_star)
    
    # 핵심 필터링: 각도 정렬 AND 행성이 관찰자와 별 사이에 있는 시야각 내
    # proj_on_los_length 가 0 (관찰자 위치)보다 크고, dist_obs_to_star (별 위치)보다 작거나 같아야 함
    # 약간의 여유를 두어 범위를 넓힘
    if not (alignment_angle < angular_tolerance_rad and \
            0.1 * dist_obs_to_star < proj_on
