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

# --- 미세중력렌즈 광도 계산 함수 (재수정) ---
def calculate_magnification(planet_pos, observer_pos, star_pos, planet_mass_ratio):
    magnification = 1.0 # 기본 광도

    # 관찰자-중심별 시선 벡터
    line_of_sight_vec = np.array(star_pos) - np.array(observer_pos)
    line_of_sight_unit = line_of_sight_vec / np.linalg.norm(line_of_sight_vec)

    # 행성 위치에서 관찰자-중심별 시선까지의 수직 거리 (impact parameter)
    vec_op = np.array(planet_pos) - np.array(observer_pos)
    proj_length = np.dot(vec_op, line_of_sight_unit)
    closest_point_on_los = np.array(observer_pos) + proj_length * line_of_sight_unit
    perpendicular_dist = np.linalg.norm(np.array(planet_pos) - closest_point_on_los)

    # 아인슈타인 반경 프록시 (시각적 효과 조절, 질량비에 비례)
    # 이 값을 조절하여 렌즈 효과의 폭과 강도를 미세 조정할 수 있습니다.
    # 목성 질량비(0.001)일 때 0.005 정도가 적절한 시작점.
    einstein_radius_visual_proxy = 0.005 * np.sqrt(planet_mass_ratio / 0.001) # 목성 질량비를 기준으로 스케일링
    
    if einstein_radius_visual_proxy <= 0: return 1.0

    u = perpendicular_dist / einstein_radius_visual_proxy # 정규화된 충격 매개변수

    # 핵심: 정렬 조건 (단일 피크 보장)
    # 행성이 관찰자와 중심별 사이를 지날 때만 증폭이 일어나도록 합니다.
    # 즉, 행성의 위치가 시선 벡터 방향으로 중심별보다 관찰자에 가까워야 합니다.
    # (proj_length가 양수여야 하며, 행성이 중심별보다 관찰자쪽에 가까이 있을 때만 유효)
    
    # 이 조건은 행성이 시선 경로를 지나갈 때만 활성화됩니다.
    # proj_length는 관찰자로부터 시선 방향으로의 행성 투영 거리.
    # 0 < proj_length < np.linalg.norm(line_of_sight_vec) (대략적으로)
    # 즉, 행성이 관찰자-중심별 사이의 구간에 있을 때.
    
    # 더 간단하게, 행성의 위치가 중심별과 관찰자 사이의 특정 '정렬 영역' 안에 있을 때만 피크를 만듭니다.
    # 행성(렌즈)이 시선 상에 광원(중심별)과 관찰자 사이에 놓여야 합니다.
    
    # Check if the planet is "in front" of the star from the observer's perspective
    # This ensures only one peak per orbit when passing the line of sight.
    
    # Calculate angular position of planet relative to star, and compare to observer's line of sight
    # Use atan2 for full 360 degree angle
    planet_angle_from_star = np.arctan2(planet_pos[1], planet_pos[0])
    
    # Normalize angles to be within [0, 2*pi)
    planet_angle_from_star = (planet_angle_from_star + 2 * np.pi) % (2 * np.pi)
    observer_angle_rad_norm = (observer_angle_rad + 2 * np.pi) % (2 * np.pi)

    # Define a small angular window around the observer's line of sight
    # This window defines *when* the microlensing event can occur.
    # A smaller window leads to a sharper, single peak.
    angular_window_rad = np.radians(5) # 5도 정도의 좁은 각도 범위
    
    # Check if the planet's angle is within this window relative to observer's angle OR observer's angle + pi (for opposite side)
    # To get a single peak, we need to ensure the planet is on the "correct" side of the star relative to the observer.
    
    # Simplified check for "alignment":
    # The event happens when the planet crosses the line segment between observer and star.
    # This means the perpendicular_dist (u) is small AND the planet is 'between' observer and star
    # in terms of projected position along the line of sight.
    
    # if u is small, it implies proximity to the line of sight.
    # now, let's filter for the specific half-orbit where it should occur.
    
    # Check if planet is on the same side of the star as the observer's projection,
    # and if it's "ahead" of the star along the observer's line of sight
    
    # dot_product = np.dot(planet_pos, line_of_sight_unit)
    # If dot_product is positive, planet is in the general direction of star from observer.
    # To ensure it's *between* observer and star, it gets complicated for orbital motion.
    
    # Let's simplify and make sure the "peak-causing" region is just one side.
    # We will rely on 'u' being small, and then add a condition based on the relative angle.
    
    angle_diff_to_observer_line = np.abs(planet_angle_from_star - observer_angle_rad_norm)
    # Normalize angle_diff to be between 0 and pi
    if angle_diff_to_observer_line > np.pi:
        angle_diff_to_observer_line = 2 * np.pi - angle_diff_to_observer_line

    # The actual physical condition for a single peak:
    # The event occurs when the planet passes through the Einstein cone/tube
    # This naturally leads to a single peak if the source is not binary.
    # The double peak usually comes from approximating the source as passing "behind" the lens from observer's perspective
    # for each side of the lens.

    # Let's try to enforce a single specific angular window for the peak
    # For example, if observer is at 90 deg (positive Y axis), the alignment happens when planet is roughly on negative Y axis.
    # The planet passes through the "line of sight" from observer to star.
    
    # To get single peak: the planet's x coordinate should align with star's x coordinate
    # (assuming observer is on y-axis, star at origin)
    # Or more generally: The component of planet_pos along the perpendicular to LOS should be small
    # AND the component along LOS should be between observer and star.
    
    # A simpler way to get single peak: only allow amplification if the planet's position is "behind" the observer's
    # line relative to the star.
    # If observer is at (obs_x, obs_y) and star is at (0,0)
    # The planet (px, py) is "between" if (px - obs_x) has opposite sign to (0 - obs_x) AND (py - obs_y) opposite sign to (0 - obs_y)
    # This is still not robust for orbital motion.

    # Let's use the current 'u' and add a filter for a half-orbit.
    # Example: If observer is at 90 deg (positive Y), alignment happens when planet's y < 0 (negative y axis)
    # This ensures it passes 'through' the star's line of sight on one side.
    
    # This condition is crucial for single peak.
    # If observer_angle_deg is 90 (looking towards 0,0 from +Y), planet_pos[1] should be negative (passing 'behind' star)
    # Generalizing: angle between line_of_sight_unit and vec_op should be close to 180 degrees.
    
    # Dot product of (planet_pos - star_pos) and line_of_sight_unit
    # This checks if the planet is "behind" the star from the observer's perspective.
    # If this dot product is negative (planet is behind the star), the peak can occur.
    
    # Vector from star to planet
    vec_sp = np.array(planet_pos) - np.array(star_pos)
    
    # Dot product between star-planet vector and observer-star vector
    # If they are roughly opposite, planet is "behind" the star from observer's view point.
    # This is a critical filter for the "single peak"
    alignment
