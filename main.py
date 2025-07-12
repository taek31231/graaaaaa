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
            0.1 * dist_obs_to_star < proj_on_los_length < 1.1 * dist_obs_to_star):
        return 1.0 # 정렬되지 않거나, 시야 범위 밖이면 증폭 없음

    # 5. 광도 증폭 계산 (A = (u^2 + 2) / (u * sqrt(u^2 + 4)))
    # u가 0에 가까워질 때 무한대 발산 방지 및 피크 높이 조절
    if u < 0.005: # 최소 u 값 설정 (이전보다 약간 높임)
        u = 0.005
    
    magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

    # 6. 현실적인 스케일로 증폭량 조정
    # 목성 질량비(0.001)에서 최대 10~20% 정도의 증폭이 나타나도록 조정
    # scaling_factor를 다시 조절하여 실제 증폭량 제어
    # (magnification - 1.0)은 순수한 증폭량 (1.0 기준)
    
    # 0.001 질량비 (목성)일 때 최대 1.15배 (15% 증폭) 정도를 목표
    target_gain_for_jupiter = 0.15 # 목표 증폭률 (15%)
    
    # u=0.005일 때의 기본 magnification
    base_u_mag = (0.005**2 + 2) / (0.005 * np.sqrt(0.005**2 + 4))
    
    # scaling_factor를 역산하여 목표 증폭률에 맞춤
    scaling_factor = target_gain_for_jupiter / (base_u_mag - 1.0) / (0.001 / 0.001)
    
    magnification = 1.0 + (magnification - 1.0) * scaling_factor * (planet_mass_ratio / 0.001)
    
    # u가 너무 커지면 증폭 없음 (선명한 단일 피크 유지)
    if u > 3.0: # 이 값 이상이면 증폭 효과 미미하므로 1.0으로 강제
        magnification = 1.0

    return max(1.0, magnification) # 광도는 1.0 미만이 될 수 없음


# --- 데이터 준비 ---
frames_data = []
times = np.arange(orbital_period)

# 관찰자 위치 계산 (고정)
observer_angle_rad = np.radians(observer_angle_deg)
observer_x = R_OBSERVER_DIST * np.cos(observer_angle_rad)
observer_y = R_OBSERVER_DIST * np.sin(observer_angle_rad)
observer_pos = (observer_x, observer_y)
star_pos = (0, 0) # 중심별은 항상 (0,0)에 고정

# 모든 프레임의 광도 값을 미리 계산하여 Y축 범위 설정 및 애니메이션 데이터 준비
all_magnifications = []
for t_val in times:
    angle_val = 2 * np.pi * t_val / orbital_period
    planet_x_val = R_ORBIT * np.cos(angle_val)
    planet_y_val = R_ORBIT * np.sin(angle_val)
    all_magnifications.append(calculate_magnification(
        (planet_x_val, planet_y_val), observer_pos, star_pos, planet_mass_ratio
    ))

# Y축 범위 미리 설정 (실제 계산된 광도 값을 기반으로)
# 증폭이 없을 경우를 대비하여 최소값과 최대값 설정
min_mag = min(all_magnifications) * 0.99
max_mag = max(all_magnifications) * 1.01
# 피크가 너무 낮아서 안 보일 경우를 대비하여 최소 범위 보장
if max_mag < 1.01: max_mag = 1.01 # 최소 1% 증폭은 보이도록
if min_mag > 0.99: min_mag = 0.99 # 최소 1% 감소도 보이도록

# --- 초기 그래프 생성 ---
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("행성 공전 시뮬레이션", "미세중력렌즈 광도 변화"),
                    specs=[[{'type': 'xy'}, {'type': 'xy'}]])

# 초기 트레이스 생성 순서 (인덱스 0부터 시작)
# 이 순서가 frames_data 내에서도 일치해야 합니다.

# Trace 0: 중심별 (왼쪽 그래프)
fig.add_trace(go.Scatter(x=[star_pos[0]], y=[star_pos[1]], mode='markers',
                         marker=dict(size=20, color='gold'),
                         name='중심별', showlegend=True), row=1, col=1)

# Trace 1: 행성 (왼쪽 그래프, 초기 위치)
fig.add_trace(go.Scatter(x=[R_ORBIT * np.cos(0)], y=[R_ORBIT * np.sin(0)], mode='markers',
                         marker=dict(size=8, color='blue'),
                         name='행성', showlegend=True), row=1, col=1)

# Trace 2: 관찰자 위치 (왼쪽 그래프)
fig.add_trace(go.Scatter(x=[observer_x], y=[observer_y], mode='markers',
                         marker=dict(size=10, color='purple', symbol='star'),
                         name='관찰자', showlegend=True), row=1, col=1)

# Trace 3: 관찰자 시선 (왼쪽 그래프)
fig.add_trace(go.Scatter(x=[observer_x, star_pos[0]], y=[observer_y, star_pos[1]],
                         mode='lines', line=dict(color='red', dash='dash'),
                         name='관찰자 시선', showlegend=True), row=1, col=1)

# Trace 4: 광도 그래프 (오른쪽 그래프, 초기 데이터)
fig.add_trace(go.Scatter(x=[0], y=[all_magnifications[0]], mode='lines',
                         line=dict(color='green'),
                         name='광도', showlegend=True,
                         xaxis='x2', yaxis='y2'), row=1, col=2) # 광도 그래프는 x2, y2 축 사용을 명시적으로!

# 왼쪽 그래프 축 설정
fig.update_xaxes(range=[-R_ORBIT * 1.2, R_ORBIT * 1.2], title_text="X 좌표", row=1, col=1)
fig.update_yaxes(range=[-R_ORBIT * 1.2, R_ORBIT * 1.2], scaleanchor="x", scaleratio=1, title_text="Y 좌표", row=1, col=1)
fig.update_layout(showlegend=True, legend=dict(x=0.01, y=0.99))

# 오른쪽 그래프 축 설정
fig.update_xaxes(range=[0, orbital_period], title_text="시간 (프레임)", row=1, col=2)
fig.update_yaxes(range=[min_mag, max_mag], title_text="상대 광도", row=1, col=2)


# --- 애니메이션 프레임 데이터 구성 ---
for t in times:
    angle = 2 * np.pi * t / orbital_period
    planet_x = R_ORBIT * np.cos(angle)
    planet_y = R_ORBIT * np.sin(angle)
    
    # Plotly Frame 데이터는 해당 프레임에서 표시될 모든 트레이스를 포함해야 합니다.
    # 각 트레이스는 생성 시의 인덱스와 동일한 순서로 와야 합니다.
    frames_data.append(
        go.Frame(
            data=[
                # Trace 0: 중심별 (왼쪽 그래프)
                go.Scatter(x=[star_pos[0]], y=[star_pos[1]], mode='markers', marker=dict(size=20, color='gold'), showlegend=False),
                # Trace 1: 행성 (왼쪽 그래프, 위치 업데이트)
                go.Scatter(x=[planet_x], y=[planet_y], mode='markers', marker=dict(size=8, color='blue'), showlegend=False),
                # Trace 2: 관찰자 (왼쪽 그래프)
                go.Scatter(x=[observer_x], y=[observer_y], mode='markers', marker=dict(size=10, color='purple', symbol='star'), showlegend=False),
                # Trace 3: 관찰자 시선 (왼쪽 그래프)
                go.Scatter(x=[observer_x, star_pos[0]], y=[observer_y, star_pos[1]], mode='lines', line=dict(color='red', dash='dash'), showlegend=False),
                # Trace 4: 광도 그래프 (오른쪽 그래프, 데이터 업데이트)
                go.Scatter(x=times[:t+1], y=all_magnifications[:t+1], mode='lines', line=dict(color='green'), showlegend=False, xaxis='x2', yaxis='y2')
            ],
            name=f'frame_{t}'
        )
    )

# --- 애니메이션 설정 ---
fig.frames = frames_data

# 애니메이션 재생/일시정지 버튼 설정
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.01,
            y=1.05,
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )
    ]
)

# 타임라인 슬라이더 설정
sliders = [
    dict(
        steps=[
            dict(
                method="animate",
                args=[
                    [f"frame_{t}"],
                    {"mode": "immediate", "frame": {"duration": 50, "redraw": True}, "transition": {"duration": 0}}
                ],
                label=str(t)
            ) for t in times
        ],
        transition={"duration": 0},
        x=0.08,
        y=0,
        currentvalue={"font": {"size": 12}, "prefix": "프레임: ", "visible": True},
        len=0.92,
    )
]
fig.update_layout(sliders=sliders)

# --- Streamlit 앱에 Plotly 그래프 표시 ---
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("시뮬레이션 설명:")
st.markdown("""
- **왼쪽 그래프**: 항성(노란색) 주위를 공전하는 행성(파란색)을 보여줍니다. 보라색 별은 관찰자의 위치이며, 붉은 점선은 관찰자로부터 항성으로 향하는 시선 방향을 나타냅니다.
- **오른쪽 그래프**: 시간에 따른 항성의 상대 광도 변화를 나타냅니다. 행성이 관찰자와 항성 사이를 지나가면서 미세중력렌즈 효과에 의해 항성의 광도가 일시적으로 증가하는 것을 시뮬레이션합니다.
- **광도 계산**: `calculate_magnification` 함수를 개선하여 **관찰자와 중심별 사이를 행성이 지나갈 때만 광도가 증가**하도록 설정했으며, **단일 피크**가 나타나도록 조건을 조정했습니다. 또한, **행성의 질량비에 따른 현실적인 광도 변화 스케일**을 반영했습니다 (대략 수십 퍼센트 이내).
""")
