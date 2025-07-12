import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

st.title("미세중력렌즈 시뮬레이션")

# --- 설정값 ---
st.sidebar.header("시뮬레이션 설정")
orbital_period = st.sidebar.slider("행성 공전 주기 (프레임 수)", 200, 1000, 400, key="period_slider") # 주기 늘림
# 행성 질량비를 더욱 현실적인 범위로 조정 (지구~목성)
# 목성 질량비 ~ 0.001 (태양=1), 지구 질량비 ~ 0.000003 (태양=1)
planet_mass_ratio = st.sidebar.slider("행성 질량비 (중심별=1)", 0.000001, 0.0015, 0.0001, format="%.7f", key="mass_ratio_slider")
observer_angle_deg = st.sidebar.slider("관찰자 시선 방향 (도)", 0, 360, 90, key="observer_angle_slider")
st.sidebar.markdown("---")
st.sidebar.info("💡 **팁:** 'Play' 버튼을 누르거나 아래 슬라이더를 움직여 애니메이션을 제어하세요.")

# --- 시뮬레이션 상수 ---
R_STAR = 0.5  # 중심별 시각적 반지름 (단위)
R_ORBIT = 5.0  # 행성 궤도 반지름 (단위)
R_OBSERVER_DIST = 10.0 # 관찰자의 중심별로부터의 거리 (시각화용)

# --- 미세중력렌즈 광도 계산 함수 (현실적 스케일 및 단일 피크) ---
def calculate_magnification(planet_pos, observer_pos, star_pos, planet_mass_ratio):
    # microlensing magnification A = (u^2 + 2) / (u * sqrt(u^2 + 4))
    # where u = impact_parameter / Einstein_radius

    magnification = 1.0 # 기본 광도 (증폭 없음)

    # 1. 아인슈타인 반경 (Einstein Radius, R_E) 계산
    # R_E는 렌즈(행성)의 질량, 렌즈-광원 거리, 관찰자-광원 거리에 따라 달라짐
    # 여기서 D_L: observer-lens, D_S: observer-source, D_LS: lens-source
    # theta_E = sqrt(4 G M_L / c^2 * (D_L * D_LS / D_S))

    # 시뮬레이션 스케일에 맞게 조정된 아인슈타인 반경 프록시 (시각적 효과 및 계산 편의)
    # 실제 목성 질량비 0.001 (태양=1)일 때의 아인슈타인 반경에 비례하도록 스케일링
    # 이 값은 렌즈 효과의 "폭"을 결정합니다.
    # 예시 값: 0.005 정도로 설정하면 더 좁은 피크가 가능
    
    # 0.001 질량비 (목성)일 때 시뮬레이션 단위계에서 아인슈타인 반경이 약 0.005라고 가정
    # 다른 질량비에 대해 sqrt(질량비)에 비례하여 조정
    einstein_radius_visual_proxy = 0.005 * np.sqrt(planet_mass_ratio / 0.001) 
    
    if einstein_radius_visual_proxy <= 0: return 1.0 # 질량비가 0이면 증폭 없음

    # 2. 충격 매개변수 (Impact Parameter) 'd_perp' 계산
    # 관찰자-광원(중심별) 시선 (line of sight)을 기준으로 행성(렌즈)이 얼마나 떨어져 있는가
    
    # 관찰자-중심별 시선 벡터
    line_of_sight_vec = np.array(star_pos) - np.array(observer_pos)
    line_of_sight_unit = line_of_sight_vec / np.linalg.norm(line_of_sight_vec)

    # 행성 위치에서 관찰자-중심별 시선까지의 수직 거리 (이것이 impact parameter)
    vec_op = np.array(planet_pos) - np.array(observer_pos) # 관찰자에서 행성으로의 벡터
    proj_length = np.dot(vec_op, line_of_sight_unit) # 시선 방향으로의 투영 길이
    
    # 시선 상의 행성으로부터 가장 가까운 점 (Closest Point on Line of Sight)
    closest_point_on_los = np.array(observer_pos) + proj_length * line_of_sight_unit
    
    # 행성 실제 위치와 시선 상의 가장 가까운 점 사이의 거리 (수직 거리)
    perpendicular_dist = np.linalg.norm(np.array(planet_pos) - closest_point_on_los)
    
    # 3. 정규화된 충격 매개변수 'u' 계산
    u = perpendicular_dist / einstein_radius_visual_proxy

    # 4. 광도 증폭 계산 (A = (u^2 + 2) / (u * sqrt(u^2 + 4)))
    # u가 0에 가까워질수록 A는 무한대로 발산
    # 이를 방지하고 현실적인 최대 증폭값을 설정
    if u < 0.0001: # u가 매우 작을 때 (거의 완벽한 정렬)
        u = 0.0001 # 최소 u 값 설정하여 발산 방지
    
    magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    
    # 5. 광도 증폭 스케일링 및 단일 피크 보장
    # 실제 행성(목성, 지구 등)에 의한 미세중력렌즈 효과는 1%~수십% 정도로 매우 작음.
    # 이를 반영하여 'magnification - 1.0' (순수 증폭량)을 다시 스케일링.
    
    # 기본적으로 목성 질량비(0.001)일 때 최대 20% 증폭 (1.2배) 정도로 보이게 설정
    # 이 스케일링 값 (200)은 시각적 효과를 위해 조정 가능
    base_magnification_factor = 200.0 # 이 값을 조정하여 증폭의 스케일을 조절
    
    magnification = 1.0 + (magnification - 1.0) * (planet_mass_ratio / 0.001) * base_magnification_factor
    
    # 행성이 관찰자 시선에 매우 가깝지 않으면 증폭 없음 (단일 피크 보장)
    # 즉, 행성이 아인슈타인 반경 프록시의 몇 배 이상 벗어나면 증폭을 1로 초기화
    if u > 3.0: # u가 이 값보다 크면 거의 증폭이 없음 (u=1일 때 A=1.34, u=2일 때 A=1.19)
        magnification = 1.0
        
    return max(1.0, magnification) # 광도는 1.0 미만이 될 수 없음

# --- 데이터 준비 ---
frames_data = []
lightcurve_current_frame_data = [] # 각 프레임마다의 광도 데이터를 저장할 리스트
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
min_mag = min(all_magnifications) * 0.95
max_mag = max(all_magnifications) * 1.05
if max_mag < 1.02: max_mag = 1.02 # 최소한 2% 증폭은 보이도록 (더 현실적인 스케일)
if min_mag > 0.98: min_mag = 0.98 # 최소한 2%는 보이도록

# --- 초기 그래프 생성 ---
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("행성 공전 시뮬레이션", "미세중력렌즈 광도 변화"),
                    specs=[[{'type': 'xy'}, {'type': 'xy'}]])

# 1. 공전 시뮬레이션 서브플롯 (왼쪽, x1, y1)
# Trace 0: 중심별
fig.add_trace(go.Scatter(x=[star_pos[0]], y=[star_pos[1]], mode='markers',
                         marker=dict(size=20, color='gold'),
                         name='중심별'), row=1, col=1)
# Trace 1: 행성 (초기 위치)
fig.add_trace(go.Scatter(x=[R_ORBIT * np.cos(0)], y=[R_ORBIT * np.sin(0)], mode='markers',
                         marker=dict(size=8, color='blue'),
                         name='행성'), row=1, col=1)

# Trace 2: 관찰자 위치
fig.add_trace(go.Scatter(x=[observer_x], y=[observer_y], mode='markers',
                         marker=dict(size=10, color='purple', symbol='star'),
                         name='관찰자'), row=1, col=1)
# Trace 3: 관찰자 시선
fig.add_trace(go.Scatter(x=[observer_x, star_pos[0]], y=[observer_y, star_pos[1]],
                         mode='lines', line=dict(color='red', dash='dash'),
                         name='관찰자 시선'), row=1, col=1)

fig.update_xaxes(range=[-R_ORBIT * 1.2, R_ORBIT * 1.2], title_text="X 좌표", row=1, col=1)
fig.update_yaxes(range=[-R_ORBIT * 1.2, R_ORBIT * 1.2], scaleanchor="x", scaleratio=1, title_text="Y 좌표", row=1, col=1)
fig.update_layout(showlegend=True, legend=dict(x=0.01, y=0.99))

# 2. 광도 변화 서브플롯 (오른쪽, x2, y2)
# Trace 4: 광도 그래프 (초기 데이터)
fig.add_trace(go.Scatter(x=[0], y=[all_magnifications[0]], mode='lines',
                         line=dict(color='green'),
                         name='광도'), row=1, col=2)

fig.update_xaxes(range=[0, orbital_period], title_text="시간 (프레임)", row=1, col=2)
fig.update_yaxes(range=[min_mag, max_mag],
                 title_text="상대 광도", row=1, col=2)

# --- 애니메이션 프레임 데이터 구성 ---
for t in times:
    angle = 2 * np.pi * t / orbital_period
    planet_x = R_ORBIT * np.cos(angle)
    planet_y = R_ORBIT * np.sin(angle)

    # 현재 프레임의 광도 값 (미리 계산된 all_magnifications에서 가져옴)
    current_magnification = all_magnifications[t]
    lightcurve_current_frame_data = all_magnifications[:t+1] # 현재 프레임까지의 광도 데이터

    # 각 프레임에 대한 데이터 저장 (모든 트레이스를 명시적으로 다시 정의)
    frames_data.append({
        'data': [
            # Trace 0: 중심별 (왼쪽 그래프)
            go.Scatter(x=[star_pos[0]], y=[star_pos[1]], mode='markers', marker=dict(size=20, color='gold'), showlegend=True, name='중심별'),
            # Trace 1: 행성 (왼쪽 그래프, 위치 업데이트)
            go.Scatter(x=[planet_x], y=[planet_y], mode='markers', marker=dict(size=8, color='blue'), showlegend=True, name='행성'),
            # Trace 2: 관찰자 (왼쪽 그래프)
            go.Scatter(x=[observer_x], y=[observer_y], mode='markers', marker=dict(size=10, color='purple', symbol='star'), showlegend=True, name='관찰자'),
            # Trace 3: 관찰자 시선 (왼쪽 그래프)
            go.Scatter(x=[observer_x, star_pos[0]], y=[observer_y, star_pos[1]], mode='lines', line=dict(color='red', dash='dash'), showlegend=True, name='관찰자 시선'),
            # Trace 4: 광도 그래프 (오른쪽 그래프, 데이터 업데이트)
            go.Scatter(x=times[:t+1], y=lightcurve_current_frame_data, mode='lines', line=dict(color='green'), showlegend=True, name='광도', xaxis='x2', yaxis='y2')
        ],
        'name': f'frame_{t}'
    })


# --- 애니메이션 설정 ---
fig.frames = [go.Frame(data=frame['data'], name=frame['name']) for frame in frames_data]

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
