"""
포식자-피식자 개체수 시뮬레이션 앱 (로트카-볼테라 모델)
통합과학2 탐구 보고서용 / 필요한 라이브러리: streamlit (pandas는 함께 설치됨)
"""
import math
 
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="포식자-피식자 시뮬레이션", page_icon="🦊", layout="wide")
 
 
# ------------------------------------------------------------
# 1. 모델과 계산 방법 (파이썬 프로그램과 같은 코드)
# ------------------------------------------------------------
def derivs(x, y, a, b, c, d, K):
    """개체수 변화율. K가 None이 아니면 먹이 한계가 적용된다."""
    if K is None:
        dx = a * x - b * x * y
    else:
        dx = a * x * (1 - x / K) - b * x * y
    dy = d * x * y - c * y
    return dx, dy
 
 
@st.cache_data
def simulate(a, b, c, d, x0, y0, dt, t_end, method, K):
    steps = int(t_end / dt)
    x, y = float(x0), float(y0)
    times, prey, pred = [0.0], [x], [y]
    for i in range(steps):
        if method == "euler":
            dx, dy = derivs(x, y, a, b, c, d, K)
            x += dx * dt
            y += dy * dt
        else:  # 룽게-쿠타 4차
            k1x, k1y = derivs(x, y, a, b, c, d, K)
            k2x, k2y = derivs(x + k1x * dt / 2, y + k1y * dt / 2, a, b, c, d, K)
            k3x, k3y = derivs(x + k2x * dt / 2, y + k2y * dt / 2, a, b, c, d, K)
            k4x, k4y = derivs(x + k3x * dt, y + k3y * dt, a, b, c, d, K)
            x += (k1x + 2 * k2x + 2 * k3x + k4x) * dt / 6
            y += (k1y + 2 * k2y + 2 * k3y + k4y) * dt / 6
        times.append((i + 1) * dt)
        prey.append(x)
        pred.append(y)
    return times, prey, pred
 
 
def find_period(times, values):
    """정점 사이 시간 간격의 평균 = 변동 주기"""
    peaks = [times[i] for i in range(1, len(values) - 1)
             if values[i - 1] < values[i] > values[i + 1]]
    if len(peaks) < 2:
        return None
    gaps = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    return sum(gaps) / len(gaps)
 
 
def conserved(x, y, a, b, c, d):
    """이론상 일정해야 하는 값. 계산 오차가 크면 처음 값과 달라진다."""
    return d * x - c * math.log(x) + b * y - a * math.log(y)
 
 
def thin(times, *series, every=0.05):
    """그래프를 가볍게 하려고 0.05 시간 간격으로만 뽑는다."""
    stride = max(1, int(every / (times[1] - times[0])))
    return [times[::stride]] + [s[::stride] for s in series]
 
 
def summary(times, prey, pred):
    period = find_period(times, prey)
    return {
        "피식자 최대": round(max(prey), 1),
        "피식자 최소": round(min(prey), 2),
        "포식자 최대": round(max(pred), 1),
        "변동 주기": round(period, 2) if period else None,
    }
 
 
# ------------------------------------------------------------
# 2. 왼쪽 메뉴: 값 바꾸기
# ------------------------------------------------------------
st.title("🦊🐇 포식자-피식자 개체수 시뮬레이션")
st.caption("로트카-볼테라 모델을 파이썬으로 구현한 통합과학2 탐구 프로그램")
 
with st.sidebar:
    st.header("값 바꿔 보기")
    a = st.slider("피식자 번식률 a", 0.5, 2.0, 1.0, 0.1)
    b = st.slider("포식률 b", 0.05, 0.20, 0.10, 0.01)
    c = st.slider("포식자 사망률 c", 0.5, 3.0, 1.5, 0.1)
    d = st.slider("포식자 증가율 d", 0.02, 0.15, 0.075, 0.005)
    x0 = st.slider("피식자 처음 개체수", 1, 60, 10)
    y0 = st.slider("포식자 처음 개체수", 1, 40, 5)
    t_end = st.slider("시뮬레이션 시간", 10, 60, 30)
    use_k = st.checkbox("먹이 한계(환경수용력 K) 적용")
    K = st.slider("환경수용력 K", 20, 100, 40) if use_k else None
    method_label = st.radio("계산 방법", ["오일러 방법", "룽게-쿠타 (더 정확)"])
    method = "euler" if method_label == "오일러 방법" else "rk4"
    dt = st.select_slider("시간 간격 dt (작을수록 정확)", options=[0.01, 0.005, 0.001], value=0.001)
 
tab1, tab2, tab3, tab4 = st.tabs(["시뮬레이션", "변인 비교", "계산 오차", "원리 설명"])
 
# ------------------------------------------------------------
# 3. 탭1: 기본 시뮬레이션
# ------------------------------------------------------------
with tab1:
    times, prey, pred = simulate(a, b, c, d, x0, y0, dt, t_end, method, K)
    s = summary(times, prey, pred)
 
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("피식자 최대", s["피식자 최대"])
    m2.metric("피식자 최소", s["피식자 최소"])
    m3.metric("포식자 최대", s["포식자 최대"])
    m4.metric("변동 주기", s["변동 주기"] if s["변동 주기"] else "-")
 
    t_ds, prey_ds, pred_ds = thin(times, prey, pred)
    left, right = st.columns(2)
    with left:
        st.subheader("시간에 따른 개체수")
        df = pd.DataFrame({"피식자": prey_ds, "포식자": pred_ds}, index=pd.Index(t_ds, name="시간"))
        st.line_chart(df)
    with right:
        st.subheader("피식자-포식자 관계")
        df2 = pd.DataFrame({"피식자 수": prey_ds, "포식자 수": pred_ds})
        st.line_chart(df2, x="피식자 수", y="포식자 수")
 
    if K is None:
        st.info(f"균형 개체수: 피식자 {c / d:.1f}마리, 포식자 {a / b:.1f}마리. "
                "이 값에서 시작하면 개체수가 변하지 않는다. 처음 값이 다르면 이 값 주변을 돌며 오르내린다.")
    else:
        st.info("먹이 한계를 적용하면 피식자가 무한히 늘지 못해서 변동 폭이 줄어든다.")
 
# ------------------------------------------------------------
# 4. 탭2: 변인 비교 (하나만 바꾸고 나머지는 고정)
# ------------------------------------------------------------
with tab2:
    st.write("하나의 값만 바꾸고 나머지는 왼쪽 메뉴의 값으로 고정해서 결과를 비교한다.")
    var = st.radio("바꿀 변인", ["포식자 처음 개체수 (y0)", "피식자 번식률 (a)"], horizontal=True)
    if var.startswith("포식자"):
        chosen = st.multiselect("비교할 값", [1, 2, 3, 5, 8, 10, 15, 20], default=[2, 5, 10])
    else:
        chosen = st.multiselect("비교할 값", [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], default=[0.8, 1.0, 1.2])
 
    if not chosen:
        st.info("비교할 값을 하나 이상 골라 주세요.")
    else:
        prey_cols, pred_cols, rows, t_ref = {}, {}, [], None
        for v in sorted(chosen):
            if var.startswith("포식자"):
                t, p1, p2 = simulate(a, b, c, d, x0, v, dt, t_end, method, K)
                label = f"y0={v}"
            else:
                t, p1, p2 = simulate(v, b, c, d, x0, y0, dt, t_end, method, K)
                label = f"a={v}"
            td, p1d, p2d = thin(t, p1, p2)
            t_ref = td
            prey_cols[label], pred_cols[label] = p1d, p2d
            rows.append({"조건": label, **summary(t, p1, p2)})
 
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("피식자 개체수")
            st.line_chart(pd.DataFrame(prey_cols, index=pd.Index(t_ref, name="시간")))
        with g2:
            st.subheader("포식자 개체수")
            st.line_chart(pd.DataFrame(pred_cols, index=pd.Index(t_ref, name="시간")))
        st.subheader("결과 표")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
 
# ------------------------------------------------------------
# 5. 탭3: 계산 방법에 따른 오차
# ------------------------------------------------------------
with tab3:
    st.write("계산이 정확하다면 이론상 일정해야 하는 값(보존량)이 처음과 얼마나 달라지는지로 오차를 비교한다. "
             "값이 클수록 오차가 크다. (기본값 a, b, c, d와 처음 개체수 사용)")
    rows = []
    for label, m, step in [("오일러 dt=0.01", "euler", 0.01),
                           ("오일러 dt=0.001", "euler", 0.001),
                           ("룽게-쿠타 dt=0.01", "rk4", 0.01)]:
        t, p1, p2 = simulate(a, b, c, d, x0, y0, step, t_end, m, None)
        change = abs(conserved(p1[-1], p2[-1], a, b, c, d) - conserved(p1[0], p2[0], a, b, c, d))
        rows.append({"계산 방법": label, "보존량 변화(오차)": round(change, 5), "피식자 최대": round(max(p1), 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("먹이 한계(K)는 적용하지 않고 비교한다.")
 
# ------------------------------------------------------------
# 6. 탭4: 원리 설명
# ------------------------------------------------------------
with tab4:
    st.markdown("""
**모델의 기본 생각**: 지금 개체수를 알면 아주 짧은 시간 뒤의 개체수를 계산할 수 있다.
 
- 다음 피식자 수 = 지금 피식자 수 + 태어나는 수(a·x) − 잡아먹히는 수(b·x·y)
- 다음 포식자 수 = 지금 포식자 수 + 먹고 늘어나는 수(d·x·y) − 죽는 수(c·y)
 
포식자와 피식자는 **만나야** 잡아먹으므로 만나는 횟수는 두 개체수를 곱한 값(x·y)에 비례한다고 본다.
 
**개체수가 오르내리는 이유**: 피식자가 늘면 포식자가 따라 늘고, 포식자가 많아지면 피식자가 줄고,
먹이가 줄어든 포식자도 다시 줄어든다. 이 과정이 반복되며 생태계 평형이 유지된다.
 
**한계**: 먹이가 무한하고, 포식자가 한 종류의 피식자만 먹고, 환경이 변하지 않는다고 가정한 단순한 모델이다.
""")
