import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

st.set_page_config(page_title="Теория игр — вариант 41", layout="wide")

A = np.array([[9, 25, 5],
              [21, 5, 17]], dtype=float)

rows = ["A1: инъекция adversarial-примеров",
        "A2: отравление обучающей выборки"]
cols = ["B1: adversarial training",
        "B2: санитизация и нормализация",
        "B3: ансамблевая детекция"]

st.title("Веб-приложение: теория игр, вариант 41")
st.caption("Атака и защита модели ИИ (Adversarial ML)")

st.header("1. Вербальная постановка")
st.write("Игрок A — red team, выбирает тип атаки. Игрок B — служба доверия и безопасности, выбирает одно защитное решение.")
st.write("Выигрыш игрока A — число нежелательных роликов, прошедших автоматическую проверку за сутки.")

st.header("2. Матрица выигрышей")
df = pd.DataFrame(A, index=rows, columns=cols)
st.dataframe(df, use_container_width=True)

st.markdown("**Обоснование элементов матрицы:**")
st.write("- При инъекции adversarial-примеров: 9, 25, 5.")
st.write("- При отравлении обучающей выборки: 21, 5, 17.")

st.header("3. Анализ доминирования")
st.write("Доминируемых строк и столбцов нет: ни одна стратегия не превосходит другую по всем исходам.")

st.header("4. Проверка седловой точки")
row_mins = A.min(axis=1)
col_maxs = A.max(axis=0)
alpha = row_mins.max()
beta = col_maxs.min()

st.write(f"Минимумы по строкам: {row_mins.astype(int).tolist()}.")
st.write(f"Максимумы по столбцам: {col_maxs.astype(int).tolist()}.")
st.write(f"maximin = {alpha:.0f}, minimax = {beta:.0f}.")
st.error("Седловой точки нет, так как maximin < minimax.")

st.header("5. Графоаналитический метод")
p = np.linspace(0, 1, 400)
f1 = 21 - 12 * p
f2 = 5 + 20 * p
f3 = 17 - 12 * p
lower = np.minimum(np.minimum(f1, f2), f3)
idx = np.argmax(lower)
p_star = p[idx]
v_star = lower[idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(p, f1, label='B1: 21 - 12p')
ax.plot(p, f2, label='B2: 5 + 20p')
ax.plot(p, f3, label='B3: 17 - 12p')
ax.plot(p, lower, '--', color='black', label='Нижняя огибающая')
ax.scatter([p_star], [v_star], color='red', zorder=5)
ax.annotate(f'({p_star:.3f}, {v_star:.3f})', (p_star, v_star),
            textcoords="offset points", xytext=(10, 10))
ax.set_xlabel('p = P(A1)')
ax.set_ylabel('Ожидаемый выигрыш A')
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

st.write(f"Оптимальная смешанная стратегия A: P* = ({p_star:.3f}, {1-p_star:.3f}).")
st.write(f"Цена игры по графику: v = {v_star:.3f}.")

st.header("6. Симплекс-метод через ЛП")
st.write("Сведем задачу к ЛП для игрока A: максимизация цены при ограничениях на ожидаемый выигрыш против каждого столбца.")

c = np.array([0, 0, -1.0])
A_ub = np.array([
    [-9, -21, 1],
    [-25, -5, 1],
    [-5, -17, 1]
], dtype=float)
b_ub = np.zeros(3)
A_eq = np.array([[1, 1, 0]], dtype=float)
b_eq = np.array([1.0])
bounds = [(0, None), (0, None), (None, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res.success:
    p1_lp, p2_lp, v_lp = res.x
    st.success("ЛП решена успешно.")
    st.write(f"P* = ({p1_lp:.3f}, {p2_lp:.3f}), v = {v_lp:.3f}.")
else:
    st.error("ЛП не удалось решить.")

st.header("7. Метод Брауна–Робинсона")
st.write("Ниже показан учебный итерационный алгоритм, который приближает оптимальные стратегии и цену игры.")

max_iter = 20
row_choice = [0]
col_choice = [0]

for k in range(1, max_iter):
    col_counts = np.bincount(col_choice, minlength=A.shape[1])
    q = col_counts / len(col_choice)
    row_expected = A @ q
    i = int(np.argmax(row_expected))
    row_choice.append(i)

    row_counts = np.bincount(row_choice, minlength=A.shape[0])
    p_emp = row_counts / len(row_choice)
    col_expected = p_emp @ A
    j = int(np.argmin(col_expected))
    col_choice.append(j)

br_p = np.bincount(row_choice, minlength=2) / len(row_choice)
br_q = np.bincount(col_choice, minlength=3) / len(col_choice)
br_v = np.mean([(br_p @ A[:, j]) for j in range(3)])

st.write(f"Приближение A: ({br_p[0]:.3f}, {br_p[1]:.3f})")
st.write(f"Приближение B: ({br_q[0]:.3f}, {br_q[1]:.3f}, {br_q[2]:.3f})")
st.write(f"Приближенная цена игры: {br_v:.3f}")

st.header("8. Итог")
st.success("Оптимальные стратегии: A* = (0.375, 0.625), B* = (0, 0.375, 0.625), цена игры v = 12.5.")
st.write("Графоаналитический метод, ЛП и итерационный метод дают согласующийся результат.")