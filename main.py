import math
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as mb
import numpy as np

# ══════════════════════════════════════════════════════════════
# КОНСТАНТА ДОПУСТИМОЙ ПОГРЕШНОСТИ
# ══════════════════════════════════════════════════════════════
TOLERANCE = 0.05  # ±5%

# ══════════════════════════════════════════════════════════════
# РАСЧЁТНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════
def calculate_bmin(lambda_, teta, nu, omega, min_or_average=False):
    A = sum(l * t for l, t in zip(lambda_, teta))
    C = sum(l * t ** 2 * (1 + v ** 2) for l, t, v in zip(lambda_, teta, nu))
    D = sum(l * t * o for l, t, o in zip(lambda_, teta, omega))
    Bmin = round(0.5 * A + math.sqrt(0.25 * A * (A + 2 * C / D)))
    if min_or_average:
        Bmin = math.ceil((Bmin + 20001) / 100000.0) * 100000
    return A, C, D, Bmin

def calculate_service(lambda_, teta, nu, omega, q, B):
    n = len(lambda_)
    V = [teta[k] / B for k in range(n)]
    V2 = [V[k] ** 2 * (1 + nu[k] ** 2) for k in range(n)]
    p = [lambda_[k] * V[k] for k in range(n)]
    R = sum(p)

    W = [0.0] * n
    u = [0.0] * n
    Wreserv = [0.0] * n
    P_prob = [0.0] * n

    for k in range(n):
        sum1 = sum(q[i][k] * (q[i][k] - 1) * p[i] for i in range(n))
        sum3 = sum(
            (2 - q[k][i]) * (1 + q[k][i]) * lambda_[i] * teta[i] ** 2 * (1 + nu[i] ** 2)
            for i in range(n)
        )
        sum4 = sum(q[i][k] * (3 - q[i][k]) * p[i] for i in range(n))
        sum5 = sum((1 - q[k][i]) * (2 - q[k][i]) * p[i] for i in range(n))

        denom1 = B * (2 - sum1)
        denom2 = B ** 2 * (2 - sum4) * (2 - sum5)

        term1 = (teta[k] * sum1) / denom1 if denom1 != 0 else 0
        term2 = sum3 / denom2 if denom2 != 0 else 0
        W[k] = term1 + term2
        u[k] = V[k] + W[k]
        Wreserv[k] = omega[k] - W[k]

    F = 0.0
    for k in range(n):
        if W[k] > 0:
            P_prob[k] = R * math.exp(-R * (omega[k] / W[k]))
        else:
            P_prob[k] = 0.0
        F += lambda_[k] * P_prob[k]

    L = R ** 2 / (1 - R) if R < 1 else float('inf')
    LambdaSum = sum(lambda_)

    return {
        'V': V, 'V2': V2, 'p': p, 'W': W, 'u': u,
        'Wreserv': Wreserv, 'P': P_prob, 'R': R, 'L': L,
        'F': F, 'LambdaSum': LambdaSum
    }

def calculate_penalty(lambda_, teta, nu, omega, q, B):
    return calculate_service(lambda_, teta, nu, omega, q, B)['F']

# ══════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ МАТРИЦ ПРИОРИТЕТОВ
# ══════════════════════════════════════════════════════════════
def build_matrix_bp(n):
    return [[0] * n for _ in range(n)]

def build_matrix_op(teta, B):
    n = len(teta)
    V = [teta[k] / B for k in range(n)]
    indices = sorted(range(n), key=lambda i: V[i])
    q = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and indices.index(i) < indices.index(j):
                q[i][j] = 1
    return q

def build_matrix_ap(teta, B):
    q = build_matrix_op(teta, B)
    n = len(teta)
    for i in range(n):
        for j in range(n):
            if q[i][j] == 1:
                q[i][j] = 2
    return q

def build_matrix_sp(lambda_, teta, nu, omega, B):
    n = len(lambda_)
    q = build_matrix_ap(teta, B)
    Fprev = calculate_penalty(lambda_, teta, nu, omega, q, B)
    for i in range(n):
        for j in range(n):
            if q[i][j] == 2:
                q[i][j] = 1
                Fnew = calculate_penalty(lambda_, teta, nu, omega, q, B)
                if Fnew > Fprev:
                    q[i][j] = 2
                else:
                    Fprev = Fnew
    return q

# ══════════════════════════════════════════════════════════════
# РАСЧЁТ КОНФИГУРАЦИИ ВС (ГЛАВА 4)
# ══════════════════════════════════════════════════════════════
def calculate_chapter4(lambda_, teta, nu, omega, F_matrix, M_arr, L_arr,
                       ST, HD, HDLength, STLength,
                       SCK, SHD, SST, Dp, VHD, VST, B_user=None):
    n_tasks = len(lambda_)
    n_files = len(M_arr)
    results = {}

    lambda0 = sum(lambda_)
    results['lambda0'] = lambda0

    R0 = sum(l * t for l, t in zip(lambda_, teta)) / lambda0
    results['R0'] = R0

    Dcurrent = [0.0] * n_files
    for j in range(n_files):
        Dcurrent[j] = math.ceil(
            sum(lambda_[i] * F_matrix[i][j] for i in range(n_tasks)) / lambda0
        )
    results['Dcurrent'] = Dcurrent

    D0 = sum(Dcurrent)
    results['D0'] = D0

    Pcurrent = [(d / (D0 + 1)) for d in Dcurrent]
    results['Pcurrent'] = Pcurrent

    Rsr = R0 / (D0 + 1)
    results['Rsr'] = Rsr

    P0 = 1 / (D0 + 1)
    results['P0'] = P0 * 100

    Vpr_calc = 1.1 * lambda0 * R0
    results['Vpr_calc'] = Vpr_calc

    if B_user and B_user > 0:
        Vpr_final = B_user
    else:
        _, _, _, Bmin_rounded = calculate_bmin(lambda_, teta, nu, omega, min_or_average=True)
        Vpr_final = Bmin_rounded
    results['Vpr_final'] = Vpr_final

    lambdaNi = [lambda0 * d for d in Dcurrent]
    results['lambdaNi'] = lambdaNi

    flag = [0] * n_files
    file_T = [0.0] * n_files
    for i in range(n_files):
        if lambdaNi[i] == 0:
            flag[i] = 3
            file_T[i] = float('inf')
            continue
        T = 1.0 / lambdaNi[i]
        file_T[i] = T
        if T < ST and M_arr[i] <= HDLength:
            flag[i] = 1
        else:
            flag[i] = 2
    results['flag'] = flag
    results['file_T'] = file_T

    DHD = sum(Dcurrent[i] for i in range(n_files) if flag[i] == 1)
    DST = sum(Dcurrent[i] for i in range(n_files) if flag[i] == 2)
    results['DHD'] = DHD
    results['DST'] = DST

    lambdaHD = lambda0 * DHD
    lambdaST = lambda0 * DST
    results['lambdaHD'] = lambdaHD
    results['lambdaST'] = lambdaST

    Z1HD = math.ceil(lambdaHD * HD / 0.5) if lambdaHD > 0 else 0
    Z1ST = math.ceil(lambdaST * ST / 0.5) if lambdaST > 0 else 0
    results['Z1HD'] = Z1HD
    results['Z1ST'] = Z1ST

    sum_M_HD = sum(M_arr[i] for i in range(n_files) if flag[i] == 1)
    sum_M_ST = sum(M_arr[i] for i in range(n_files) if flag[i] == 2)
    Z2HD = math.ceil(sum_M_HD / HDLength) if sum_M_HD > 0 else 0
    Z2ST = math.ceil(sum_M_ST / STLength) if sum_M_ST > 0 else 0
    results['Z2HD'] = Z2HD
    results['Z2ST'] = Z2ST
    results['sum_M_HD'] = sum_M_HD
    results['sum_M_ST'] = sum_M_ST

    ZHD = max(Z1HD, Z2HD)
    ZST = max(Z1ST, Z2ST)
    if lambdaHD > 0 and ZHD == 0:
        ZHD = 1
    if lambdaST > 0 and ZST == 0:
        ZST = 1
    results['ZHD'] = ZHD
    results['ZST'] = ZST

    lambdaCK = lambda0 * D0
    results['lambdaCK'] = lambdaCK

    PHD_dol = sum(Pcurrent[i] for i in range(n_files) if flag[i] == 1) if D0 > 0 else 0
    PST_dol = sum(Pcurrent[i] for i in range(n_files) if flag[i] == 2) if D0 > 0 else 0
    results['PHD_dol'] = PHD_dol
    results['PST_dol'] = PST_dol

    if PHD_dol > 0:
        LHD = sum(L_arr[i] * Pcurrent[i] for i in range(n_files) if flag[i] == 1) / PHD_dol
    else:
        LHD = 0
    if PST_dol > 0:
        LST = sum(L_arr[i] * Pcurrent[i] for i in range(n_files) if flag[i] == 2) / PST_dol
    else:
        LST = 0
    results['LHD'] = LHD
    results['LST'] = LST

    VHD_kBps = VHD * 1000
    VST_kBps = VST * 1000
    TCK = (
        (LHD * PHD_dol) / (VHD_kBps * 1024) +
        (LST * PST_dol) / (VST_kBps * 1024)
    ) if (PHD_dol + PST_dol) > 0 else 0
    results['TCK'] = TCK

    ZCK = math.ceil(lambdaCK * TCK / 0.7) if lambdaCK * TCK > 0 else 1
    results['ZCK'] = ZCK

    denom_pr = Vpr_final - lambda0 * R0
    Upr = R0 / denom_pr if denom_pr > 0 else float('inf')
    results['Upr'] = Upr

    if ZST > 0 and lambdaST > 0:
        UST = ST / (1 - (lambdaST * ST / ZST))
    else:
        UST = 0.0
    results['UST'] = UST

    if ZHD > 0 and lambdaHD > 0:
        UHD = HD / (1 - (lambdaHD * HD / ZHD))
    else:
        UHD = 0.0
    results['UHD'] = UHD

    if ZCK > 0 and lambdaCK > 0:
        UCK = TCK / (1 - (lambdaCK * TCK / ZCK))
    else:
        UCK = 0.0
    results['UCK'] = UCK

    Umin = Upr + DHD * UHD + DST * UST + D0 * UCK
    results['Umin'] = Umin

    Smin = (Vpr_final / 1000) * Dp + ZCK * SCK + ZHD * SHD + ZST * SST
    results['Smin'] = Smin

    return results

# ══════════════════════════════════════════════════════════════
# ГЛАВА 6 - СИНТЕЗ ПО ВРЕМЕНИ
# Цель: U_actual = U_target ±5%
# ══════════════════════════════════════════════════════════════
def calculate_synthesis_by_time(res_ch4, U_target, DK):
    lambda0  = res_ch4['lambda0']
    R0       = res_ch4['R0']
    D0       = res_ch4['D0']
    DHD      = res_ch4['DHD']
    DST      = res_ch4['DST']
    lambdaHD = res_ch4['lambdaHD']
    lambdaST = res_ch4['lambdaST']
    lambdaCK = res_ch4['lambdaCK']
    TCK      = res_ch4['TCK']
    # Время
    HD_time  = 0.00007
    ST_time  = 0.001
    # Берём Z напрямую из главы 4 как фиксированные
    ZHD_fixed = res_ch4['ZHD']
    ZST_fixed = res_ch4['ZST']
    ZCK_fixed = res_ch4['ZCK']
    # Цена
    SHD_val   = 25
    SST_val   = 40
    SCK_val   = 140

    # Z используются из главы 4, без изменений
    zhd = ZHD_fixed
    zst = ZST_fixed
    zck = ZCK_fixed

    # Минимально достижимое время
    U_inf = DHD * HD_time + DST * ST_time + D0 * TCK
    if U_target <= U_inf + 1e-9:
        return {
            'error': (
                f"Заданное время U_target = {U_target:.6f} с "
                f"меньше или равно минимально достижимому {U_inf:.6f} с. "
                f"Увеличьте U_target."
            )
        }

    # Считаем время на устройствах с Z из главы 4
    def calc_U_typical(zh, zs, zc):
        U_hd = 0.0
        U_st = 0.0
        U_ck = 0.0
        if lambdaHD > 0 and zh > lambdaHD * HD_time:
            U_hd = (DHD * zh * HD_time) / (zh - lambdaHD * HD_time)
        if lambdaST > 0 and zs > lambdaST * ST_time:
            U_st = (DST * zs * ST_time) / (zs - lambdaST * ST_time)
        if lambdaCK > 0 and zc > lambdaCK * TCK:
            U_ck = (D0 * zc * TCK) / (zc - lambdaCK * TCK)
        return U_hd + U_st + U_ck

    U_typ = calc_U_typical(zhd, zst, zck)

    # Проверка: хватает ли времени для процессора
    U_pr_need = U_target - U_typ
    if U_pr_need <= 1e-12:
        return {
            'error': (
                f"Время на устройствах из гл.4 "
                f"U_typ = {U_typ:.6f} с уже превышает "
                f"U_target = {U_target:.6f} с. "
                f"Увеличьте U_target."
            )
        }

    # Быстродействие процессора из остаточного времени
    B_new = lambda0 * R0 + R0 / U_pr_need

    if B_new > lambda0 * R0:
        U_pr_actual = R0 / (B_new - lambda0 * R0)
    else:
        U_pr_actual = float('inf')

    U_actual = U_pr_actual + U_typ

    # Погрешность
    delta = abs(U_actual - U_target) / U_target if U_target != 0 else 0

    # Стоимость
    S_total = (B_new / 1000) * DK + zhd * SHD_val + zst * SST_val + zck * SCK_val

    return {
        'Zhd_round'        : zhd,
        'Zst_round'        : zst,
        'Zck_round'        : zck,
        'U_typical_actual' : U_typ,
        'U_pr_rem'         : U_pr_need,
        'B_new'            : B_new,
        'U_actual'         : U_actual,
        'U_target'         : U_target,
        'delta_pct'        : delta * 100,
        'S_total'          : S_total,
    }

# ══════════════════════════════════════════════════════════════
# ГЛАВА 7 - СИНТЕЗ ПО СТОИМОСТИ
# Цель: S_actual = S_target (присваивание)
# ══════════════════════════════════════════════════════════════
def calculate_synthesis_by_cost(res_ch4, S_target, DK):
    lambda0  = res_ch4['lambda0']
    R0       = res_ch4['R0']
    D0       = res_ch4['D0']
    DHD      = res_ch4['DHD']
    DST      = res_ch4['DST']
    lambdaHD = res_ch4['lambdaHD']
    lambdaST = res_ch4['lambdaST']
    lambdaCK = res_ch4['lambdaCK']
    TCK      = res_ch4['TCK']
    # Время
    HD_time  = 0.00007
    ST_time  = 0.001
    # Берём Z напрямую из главы 4 (как в главе 6)
    ZHD_fixed = res_ch4['ZHD']
    ZST_fixed = res_ch4['ZST']
    ZCK_fixed = res_ch4['ZCK']
    # Цена
    SHD_val   = 25
    SST_val   = 40
    SCK_val   = 140
    Vpr_base  = res_ch4['Vpr_final']

    # Z используются из главы 4, без изменений
    zhd = ZHD_fixed
    zst = ZST_fixed
    zck = ZCK_fixed

    # Стоимость устройств
    S_devices = zhd * SHD_val + zst * SST_val + zck * SCK_val

    # Остаток бюджета на процессор
    S_pr_budget = S_target - S_devices

    if S_pr_budget <= 0:
        return {
            'error': (
                f"Стоимость устройств ({S_devices:.2f}) "
                f">= S_target ({S_target:.2f}). "
                f"Увеличьте S_target."
            )
        }

    # Быстродействие из бюджета на процессор
    if DK > 0:
        Vpr = (S_pr_budget * 1000.0) / DK
    else:
        Vpr = Vpr_base
    Vpr = max(Vpr, Vpr_base)

    # Фактическая стоимость (может отличаться если Vpr скорректирован)
    S_actual = (Vpr / 1000.0) * DK + S_devices

    # Погрешность
    delta = abs(S_actual - S_target) / S_target if S_target != 0 else 0

    # Время ответа
    U_hd = 0.0
    U_st = 0.0
    U_ck = 0.0
    if lambdaHD > 0 and zhd > lambdaHD * HD_time:
        U_hd = (DHD * zhd * HD_time) / (zhd - lambdaHD * HD_time)
    if lambdaST > 0 and zst > lambdaST * ST_time:
        U_st = (DST * zst * ST_time) / (zst - lambdaST * ST_time)
    if lambdaCK > 0 and zck > lambdaCK * TCK:
        U_ck = (D0 * zck * TCK) / (zck - lambdaCK * TCK)
    if Vpr > lambda0 * R0:
        U_pr = R0 / (Vpr - lambda0 * R0)
    else:
        U_pr = float('inf')
    U_actual = U_pr + U_hd + U_st + U_ck

    return {
        'Zhd_round'   : zhd,
        'Zst_round'   : zst,
        'Zck_round'   : zck,
        'Vpr_final'   : Vpr,
        'S_devices'   : S_devices,
        'S_pr_budget' : S_pr_budget,
        'U_actual'    : U_actual,
        'S_actual'    : S_actual,
        'S_target'    : S_target,
        'delta_pct'   : delta * 100,
    }

# ══════════════════════════════════════════════════════════════
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ══════════════════════════════════════════════════════════════
def main():
    lambda_ = [0.6, 0.5, 1.8, 1.1, 1.6]
    teta    = [73550, 20000, 65750, 22550, 18360]
    nu      = [0.90, 0.85, 1.0, 0.75, 0.90]
    omega   = [0.5, 0.4, 0.1, 0.4, 1.3]
    n       = len(lambda_)

    F_matrix = [
        [11, 4, 7, 2, 3, 0, 0, 0, 0, 0],
        [0, 12, 7, 2, 7, 3, 0, 0, 0, 0],
        [0, 0, 0, 4, 10, 6, 2, 1, 0, 0],
        [10, 0, 0, 8, 0, 6, 2, 4, 0, 0],
        [1, 2, 4, 6, 8, 10, 0, 0, 0, 0]
    ]
    M_arr   = [5, 6, 7, 8, 9, 10, 9, 8, 7, 6] # хар-ка файлов - длина файла
    L_arr   = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9] # хар-ка файлов - ср. длина записи
    n_files = len(M_arr)
    # СРЕДНЕЕ ВРЕМЯ ДОСТУПА К ДАННЫМ (МС)
    ST       = 0.001 # ST
    HD       = 0.00007 # HD
    # ЕМКОСТЬ
    HDLength = 80
    STLength = 220
    #СТОИМОСТЬ ТИПОВЫХ У-ЙВ
    SCK      = 140
    SHD      = 25
    SST      = 40
    # СТОИМОСТЬ КОЭФ. ПРОЦЕССОРА
    Dp       = 5
    # СКОРОСТЬ ПЕРЕДАЧИ ДАННЫХ
    VHD      = 180
    VST      = 22

    headers_input    = ["λ (lambda)", "θ (teta)", "ν (nu)", "ω (omega)"]
    discipline_names = {
        0: "БП — Бесприоритетная",
        1: "ОП — Относительный приоритет",
        2: "АП — Абсолютный приоритет",
        3: "СП — Смешанный приоритет",
    }

    root = tk.Tk()
    root.title("СВКС — Расчёт быстродействия, дисциплин обслуживания и конфигурации ВС")
    root.geometry("1050x850")
    root.resizable(True, True)
    root.configure(bg="#f0f0f0")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Title.TLabel",     font=("Segoe UI", 14, "bold"), background="#f0f0f0")
    style.configure("Sub.TLabel",       font=("Segoe UI", 11, "bold"), background="#f0f0f0")
    style.configure("Res.TLabel",       font=("Consolas", 11),         background="#f0f0f0")
    style.configure("Treeview",         font=("Consolas", 10), rowheight=26)
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    # ══════════════════════════════════════════════════════════
    # ВКЛАДКА Bmin
    # ══════════════════════════════════════════════════════════
    tab_bmin = ttk.Frame(notebook)
    notebook.add(tab_bmin, text="  Bmin  ")

    ttk.Label(tab_bmin,
              text="Расчёт минимального быстродействия (Bmin)",
              style="Title.TLabel").pack(pady=(14, 8))

    frame_input1 = ttk.LabelFrame(tab_bmin, text="Входные данные", padding=10)
    frame_input1.pack(padx=16, pady=(0, 8), fill="x")

    cols1 = ("idx",) + tuple(headers_input)
    tree1 = ttk.Treeview(frame_input1, columns=cols1, show="headings", height=n)
    tree1.heading("idx", text="№")
    tree1.column("idx", width=40, anchor="center")
    for h in headers_input:
        tree1.heading(h, text=h)
        tree1.column(h, width=150, anchor="center")
    for i in range(n):
        tree1.insert("", "end", values=(i + 1, lambda_[i], teta[i], nu[i], omega[i]))
    tree1.pack(fill="x")

    result_bmin_var = tk.StringVar(value="Нажмите кнопку для расчёта")

    def on_calculate_bmin():
        A, C, D, Bmin = calculate_bmin(lambda_, teta, nu, omega, min_or_average=False)
        B = math.ceil((Bmin + 20001) / 100000.0) * 100000
        txt = (
            f"A = {round(A)}\n"
            f"C = {round(C)}\n"
            f"D = {round(D)}\n"
            f"──────────────────────────\n"
            f"Bmin = {Bmin}\n"
            f"B (округлённое) = {int(B)}"
        )
        result_bmin_var.set(txt)

    ttk.Button(tab_bmin, text="Рассчитать Bmin",
               command=on_calculate_bmin).pack(pady=8)

    frame_res1 = ttk.LabelFrame(tab_bmin, text="Результаты", padding=10)
    frame_res1.pack(padx=16, pady=(0, 16), fill="x")
    ttk.Label(frame_res1, textvariable=result_bmin_var,
              style="Res.TLabel", justify="left").pack(anchor="w")

    # ══════════════════════════════════════════════════════════
    # ВКЛАДКА Дисциплины обслуживания
    # ══════════════════════════════════════════════════════════
    tab_do = ttk.Frame(notebook)
    notebook.add(tab_do, text="  Дисциплины обслуживания  ")

    ttk.Label(tab_do,
              text="Расчёт дисциплин обслуживания",
              style="Title.TLabel").pack(pady=(10, 6))

    frame_top = ttk.Frame(tab_do)
    frame_top.pack(padx=16, pady=(0, 6), fill="x")

    ttk.Label(frame_top, text="B:", font=("Segoe UI", 11)).pack(side="left")
    entry_B = ttk.Entry(frame_top, width=14, font=("Consolas", 11))
    entry_B.pack(side="left", padx=(4, 16))

    def fill_b_from_bmin():
        _, _, _, Bmin = calculate_bmin(lambda_, teta, nu, omega, min_or_average=False)
        B_rounded = int(math.ceil((Bmin + 20001) / 100000.0) * 100000)
        entry_B.delete(0, tk.END)
        entry_B.insert(0, str(B_rounded))

    ttk.Button(frame_top, text="← Bmin",
               command=fill_b_from_bmin).pack(side="left", padx=(0, 20))

    ttk.Label(frame_top, text="Дисциплина:",
              font=("Segoe UI", 11)).pack(side="left")

    combo_disc = ttk.Combobox(
        frame_top, state="readonly", width=32, font=("Segoe UI", 10),
        values=[
            "БП — Бесприоритетная",
            "ОП — Относительный приоритет",
            "АП — Абсолютный приоритет",
            "СП — Смешанный приоритет",
            "Все дисциплины (сравнение)"
        ]
    )
    combo_disc.current(0)
    combo_disc.pack(side="left", padx=4)

    frame_matrix = ttk.LabelFrame(
        tab_do,
        text="Матрица приоритетов q[i,j] (клик по ячейке для изменения)",
        padding=8
    )
    frame_matrix.pack(padx=16, pady=(0, 4), fill="x")

    matrix_labels = []
    for j in range(n):
        lbl = ttk.Label(frame_matrix, text=f"  {j+1} ",
                        font=("Consolas", 11, "bold"), anchor="center", width=4)
        lbl.grid(row=0, column=j + 1, padx=2, pady=2)

    for i in range(n):
        ttk.Label(frame_matrix, text=f"{i+1}",
                  font=("Consolas", 11, "bold"),
                  anchor="center", width=3).grid(row=i + 1, column=0, padx=2, pady=2)
        row_labels = []
        for j in range(n):
            lbl = tk.Label(frame_matrix, text="0",
                           font=("Consolas", 13, "bold"),
                           bg="#ffffff", relief="groove",
                           width=4, height=1, anchor="center")
            lbl.grid(row=i + 1, column=j + 1, padx=2, pady=2)
            row_labels.append(lbl)
        matrix_labels.append(row_labels)

    color_map = {0: "#ffffff", 1: "#b3e5fc", 2: "#ffcdd2"}

    def display_matrix(q):
        for i in range(n):
            for j in range(n):
                val = q[i][j]
                matrix_labels[i][j].config(
                    text=str(val), bg=color_map.get(val, "#ffffff"))

    def get_current_matrix():
        q = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                q[i][j] = int(matrix_labels[i][j].cget("text"))
        return q

    def on_cell_click(i, j):
        current = int(matrix_labels[i][j].cget("text"))
        new_val = (current + 1) % 3
        matrix_labels[i][j].config(
            text=str(new_val), bg=color_map.get(new_val, "#ffffff"))

    for i in range(n):
        for j in range(n):
            matrix_labels[i][j].bind(
                "<Button-1>", lambda e, i=i, j=j: on_cell_click(i, j))

    frame_calc = ttk.LabelFrame(tab_do, text="Результаты расчёта", padding=8)
    frame_calc.pack(padx=16, pady=(0, 4), fill="both", expand=True)

    calc_cols = ("idx", "V", "V2(1+v2)", "p", "W", "u", "w-W", "P(w<W)")
    tree_calc = ttk.Treeview(frame_calc, columns=calc_cols,
                              show="headings", height=n)
    for c in calc_cols:
        tree_calc.heading(c, text=c)
        w = 40 if c == "idx" else 110
        tree_calc.column(c, width=w, anchor="center")
    tree_calc.pack(fill="x", pady=(0, 6))

    text_summary = tk.Text(frame_calc, height=10, font=("Consolas", 10),
                           bg="#fafafa", relief="groove", wrap="word")
    text_summary.pack(fill="both", expand=True)

    def get_sorted_indices_by_wreserv(result):
        wres = result['Wreserv']
        indices = list(range(len(wres)))
        indices.sort(key=lambda i: wres[i], reverse=True)
        return indices

    def display_results(result, discipline_idx, B_val):
        for item in tree_calc.get_children():
            tree_calc.delete(item)
        sorted_idx = get_sorted_indices_by_wreserv(result)
        for orig_idx in sorted_idx:
            tree_calc.insert("", "end", values=(
                orig_idx + 1,
                f"{result['V'][orig_idx]:.5f}",
                f"{result['V2'][orig_idx]:.5f}",
                f"{result['p'][orig_idx]:.5f}",
                f"{result['W'][orig_idx]:.5f}",
                f"{result['u'][orig_idx]:.5f}",
                f"{result['Wreserv'][orig_idx]:.5f}",
                f"{result['P'][orig_idx]:.7f}"
            ))
        text_summary.delete("1.0", tk.END)
        text_summary.insert(
            tk.END,
            f"Дисциплина: {discipline_names.get(discipline_idx, '?')}\n"
            f"B = {B_val}\n{'─'*50}\n"
            f"Суммарная интенсивность = {result['LambdaSum']:.5f}\n"
            f"Суммарная загрузка  R   = {result['R']:.5f}\n"
            f"Длина очереди        L   = {result['L']:.5f}\n"
            f"Функция штрафа      F   = {result['F']:.7f}\n"
            f"\nЗадачи отсортированы по убыванию w-W:\n"
        )
        for orig_idx in sorted_idx:
            text_summary.insert(
                tk.END,
                f"  Задача {orig_idx+1}: w-W = {result['Wreserv'][orig_idx]:.5f}\n"
            )

    def display_comparison(results_all, B_val):
        for item in tree_calc.get_children():
            tree_calc.delete(item)
        text_summary.delete("1.0", tk.END)
        text_summary.insert(
            tk.END,
            f"=== СРАВНЕНИЕ ВСЕХ ДИСЦИПЛИН при B = {B_val} ===\n\n"
        )
        bp_res = results_all[0]
        sorted_idx = get_sorted_indices_by_wreserv(bp_res)
        for disc_idx, disc_name in discipline_names.items():
            res = results_all[disc_idx]
            text_summary.insert(
                tk.END,
                f"> {disc_name}\n"
                f"  R={res['R']:.5f}  L={res['L']:.5f}  F={res['F']:.7f}\n"
            )
            for orig_idx in sorted_idx:
                text_summary.insert(
                    tk.END,
                    f"    Задача {orig_idx+1}: "
                    f"V={res['V'][orig_idx]:.5f}  "
                    f"W={res['W'][orig_idx]:.5f}  "
                    f"u={res['u'][orig_idx]:.5f}  "
                    f"w-W={res['Wreserv'][orig_idx]:.5f}  "
                    f"P={res['P'][orig_idx]:.7f}\n"
                )
            text_summary.insert(tk.END, "\n")

        text_summary.insert(
            tk.END,
            f"{'─'*60}\n"
            f"{'Дисциплина':<30} {'R':>8} {'L':>10} {'F':>12}\n"
            f"{'─'*60}\n"
        )
        for disc_idx, disc_name in discipline_names.items():
            res = results_all[disc_idx]
            text_summary.insert(
                tk.END,
                f"{disc_name:<30} {res['R']:>8.5f} {res['L']:>10.5f} {res['F']:>12.7f}\n"
            )
        text_summary.insert(tk.END, f"{'─'*60}\n")
        best = min(results_all, key=lambda d: results_all[d]['F'])
        text_summary.insert(
            tk.END,
            f"\nЛучшая дисциплина по F: {discipline_names[best]} "
            f"(F = {results_all[best]['F']:.7f})\n"
        )

    def on_calculate_do():
        try:
            B_val = float(entry_B.get())
            if B_val <= 0:
                raise ValueError
        except Exception:
            text_summary.delete("1.0", tk.END)
            text_summary.insert(tk.END, "Ошибка: введите положительное B!")
            return

        sel = combo_disc.current()
        if sel <= 3:
            if sel == 0:
                q = build_matrix_bp(n)
            elif sel == 1:
                q = build_matrix_op(teta, B_val)
            elif sel == 2:
                q = build_matrix_ap(teta, B_val)
            else:
                q = build_matrix_sp(lambda_, teta, nu, omega, B_val)
            display_matrix(q)
            result = calculate_service(
                lambda_, teta, nu, omega, get_current_matrix(), B_val)
            display_results(result, sel, B_val)
        else:
            matrices = {
                0: build_matrix_bp(n),
                1: build_matrix_op(teta, B_val),
                2: build_matrix_ap(teta, B_val),
                3: build_matrix_sp(lambda_, teta, nu, omega, B_val)
            }
            results_all = {
                d: calculate_service(lambda_, teta, nu, omega, q, B_val)
                for d, q in matrices.items()
            }
            display_matrix(matrices[3])
            display_comparison(results_all, B_val)

    def reset_matrix():
        sel = combo_disc.current()
        try:
            B_val = float(entry_B.get())
            if B_val <= 0:
                raise ValueError
        except Exception:
            return
        if sel == 0:
            q = build_matrix_bp(n)
        elif sel == 1:
            q = build_matrix_op(teta, B_val)
        elif sel == 2:
            q = build_matrix_ap(teta, B_val)
        elif sel == 3:
            q = build_matrix_sp(lambda_, teta, nu, omega, B_val)
        else:
            return
        display_matrix(q)

    ttk.Button(frame_top, text="Сбросить матрицу",
               command=reset_matrix).pack(side="left", padx=10)
    ttk.Button(tab_do, text="  Расчёт  ",
               command=on_calculate_do).pack(pady=(0, 8))
    fill_b_from_bmin()

    # ══════════════════════════════════════════════════════════
    # ВКЛАДКА Конфигурация ВС (Глава 4)
    # ══════════════════════════════════════════════════════════
    tab_ch4 = ttk.Frame(notebook)
    notebook.add(tab_ch4, text="  Конфигурация ВС  ")

    canvas_ch4    = tk.Canvas(tab_ch4, bg="#f0f0f0", highlightthickness=0)
    scrollbar_ch4 = ttk.Scrollbar(tab_ch4, orient="vertical",
                                   command=canvas_ch4.yview)
    scroll_frame_ch4 = ttk.Frame(canvas_ch4)
    scroll_frame_ch4.bind(
        "<Configure>",
        lambda e: canvas_ch4.configure(scrollregion=canvas_ch4.bbox("all"))
    )
    canvas_ch4.create_window((0, 0), window=scroll_frame_ch4, anchor="nw")
    canvas_ch4.configure(yscrollcommand=scrollbar_ch4.set)
    scrollbar_ch4.pack(side="right", fill="y")
    canvas_ch4.pack(side="left", fill="both", expand=True)

    def _on_mousewheel(event):
        canvas_ch4.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas_ch4.bind("<MouseWheel>", _on_mousewheel)

    ttk.Label(scroll_frame_ch4,
              text="Расчёт конфигурации ВС (Глава 4)",
              style="Title.TLabel").pack(pady=(14, 8))

    frame_f_input = ttk.LabelFrame(
        scroll_frame_ch4, text="Матрица обращений к файлам", padding=8)
    frame_f_input.pack(padx=16, pady=(0, 8), fill="x")

    f_cols = ("Задача",) + tuple(f"Ф{j+1}" for j in range(n_files))
    tree_f = ttk.Treeview(frame_f_input, columns=f_cols,
                           show="headings", height=n)
    tree_f.heading("Задача", text="Задача")
    tree_f.column("Задача", width=60, anchor="center")
    for j in range(n_files):
        tree_f.heading(f"Ф{j+1}", text=f"Ф{j+1}")
        tree_f.column(f"Ф{j+1}", width=55, anchor="center")
    for i in range(n):
        vals = (f"З{i+1}",) + tuple(F_matrix[i][j] for j in range(n_files))
        tree_f.insert("", "end", values=vals)
    tree_f.pack(fill="x")

    frame_file_params = ttk.LabelFrame(
        scroll_frame_ch4, text="Параметры файлов", padding=8)
    frame_file_params.pack(padx=16, pady=(0, 8), fill="x")

    fp_cols = ("Файл", "M (размер)", "L (длина записи)")
    tree_fp = ttk.Treeview(frame_file_params, columns=fp_cols,
                            show="headings", height=n_files)
    for c in fp_cols:
        tree_fp.heading(c, text=c)
        tree_fp.column(c, width=120, anchor="center")
    for j in range(n_files):
        tree_fp.insert("", "end", values=(f"Файл {j+1}", M_arr[j], L_arr[j]))
    tree_fp.pack(fill="x")

    frame_consts = ttk.LabelFrame(
        scroll_frame_ch4, text="Параметры устройств", padding=8)
    frame_consts.pack(padx=16, pady=(0, 8), fill="x")

    consts_text = (
        f"HD: время доступа = {HD} с, ёмкость = {HDLength} МБ, "
        f"скорость = {VHD} Кбайт/мс, стоимость = {SHD}\n"
        f"ST: время доступа = {ST} с, ёмкость = {STLength} МБ, "
        f"скорость = {VST} Кбайт/мс, стоимость = {SST}\n"
        f"Канал: стоимость = {SCK}     Процессор: стоимость 1000 оп/с = {Dp}"
    )
    ttk.Label(frame_consts, text=consts_text,
              font=("Consolas", 10), background="#f0f0f0",
              justify="left").pack(anchor="w")

    frame_ch4_res = ttk.LabelFrame(
        scroll_frame_ch4, text="Результаты расчёта", padding=8)
    frame_ch4_res.pack(padx=16, pady=(0, 8), fill="both", expand=True)

    text_ch4 = tk.Text(frame_ch4_res, height=38, font=("Consolas", 10),
                       bg="#fafafa", relief="groove", wrap="word")
    text_ch4.pack(fill="both", expand=True)

    def on_calculate_ch4():
        try:
            b_val = float(entry_B.get())
            if b_val <= 0:
                b_val = None
        except Exception:
            b_val = None

        res = calculate_chapter4(
            lambda_, teta, nu, omega, F_matrix, M_arr, L_arr,
            ST, HD, HDLength, STLength,
            SCK, SHD, SST, Dp, VHD, VST, B_user=b_val
        )

        text_ch4.delete("1.0", tk.END)
        sep = "─" * 60

        text_ch4.insert(tk.END,
            f"{'='*60}\n  1. ПАРАМЕТРЫ СРЕДНЕЙ ЗАДАЧИ\n{'='*60}\n\n")
        text_ch4.insert(tk.END,
            f"Интенсивность потока заявок  lambda0 = {res['lambda0']:.4f}\n")
        text_ch4.insert(tk.END,
            f"Средняя трудоёмкость          R0 = {res['R0']:.0f}\n\n")
        text_ch4.insert(tk.END, "Обращения к файлам D[j]:\n")
        for j in range(n_files):
            text_ch4.insert(tk.END,
                f"  Файл {j+1:2d}: D = {res['Dcurrent'][j]:.0f}\n")
        text_ch4.insert(tk.END,
            f"  Суммарно D0 = {res['D0']:.0f}\n\n")
        text_ch4.insert(tk.END, "Вероятности обращений P[j]:\n")
        for j in range(n_files):
            text_ch4.insert(tk.END,
                f"  Файл {j+1:2d}: P = {res['Pcurrent'][j]:.4f}\n")
        text_ch4.insert(tk.END,
            f"\nТрудоёмкость этапа счёта  Rsr = {res['Rsr']:.1f}\n")
        text_ch4.insert(tk.END,
            f"Вероятность выхода         P0 = {res['P0']:.2f}%\n")

        text_ch4.insert(tk.END,
            f"\n{'='*60}\n  2. МИНИМАЛЬНАЯ СКОРОСТЬ ПРОЦЕССОРА\n{'='*60}\n\n")
        text_ch4.insert(tk.END,
            f"Минимальное быстродействие  Vpr_calc = {res['Vpr_calc']:.1f} оп/с\n")
        text_ch4.insert(tk.END,
            f"ИСПОЛЬЗУЕМОЕ (из Bmin/Поля)  B_final  = {res['Vpr_final']:.0f} оп/с\n")

        text_ch4.insert(tk.END,
            f"\n{'='*60}\n  3. РАЗМЕЩЕНИЕ ФАЙЛОВ\n{'='*60}\n\n")
        text_ch4.insert(tk.END, "Интенсивности обращений и размещение:\n")
        flag_names = {1: "HD", 2: "ST", 3: "не исп."}
        for j in range(n_files):
            lni  = res['lambdaNi'][j]
            t_v  = res['file_T'][j]
            fl   = res['flag'][j]
            if fl == 3:
                text_ch4.insert(tk.END,
                    f"  Файл {j+1:2d}: lambda_N={lni:.4f},  T=---  → {flag_names[fl]}\n")
            else:
                text_ch4.insert(tk.END,
                    f"  Файл {j+1:2d}: lambda_N={lni:.4f},  T={t_v:.4f} с  → {flag_names[fl]}\n")
        text_ch4.insert(tk.END,
            f"\nОбращений: HD={res['DHD']:.0f},  ST={res['DST']:.0f}\n")
        text_ch4.insert(tk.END,
            f"Интенсивности: lambda_HD={res['lambdaHD']:.2f},  lambda_ST={res['lambdaST']:.2f}\n")

        text_ch4.insert(tk.END,
            f"\n{'='*60}\n  4. КОЛИЧЕСТВО ВЗУ\n{'='*60}\n\n")
        text_ch4.insert(tk.END,
            f"По загрузке:  HD={res['Z1HD']},  ST={res['Z1ST']}\n")
        text_ch4.insert(tk.END,
            f"По объёму:     HD={res['Z2HD']} (M={res['sum_M_HD']}),  "
            f"ST={res['Z2ST']} (M={res['sum_M_ST']})\n")
        text_ch4.insert(tk.END,
            f"{sep}\nИТОГО:        HD={res['ZHD']},  ST={res['ZST']}\n")

        text_ch4.insert(tk.END,
            f"\n{'='*60}\n  5. СЕЛЕКТОРНЫЕ КАНАЛЫ\n{'='*60}\n\n")
        text_ch4.insert(tk.END,
            f"Интенсивность каналов   lambda_CK = {res['lambdaCK']:.2f}\n")
        text_ch4.insert(tk.END,
            f"Вероятности: P_HD={res['PHD_dol']*100:.2f}%,  P_ST={res['PST_dol']*100:.2f}%\n")
        text_ch4.insert(tk.END,
            f"Средние длины записей: L_HD={res['LHD']:.2f} байт,  L_ST={res['LST']:.2f} байт\n")
        text_ch4.insert(tk.END,
            f"Время передачи через канал T_CK = {res['TCK']:.6f} с\n")
        text_ch4.insert(tk.END,
            f"{sep}\nКоличество каналов Z_CK = {res['ZCK']}\n")

        text_ch4.insert(tk.END,
            f"\n{'='*60}\n  6. СРЕДНЕЕ ВРЕМЯ ОТВЕТА\n{'='*60}\n\n")
        text_ch4.insert(tk.END,
            f"Время в процессоре U_pr = {res['Upr']:.6f} с\n")
        text_ch4.insert(tk.END,
            f"Время в HD         U_HD = {res['UHD']:.6f} с\n")
        text_ch4.insert(tk.END,
            f"Время в ST         U_ST = {res['UST']:.6f} с\n")
        text_ch4.insert(tk.END,
            f"Время в канале     U_CK = {res['UCK']:.6f} с\n")
        text_ch4.insert(tk.END,
            f"{sep}\nСРЕДНЕЕ ВРЕМЯ ОТВЕТА U_min = {res['Umin']:.4f} с\n")

        text_ch4.insert(tk.END,
            f"\n{'='*60}\n  7. СТОИМОСТЬ\n{'='*60}\n\n")
        cost_pr = (res['Vpr_final'] / 1000) * Dp
        cost_ck = res['ZCK'] * SCK
        cost_hd = res['ZHD'] * SHD
        cost_st = res['ZST'] * SST
        text_ch4.insert(tk.END,
            f"Процессор: ({res['Vpr_final']:.1f}/1000) × {Dp} = {cost_pr:.2f}\n")
        text_ch4.insert(tk.END,
            f"Каналы:    {res['ZCK']} × {SCK} = {cost_ck:.2f}\n")
        text_ch4.insert(tk.END,
            f"HD:        {res['ZHD']} × {SHD} = {cost_hd:.2f}\n")
        text_ch4.insert(tk.END,
            f"ST:        {res['ZST']} × {SST} = {cost_st:.2f}\n")
        text_ch4.insert(tk.END,
            f"{sep}\nИТОГО СТОИМОСТЬ S_min = {res['Smin']:.2f} ед. изм.\n")

    ttk.Button(scroll_frame_ch4,
               text="  Рассчитать конфигурацию ВС  ",
               command=on_calculate_ch4).pack(pady=(0, 4))

    # ══════════════════════════════════════════════════════════
    # ВКЛАДКА Графики (Глава 5)
    # ══════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        try:
            import subprocess, sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "matplotlib"])
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            MATPLOTLIB_AVAILABLE = True
        except Exception:
            MATPLOTLIB_AVAILABLE = False

    tab_graphics = ttk.Frame(notebook)
    notebook.add(tab_graphics, text="  Глава 5 (Графики)  ")

    if not MATPLOTLIB_AVAILABLE:
        ttk.Label(
            tab_graphics,
            text="Библиотека matplotlib не установлена.\nГрафики недоступны.",
            foreground="red", justify="center"
        ).pack(pady=20)
    else:
        try:
            plt.rcParams['font.family']      = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass

        def get_current_B():
            try:
                b = float(entry_B.get())
                if b <= 0:
                    raise ValueError
                return b
            except Exception:
                _, _, _, bmin = calculate_bmin(
                    lambda_, teta, nu, omega, min_or_average=False)
                return math.ceil((bmin + 20001) / 100000.0) * 100000

        def plot_w_vs_load():
            B_fixed = get_current_B()
            multipliers = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
            q_bp = build_matrix_bp(n)
            q_op = build_matrix_op(teta, B_fixed)
            q_ap = build_matrix_ap(teta, B_fixed)
            q_sp = build_matrix_sp(lambda_, teta, nu, omega, B_fixed)
            disciplines = [
                ("БП", q_bp, 'red'),
                ("ОП", q_op, 'blue'),
                ("АП", q_ap, 'green'),
                ("СП", q_sp, 'purple')
            ]
            w_data = {name: [[] for _ in range(n)] for name, _, _ in disciplines}
            for mult in multipliers:
                lambda_scaled = [l * (1 + mult) for l in lambda_]
                for name, q, color in disciplines:
                    try:
                        res = calculate_service(
                            lambda_scaled, teta, nu, omega, q, B_fixed)
                        for task in range(n):
                            w_data[name][task].append(res['W'][task])
                    except Exception:
                        for task in range(n):
                            w_data[name][task].append(float('nan'))

            fig = Figure(figsize=(15, 10))
            for task in range(n):
                ax = fig.add_subplot(2, 3, task + 1)
                for name, q, color in disciplines:
                    y_vals = w_data[name][task]
                    valid = [
                        (multipliers[i], y_vals[i])
                        for i in range(len(multipliers))
                        if not np.isnan(y_vals[i])
                    ]
                    if valid:
                        xv, yv = zip(*valid)
                        ax.plot(xv, yv, marker='o', label=name,
                                color=color, linewidth=2)
                ax.set_xlabel("Множитель интенсивности (1+mult)", fontsize=10)
                ax.set_ylabel("Время ожидания W", fontsize=10)
                ax.set_title(f"Задача {task+1}", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
            fig.tight_layout()

            win = tk.Toplevel(root)
            win.title("Зависимость W от интенсивности λ")
            cv = FigureCanvasTkAgg(fig, master=win)
            cv.draw()
            cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            tk.Button(win, text="Закрыть", command=win.destroy).pack(pady=5)

        def plot_w_vs_b():
            B_values = [
                1000000, 1250000, 1500000,
                1750000, 2000000, 2250000, 2500000
            ]
            w_data = {name: [[] for _ in range(n)]
                      for name in ["БП", "ОП", "АП", "СП"]}
            for B in B_values:
                q_bp = build_matrix_bp(n)
                q_op = build_matrix_op(teta, B)
                q_ap = build_matrix_ap(teta, B)
                q_sp = build_matrix_sp(lambda_, teta, nu, omega, B)
                disciplines = [
                    ("БП", q_bp), ("ОП", q_op),
                    ("АП", q_ap), ("СП", q_sp)
                ]
                for name, q in disciplines:
                    try:
                        res = calculate_service(lambda_, teta, nu, omega, q, B)
                        for task in range(n):
                            w_data[name][task].append(res['W'][task])
                    except Exception:
                        for task in range(n):
                            w_data[name][task].append(float('nan'))

            colors = {"БП": 'red', "ОП": 'blue', "АП": 'green', "СП": 'purple'}
            fig = Figure(figsize=(15, 10))
            for task in range(n):
                ax = fig.add_subplot(2, 3, task + 1)
                for name in w_data.keys():
                    y_vals = w_data[name][task]
                    valid = [
                        (B_values[i], y_vals[i])
                        for i in range(len(B_values))
                        if not np.isnan(y_vals[i])
                    ]
                    if valid:
                        xv, yv = zip(*valid)
                        ax.plot(xv, yv, marker='o', label=name,
                                color=colors[name], linewidth=2)
                ax.set_xlabel("Быстродействие B (оп/с)", fontsize=10)
                ax.set_ylabel("Время ожидания W", fontsize=10)
                ax.set_title(f"Задача {task+1}", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
            fig.tight_layout()

            win = tk.Toplevel(root)
            win.title("Зависимость W от быстродействия B")
            cv = FigureCanvasTkAgg(fig, master=win)
            cv.draw()
            cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            tk.Button(win, text="Закрыть", command=win.destroy).pack(pady=5)

        def plot_discipline_comparison():
            B_fixed = get_current_B()
            q_bp = build_matrix_bp(n)
            q_op = build_matrix_op(teta, B_fixed)
            q_ap = build_matrix_ap(teta, B_fixed)
            q_sp = build_matrix_sp(lambda_, teta, nu, omega, B_fixed)
            res_bp = calculate_service(lambda_, teta, nu, omega, q_bp, B_fixed)
            res_op = calculate_service(lambda_, teta, nu, omega, q_op, B_fixed)
            res_ap = calculate_service(lambda_, teta, nu, omega, q_ap, B_fixed)
            res_sp = calculate_service(lambda_, teta, nu, omega, q_sp, B_fixed)
            w_bp = sorted(res_bp['W'])
            w_op = sorted(res_op['W'])
            w_ap = sorted(res_ap['W'])
            w_sp = sorted(res_sp['W'])

            fig = Figure(figsize=(8, 6))
            ax  = fig.add_subplot(111)
            x   = list(range(1, n + 1))
            ax.plot(x, w_bp, marker='o', label="БП", color='red',    linewidth=2)
            ax.plot(x, w_op, marker='s', label="ОП", color='blue',   linewidth=2)
            ax.plot(x, w_ap, marker='^', label="АП", color='green',  linewidth=2)
            ax.plot(x, w_sp, marker='D', label="СП", color='purple', linewidth=2)
            ax.set_xlabel("Номер задачи (сортировка W по возрастанию)", fontsize=11)
            ax.set_ylabel("Время ожидания W", fontsize=11)
            ax.set_title("Сравнение дисциплин обслуживания", fontsize=13)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

            win = tk.Toplevel(root)
            win.title("Сравнение дисциплин")
            cv = FigureCanvasTkAgg(fig, master=win)
            cv.draw()
            cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            tk.Button(win, text="Закрыть", command=win.destroy).pack(pady=5)

        ttk.Label(tab_graphics,
                  text="Построение графиков (Глава 5)",
                  style="Title.TLabel").pack(pady=(14, 8))

        info_frame = ttk.LabelFrame(tab_graphics, text="Информация", padding=8)
        info_frame.pack(padx=16, pady=(0, 12), fill="x")
        ttk.Label(info_frame, text=(
            "• График 1: зависимость W от множителя интенсивности (1+mult).\n"
            "• График 2: зависимость W от быстродействия B (фикс. значения).\n"
            "• График 3: сравнение БП, ОП, АП, СП (все задачи, сортировка W)."
        ), justify="left", font=("Segoe UI", 10)).pack(anchor="w")

        btn_frame = ttk.Frame(tab_graphics)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame,
                   text="Зависимость W от интенсивности λ",
                   command=plot_w_vs_load, width=45).pack(pady=4)
        ttk.Button(btn_frame,
                   text="Зависимость W от быстродействия B",
                   command=plot_w_vs_b, width=45).pack(pady=4)
        ttk.Button(btn_frame,
                   text="Сравнение дисциплин обслуживания",
                   command=plot_discipline_comparison, width=45).pack(pady=4)

    # ══════════════════════════════════════════════════════════
    # ВКЛАДКА Глава 6 – Синтез по времени
    # ══════════════════════════════════════════════════════════
    tab_ch6 = ttk.Frame(notebook)
    notebook.add(tab_ch6, text="  Глава 6 — Синтез по времени  ")

    frame_ch6_input = ttk.LabelFrame(
        tab_ch6, text="Параметры синтеза", padding=8)
    frame_ch6_input.pack(padx=16, pady=(8, 8), fill="x")

    ttk.Label(frame_ch6_input,
              text="Предельное время ответа U_target (с):").grid(
        row=0, column=0, padx=5, pady=5, sticky="e")
    entry_U_target = ttk.Entry(frame_ch6_input, width=15)
    entry_U_target.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    entry_U_target.insert(0, "0.5")

    ttk.Label(frame_ch6_input,
              text="Стоимость 1000 оп/с (DK):").grid(
        row=1, column=0, padx=5, pady=5, sticky="e")
    entry_DK6 = ttk.Entry(frame_ch6_input, width=15)
    entry_DK6.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    entry_DK6.insert(0, str(Dp))

    frame_ch6_res = ttk.LabelFrame(
        tab_ch6, text="Результаты синтеза", padding=8)
    frame_ch6_res.pack(padx=16, pady=(0, 8), fill="both", expand=True)

    text_ch6 = tk.Text(frame_ch6_res, height=30, font=("Consolas", 10),
                       bg="#fafafa", relief="groove", wrap="word")
    text_ch6.pack(fill="both", expand=True)

    def on_calculate_ch6():
        try:
            B_cur = float(entry_B.get())
            if B_cur <= 0:
                raise ValueError
        except Exception:
            text_ch6.delete("1.0", tk.END)
            text_ch6.insert(
                tk.END,
                "Ошибка: введите положительное B на вкладке «Дисциплины»"
            )
            return

        res_ch4 = calculate_chapter4(
            lambda_, teta, nu, omega, F_matrix, M_arr, L_arr,
            ST, HD, HDLength, STLength,
            SCK, SHD, SST, Dp, VHD, VST, B_user=B_cur
        )

        try:
            U_target = float(entry_U_target.get())
            if U_target <= 0:
                raise ValueError
            DK = float(entry_DK6.get())
            if DK <= 0:
                DK = Dp
        except Exception:
            text_ch6.delete("1.0", tk.END)
            text_ch6.insert(
                tk.END,
                "Ошибка: введите положительные числа для U_target и DK"
            )
            return

        res = calculate_synthesis_by_time(res_ch4, U_target, DK)
        text_ch6.delete("1.0", tk.END)

        if 'error' in res:
            text_ch6.insert(tk.END, f"ОШИБКА:\n{res['error']}")
            return

        sep = "─" * 60

        text_ch6.insert(tk.END,
            "=== ГЛАВА 6 — СИНТЕЗ ПО ЗАДАННОМУ ВРЕМЕНИ ===\n\n")
        text_ch6.insert(tk.END,
            f"Заданное время ответа  U_target = {res['U_target']:.6f} с\n\n")

        text_ch6.insert(tk.END, sep + "\n")
        text_ch6.insert(tk.END,
            "ИСПОЛЬЗУЕМЫЕ Z (из главы 4):\n")
        text_ch6.insert(tk.END,
            f"  HD  = {res['Zhd_round']}\n"
            f"  ST  = {res['Zst_round']}\n"
            f"  CK  = {res['Zck_round']}\n\n")

        text_ch6.insert(tk.END,
            f"Время на типовых устройствах U_typ = "
            f"{res['U_typical_actual']:.6f} с\n")
        text_ch6.insert(tk.END,
            f"Остаточное время для процессора:\n"
            f"  U_pr_rem = U_target - U_typ = "
            f"{res['U_target']:.6f} - {res['U_typical_actual']:.6f} = "
            f"{res['U_pr_rem']:.6f} с\n\n")
        text_ch6.insert(tk.END,
            f"Скорректированное быстродействие:\n"
            f"  Vpr' = λ₀·R₀ + R₀/U_pr_rem =\n"
            f"       = {res['B_new']:.2f} оп/с\n\n")

        text_ch6.insert(tk.END, sep + "\n")

        delta = res['delta_pct']
        if delta <= TOLERANCE * 100:
            status = f"✔ В пределах ±{TOLERANCE*100:.0f}% погрешности"
        else:
            status = f"✘ ВЫХОД за пределы ±{TOLERANCE*100:.0f}% погрешности!"

        text_ch6.insert(tk.END,
            f"U_actual  = {res['U_actual']:.6f} с\n"
            f"U_target  = {res['U_target']:.6f} с\n"
            f"Погрешность δ = {delta:.2f}%   {status}\n")
        text_ch6.insert(tk.END, sep + "\n")
        text_ch6.insert(tk.END,
            f"Стоимость конфигурации S_total = {res['S_total']:.2f} ед. изм.\n")

    ttk.Button(tab_ch6, text="  Выполнить синтез  ",
               command=on_calculate_ch6).pack(pady=8)

    # ══════════════════════════════════════════════════════════
    # ВКЛАДКА Глава 7 – Синтез по стоимости
    # ══════════════════════════════════════════════════════════
    tab_ch7 = ttk.Frame(notebook)
    notebook.add(tab_ch7, text="  Глава 7 — Синтез по стоимости  ")

    frame_ch7_input = ttk.LabelFrame(
        tab_ch7, text="Параметры синтеза", padding=8)
    frame_ch7_input.pack(padx=16, pady=(8, 8), fill="x")

    ttk.Label(frame_ch7_input,
              text="Желаемая стоимость S_target (ед. изм.):").grid(
        row=0, column=0, padx=5, pady=5, sticky="e")
    entry_S_target = ttk.Entry(frame_ch7_input, width=15)
    entry_S_target.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    entry_S_target.insert(0, "100000")

    ttk.Label(frame_ch7_input,
              text="Стоимость 1000 оп/с (DK):").grid(
        row=1, column=0, padx=5, pady=5, sticky="e")
    entry_DK7 = ttk.Entry(frame_ch7_input, width=15)
    entry_DK7.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    entry_DK7.insert(0, str(Dp))

    frame_ch7_res = ttk.LabelFrame(
        tab_ch7, text="Результаты синтеза", padding=8)
    frame_ch7_res.pack(padx=16, pady=(0, 8), fill="both", expand=True)

    text_ch7 = tk.Text(frame_ch7_res, height=30, font=("Consolas", 10),
                       bg="#fafafa", relief="groove", wrap="word")
    text_ch7.pack(fill="both", expand=True)

    def on_calculate_ch7():
        try:
            B_cur = float(entry_B.get())
            if B_cur <= 0:
                raise ValueError
        except Exception:
            text_ch7.delete("1.0", tk.END)
            text_ch7.insert(
                tk.END,
                "Ошибка: введите положительное B на вкладке «Дисциплины»"
            )
            return

        res_ch4 = calculate_chapter4(
            lambda_, teta, nu, omega, F_matrix, M_arr, L_arr,
            ST, HD, HDLength, STLength,
            SCK, SHD, SST, Dp, VHD, VST, B_user=B_cur
        )

        try:
            S_target = float(entry_S_target.get())
            if S_target <= 0:
                raise ValueError
            DK = float(entry_DK7.get())
            if DK <= 0:
                DK = Dp
        except Exception:
            text_ch7.delete("1.0", tk.END)
            text_ch7.insert(
                tk.END,
                "Ошибка: введите положительные числа для S_target и DK"
            )
            return

        res = calculate_synthesis_by_cost(res_ch4, S_target, DK)
        text_ch7.delete("1.0", tk.END)

        if 'error' in res:
            text_ch7.insert(tk.END, f"ОШИБКА:\n{res['error']}")
            return

        sep = "─" * 60

        text_ch7.insert(tk.END,
            "=== ГЛАВА 7 — СИНТЕЗ ПО ЗАДАННОЙ СТОИМОСТИ ===\n\n")
        text_ch7.insert(tk.END,
            f"Заданная стоимость  S_target = {res['S_target']:.2f} ед. изм.\n\n")

        text_ch7.insert(tk.END, sep + "\n")
        text_ch7.insert(tk.END,
            "ИСПОЛЬЗУЕМЫЕ Z (из главы 4):\n")
        text_ch7.insert(tk.END,
            f"  HD  = {res['Zhd_round']}\n"
            f"  ST  = {res['Zst_round']}\n"
            f"  CK  = {res['Zck_round']}\n\n")

        text_ch7.insert(tk.END,
            f"Стоимость устройств:\n"
            f"  {res['Zhd_round']} × {65} + {res['Zst_round']} × {50} + "
            f"{res['Zck_round']} × {120} = {res['S_devices']:.2f} ед. изм.\n\n")

        text_ch7.insert(tk.END,
            f"Остаток бюджета на процессор:\n"
            f"  S_pr_budget = S_target - S_devices =\n"
            f"              = {res['S_target']:.2f} - {res['S_devices']:.2f} = "
            f"{res['S_pr_budget']:.2f} ед. изм.\n\n")

        text_ch7.insert(tk.END,
            f"Быстродействие процессора:\n"
            f"  Vpr' = (S_pr_budget × 1000) / DK =\n"
            f"       = ({res['S_pr_budget']:.2f} × 1000) / {DK} = "
            f"{res['Vpr_final']:.2f} оп/с\n\n")

        text_ch7.insert(tk.END, sep + "\n")

        delta = res['delta_pct']
        if delta <= TOLERANCE * 100:
            status = f"✔ В пределах ±{TOLERANCE*100:.0f}% погрешности"
        else:
            status = f"✘ ВЫХОД за пределы ±{TOLERANCE*100:.0f}% погрешности!"

        text_ch7.insert(tk.END,
            f"S_actual  = {res['S_actual']:.2f} ед. изм.\n"
            f"S_target  = {res['S_target']:.2f} ед. изм.\n"
            f"Погрешность δ = {delta:.2f}%   {status}\n")
        text_ch7.insert(tk.END, sep + "\n")
        text_ch7.insert(tk.END,
            f"Достигнутое время ответа U_actual = {res['U_actual']:.6f} с\n")

    ttk.Button(tab_ch7, text="  Выполнить синтез  ",
               command=on_calculate_ch7).pack(pady=8)

    root.mainloop()


if __name__ == "__main__":
    main()
