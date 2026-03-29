"""
Simulateur de panneau solaire — Modèle 1 diode / 5 paramètres
Courbes I=f(V) et P=f(V) réalistes avec coude visible.
Lancer : streamlit run solar_panel_app.py
Dépendances : streamlit, numpy, scipy, matplotlib
"""

import streamlit as st
import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

# ── Constantes physiques ──────────────────────────────────────────────
q = 1.602176634e-19   # charge élémentaire (C)
k = 1.380649e-23       # constante de Boltzmann (J/K)

# ── Modèle à 1 diode ─────────────────────────────────────────────────

def extract_params(Voc, Isc, Vmpp, Impp, Ns, T_stc=25.0):
    """
    Extraction itérative des 5 paramètres (Iph, I0, n, Rs, Rsh)
    à partir des données STC du datasheet.
    Méthode inspirée de Villalva et al. (2009).
    """
    T = T_stc + 273.15
    Vt = k * T / q  # tension thermique d'une cellule

    # Facteur d'idéalité initial (entre 1 et 2)
    # On le choisit pour que le coude soit bien marqué
    n = 1.2

    Vt_mod = n * Ns * Vt  # tension thermique du module

    # Courant de saturation inverse depuis Voc
    I0 = Isc / (np.exp(Voc / Vt_mod) - 1)

    # Photo-courant ≈ Isc (approximation initiale)
    Iph = Isc

    # Résistance shunt initiale (grande)
    Rsh = 500.0

    # ── Recherche itérative de Rs pour que P(Vmpp) = Vmpp*Impp ──
    # On ajuste Rs et Rsh simultanément.

    best_Rs = 0.0
    Pmpp_target = Vmpp * Impp

    for iteration in range(200):
        # Bornes de Rs
        Rs_min = 0.0
        Rs_max = (Voc - Vmpp) / Impp  # borne supérieure physique

        def power_error(Rs):
            """Erreur sur la puissance au MPP pour un Rs donné."""
            # Recalcul de Rsh pour satisfaire le point MPP
            # I(Vmpp) doit = Impp
            num = Vmpp + Impp * Rs - Voc
            den = Vt_mod
            exp_mpp = np.exp(num / den)
            exp_oc = np.exp(Voc / Vt_mod)

            # Depuis l'équation I = Iph - I0*(exp((V+I*Rs)/(n*Ns*Vt))-1) - (V+I*Rs)/Rsh
            # Au point MPP : Impp = Iph - I0*(exp_mpp - 1) - (Vmpp + Impp*Rs)/Rsh
            # => Rsh = (Vmpp + Impp*Rs) / (Iph - I0*(exp_mpp - 1) - Impp)
            denom = Iph - I0 * (exp_mpp - 1) - Impp
            if denom <= 0:
                return 1e6
            Rsh_calc = (Vmpp + Impp * Rs) / denom
            if Rsh_calc < 10:
                Rsh_calc = 10

            # Recalcul de Iph avec Rsh
            Iph_calc = Isc * (1 + Rs / Rsh_calc)

            # Recalcul de I0 avec Iph_calc
            I0_calc = (Iph_calc - Voc / Rsh_calc) / (np.exp(Voc / Vt_mod) - 1)
            if I0_calc <= 0:
                return 1e6

            # Calculer I au point Vmpp avec ces paramètres
            def current_eq(I):
                return Iph_calc - I0_calc * (np.exp((Vmpp + I * Rs) / Vt_mod) - 1) - (Vmpp + I * Rs) / Rsh_calc - I

            try:
                I_calc = brentq(current_eq, 0, Isc * 1.5, xtol=1e-10)
            except:
                return 1e6

            P_calc = Vmpp * I_calc
            return abs(P_calc - Pmpp_target)

        # Recherche du meilleur Rs
        try:
            result = minimize_scalar(power_error, bounds=(Rs_min, Rs_max), method='bounded',
                                     options={'xatol': 1e-8, 'maxiter': 500})
            best_Rs = result.x
        except:
            best_Rs = 0.01 * Ns

        # Mettre à jour Rsh et Iph avec le Rs trouvé
        Rs = best_Rs
        num = Vmpp + Impp * Rs - Voc
        exp_mpp = np.exp(num / Vt_mod)
        denom = Iph - I0 * (exp_mpp - 1) - Impp
        if denom > 0:
            Rsh = (Vmpp + Impp * Rs) / denom
        if Rsh < 10:
            Rsh = 10

        Iph = Isc * (1 + Rs / Rsh)
        I0 = (Iph - Voc / Rsh) / (np.exp(Voc / Vt_mod) - 1)
        if I0 <= 0:
            I0 = 1e-12

        # Vérifier convergence
        def current_eq_check(I):
            return Iph - I0 * (np.exp((Vmpp + I * Rs) / Vt_mod) - 1) - (Vmpp + I * Rs) / Rsh - I
        try:
            I_check = brentq(current_eq_check, 0, Isc * 1.5, xtol=1e-12)
            P_check = Vmpp * I_check
            if abs(P_check - Pmpp_target) / Pmpp_target < 1e-6:
                break
        except:
            pass

    return Iph, I0, n, Rs, Rsh


def iv_curve(Voc, Isc, Vmpp, Impp, Ns, T_cell, G, AM,
             mu_isc=0.0005, mu_voc=-0.003):
    """
    Calcule la courbe I-V complète pour des conditions (T, G, AM) données.
    Retourne V, I, P (arrays).
    """
    T_stc = 25.0
    G_stc = 1000.0

    # Extraction des paramètres STC
    Iph_stc, I0_stc, n, Rs, Rsh = extract_params(Voc, Isc, Vmpp, Impp, Ns, T_stc)

    # ── Corrections pour T et G ──
    dT = T_cell - T_stc

    # Photo-courant corrigé
    Iph = (Iph_stc + mu_isc * Isc * dT) * (G / G_stc)

    # Correction AM (effet spectral simplifié)
    if AM > 0:
        am_factor = min(1.0, 1.0 / (1.0 + 0.02 * (AM - 1.5)))
        Iph *= am_factor

    # Tension thermique corrigée
    T_K = (T_cell + 273.15)
    Vt = k * T_K / q
    Vt_mod = n * Ns * Vt

    # Courant de saturation corrigé (très sensible à T)
    T_stc_K = T_stc + 273.15
    I0 = I0_stc * (T_K / T_stc_K) ** 3 * np.exp(
        (q * 1.12 / (n * k)) * (1.0 / T_stc_K - 1.0 / T_K)
    )

    # Voc corrigé (estimation pour la plage de tension)
    Voc_corr = Voc + mu_voc * Voc * dT + Vt_mod * np.log(max(G / G_stc, 0.01))
    Voc_corr = max(Voc_corr, 0.1)

    # ── Calcul de la courbe I-V ──
    # On balaie V de 0 à un peu au-delà de Voc_corr
    V = np.linspace(0, Voc_corr * 1.05, 500)
    I_out = np.zeros_like(V)

    for i, v in enumerate(V):
        def f(I_val):
            return Iph - I0 * (np.exp((v + I_val * Rs) / Vt_mod) - 1) - (v + I_val * Rs) / Rsh - I_val

        try:
            # Chercher I entre 0 et Iph
            if f(0) < 0:
                I_out[i] = 0.0
            else:
                I_out[i] = brentq(f, 0, Iph * 1.5, xtol=1e-10)
        except:
            I_out[i] = 0.0

        if I_out[i] < 0:
            I_out[i] = 0.0

    P = V * I_out
    return V, I_out, P


# ══════════════════════════════════════════════════════════════════════
# Interface Streamlit
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Simulateur PV — Modèle 1 Diode", layout="wide")
st.title("☀️ Simulateur Panneau Solaire — Modèle 1 Diode / 5 Paramètres")
st.markdown("""
Courbes **I = f(V)** et **P = f(V)** réalistes.
Comparez deux configurations en modifiant un paramètre.
""")

st.sidebar.header("⚙️ Paramètres du panneau (STC)")

Voc = st.sidebar.number_input("Voc (V)", value=40.0, step=0.1, format="%.2f")
Isc = st.sidebar.number_input("Isc (A)", value=9.5, step=0.1, format="%.2f")
Vmpp = st.sidebar.number_input("Vmpp (V)", value=32.0, step=0.1, format="%.2f")
Impp = st.sidebar.number_input("Impp (A)", value=8.9, step=0.1, format="%.2f")
Ns = st.sidebar.number_input("Nombre de cellules (Ns)", value=60, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Coefficients de température")
mu_isc = st.sidebar.number_input("μ_Isc (%/°C)", value=0.05, step=0.01, format="%.3f") / 100
mu_voc = st.sidebar.number_input("μ_Voc (%/°C)", value=-0.30, step=0.01, format="%.3f") / 100

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Vérification paramètres extraits")
try:
    Iph_ext, I0_ext, n_ext, Rs_ext, Rsh_ext = extract_params(Voc, Isc, Vmpp, Impp, Ns)
    st.sidebar.write(f"**Iph** = {Iph_ext:.4f} A")
    st.sidebar.write(f"**I₀** = {I0_ext:.2e} A")
    st.sidebar.write(f"**n** = {n_ext:.2f}")
    st.sidebar.write(f"**Rs** = {Rs_ext:.4f} Ω")
    st.sidebar.write(f"**Rsh** = {Rsh_ext:.2f} Ω")
except Exception as e:
    st.sidebar.error(f"Erreur extraction : {e}")

# ── Deux configurations ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Configuration A")
    T_a = st.number_input("Température cellule (°C)", value=25.0, step=1.0, key="T_a")
    G_a = st.number_input("Éclairement (W/m²)", value=1000.0, step=50.0, key="G_a")
    AM_a = st.number_input("Air Mass (AM)", value=1.5, step=0.1, key="AM_a")

with col2:
    st.subheader("🔴 Configuration B")
    T_b = st.number_input("Température cellule (°C)", value=45.0, step=1.0, key="T_b")
    G_b = st.number_input("Éclairement (W/m²)", value=800.0, step=50.0, key="G_b")
    AM_b = st.number_input("Air Mass (AM)", value=1.5, step=0.1, key="AM_b")

# ── Calcul des courbes ──
try:
    V_a, I_a, P_a = iv_curve(Voc, Isc, Vmpp, Impp, Ns, T_a, G_a, AM_a, mu_isc, mu_voc)
    V_b, I_b, P_b = iv_curve(Voc, Isc, Vmpp, Impp, Ns, T_b, G_b, AM_b, mu_isc, mu_voc)

    # Points MPP
    idx_a = np.argmax(P_a)
    idx_b = np.argmax(P_b)

    # ── Graphiques ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # I = f(V)
    ax1.plot(V_a, I_a, 'b-', linewidth=2.2, label=f"A : {G_a:.0f} W/m², {T_a:.0f}°C")
    ax1.plot(V_b, I_b, 'r-', linewidth=2.2, label=f"B : {G_b:.0f} W/m², {T_b:.0f}°C")
    ax1.plot(V_a[idx_a], I_a[idx_a], 'bo', markersize=10, zorder=5)
    ax1.plot(V_b[idx_b], I_b[idx_b], 'ro', markersize=10, zorder=5)
    ax1.annotate(f"MPP\n({V_a[idx_a]:.1f}V, {I_a[idx_a]:.2f}A)",
                 xy=(V_a[idx_a], I_a[idx_a]), fontsize=8,
                 textcoords="offset points", xytext=(10, 5),
                 arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
    ax1.annotate(f"MPP\n({V_b[idx_b]:.1f}V, {I_b[idx_b]:.2f}A)",
                 xy=(V_b[idx_b], I_b[idx_b]), fontsize=8,
                 textcoords="offset points", xytext=(10, -20),
                 arrowprops=dict(arrowstyle='->', color='red'), color='red')
    ax1.set_xlabel("Tension V (V)", fontsize=12)
    ax1.set_ylabel("Courant I (A)", fontsize=12)
    ax1.set_title("Caractéristique I = f(V)", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)

    # P = f(V)
    ax2.plot(V_a, P_a, 'b-', linewidth=2.2, label=f"A : Pmax = {P_a[idx_a]:.1f} W")
    ax2.plot(V_b, P_b, 'r-', linewidth=2.2, label=f"B : Pmax = {P_b[idx_b]:.1f} W")
    ax2.plot(V_a[idx_a], P_a[idx_a], 'bo', markersize=10, zorder=5)
    ax2.plot(V_b[idx_b], P_b[idx_b], 'ro', markersize=10, zorder=5)
    ax2.annotate(f"MPP\n{P_a[idx_a]:.1f} W",
                 xy=(V_a[idx_a], P_a[idx_a]), fontsize=9,
                 textcoords="offset points", xytext=(10, 5),
                 arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
    ax2.annotate(f"MPP\n{P_b[idx_b]:.1f} W",
                 xy=(V_b[idx_b], P_b[idx_b]), fontsize=9,
                 textcoords="offset points", xytext=(10, -15),
                 arrowprops=dict(arrowstyle='->', color='red'), color='red')
    ax2.set_xlabel("Tension V (V)", fontsize=12)
    ax2.set_ylabel("Puissance P (W)", fontsize=12)
    ax2.set_title("Caractéristique P = f(V)", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    st.pyplot(fig)

    # ── Tableau récapitulatif ──
    st.markdown("### 📊 Résumé")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("Pmax A", f"{P_a[idx_a]:.1f} W")
        st.metric("Vmpp A", f"{V_a[idx_a]:.1f} V")
        st.metric("Impp A", f"{I_a[idx_a]:.2f} A")
        ff_a = P_a[idx_a] / (Voc * Isc) * 100
        st.metric("Fill Factor A", f"{ff_a:.1f} %")
    with col_r2:
        st.metric("Pmax B", f"{P_b[idx_b]:.1f} W")
        st.metric("Vmpp B", f"{V_b[idx_b]:.1f} V")
        st.metric("Impp B", f"{I_b[idx_b]:.2f} A")
        ff_b = P_b[idx_b] / (Voc * Isc) * 100
        st.metric("Fill Factor B", f"{ff_b:.1f} %")

except Exception as e:
    st.error(f"Erreur de calcul : {e}")
    st.exception(e)
