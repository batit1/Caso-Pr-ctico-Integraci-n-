import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS DEL PROBLEMA
# =============================================================================
R = 4.10
a = 0.55
rho0 = 0.1175

# =============================================================================
# ACTIVIDAD 1 — FORMULACIÓN
# =============================================================================
# La integral a calcular es:
#   I = ∫₀^∞  r² / (1 + exp((r-R)/a))  dr
#
# En NumPy se expresa como:

def g(r):
    """Función del integrando."""
    return r**2 / (1 + np.exp((r - R) / a))

# El número de nucleones es:  N = 4π·ρ₀·I

# Intervalo adecuado: la función g(r) → 0 para r >> R (asíntota horizontal).
# La cola derecha decae rápidamente para r > R + varios·a.
# Como R=4.10 y a=0.55, para r > 10 el integrando es prácticamente cero.
# → Truncamos el límite superior en b = 10 (se justifica en Actividad 2).

print("=" * 60)
print("ACTIVIDAD 1 — FORMULACIÓN")
print("=" * 60)
print(f"g(r) = r² / (1 + exp((r - {R}) / {a}))")
print(f"Parámetros: R={R}, a={a}, ρ₀={rho0}")
print("La integral de volumen es:  ∫_V ρ(r) dV = 4π·ρ₀·∫₀^b g(r) dr")
print()

# Gráfica de g(r)
r_vals = np.linspace(0, 12, 500)
plt.figure(figsize=(8, 4))
plt.plot(r_vals, g(r_vals), 'b-', linewidth=2)
plt.axvline(10, color='r', linestyle='--', label='b = 10 (truncamiento)')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.title('Función del integrando  g(r) = r² / (1 + exp((r−R)/a))')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('actividad1_grafica_g.png', dpi=120)
plt.show()
print("  → Gráfica guardada: actividad1_grafica_g.png\n")

# =============================================================================
# ACTIVIDAD 2 — ANÁLISIS DEL LÍMITE SUPERIOR (ASÍNTOTA HORIZONTAL)
# =============================================================================

def trapecio_simple(f, a_lim, b_lim):
    """Regla del trapecio simple sobre [a_lim, b_lim]."""
    return (b_lim - a_lim) * (f(a_lim) + f(b_lim)) / 2

def punto_medio_simple(f, a_lim, b_lim):
    """Regla del punto medio simple sobre [a_lim, b_lim]."""
    return (b_lim - a_lim) * f((a_lim + b_lim) / 2)

def trapecio_compuesto(f, a_lim, b_lim, n):
    """Regla del trapecio compuesta con n subintervalos."""
    x = np.linspace(a_lim, b_lim, n + 1)
    y = f(x)
    h = (b_lim - a_lim) / n
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)

def punto_medio_compuesto(f, a_lim, b_lim, n):
    """Regla del punto medio compuesta con n subintervalos."""
    h = (b_lim - a_lim) / n
    x_mid = a_lim + (np.arange(n) + 0.5) * h
    return h * np.sum(f(x_mid))

def simpson_simple(f, a_lim, b_lim):
    """Regla de Simpson simple sobre [a_lim, b_lim]."""
    m = (a_lim + b_lim) / 2
    return (b_lim - a_lim) / 6 * (f(a_lim) + 4*f(m) + f(b_lim))

def simpson_compuesto(f, a_lim, b_lim, n):
    """Regla de Simpson compuesta con n subintervalos (n debe ser par)."""
    if n % 2 != 0:
        n += 1
    x = np.linspace(a_lim, b_lim, n + 1)
    y = f(x)
    h = (b_lim - a_lim) / n
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

def abierta_dos_puntos(f, a_lim, b_lim):
    """Fórmula abierta a dos puntos (Newton-Cotes abierta de orden 1)."""
    h = (b_lim - a_lim) / 3
    return (b_lim - a_lim) / 2 * (f(a_lim + h) + f(a_lim + 2*h))

# --- Barrido del límite superior b desde 1 hasta 15 en pasos de 0.5 ---
print("=" * 60)
print("ACTIVIDAD 2 — ANÁLISIS DEL LÍMITE SUPERIOR")
print("=" * 60)

# Usamos integración compuesta (n=1000) como referencia de alta precisión
N_REF = 1000
b_values = np.arange(1, 15.5, 0.5)

I_trap   = []
I_pmedio = []

for b in b_values:
    I_trap.append(trapecio_compuesto(g, 0, b, N_REF))
    I_pmedio.append(punto_medio_compuesto(g, 0, b, N_REF))

print(f"\n{'b':>6}  {'Trapecio (N=1000)':>20}  {'Punto Medio (N=1000)':>22}")
print("-" * 55)
for b, it, ip in zip(b_values, I_trap, I_pmedio):
    print(f"{b:>6.1f}  {it:>20.8f}  {ip:>22.8f}")

# Gráfica convergencia con b
plt.figure(figsize=(8, 4))
plt.plot(b_values, I_trap,   'b-o', markersize=4, label='Trapecio compuesto')
plt.plot(b_values, I_pmedio, 'r-s', markersize=4, label='Punto medio compuesto')
plt.xlabel('Límite superior b')
plt.ylabel('Valor de la integral')
plt.title('Convergencia de la integral al aumentar b')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('actividad2_convergencia_b.png', dpi=120)
plt.show()
print("\n  → Gráfica guardada: actividad2_convergencia_b.png")

# Determinación del b adecuado (cambio relativo < 1e-6)
tol = 1e-6
b_elegido = None
for i in range(1, len(I_trap)):
    cambio_rel = abs(I_trap[i] - I_trap[i-1]) / abs(I_trap[i-1] + 1e-20)
    if cambio_rel < tol:
        b_elegido = b_values[i]
        break

print(f"\n  → Límite superior elegido: b = {b_elegido}")
print(f"    Justificación: para b ≥ {b_elegido} el cambio relativo entre pasos")
print(f"    consecutivos es < {tol:.0e}, indicando convergencia.")
print(f"    Coincide con r >> R+5a = {R + 5*a:.2f}, donde la función es ≈ 0.\n")

# Para el resto del práctico usamos b = 10
b = 10.0

# Comparación trapecio vs punto medio
print("\n  Comparación Trapecio vs Punto Medio:")
print("  - Ambos convergen al mismo valor de b ≈ 9-10.")
print("  - El punto medio compuesto converge ligeramente más rápido")
print("    porque tiene error O(h²) igual al trapecio pero con constante menor.")
print("  - El trapecio puede sobreestimar en la cola decreciente.")

# =============================================================================
# ACTIVIDAD 3 — FÓRMULAS SIMPLES (b = 10)
# =============================================================================
print()
print("=" * 60)
print("ACTIVIDAD 3 — FÓRMULAS SIMPLES  (b = 10)")
print("=" * 60)

I_trap_s   = trapecio_simple(g, 0, b)
I_simp_s   = simpson_simple(g, 0, b)
I_pmedio_s = punto_medio_simple(g, 0, b)
I_ab2p     = abierta_dos_puntos(g, 0, b)

# Valor de referencia con Simpson compuesto N=100000
I_ref = simpson_compuesto(g, 0, b, 100000)

print(f"\n  Valor de referencia (Simpson compuesto, N=100000): {I_ref:.10f}")
print()
print(f"  {'Fórmula':<28}  {'Resultado':>14}  {'Error absoluto':>16}")
print("  " + "-" * 62)

resultados_simples = [
    ("Trapecio simple",          I_trap_s),
    ("Simpson simple",           I_simp_s),
    ("Punto medio simple",       I_pmedio_s),
    ("Abierta dos puntos",       I_ab2p),
]

for nombre, val in resultados_simples:
    err = abs(val - I_ref)
    print(f"  {nombre:<28}  {val:>14.6f}  {err:>16.6f}")

print()
print("  ANÁLISIS:")
print("  · Fórmulas cerradas (trapecio, Simpson): usan los extremos del intervalo.")
print("    El trapecio es O(h³) y Simpson O(h⁵) en un solo intervalo.")
print("  · Fórmulas abiertas (punto medio, a 2 puntos): no usan los extremos.")
print("    Útiles cuando los extremos son singulares o difíciles de evaluar.")
print("  · Simpson simple es la más precisa de las cerradas (orden más alto).")
print("  · El punto medio simple sorprende: su error es similar al trapecio")
print("    porque ambos tienen el mismo orden de error O(h³), pero con distinto")
print("    signo: ε_trap ≈ -2·ε_pmedio (los errores se cancelan en Simpson).")
print("  · La fórmula abierta a dos puntos es menos precisa que el punto medio")
print("    porque usa dos puntos interiores en lugar del punto central óptimo.")

# =============================================================================
# ACTIVIDAD 4 — CONVERGENCIA EN INTEGRACIÓN COMPUESTA
# =============================================================================
print()
print("=" * 60)
print("ACTIVIDAD 4 — CONVERGENCIA EN INTEGRACIÓN COMPUESTA  (b = 10)")
print("=" * 60)

potencias = [0, 1, 2, 3, 4, 5]
N_vals    = [10**p for p in potencias]

I_simp_comp  = []
I_pmedio_comp = []

for N in N_vals:
    I_simp_comp.append(simpson_compuesto(g, 0, b, N if N >= 2 else 2))
    I_pmedio_comp.append(punto_medio_compuesto(g, 0, b, N if N >= 1 else 1))

# Mejor aproximación = mayor N de Simpson
I_verdadero = I_simp_comp[-1]

print(f"\n  Valor 'verdadero' (referencia): {I_verdadero:.12f}")
print()
print(f"  {'N':>8}  {'Simpson comp.':>16}  {'ε_simp':>14}  {'Pmedio comp.':>16}  {'ε_pmedio':>14}")
print("  " + "-" * 76)

for i, N in enumerate(N_vals):
    err_s = abs(I_simp_comp[i]  - I_verdadero)
    err_p = abs(I_pmedio_comp[i] - I_verdadero)
    print(f"  {N:>8}  {I_simp_comp[i]:>16.10f}  {err_s:>14.2e}  {I_pmedio_comp[i]:>16.10f}  {err_p:>14.2e}")

print()
print("  VERIFICACIÓN DE LAS RELACIONES DE ERROR:")

# Simpson compuesto: ε_comp ≈ ε_simple / N^4
print("\n  Simpson compuesto  → ε_comp ≈ ε_simple / N⁴")
err_simp_simple = abs(I_simp_s - I_verdadero)
print(f"    ε_simple (Simpson) = {err_simp_simple:.4e}")
for i, N in enumerate(N_vals[1:], start=1):
    err_c = abs(I_simp_comp[i] - I_verdadero)
    predicho = err_simp_simple / N**4
    print(f"    N={N:>6}: ε_comp={err_c:.3e}  predicho={predicho:.3e}  ratio={err_c/predicho:.3f}" if predicho > 0 else "")

# Punto medio compuesto: ε_comp ≈ ε_simple / N^2
print("\n  Punto medio compuesto → ε_comp ≈ ε_simple / N²")
err_pm_simple = abs(I_pmedio_s - I_verdadero)
print(f"    ε_simple (P.Medio)  = {err_pm_simple:.4e}")
for i, N in enumerate(N_vals[1:], start=1):
    err_c = abs(I_pmedio_comp[i] - I_verdadero)
    predicho = err_pm_simple / N**2
    print(f"    N={N:>6}: ε_comp={err_c:.3e}  predicho={predicho:.3e}  ratio={err_c/predicho:.3f}" if predicho > 0 else "")

# Gráfica de convergencia
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

N_plot = N_vals[1:]  # excluir N=1 para la gráfica log-log
err_s_plot = [abs(I_simp_comp[i]  - I_verdadero) for i in range(1, len(N_vals))]
err_p_plot = [abs(I_pmedio_comp[i] - I_verdadero) for i in range(1, len(N_vals))]

axes[0].loglog(N_plot, err_s_plot, 'b-o', label='Simpson compuesto')
axes[0].loglog(N_plot, [err_simp_simple / n**4 for n in N_plot], 'b--', label='~1/N⁴')
axes[0].set_xlabel('N (número de subintervalos)')
axes[0].set_ylabel('Error absoluto')
axes[0].set_title('Convergencia Simpson compuesto')
axes[0].legend()
axes[0].grid(True, which='both')

axes[1].loglog(N_plot, err_p_plot, 'r-s', label='Punto medio compuesto')
axes[1].loglog(N_plot, [err_pm_simple / n**2 for n in N_plot], 'r--', label='~1/N²')
axes[1].set_xlabel('N (número de subintervalos)')
axes[1].set_ylabel('Error absoluto')
axes[1].set_title('Convergencia Punto Medio compuesto')
axes[1].legend()
axes[1].grid(True, which='both')

plt.tight_layout()
plt.savefig('actividad4_convergencia.png', dpi=120)
plt.show()
print("\n  → Gráfica guardada: actividad4_convergencia.png")

# =============================================================================
# RESULTADO FÍSICO FINAL
# =============================================================================
print()
print("=" * 60)
print("RESULTADO FÍSICO — Número de nucleones del Ca-40")
print("=" * 60)
N_nucleones = 4 * np.pi * rho0 * I_verdadero
print(f"  I = {I_verdadero:.8f}")
print(f"  N = 4π·ρ₀·I = 4π × {rho0} × {I_verdadero:.6f}")
print(f"  N ≈ {N_nucleones:.4f}  nucleones")
print(f"\n  El valor esperado para Ca-40 es 40 nucleones.")
print(f"  Diferencia: {abs(N_nucleones - 40):.4f}  ({abs(N_nucleones-40)/40*100:.2f}%)")
print()
print("  Nota: la pequeña diferencia se debe al truncamiento del límite")
print("  superior y a los parámetros aproximados del modelo de Fermi.")
