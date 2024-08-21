import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Parámetros iniciales
theta = np.pi / 3  # Ángulo del cono
h2 = 4  # Cuadrado del momento angular Lz
energia = 10  # Energía

# Función de energía potencial
f = lambda r: h2 / (2 * r*2 * np.sin(theta)*2) + 9.8 * r * np.cos(theta)
r = np.linspace(0.2, 3, 400)
plt.plot(r, f(r), color='r')
plt.ylim([0, 20])
plt.axhline(y=energia, color='k')

# Encontrar raíces
g = lambda r: f(r) - energia
r1 = fsolve(g, 0.4)[0]
r2 = fsolve(g, 2)[0]
plt.axvline(x=r1, ymin=0, ymax=energia/20, linestyle='--')
plt.axvline(x=r2, ymin=0, ymax=energia/20, linestyle='--')

plt.grid(True)
plt.xlabel('r')
plt.ylabel('V(r)')
plt.title('Energía potencial')
plt.show()

# Definición de nuevos parámetros
r1 = 4
r2 = 4
theta = np.pi / 6  # Nuevo ángulo del cono
h2 = 2 * 9.8 * r1*2 * r2 * np.sin(theta)*2 * np.cos(theta) / (r1 + r2)
x0 = [r1, 0, 0]  # Condiciones iniciales

# Ecuaciones diferenciales
def fg(t, x):
    dxdt = [x[1], 
            h2 / (x[0]*3 * np.sin(theta)*2) - 9.8 * np.cos(theta), 
            np.sqrt(h2) / (x[0]*2 * np.sin(theta)*2)]
    return dxdt

t_span = [0, 20]
sol = solve_ivp(fg, t_span, x0, t_eval=np.linspace(0, 20, 400))

xp = sol.y[0] * np.cos(sol.y[2]) * np.sin(theta)
yp = sol.y[0] * np.sin(sol.y[2]) * np.sin(theta)
zp = sol.y[0] * np.cos(theta)

# Graficar superficie cónica y trayectoria
phi = np.linspace(0, 2 * np.pi, 40)
r = np.linspace(0, 4)
phi, r = np.meshgrid(phi, r)
x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color=[0.6, 0.6, 0.6], alpha=0.5)

# Graficar trayectoria
ax.plot(xp, yp, zp, color=[.7, 0, 0], linewidth=1.5)

plt.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Movimiento en una superficie cónica')
plt.show()