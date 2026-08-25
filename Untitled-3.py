#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIGURACIÓN DEL SISTEMA
# ---------------------------------------------------------
L = 50         # Red de 50x50 para que los dominios visuales se vean espectaculares
N = L ** D if 'D' in locals() else L**2
T_fija = 1.0   # <--- CAMBIA ESTO: Usa 1.0 para fase ordenada y 4.0 para desordenada
steps = 1000   # Pasos de Monte Carlo para ver la evolución
# ---------------------------------------------------------

def INIT(L):
    """Genera una configuración inicial completamente aleatoria (alta energía)"""
    return np.random.choice([-1, 1], size=(L, L))

def METROPOLIS(spin, beta):
    """Algoritmo de Metropolis para un barrido completo de la red"""
    for _ in range(L**2):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        s = spin[x, y]
        
        # Condiciones de frontera periódicas
        R = (spin[(x + 1) % L, y] + spin[(x - 1) % L, y] +
             spin[x, (y + 1) % L] + spin[x, (y - 1) % L])
        
        dE = 2 * s * R
        if dE <= 0:
            spin[x, y] = -s
        elif np.random.rand() < np.exp(-dE * beta):
            spin[x, y] = -s
    return spin

# ---------------------------------------------------------
# PROCESO DE SIMULACIÓN Y VISUALIZACIÓN
# ---------------------------------------------------------
spin = INIT(L)
beta = 1.0 / T_fija

# Configurar la gráfica interactiva
plt.ion()  # Activa el modo interactivo para ver la animación "en vivo"
fig, ax = plt.subplots(figsize=(6, 6))

# Dibujar la malla inicial
# cmap='gray' mostrará los espines +1 como blanco y -1 como negro
img = ax.imshow(spin, cmap='gray', vmin=-1, vmax=1)
ax.set_title(f'Evolución de la Red a T = {T_fija:.2f}')
plt.axis('off')

print(f"Simulando evolución a T = {T_fija}... Observa la ventana de gráficos.")

for step in range(steps):
    METROPOLIS(spin, beta)
    
    # Actualizar la imagen en pantalla cada 5 pasos para que no vaya lento
    if step % 5 == 0:
        img.set_data(spin)
        ax.set_title(f'Modelo de Ising 2D ({L}x{L}) | Paso MC: {step} | T = {T_fija:.1f}')
        plt.draw()
        plt.pause(0.01)

plt.ioff() # Desactivar interactividad al terminar
print("Simulación terminada.")

# Guardar el estado final para tu reporte en LaTeX
nombre_imagen = f'malla_ising_T_{T_fija:.1f}.png'
plt.savefig(nombre_imagen, dpi=300, bbox_inches='tight')
print(f"Configuración final guardada como '{nombre_imagen}'")
plt.show()
