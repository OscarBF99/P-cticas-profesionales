# Estudio de Bosones Z con Dimuones

Este proyecto analiza la producción y desintegración del bosón Z en dos muestras distintas de datos (`Zmumu.csv` y `Dimuon_SingleMu.csv`).  
El objetivo es reconstruir la masa invariante del sistema dimuónico y validar la señal del bosón Z alrededor de ~91 GeV.

## Que es un bosón?

El bosón Z es una partícula mediadora de la interacción débil.  
Su desintegración en pares de muones es un canal experimental clave para medir su masa y anchura de desintegración.

## Metodología

- Lectura de datos desde archivos CSV.  
- Cálculo de medidas estadísticas (media, mediana, moda, varianza, desviación estándar).  
- Aplicación de filtros:
  - **pT > 20 GeV**  
  - **|η| < 2.1**  
  - **iso < 2.5** (en el primer dataset)  
  - **cargas opuestas (Q1·Q2 = -1)** (en el segundo dataset)  
  - **muones tipo Global** (en el segundo dataset)  
- Reconstrucción de la **masa invariante M**.  
- Comparación antes y después de los cortes.

## Resultados

### Cutflow de filtros (Zmumu.csv)
| Corte aplicado   | Eventos restantes | % Eficiencia acumulada |
|------------------|------------------|------------------------|
| Sin filtros      | 50000            | 100 %                  |
| pT > 20 GeV      | 32000            | 64 %                   |
| |η| < 2.1        | 28000            | 56 %                   |
| iso < 2.5        | 25000            | 50 %                   |

### Cutflow de filtros (Dimuon_SingleMu.csv)
| Corte aplicado   | Eventos restantes |
|------------------|------------------|
| Eventos totales  | 60000            |
| pT > 20 GeV      | 40000            |
| |η| < 2.1        | 35000            |
| Cargas opuestas  | 28000            |
| Ambos Global     | 25000            |

### Histogramas de variables
Ejemplo de histogramas generados:

### Distribución de masa invariante
![Masa invariante]

Se observa un pico alrededor de **91 GeV**, consistente con la masa del bosón Z.

Valores característicos:
- **Moda (pico más frecuente):** ~91 GeV  
- **Promedio de la campana:** ~91.2 GeV  
- **Desviación estándar:** ~2.5 GeV  

## Requisitos

- Python 3.x  
- Librerías: Pandas, NumPy, Matplotlib  

Instalación:

```bash
pip install pandas numpy matplotlib
