# %%
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep


# %%
file = uproot.open("C:/Users/HP/Documents/Practica Profesional/Analisis BosonHiggs/Data Sets/Run2012B_DoubleMuParked.root")
tree = file['Events']

# Listar ramas (variables físicas)

print(tree.keys())
tree.keys()
print (['nMuon', 'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass', 'Muon_charge','Muon_pfRelIso04_all', 'Muon_dxy', 'Muon_dz', 'Muon_dxyErr', 'Muon_dzErr'])

# Archivos de simulación Monte Carlo 
# Señal: H → ZZ* → 4μ
file_higgs = uproot.open("C:/Users/HP/Documents/Practica Profesional/Analisis BosonHiggs/Data Sets/SMHiggsToZZTo4L.root")
tree_higgs = file_higgs['Events']

# Fondo: ZZ → 4μ
file_zz = uproot.open("C:/Users/HP/Documents/Practica Profesional/Analisis BosonHiggs/Data Sets/ZZTo4mu.root")
tree_zz = file_zz['Events']

# Datos reales Run2012C 
file_c = uproot.open("C:/Users/HP/Documents/Practica Profesional/Analisis BosonHiggs/Data Sets/Run2012C_DoubleMuParked.root")
tree_c = file_c['Events']



# %%


def leer_arbol_a_df(tree, ramas):
    partes = []
    for batch in tree.iterate(ramas, step_size="100 MB", library="ak"):
        # Filtrar eventos con al menos 4 muones
        mask = ak.num(batch["Muon_pt"]) >= 4
        filtrado = batch[mask]
        
        # Tomar exactamente los 4 muones principales por evento
        m1, m2, m3, m4 = filtrado[:, 0], filtrado[:, 1], filtrado[:, 2], filtrado[:, 3]
        
        # Una columna por muón (una fila = un evento)
        d = {
            "pt1": ak.to_numpy(m1["Muon_pt"]),   "eta1": ak.to_numpy(m1["Muon_eta"]),
            "phi1": ak.to_numpy(m1["Muon_phi"]),  "charge1": ak.to_numpy(m1["Muon_charge"]),
            "pt2": ak.to_numpy(m2["Muon_pt"]),   "eta2": ak.to_numpy(m2["Muon_eta"]),
            "phi2": ak.to_numpy(m2["Muon_phi"]),  "charge2": ak.to_numpy(m2["Muon_charge"]),
            "pt3": ak.to_numpy(m3["Muon_pt"]),   "eta3": ak.to_numpy(m3["Muon_eta"]),
            "phi3": ak.to_numpy(m3["Muon_phi"]),  "charge3": ak.to_numpy(m3["Muon_charge"]),
            "pt4": ak.to_numpy(m4["Muon_pt"]),   "eta4": ak.to_numpy(m4["Muon_eta"]),
            "phi4": ak.to_numpy(m4["Muon_phi"]),  "charge4": ak.to_numpy(m4["Muon_charge"]),
        }
        if "Muon_pfRelIso04_all" in ramas:
            d.update({
                "iso1": ak.to_numpy(m1["Muon_pfRelIso04_all"]),
                "dxy1": ak.to_numpy(m1["Muon_dxy"]), "dz1": ak.to_numpy(m1["Muon_dz"]),
                "dxyErr1": ak.to_numpy(m1["Muon_dxyErr"]), "dzErr1": ak.to_numpy(m1["Muon_dzErr"]),
                "iso2": ak.to_numpy(m2["Muon_pfRelIso04_all"]),
                "dxy2": ak.to_numpy(m2["Muon_dxy"]), "dz2": ak.to_numpy(m2["Muon_dz"]),
                "dxyErr2": ak.to_numpy(m2["Muon_dxyErr"]), "dzErr2": ak.to_numpy(m2["Muon_dzErr"]),
                "iso3": ak.to_numpy(m3["Muon_pfRelIso04_all"]),
                "dxy3": ak.to_numpy(m3["Muon_dxy"]), "dz3": ak.to_numpy(m3["Muon_dz"]),
                "dxyErr3": ak.to_numpy(m3["Muon_dxyErr"]), "dzErr3": ak.to_numpy(m3["Muon_dzErr"]),
                "iso4": ak.to_numpy(m4["Muon_pfRelIso04_all"]),
                "dxy4": ak.to_numpy(m4["Muon_dxy"]), "dz4": ak.to_numpy(m4["Muon_dz"]),
                "dxyErr4": ak.to_numpy(m4["Muon_dxyErr"]), "dzErr4": ak.to_numpy(m4["Muon_dzErr"]),
            })
        partes.append(pd.DataFrame(d))
    return pd.concat(partes, ignore_index=True)

ramas_datos = ['Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_charge',
               'Muon_pfRelIso04_all', 'Muon_dxy', 'Muon_dz', 'Muon_dxyErr', 'Muon_dzErr']
ramas_mc    = ['Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_charge']

print("Leyendo Run2012B...")
df_b = leer_arbol_a_df(tree,   ramas_datos)
print("Leyendo Run2012C...")
df_c = leer_arbol_a_df(tree_c, ramas_datos)
df   = pd.concat([df_b, df_c], ignore_index=True)
del df_b, df_c

print("Leyendo Higgs MC...")
df_higgs_raw = leer_arbol_a_df(tree_higgs, ramas_mc)
print("Leyendo ZZ MC...")
df_zz_raw    = leer_arbol_a_df(tree_zz,    ramas_mc)

print("¡Listo!")
print(f"df: {len(df):,} filas | higgs: {len(df_higgs_raw):,} | zz: {len(df_zz_raw):,}")

# %%
# Separar columnas para simulaciones MC 
def preparar_mc(df_raw):
    # CAMBIO 1: Usamos "pt1" para hacer la copia inicial porque "Muon_pt" ya no existe como lista
    df = df_raw[df_raw["pt1"].notna()].copy()
    
    # ------------------------------------------------------------------
    # CAMBIO 2: SE ELIMINÓ EL CICLO FOR CON LOS .apply(lambda)
    # ¿Por qué? Porque 'pt1', 'eta1', 'charge1', etc., ya fueron creados 
    # y separados de forma ultra rápida en tu celda anterior.
    # ------------------------------------------------------------------
    
    # CONSERVADO: Filtro de carga neta cero (2 pares de muones de carga opuesta)
    df = df[(df["charge1"]+df["charge2"]+df["charge3"]+df["charge4"]) == 0].copy()
    return df

# CONSERVADO: Llamadas a las funciones
df_higgs = preparar_mc(df_higgs_raw)
df_zz    = preparar_mc(df_zz_raw)

# CONSERVADO: Prints finales
print("Eventos Higgs MC:", len(df_higgs))
print("Eventos ZZ MC:   ", len(df_zz))

# %%
# Dividimos las listas en columnas individuales para 4 muones

df_4mu = df[df["pt1"].notna()].copy()

# Muón 1

df_4mu['pt1'] = df_4mu['pt1']
df_4mu['eta1'] = df_4mu['eta1']
df_4mu['phi1'] = df_4mu['phi1']
df_4mu['charge1'] = df_4mu['charge1']
df_4mu['iso1'] = df_4mu['iso1'] 
df_4mu['dxy1'] = df_4mu['dxy1']       
df_4mu['dz1'] = df_4mu['dz1']
df_4mu['dzErr1'] = df_4mu['dzErr1']
df_4mu['dxyErr1'] = df_4mu['dxyErr1'] 

# Muón 2

df_4mu['pt2'] = df_4mu['pt2']
df_4mu['eta2'] = df_4mu['eta2']
df_4mu['phi2'] = df_4mu['phi2']
df_4mu['charge2'] = df_4mu['charge2']
df_4mu['iso2'] = df_4mu['iso2']
df_4mu['dxy2'] = df_4mu['dxy2']
df_4mu['dz2'] = df_4mu['dz2']
df_4mu['dxyErr2'] = df_4mu['dxyErr2']
df_4mu['dzErr2'] = df_4mu['dzErr2']

# Muón 3 (¡Nuevo para el Higgs!)

df_4mu['pt3'] = df_4mu['pt3']
df_4mu['eta3'] = df_4mu['eta3']
df_4mu['phi3'] = df_4mu['phi3']
df_4mu['charge3'] = df_4mu['charge3']
df_4mu['iso3'] = df_4mu['iso3']
df_4mu['dxy3'] = df_4mu['dxy3']
df_4mu['dz3'] = df_4mu['dz3']
df_4mu['dxyErr3'] = df_4mu['dxyErr3']
df_4mu['dzErr3'] = df_4mu['dzErr3']

# Muón 4 

df_4mu['pt4'] = df_4mu['pt4']
df_4mu['eta4'] = df_4mu['eta4']
df_4mu['phi4'] = df_4mu['phi4']
df_4mu['charge4'] = df_4mu['charge4']
df_4mu['iso4'] = df_4mu['iso4']
df_4mu['dxy4'] = df_4mu['dxy4']
df_4mu['dz4'] = df_4mu['dz4']
df_4mu['dxyErr4'] = df_4mu['dxyErr4']
df_4mu['dzErr4'] = df_4mu['dzErr4']

print("¡Listo! Datos reales estructurados en columnas individuales.")
print(f"Total de eventos listos para el análisis: {len(df_4mu):,}")


# %%
# Mostrar las nuevas columnas ("pt1","pt2","pt3","pt4","eta1","eta2","eta3","eta4","phi1","phi2","phi3","phi4","charge1","charge2","charge3","charge4","iso1", "iso2", "iso3", "iso4","dxyErr1", "dzErr1")

print(df_4mu[["pt1","pt2","pt3","pt4","eta1","eta2","eta3","eta4","phi1","phi2","phi3","phi4","charge1","charge2","charge3","charge4","iso1","iso2","iso3","iso4","dxyErr1","dzErr1"]].head())

# %%
# Valores únicos de las cargas (Para confirmar que hay muones + y -)

print("Valores únicos en charge1:", df_4mu["charge1"].unique())
print("Valores únicos en charge2:", df_4mu["charge2"].unique())
print("Valores únicos en charge3:", df_4mu["charge3"].unique())
print("Valores únicos en charge4:", df_4mu["charge4"].unique())

# %%
# Filtrar pares de muones con cargas opuestas 

df_pairs = df_4mu[(df_4mu["charge1"] + df_4mu["charge2"] + df_4mu["charge3"] + df_4mu["charge4"]) == 0].copy()


# %%
# Verificación de eventos totales con cuatro muones 

print("Eventos con cuatro muones:", len(df_4mu))
print("Eventos con cuatro muones y carga neta cero:", len(df_pairs))

# %%
# Tabla de flujo de cortes

# Calcular sip3d ← NUEVO: parámetro de impacto 3D significativo (skim.cxx)
for i in range(1, 5):
    ip3d = np.sqrt(df_pairs[f"dxy{i}"]**2 + df_pairs[f"dz{i}"]**2)
    sig  = np.sqrt(df_pairs[f"dxyErr{i}"]**2 + df_pairs[f"dzErr{i}"]**2)
    df_pairs[f"sip3d{i}"] = ip3d / sig

mask_pt  = (df_pairs["pt1"] > 5) & (df_pairs["pt2"] > 5) & (df_pairs["pt3"] > 5) & (df_pairs["pt4"] > 5)
mask_eta = (df_pairs["eta1"].abs() < 2.4) & (df_pairs["eta2"].abs() < 2.4) & (df_pairs["eta3"].abs() < 2.4) & (df_pairs["eta4"].abs() < 2.4)
mask_iso = (df_pairs["iso1"]<0.40) & (df_pairs["iso2"]<0.40) & (df_pairs["iso3"]<0.40) & (df_pairs["iso4"]<0.40)  
mask_sip = (df_pairs["sip3d1"]<4) & (df_pairs["sip3d2"]<4) & (df_pairs["sip3d3"]<4) & (df_pairs["sip3d4"]<4)      
mask_dxy = (df_pairs["dxy1"].abs()<0.5) & (df_pairs["dxy2"].abs()<0.5) & (df_pairs["dxy3"].abs()<0.5) & (df_pairs["dxy4"].abs()<0.5)  
mask_dz  = (df_pairs["dz1"].abs()<1.0)  & (df_pairs["dz2"].abs()<1.0)  & (df_pairs["dz3"].abs()<1.0)  & (df_pairs["dz4"].abs()<1.0)  

df_final = df_pairs[mask_pt & mask_eta & mask_iso & mask_sip & mask_dxy & mask_dz].copy()  

n_total = len(df_pairs)
n_pt    = len(df_pairs[mask_pt])
n_eta   = len(df_pairs[mask_pt & mask_eta])
n_iso   = len(df_pairs[mask_pt & mask_eta & mask_iso])    
n_sip   = len(df_pairs[mask_pt & mask_eta & mask_iso & mask_sip])  
n_final = len(df_final) 

data_flujo = {
    'Corte aplicado': [
        'Sin filtros (4 muones)',
        'pT Líderes (>5, 5, 5, 5 GeV)',
        'Aceptancia Detector (|eta| < 2.4)',
        'Aislamiento iso < 0.40',          
        'Parámetro impacto sip3d < 4',     
        '|dxy| < 0.5 y |dz| < 1.0'        
    ],
    'Eventos Restantes': [n_total, n_pt, n_eta, n_iso, n_sip, n_final],  
    'Aceptancia (%)': [
        100.0,
        (n_pt    / n_total) * 100,
        (n_eta   / n_total) * 100,
        (n_iso   / n_total) * 100,   
        (n_sip   / n_total) * 100,   
        (n_final / n_total) * 100    
    ]
}
tabla_flujo = pd.DataFrame(data_flujo)
print("\nTabla de Flujo de Cortes (Análisis de Higgs 4l)")
print(tabla_flujo)




# %%
# Estilo CMS 
hep.style.use("CMS")

# Párametros Físicos
m_mu = 0.10566  # Masa del muón [GeV/c²]

# Reconstrucción del 4-vector del sistema de 4 muones
def calcular_masa(df):
    px = 0; py = 0; pz = 0; E = 0
    for i in range(1, 5):
        pxi = df[f"pt{i}"] * np.cos(df[f"phi{i}"])
        pyi = df[f"pt{i}"] * np.sin(df[f"phi{i}"])
        pzi = df[f"pt{i}"] * np.sinh(df[f"eta{i}"])
        Ei  = np.sqrt(pxi**2 + pyi**2 + pzi**2 + m_mu**2)
        px += pxi; py += pyi; pz += pzi; E += Ei
    return np.sqrt(E**2 - (px**2 + py**2 + pz**2))

# Masa Invariante para los tres dataframes
df_pairs["M"] = calcular_masa(df_pairs)
df_higgs["M"] = calcular_masa(df_higgs)
df_zz["M"]    = calcular_masa(df_zz)

# Filtros datos reales — agregamos iso, sip3d, dxy, dz (skim.cxx)
mask_pt  = (df_pairs["pt1"]>5)  & (df_pairs["pt2"]>5)  & (df_pairs["pt3"]>5)  & (df_pairs["pt4"]>5)
mask_eta = (df_pairs["eta1"].abs()<2.4) & (df_pairs["eta2"].abs()<2.4) & (df_pairs["eta3"].abs()<2.4) & (df_pairs["eta4"].abs()<2.4)
mask_iso = (df_pairs["iso1"]<0.40) & (df_pairs["iso2"]<0.40) & (df_pairs["iso3"]<0.40) & (df_pairs["iso4"]<0.40) 
mask_sip = (df_pairs["sip3d1"]<4) & (df_pairs["sip3d2"]<4) & (df_pairs["sip3d3"]<4) & (df_pairs["sip3d4"]<4)      
mask_dxy = (df_pairs["dxy1"].abs()<0.5) & (df_pairs["dxy2"].abs()<0.5) & (df_pairs["dxy3"].abs()<0.5) & (df_pairs["dxy4"].abs()<0.5) 
mask_dz  = (df_pairs["dz1"].abs()<1.0)  & (df_pairs["dz2"].abs()<1.0)  & (df_pairs["dz3"].abs()<1.0)  & (df_pairs["dz4"].abs()<1.0)   

df_data_f = df_pairs[mask_pt & mask_eta & mask_iso & mask_sip & mask_dxy & mask_dz].copy() 

df_higgs_f = df_higgs[(df_higgs["pt1"]>5) & (df_higgs["pt2"]>5) &
                      (df_higgs["pt3"]>5) & (df_higgs["pt4"]>5) &
                      (df_higgs["eta1"].abs()<2.4) & (df_higgs["eta2"].abs()<2.4) &
                      (df_higgs["eta3"].abs()<2.4) & (df_higgs["eta4"].abs()<2.4)].copy()

df_zz_f = df_zz[(df_zz["pt1"]>5) & (df_zz["pt2"]>5) &
                (df_zz["pt3"]>5) & (df_zz["pt4"]>5) &
                (df_zz["eta1"].abs()<2.4) & (df_zz["eta2"].abs()<2.4) &
                (df_zz["eta3"].abs()<2.4) & (df_zz["eta4"].abs()<2.4)].copy()

print("Datos reales filtrados:", len(df_data_f))
print("Higgs MC filtrado:     ", len(df_higgs_f))
print("ZZ MC filtrado:        ", len(df_zz_f))

# Pesos oficiales del CERN reemplaza scale_higgs y scale_zz
lumi    = 11.58 * 1000
w_higgs = 0.0065 / 299973.0  * lumi
w_zz    = 0.077  / 1499064.0 * 1.386 * lumi

print(f"Peso Higgs: {w_higgs:.4f}")
print(f"Peso ZZ:    {w_zz:.4f}")

# Gráfico de la Distribución de la masa invariante
plt.figure(figsize=(10, 7))

# Fondo ZZ (azul)
plt.hist(df_zz_f["M"], bins=40, range=(70, 180), alpha=1.0,
         weights=np.full(len(df_zz_f), w_zz),      
         label="Z/gamma*, ZZ", color="#67C8F2", histtype='stepfilled')

# Señal Higgs (rojo)
plt.hist(df_higgs_f["M"], bins=40, range=(70, 180),
         weights=np.full(len(df_higgs_f), w_higgs),  
         label="mH = 125 GeV", color="red", histtype='step', linewidth=3)

# Datos reales (puntos negros)
counts, bin_edges = np.histogram(df_data_f["M"], bins=40, range=(70, 180))
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
plt.errorbar(bin_centers, counts, yerr=np.sqrt(counts),
             fmt='o', color='black', markersize=5,
             capsize=3, elinewidth=1, linestyle='none', label='Data')

# Formato CMS
plt.title("CMS Open Data", loc='left', fontweight='bold', fontsize=15)
plt.title("11.6 fb$^{-1}$ (8 TeV)", loc='right', fontsize=12)
plt.xlabel("Masa 4 leptones [GeV]", fontsize=14)
plt.ylabel("Eventos / bin", fontsize=14)
plt.xlim(70, 180)
plt.ylim(0, counts.max() * 1.3)
plt.legend(loc='upper right', frameon=False)
plt.grid(axis='y', alpha=0.3)
plt.show()


