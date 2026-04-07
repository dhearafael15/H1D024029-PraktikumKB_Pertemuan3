import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

barang_terjual = ctrl.Antecedent(np.arange(0, 101, 1), 'barang_terjual')
permintaan = ctrl.Antecedent(np.arange(0, 301, 1), 'permintaan')
harga = ctrl.Antecedent(np.arange(0, 100001, 1), 'harga')
profit = ctrl.Antecedent(np.arange(0, 4000001, 1), 'profit')
stok = ctrl.Consequent(np.arange(0, 1001, 1), 'stok')

# Membership functions
barang_terjual['rendah'] = fuzz.trimf(barang_terjual.universe, [0, 0, 50])
barang_terjual['sedang'] = fuzz.trimf(barang_terjual.universe, [30, 50, 70])
barang_terjual['tinggi'] = fuzz.trimf(barang_terjual.universe, [60, 100, 100])

permintaan['rendah'] = fuzz.trimf(permintaan.universe, [0, 0, 150])
permintaan['sedang'] = fuzz.trimf(permintaan.universe, [100, 150, 200])
permintaan['tinggi'] = fuzz.trimf(permintaan.universe, [180, 300, 300])

harga['murah'] = fuzz.trimf(harga.universe, [0, 0, 50000])
harga['sedang'] = fuzz.trimf(harga.universe, [30000, 50000, 70000])
harga['mahal'] = fuzz.trimf(harga.universe, [60000, 100000, 100000])

profit['rendah'] = fuzz.trimf(profit.universe, [0, 0, 1500000])
profit['sedang'] = fuzz.trimf(profit.universe, [1000000, 2000000, 3000000])
profit['tinggi'] = fuzz.trapmf(profit.universe, [2500000, 3000000, 4000000, 4000000])

stok['sedang'] = fuzz.trimf(stok.universe, [200, 500, 800])
stok['banyak'] = fuzz.trapmf(stok.universe, [600, 800, 1000, 1000])

# Rules
rule1 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['tinggi'] & harga['murah'] & profit['tinggi'], stok['banyak'])
rule2 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['tinggi'] & harga['murah'] & profit['sedang'], stok['sedang'])

stok_ctrl = ctrl.ControlSystem([rule1, rule2])
prediksi_stok = ctrl.ControlSystemSimulation(stok_ctrl)

prediksi_stok.input['barang_terjual'] = 80
prediksi_stok.input['permintaan'] = 255
prediksi_stok.input['harga'] = 25000
prediksi_stok.input['profit'] = 3500000

prediksi_stok.compute()
print(f"Hasil Prediksi Stok: {prediksi_stok.output['stok']:.2f}")

import matplotlib.pyplot as plt

barang_terjual.view()
permintaan.view()
harga.view()
profit.view()
stok.view()

stok.view(sim=prediksi_stok)

plt.show()
plt.figure(figsize=(10, 8))