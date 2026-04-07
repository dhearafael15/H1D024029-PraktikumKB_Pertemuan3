import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

informasi = ctrl.Antecedent(np.arange(0, 101, 1), 'informasi')
persyaratan = ctrl.Antecedent(np.arange(0, 101, 1), 'persyaratan')
petugas = ctrl.Antecedent(np.arange(0, 101, 1), 'petugas')
sarpras = ctrl.Antecedent(np.arange(0, 101, 1), 'sarpras')
kepuasan = ctrl.Consequent(np.arange(0, 401, 1), 'kepuasan')

# Membership input
for var in [informasi, persyaratan, petugas, sarpras]:
    var['tidak_memuaskan'] = fuzz.trapmf(var.universe, [0, 0, 50, 70])
    var['cukup_memuaskan'] = fuzz.trimf(var.universe, [60, 75, 90])
    var['memuaskan'] = fuzz.trapmf(var.universe, [80, 90, 100, 100])

# Membership output
kepuasan['tidak_memuaskan'] = fuzz.trapmf(kepuasan.universe, [0, 0, 80, 120])
kepuasan['kurang_memuaskan'] = fuzz.trapmf(kepuasan.universe, [80, 120, 160, 200])
kepuasan['cukup_memuaskan'] = fuzz.trimf(kepuasan.universe, [160, 200, 240])
kepuasan['memuaskan'] = fuzz.trapmf(kepuasan.universe, [200, 240, 300, 340])
kepuasan['sangat_memuaskan'] = fuzz.trapmf(kepuasan.universe, [300, 340, 400, 400])

# Rules utama (punya kamu)
r1 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['tidak_memuaskan'] & sarpras['tidak_memuaskan'], kepuasan['tidak_memuaskan'])
r2 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['tidak_memuaskan'] & sarpras['cukup_memuaskan'], kepuasan['tidak_memuaskan'])
r3 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['tidak_memuaskan'] & sarpras['memuaskan'], kepuasan['tidak_memuaskan'])
r4 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['tidak_memuaskan'], kepuasan['tidak_memuaskan'])
r5 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['cukup_memuaskan'], kepuasan['tidak_memuaskan'])
r6 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['memuaskan'], kepuasan['cukup_memuaskan'])
r7 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['memuaskan'] & sarpras['tidak_memuaskan'], kepuasan['tidak_memuaskan'])
r8 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['memuaskan'] & sarpras['cukup_memuaskan'], kepuasan['cukup_memuaskan'])
r9 = ctrl.Rule(informasi['tidak_memuaskan'] & persyaratan['tidak_memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['cukup_memuaskan'])
r10 = ctrl.Rule(informasi['cukup_memuaskan'] & persyaratan['cukup_memuaskan'] & petugas['cukup_memuaskan'] & sarpras['memuaskan'], kepuasan['memuaskan'])
r11 = ctrl.Rule(informasi['cukup_memuaskan'] & persyaratan['cukup_memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['memuaskan'])
r12 = ctrl.Rule(informasi['cukup_memuaskan'] & persyaratan['memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['sangat_memuaskan'])
r13 = ctrl.Rule(informasi['memuaskan'] & persyaratan['memuaskan'] & petugas['memuaskan'] & sarpras['memuaskan'], kepuasan['sangat_memuaskan'])

# 🔥 RULE TAMBAHAN (BIAR PASTI ADA OUTPUT)
r14 = ctrl.Rule(
    informasi['cukup_memuaskan'] |
    persyaratan['cukup_memuaskan'] |
    petugas['cukup_memuaskan'] |
    sarpras['cukup_memuaskan'],
    kepuasan['cukup_memuaskan']
)

r15 = ctrl.Rule(
    informasi['memuaskan'] &
    sarpras['memuaskan'],
    kepuasan['memuaskan']
)

# Control system
kepuasan_ctrl = ctrl.ControlSystem([r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15])
prediksi_kepuasan = ctrl.ControlSystemSimulation(kepuasan_ctrl)

# Input
prediksi_kepuasan.input['informasi'] = 80
prediksi_kepuasan.input['persyaratan'] = 60
prediksi_kepuasan.input['petugas'] = 50
prediksi_kepuasan.input['sarpras'] = 90

# Compute
prediksi_kepuasan.compute()

# Output
print(f"Tingkat Kepuasan Pelayanan: {prediksi_kepuasan.output['kepuasan']:.2f}")

# Grafik
kepuasan.view(sim=prediksi_kepuasan)
plt.show()