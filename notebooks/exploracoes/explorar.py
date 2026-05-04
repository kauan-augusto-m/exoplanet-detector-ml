import lightkurve as lk
import matplotlib.pyplot as plt

resultado = lk.search_lightcurve("Kepler-22", mission="Kepler")
colecao = resultado.download_all()
lc = colecao.stitch().remove_outliers(sigma=3).normalize()
# BLS = Box Least Squares, algoritmo que detecta quedas periódicas
periodograma = lc.to_periodogram(method="bls", period=range(100, 400))
periodo_encontrado = periodograma.period_at_max_power
epoch_encontrado = periodograma.transit_time_at_max_power

print(f"Período encontrado: {periodo_encontrado:.2f}")
print(f"Epoch encontrado: {epoch_encontrado}")
lc_fold = lc.fold(period=periodo_encontrado, epoch_time=epoch_encontrado)
lc_bin = lc_fold.bin(time_bin_size=0.5)

lc_bin.plot()
plt.title("Trânsito do Kepler-22b")
plt.show()