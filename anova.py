import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv("anova.txt", sep="\t")
data.boxplot(column=["KME", "KMO", "SGA(R)", "SGA(C)", "MOGAFSC"], notch=1, grid=0, sym='b+',
             widths=(0.4, 0.4, 0.4, 0.4, 0.48))
plt.show()

fvalue, pvalue = stats.f_oneway(data['KME'], data['MOGAFSC'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(data['KMO'], data['MOGAFSC'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(data['SGA(R)'], data['MOGAFSC'])
print(fvalue, pvalue)
fvalue, pvalue = stats.f_oneway(data['SGA(C)'], data['MOGAFSC'])
print(fvalue, pvalue)
