import numpy as np

# 乱数の種を固定して、架空の測定値を100個作り、平均と標準偏差を表示する
rng = np.random.default_rng(20261029)
values = rng.normal(loc=10.0, scale=2.0, size=100)
print(f"n={values.size}  mean={values.mean():.4f}  sd={values.std(ddof=1):.4f}")
