import numpy as np
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------
# 1) Challenger-Daten
# Temperatur in Fahrenheit, damaged = Anzahl beschädigter O-Ringe
# Zielvariable y: 1, falls damaged > 0, sonst 0
# ------------------------------------------------------------
temp = np.array([
    53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70,
    70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81
], dtype=float)

damaged = np.array([
    2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0
], dtype=int)

y = (damaged > 0).astype(float)   # binäre Zielvariable

# ------------------------------------------------------------
# 2) Standardisierung für stabileres Gradient Descent
#    Wir rechnen intern mit x_std, wandeln Koeffizienten am Ende
#    aber wieder auf die Originalskala zurück.
# ------------------------------------------------------------
mu = temp.mean()
sigma_x = temp.std()

x_std = (temp - mu) / sigma_x
X = np.column_stack([np.ones(len(x_std)), x_std])   # [1, x_std]

# ------------------------------------------------------------
# 3) Sigmoid, Kostenfunktion, Gradient
# ------------------------------------------------------------
def sigmoid(z):
    # numerisch stabile Version
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def loss(theta, X, y):
    """
    Mittlere logistische Verlustfunktion (negative log-likelihood)
    J(theta) = -(1/n) * sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ]
    """
    p = sigmoid(X @ theta)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

def gradient(theta, X, y):
    """
    Gradient der mittleren logistischen Verlustfunktion
    grad J(theta) = (1/n) * X^T (p - y)
    """
    p = sigmoid(X @ theta)
    return (X.T @ (p - y)) / len(y)

# ------------------------------------------------------------
# 4) Gradient Descent
# ------------------------------------------------------------
def gradient_descent(X, y, lr=0.1, max_iter=50000, tol=1e-10):
    theta = np.zeros(X.shape[1])
    history = []

    for i in range(max_iter):
        grad = gradient(theta, X, y)
        theta_new = theta - lr * grad

        current_loss = loss(theta_new, X, y)
        history.append(current_loss)

        if np.linalg.norm(theta_new - theta, ord=2) < tol:
            theta = theta_new
            break

        theta = theta_new

    return theta, np.array(history)

theta_std, history = gradient_descent(X, y, lr=0.1, max_iter=50000, tol=1e-12)

# ------------------------------------------------------------
# 5) Koeffizienten zurück auf Originalskala der Temperatur
#    Modell auf Standardskala:
#       logit(p) = a_std + b_std * ((temp - mu)/sigma_x)
#
#    Auf Originalskala:
#       logit(p) = a + b * temp
#    mit
#       b = b_std / sigma_x
#       a = a_std - b_std * mu / sigma_x
# ------------------------------------------------------------
a_std, b_std = theta_std
b = b_std / sigma_x
a = a_std - b_std * mu / sigma_x

print("Eigene Implementierung")
print("----------------------")
print(f"Intercept (Originalskala): {a:.12f}")
print(f"Steigung  (Originalskala): {b:.12f}")
print(f"Finale Lossfunktion:       {loss(theta_std, X, y):.12f}")
print(f"Anzahl Iterationen:        {len(history)}")

# ------------------------------------------------------------
# 6) Vergleich mit sklearn
#    Wichtig: penalty=None für unregularisierte logistische Regression
# ------------------------------------------------------------
X_sklearn = temp.reshape(-1, 1)

clf = LogisticRegression(
    penalty=None,       # unregularisiert; fairer Vergleich
    solver="lbfgs",
    fit_intercept=True,
    max_iter=10000
)
clf.fit(X_sklearn, y)

a_sklearn = clf.intercept_[0]
b_sklearn = clf.coef_[0, 0]

# sklearn-Loss auf denselben Daten berechnen
p_sklearn = clf.predict_proba(X_sklearn)[:, 1]
eps = 1e-12
loss_sklearn = -np.mean(y * np.log(p_sklearn + eps) + (1 - y) * np.log(1 - p_sklearn + eps))

print("\nsklearn")
print("-------")
print(f"Intercept:               {a_sklearn:.12f}")
print(f"Steigung:                {b_sklearn:.12f}")
print(f"Lossfunktion:            {loss_sklearn:.12f}")

print("\nDifferenzen")
print("-----------")
print(f"|Δ Intercept|:           {abs(a - a_sklearn):.12e}")
print(f"|Δ Steigung|:            {abs(b - b_sklearn):.12e}")
print(f"|Δ Loss|:                {abs(loss(theta_std, X, y) - loss_sklearn):.12e}")
