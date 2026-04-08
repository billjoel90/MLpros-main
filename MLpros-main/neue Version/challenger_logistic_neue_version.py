import numpy as np

# --------------------------------------------------
# 1) Challenger-Datensatz
# --------------------------------------------------
temp = np.array([
    53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70,
    70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81
], dtype=float)

damaged = np.array([
    2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0
], dtype=int)

# Binäre Zielvariable:
# 1 = mindestens ein O-Ring beschädigt
# 0 = kein O-Ring beschädigt
y = (damaged > 0).astype(float)

# --------------------------------------------------
# 2) Standardisierung der Temperatur
# --------------------------------------------------
mu = temp.mean()
sigma_x = temp.std()

x_std = (temp - mu) / sigma_x

# Designmatrix mit Intercept
X = np.column_stack([np.ones(len(x_std)), x_std])

# --------------------------------------------------
# 3) Sigmoid-Funktion
# --------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)   # numerische Stabilität
    return 1.0 / (1.0 + np.exp(-z))

# --------------------------------------------------
# 4) Kostenfunktion (log-loss)
# --------------------------------------------------
def loss(theta, X, y):
    p = sigmoid(X @ theta)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

# --------------------------------------------------
# 5) Gradient der Kostenfunktion
# --------------------------------------------------
def gradient(theta, X, y):
    p = sigmoid(X @ theta)
    return (X.T @ (p - y)) / len(y)

# --------------------------------------------------
# 6) Gradientenverfahren
# --------------------------------------------------
def gradient_descent(X, y, lr=0.1, max_iter=50000, tol=1e-12):
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

# Training
theta_std, history = gradient_descent(X, y, lr=0.1, max_iter=50000, tol=1e-12)

# --------------------------------------------------
# 7) Koeffizienten zurück auf Originalskala
# --------------------------------------------------
a_std, b_std = theta_std

b = b_std / sigma_x
a = a_std - (b_std * mu / sigma_x)

# finale Loss auf standardisierten Daten
final_loss = loss(theta_std, X, y)

# --------------------------------------------------
# 8) Ausgabe
# --------------------------------------------------
print("Ergebnisse der eigenen logistischen Regression")
print("----------------------------------------------")
print(f"Intercept (Originalskala): {a:.6f}")
print(f"Koeffizient Temperatur:    {b:.6f}")
print(f"Finale Lossfunktion:       {final_loss:.6f}")
print(f"Anzahl Iterationen:        {len(history)}")

# --------------------------------------------------
# 9) Vorhersagefunktion
# --------------------------------------------------
def predict_proba(temp_values, a, b):
    z = a + b * temp_values
    return sigmoid(z)

# Beispiel: Ausfallwahrscheinlichkeit bei 31°F
t_test = 31
p_test = predict_proba(np.array([t_test]), a, b)[0]
print(f"\nGeschätzte Ausfallwahrscheinlichkeit bei {t_test}°F: {p_test:.6f}")
