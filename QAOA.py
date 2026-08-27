import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


# ============================================================
# 1. PORTFOLIO DATA
# ============================================================

# Three stocks
# S1, S2, S3

mu = np.array([
    0.10,   # expected return S1
    0.15,   # expected return S2
    0.12    # expected return S3
])

# Covariance matrix
Sigma = np.array([
    [0.04,   0.01,   0.015],
    [0.01,   0.09,   0.02 ],
    [0.015,  0.02,   0.0625]
])

# Stock prices
prices = np.array([
    100,
    200,
    150
])

# Total available budget
budget = 250

# Number of stocks to select
K = 2

# Risk parameter
lam = 1.0

# Constraint penalties
P_K = 1.0
P_B = 1.0


# ============================================================
# 2. SCALE THE BUDGET
# ============================================================

# Divide all prices and budget by 50.
# This gives:
#
# prices = [2, 4, 3]
# budget = 5

scale = 50

p = prices // scale
B = budget // scale

print("Scaled prices:", p)
print("Scaled budget:", B)


# ============================================================
# 3. VARIABLES
# ============================================================

# Stock variables:
#
# x1, x2, x3
#
# Slack variables:
#
# s0, s1, s2
#
# s = s0 + 2*s1 + 4*s2
#
# Total = 6 binary variables
# Therefore = 6 qubits.

n_qubits = 6


# ============================================================
# 4. CONSTRUCT QUBO COEFFICIENTS
# ============================================================

# We represent the QUBO as
#
# C(x) = constant
#        + sum_i linear[i] * x_i
#        + sum_(i<j) quadratic[i,j] * x_i*x_j
#
# Variables:
#
# 0 -> x1
# 1 -> x2
# 2 -> x3
# 3 -> s0
# 4 -> s1
# 5 -> s2

linear = np.zeros(n_qubits)
quadratic = {}
constant = 0.0


# ============================================================
# 5. RETURN TERM
# ============================================================

# -mu^T x

for i in range(3):
    linear[i] += -mu[i]


# ============================================================
# 6. RISK TERM
# ============================================================

# lambda * x^T Sigma x
#
# Diagonal:
# Sigma_ii * xi^2 = Sigma_ii * xi
#
# Off diagonal:
# 2 Sigma_ij xi*xj

for i in range(3):

    # xi^2 = xi
    linear[i] += lam * Sigma[i, i]

    for j in range(i + 1, 3):

        coefficient = lam * 2.0 * Sigma[i, j]

        quadratic[(i, j)] = quadratic.get((i, j), 0.0) + coefficient


# ============================================================
# 7. EXACTLY K STOCKS CONSTRAINT
# ============================================================

# P_K * (x1 + x2 + x3 - K)^2
#
# General expansion:
#
# (sum xi - K)^2
#
# = sum xi^2
#   + 2 sum xi*xj
#   - 2K sum xi
#   + K^2
#
# Since xi^2 = xi:
#
# linear coefficient = 1 - 2K
#
# quadratic coefficient = 2
#
# constant = K^2

for i in range(3):

    linear[i] += P_K * (1 - 2 * K)

for i in range(3):
    for j in range(i + 1, 3):

        quadratic[(i, j)] = (
            quadratic.get((i, j), 0.0)
            + P_K * 2
        )

constant += P_K * K**2


# ============================================================
# 8. BUDGET CONSTRAINT
# ============================================================

# Budget:
#
# 2*x1 + 4*x2 + 3*x3 <= 5
#
# Introduce:
#
# s = s0 + 2*s1 + 4*s2
#
# Therefore:
#
# 2*x1 + 4*x2 + 3*x3
# + s0 + 2*s1 + 4*s2
# - 5 = 0
#
# Penalty:
#
# P_B * (...)^2

budget_coeff = np.array([
    2,   # x1
    4,   # x2
    3,   # x3
    1,   # s0
    2,   # s1
    4    # s2
])

target = B


# Add linear contribution
#
# (a^T z - B)^2
#
# = (a^T z)^2 - 2B(a^T z) + B^2
#
# For binary zi:
#
# linear coefficient = ai^2 - 2B ai

for i in range(n_qubits):

    a = budget_coeff[i]

    linear[i] += P_B * (
        a**2 - 2 * target * a
    )


# Add quadratic contribution
#
# 2 ai aj zi zj

for i in range(n_qubits):

    for j in range(i + 1, n_qubits):

        coefficient = (
            P_B
            * 2
            * budget_coeff[i]
            * budget_coeff[j]
        )

        quadratic[(i, j)] = (
            quadratic.get((i, j), 0.0)
            + coefficient
        )


# Constant B^2
constant += P_B * target**2


# ============================================================
# 9. PRINT QUBO
# ============================================================

print("\n================ QUBO ================")

print("Constant =", constant)

print("\nLinear coefficients:")

for i, value in enumerate(linear):

    print(f"x[{i}] : {value:.6f}")


print("\nQuadratic coefficients:")

for (i, j), value in quadratic.items():

    print(
        f"x[{i}] x[{j}] : {value:.6f}"
    )


# ============================================================
# 10. CONVERT QUBO -> ISING HAMILTONIAN
# ============================================================

# Mapping:
#
# x_i = (1 - Z_i)/2
#
# Linear term:
#
# a_i*x_i
#
# = a_i/2 - a_i/2 Z_i
#
#
# Quadratic term:
#
# b_ij*x_i*x_j
#
# = b_ij/4
#   (1 - Zi - Zj + ZiZj)


pauli_terms = {}
hamiltonian_constant = constant


def add_pauli(pauli_string, coefficient):

    pauli_terms[pauli_string] = (
        pauli_terms.get(pauli_string, 0.0)
        + coefficient
    )


# ------------------------------------------------------------
# Linear terms
# ------------------------------------------------------------

for i in range(n_qubits):

    a = linear[i]

    # Constant contribution
    hamiltonian_constant += a / 2

    # Z contribution
    pauli = ["I"] * n_qubits
    pauli[i] = "Z"

    add_pauli(
        "".join(pauli),
        -a / 2
    )


# ------------------------------------------------------------
# Quadratic terms
# ------------------------------------------------------------

for (i, j), b in quadratic.items():

    # Constant contribution
    hamiltonian_constant += b / 4

    # -Zi
    pauli_i = ["I"] * n_qubits
    pauli_i[i] = "Z"

    add_pauli(
        "".join(pauli_i),
        -b / 4
    )

    # -Zj
    pauli_j = ["I"] * n_qubits
    pauli_j[j] = "Z"

    add_pauli(
        "".join(pauli_j),
        -b / 4
    )

    # ZiZj
    pauli_ij = ["I"] * n_qubits
    pauli_ij[i] = "Z"
    pauli_ij[j] = "Z"

    add_pauli(
        "".join(pauli_ij),
        b / 4
    )


# ============================================================
# 11. CREATE QISKIT HAMILTONIAN
# ============================================================

hamiltonian_list = []

for pauli_string, coefficient in pauli_terms.items():

    if abs(coefficient) > 1e-12:

        hamiltonian_list.append(
            (pauli_string, coefficient)
        )


H_cost = SparsePauliOp.from_list(
    hamiltonian_list
)


print("\n================ HAMILTONIAN ================")

print(H_cost)

print("\nHamiltonian constant =", hamiltonian_constant)


# ============================================================
# 12. CLASSICAL BRUTE-FORCE CHECK
# ============================================================

print("\n================ VALID PORTFOLIOS ================")

best_cost = np.inf
best_portfolio = None

for x1 in [0, 1]:
    for x2 in [0, 1]:
        for x3 in [0, 1]:

            x = np.array([
                x1, x2, x3
            ])

            number_selected = np.sum(x)

            money = np.dot(
                prices,
                x
            )

            if number_selected != K:
                continue

            if money > budget:
                continue

            return_value = np.dot(
                mu,
                x
            )

            risk = (
                x.T
                @ Sigma
                @ x
            )

            cost = (
                -return_value
                + lam * risk
            )

            print(
                f"{x}  "
                f"cost = {cost:.6f}, "
                f"return = {return_value:.4f}, "
                f"risk = {risk:.6f}, "
                f"budget = ₹{money}"
            )

            if cost < best_cost:

                best_cost = cost
                best_portfolio = x.copy()


print("\nBest classical portfolio:")
print(best_portfolio)

print("Best cost:", best_cost)


# ============================================================
# 13. QAOA MIXER
# ============================================================

# Standard X mixer:
#
# H_M = X1 + X2 + ... + X6

def apply_mixer(qc, beta):

    for q in range(n_qubits):

        qc.rx(
            2 * beta,
            q
        )


# ============================================================
# 14. QAOA COST UNITARY
# ============================================================

# H_cost contains:
#
# h_i Zi
#
# and
#
# J_ij ZiZj
#
# The unitary is:
#
# U_C(gamma) = exp(-i gamma H_C)


def apply_cost_unitary(qc, gamma):

    # Single-qubit Z terms
    for pauli_string, coefficient in zip(
        H_cost.paulis.to_labels(),
        H_cost.coeffs
    ):

        # Remove identity-only term
        if "Z" in pauli_string:

            # Count Z operators
            z_positions = [
                i
                for i, p in enumerate(pauli_string)
                if p == "Z"
            ]

            # Single Z term
            if len(z_positions) == 1:

                q = z_positions[0]

                qc.rz(
                    2 * gamma * coefficient.real,
                    q
                )

    # Two-qubit ZZ terms
    for pauli_string, coefficient in zip(
        H_cost.paulis.to_labels(),
        H_cost.coeffs
    ):

        z_positions = [
            i
            for i, p in enumerate(pauli_string)
            if p == "Z"
        ]

        if len(z_positions) == 2:

            q1, q2 = z_positions

            qc.cx(q1, q2)

            qc.rz(
                2 * gamma * coefficient.real,
                q2
            )

            qc.cx(q1, q2)


# ============================================================
# 15. BUILD p = 1 QAOA CIRCUIT
# ============================================================

gamma = Parameter("γ")
beta = Parameter("β")

qc = QuantumCircuit(n_qubits)

# Initial state |+>^6
for q in range(n_qubits):

    qc.h(q)

# Cost layer
apply_cost_unitary(
    qc,
    gamma
)

# Mixer layer
apply_mixer(
    qc,
    beta
)

print("\n================ QAOA CIRCUIT ================")

print(qc.draw())


# ============================================================
# 16. NUMERICAL OPTIMIZATION
# ============================================================

from scipy.optimize import minimize


simulator = AerSimulator()


def expectation_value(params):

    gamma_value, beta_value = params

    circuit = qc.assign_parameters({
        gamma: gamma_value,
        beta: beta_value
    })

    # Add measurement
    measured = circuit.copy()

    measured.measure_all()

    result = simulator.run(
        measured,
        shots=4096
    ).result()

    counts = result.get_counts()

    expectation = 0.0

    # Evaluate classical energy of each bitstring
    for bitstring, count in counts.items():

        bits = bitstring.replace(" ", "")

        # Qiskit bit order is reversed
        bits = bits[::-1]

        z = np.array([
            1 if bit == "0" else -1
            for bit in bits
        ])

        energy = hamiltonian_constant

        # Linear
        for i in range(n_qubits):

            label = ["I"] * n_qubits
            label[i] = "Z"

            label = "".join(label)

            if label in pauli_terms:

                energy += (
                    pauli_terms[label]
                    * z[i]
                )

        # Quadratic
        for (i, j), b in quadratic.items():

            # Instead of reconstructing from Pauli
            # terms, directly evaluate QUBO.

            pass

        probability = count / 4096

        # Direct QUBO evaluation
        binary = np.array([
            0 if bit == "0" else 1
            for bit in bits
        ])

        qubo_energy = constant

        qubo_energy += np.dot(
            linear,
            binary
        )

        for (i, j), coeff in quadratic.items():

            qubo_energy += (
                coeff
                * binary[i]
                * binary[j]
            )

        expectation += (
            probability
            * qubo_energy
        )

    return expectation


# Try several starting points
best_result = None

for initial in [
    [0.1, 0.1],
    [0.5, 0.2],
    [1.0, 0.5],
    [1.5, 1.0]
]:

    result = minimize(
        expectation_value,
        initial,
        method="COBYLA",
        options={
            "maxiter": 50
        }
    )

    if (
        best_result is None
        or result.fun < best_result.fun
    ):

        best_result = result


print("\n================ OPTIMIZATION ================")

print("Optimal gamma =", best_result.x[0])

print("Optimal beta  =", best_result.x[1])

print("Expectation   =", best_result.fun)


# ============================================================
# 17. FINAL MEASUREMENT
# ============================================================

final_circuit = qc.assign_parameters({

    gamma: best_result.x[0],
    beta: best_result.x[1]

})

final_circuit.measure_all()

result = simulator.run(
    final_circuit,
    shots=10000
).result()

counts = result.get_counts()


print("\n================ MEASUREMENT ================")

sorted_counts = sorted(
    counts.items(),
    key=lambda x: x[1],
    reverse=True
)

for bitstring, count in sorted_counts[:15]:

    print(
        bitstring,
        " : ",
        count
    )


# ============================================================
# 18. DECODE THE MOST PROBABLE STATE
# ============================================================

best_bitstring = sorted_counts[0][0]

bits = best_bitstring.replace(" ", "")[::-1]

binary = np.array([
    int(bit)
    for bit in bits
])

stocks = binary[:3]

s0 = binary[3]
s1 = binary[4]
s2 = binary[5]

slack = (
    s0
    + 2*s1
    + 4*s2
)

print("\n================ SOLUTION ================")

print("Stock bits:", stocks)

print("Slack bits:", [s0, s1, s2])

print("Slack:", slack)

print(
    "Selected stocks:",
    [
        i + 1
        for i in range(3)
        if stocks[i] == 1
    ]
)

print(
    "Actual budget used:",
    np.dot(prices, stocks)
)

print(
    "Budget:",
    budget
)