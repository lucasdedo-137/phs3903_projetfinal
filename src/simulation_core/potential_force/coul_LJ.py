import numpy as np
import quaternion as qtn
from scipy.spatial.transform import Rotation

def build_potential_vector_force_torque_matrix(
    n: int,
    v,
    Lx: float,
    Ly: float,
    Lz: float,
    nbr_list,
    theta: float,
    r_oh: float,
    q_o: float,
    q_h: float,
    eps_LJ: float,
    sigma_LJ: float,
    k_coul: float
):
    """Calcule les potentiels, forces, torques, laplacien cartésien
    et laplacien rotationnel pour une molécule à trois sites donné,
    avec Coulomb et LJ.

    Args:
        n (int): Nombre de molécules
        v: Matrice d'état du système
        Lx, Ly, Lz (float): Taille de la boîte de simulation
        nbr_list: Liste de booléens des voisins pour chaque molécule
        theta (float): Angle H-O-H du modèle de la molécule d'eau
        r_oh (float): Distance O-H du modèle de la molécule d'eau
        q_o (float): Charge sur l'atome d'oxygène
        q_h (float): Charge sur l'atome d'hydrogène
        eps_LJ (float): epsilon de Lennard-Jones pour l'interaction O-O
        sigma_LJ (float): sigma de Lennard-Jones pour l'interaction O-O
        k_coul (float): constante coulombienne

    Returns:
        tuple:
            U        : potentiel réparti par molécule
            F        : forces par molécule
            tau      : torques par molécule
            lap_cart : laplacien cartésien réparti par molécule
            lap_rot  : laplacien rotationnel réparti par molécule
    """
    list_r = v[:3*n].reshape(n, 3)
    list_q = v[6*n:10*n].reshape(n, 4)[:, [3, 0, 1, 2]]
    L = np.array([Lx, Ly, Lz])

    s = np.sin(theta / 2)
    c = np.cos(theta / 2)

    # Positions des hydrogènes dans le repère moléculaire
    r_h1 = r_oh * np.array([0,  s, c])
    r_h2 = r_oh * np.array([0, -s, c])

    i_idx, j_idx = np.where(nbr_list)

    # Vecteur COM_j - COM_i
    r = list_r[j_idx] - list_r[i_idx]

    q_i = qtn.from_float_array(list_q[i_idx])
    q_j = qtn.from_float_array(list_q[j_idx])

    # Leviers atomiques tournés dans le labo
    u_h1 = qtn.rotate_vectors(q_i, r_h1)
    u_h2 = qtn.rotate_vectors(q_i, r_h2)
    v_h1 = qtn.rotate_vectors(q_j, r_h1)
    v_h2 = qtn.rotate_vectors(q_j, r_h2)

    # Vecteurs intersites
    rOO   = r
    rOH1  = r + v_h1
    rOH2  = r + v_h2
    rH1O  = r - u_h1
    rH1H1 = r - u_h1 + v_h1
    rH1H2 = r - u_h1 + v_h2
    rH2O  = r - u_h2
    rH2H1 = r - u_h2 + v_h1
    rH2H2 = r - u_h2 + v_h2

    r_vec = np.array([
        rOO, rOH1, rOH2,
        rH1O, rH1H1, rH1H2,
        rH2O, rH2H1, rH2H2
    ])

    # Conditions périodiques minimales
    r_vec = r_vec - L * np.round(r_vec / L)

    eps = 1e-12
    r2 = np.einsum("kij,kij->ki", r_vec, r_vec) + eps**2
    inv_r = 1.0 / np.sqrt(r2)
    r_norm = np.sqrt(r2)

    # Charges des 9 interactions
    q_left  = np.array([q_o]*3 + [q_h]*6)
    q_right = np.array([q_o, q_h, q_h] * 3)
    qq = k_coul * q_left * q_right

    # -------------------------
    # Dérivées radiales u'(r), u''(r)
    # -------------------------

    # Coulomb : u(r) = qq / r
    u_prime_coul = -qq[:, None] * inv_r**2
    u_second_coul = 2.0 * qq[:, None] * inv_r**3

    # Lennard-Jones sur OO seulement
    sig6 = sigma_LJ**6
    sig12 = sig6**2

    u_prime_LJ = np.zeros_like(inv_r)
    u_second_LJ = np.zeros_like(inv_r)

    u_prime_LJ[0] = 24.0 * eps_LJ * (
        -2.0 * sig12 * inv_r[0]**13 + sig6 * inv_r[0]**7
    )

    u_second_LJ[0] = 24.0 * eps_LJ * (
        26.0 * sig12 * inv_r[0]**14 - 7.0 * sig6 * inv_r[0]**8
    )

    u_prime = u_prime_coul + u_prime_LJ
    u_second = u_second_coul + u_second_LJ

    # -------------------------
    # Préfacteur des forces
    # F = -(u'(r)/r) * r_vec
    # -------------------------
    prefactor = -(u_prime / r_norm)   # forme (9, N_pairs)

    # Potentiel de paire
    u_pair = (
        np.einsum("k,kp->p", qq, inv_r)
        + 4.0 * eps_LJ * np.sum(sig12 * inv_r[0]**12 - sig6 * inv_r[0]**6)
    )

    U = np.zeros(n)
    np.add.at(U, i_idx, 0.5 * u_pair)
    np.add.at(U, j_idx, 0.5 * u_pair)

    # -------------------------
    # Forces
    # -------------------------
    F_pair = prefactor[:, :, None] * r_vec
    F = np.zeros((n, 3))
    for k in range(9):
        np.add.at(F, i_idx,  F_pair[k])
        np.add.at(F, j_idx, -F_pair[k])

    # -------------------------
    # Torques
    # -------------------------
    def scatter_torque(lever, F_rows, molecule_idx):
        tau_local = np.zeros((n, 3))
        for F_k in F_rows:
            np.add.at(tau_local, molecule_idx, np.cross(lever, F_k))
        return tau_local

    tau = np.zeros((n, 3))
    tau += scatter_torque(u_h1, F_pair[3:6],         i_idx)
    tau += scatter_torque(u_h2, F_pair[6:9],         i_idx)
    tau += scatter_torque(v_h1, -F_pair[[1, 4, 7]],  j_idx)
    tau += scatter_torque(v_h2, -F_pair[[2, 5, 8]],  j_idx)

    # -------------------------
    # Laplacien cartésien
    # Pour une paire : u''(r) + (2/r)u'(r)
    # Ici on le répartit par molécule comme U
    # -------------------------
    lap_pair = u_second + (2.0 / r_norm) * u_prime
    lap_pair_sum = np.sum(lap_pair, axis=0)

    lap_cart = np.zeros(n)
    np.add.at(lap_cart, i_idx, 0.5 * lap_pair_sum)
    np.add.at(lap_cart, j_idx, 0.5 * lap_pair_sum)

    # -------------------------
    # Laplacien rotationnel
    # Delta_rot,i U = sum u''(d) |r_il x r_ijlm|^2 / d^2
    # -------------------------
    lap_rot = np.zeros(n)

    # Contributions côté molécule i
    lever_list_i = [
        np.zeros_like(r),  # OO
        np.zeros_like(r),  # OH1
        np.zeros_like(r),  # OH2
        u_h1,              # H1O
        u_h1,              # H1H1
        u_h1,              # H1H2
        u_h2,              # H2O
        u_h2,              # H2H1
        u_h2               # H2H2
    ]

    for k in range(9):
        lever = lever_list_i[k]
        cross = np.cross(lever, r_vec[k])
        cross2 = np.einsum("ij,ij->i", cross, cross)
        contrib = u_second[k] * cross2 / r2[k]
        np.add.at(lap_rot, i_idx, contrib)

    # Contributions côté molécule j
    lever_list_j = [
        np.zeros_like(r),  # OO
        v_h1,              # OH1
        v_h2,              # OH2
        np.zeros_like(r),  # H1O
        v_h1,              # H1H1
        v_h2,              # H1H2
        np.zeros_like(r),  # H2O
        v_h1,              # H2H1
        v_h2               # H2H2
    ]

    for k in range(9):
        lever = lever_list_j[k]
        cross = np.cross(lever, r_vec[k])
        cross2 = np.einsum("ij,ij->i", cross, cross)
        contrib = u_second[k] * cross2 / r2[k]
        np.add.at(lap_rot, j_idx, contrib)

    return U, F, tau, lap_cart, lap_rot
