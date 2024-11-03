import numpy as np

def weno5(a, b, c, d, e):
    eps = 1.0e-6
    q1 = a / 3.0 - 7.0 / 6.0 * b + 11.0 / 6.0 * c
    q2 = -b / 6.0 + 5.0 / 6.0 * c + d / 3.0
    q3 = c / 3.0 + 5.0 / 6.0 * d - e / 6.0

    s1 = (13.0 / 12.0) * (a - 2.0 * b + c) ** 2 + 0.25 * (a - 4.0 * b + 3.0 * c) ** 2
    s2 = (13.0 / 12.0) * (b - 2.0 * c + d) ** 2 + 0.25 * (d - b) ** 2
    s3 = (13.0 / 12.0) * (c - 2.0 * d + e) ** 2 + 0.25 * (3.0 * c - 4.0 * d + e) ** 2

    a1 = 1.0 / (eps + s1) ** 2
    a2 = 6.0 / (eps + s2) ** 2
    a3 = 3.0 / (eps + s3) ** 2

    f = (a1 * q1 + a2 * q2 + a3 * q3) / (a1 + a2 + a3)
    return f


def rhs(nx, dx, u):
    r = np.zeros(nx)
    f = 0.5 * u ** 2
    cc = np.zeros(nx)

    for i in range(2, nx - 2):
        cc[i] = max(abs(u[i-2]), abs(u[i-1]), abs(u[i]), abs(u[i+1]), abs(u[i+2]))

    # Periodicity
    cc[0] = max(abs(u[-2]), abs(u[-1]), abs(u[0]), abs(u[1]), abs(u[2]))
    cc[1] = max(abs(u[-2]), abs(u[-1]), abs(u[0]), abs(u[1]), abs(u[2]))
    cc[nx - 2] = max(abs(u[nx - 3]), abs(u[nx - 2]), abs(u[nx - 1]), abs(u[0]), abs(u[1]))
    cc[nx - 1] = max(abs(u[nx - 3]), abs(u[nx - 2]), abs(u[nx - 1]), abs(u[0]), abs(u[1]))

    fp = 0.5 * (f + cc * u)
    fm = 0.5 * (f - cc * u)

    fL = np.zeros(nx + 1)
    fR = np.zeros(nx + 1)

    # Upwind reconstruction
    for i in range(nx):
        if i < 2:
            a, b, c, d, e = u[i-2], u[i-1], u[i], u[i+1], u[i+2]
        else:
            a, b, c, d, e = u[i-2], u[i-1], u[i], u[i+1], u[i+2]
        fL[i] = weno5(a, b, c, d, e)

    # Downwind reconstruction
    for i in range(nx):
        if i < 2:
            a, b, c, d, e = u[i+2], u[i+1], u[i], u[i-1], u[i-2]
        else:
            a, b, c, d, e = u[i+2], u[i+1], u[i], u[i-1], u[i-2]
        fR[i] = weno5(a, b, c, d, e)

    # Compute RHS
    for i in range(nx):
        r[i] = -(fL[i + 1] - fL[i]) / dx - (fR[i + 1] - fR[i]) / dx

    return r


def numerical(nx, ns, nt, dx, dt):
    u = np.zeros((nx, ns + 1))
    un = np.zeros(nx)
    pi = np.pi

    # Initial condition
    for i in range(nx):
        x = -0.5 * dx + (i + 1) * dx
        un[i] = np.sin(2.0 * pi * x)

    # Time integration
    k = 0
    freq = nt // ns

    for j in range(nt):
        rhs_value = rhs(nx, dx, un)
        ut = un + dt * rhs_value

        rhs_value = rhs(nx, dx, ut)
        ut = 0.75 * un + 0.25 * ut + 0.25 * dt * rhs_value

        rhs_value = rhs(nx, dx, ut)
        un = (1.0 / 3.0) * un + (2.0 / 3.0) * ut + (2.0 / 3.0) * dt * rhs_value

        if j % freq == 0:
            u[:, k] = un
            k += 1

    return u


def main():
    # Reading input file
    with open('input_sol.txt', 'r') as file:
        nx = int(file.readline())
        ns = int(file.readline())
        dt = float(file.readline())
        tm = float(file.readline())

    dx = 1.0 / nx
    ds = tm / ns
    nt = int(tm / dt)

    u = numerical(nx, ns, nt, dx, dt)

    # Write solutions to Tecplot format
    # Implement file writing similar to Fortran example here
    # ...

if __name__ == "__main__":
    main()
