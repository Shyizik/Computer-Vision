import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- МАТРИЧНІ ОПЕРАЦІЇ ---
def get_translation_matrix(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])

def get_scaling_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

def get_rotation_y_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])

# --- ІНІЦІАЛІЗАЦІЯ ФІГУР ---
# 2D Квадрат
square_coords = np.array([
    [-1, 1, 1, -1, -1],
    [-1, -1, 1, 1, -1],
    [1, 1, 1, 1, 1]
])

# 3D Піраміда
v = np.array([
    [-1, 0, -0.577, 1],
    [1, 0, -0.577, 1],
    [0, 0, 1.155, 1],
    [0, 2, 0, 1]
]).T

pyramid_faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]

# --- НАЛАШТУВАННЯ ВІЗУАЛІЗАЦІЇ ---
fig = plt.figure(figsize=(12, 6))
ax2d = fig.add_subplot(1, 2, 1)
ax2d.set_title("2D: Scale -> Move -> Scale")
ax2d.set_xlim(-10, 10);
ax2d.set_ylim(-10, 10);
ax2d.grid(True)

line2d, = ax2d.plot([], [], 'b-', lw=2)
traj_line, = ax2d.plot([], [], 'r--', lw=1)
trajectory_x, trajectory_y = [], []

ax3d = fig.add_subplot(1, 2, 2, projection='3d')
ax3d.set_title("3D: Fade & Color Change")
ax3d.set_xlim(-3, 3);
ax3d.set_ylim(-3, 3);
ax3d.set_zlim(-1, 4)
poly3d = Poly3DCollection([], alpha=0.5)
ax3d.add_collection3d(poly3d)

# --- АНІМАЦІЯ ---
scale1 = get_scaling_matrix(0.5, 0.5)
scale2 = get_scaling_matrix(1.5, 1.5)

def update(frame):
    # 2D Part
    t = frame * 0.1
    dx, dy = 4 * np.cos(t), 4 * np.sin(t)

    M_final = scale2 @ get_translation_matrix(dx, dy) @ scale1
    new_square = M_final @ square_coords

    trajectory_x.append(np.mean(new_square[0, :-1]))
    trajectory_y.append(np.mean(new_square[1, :-1]))
    if len(trajectory_x) > 100: trajectory_x.pop(0); trajectory_y.pop(0)

    line2d.set_data(new_square[0, :], new_square[1, :])
    traj_line.set_data(trajectory_x, trajectory_y)

    # 3D Part
    transformed_v = (get_rotation_y_matrix(frame * 0.05) @ v).T
    verts = [[transformed_v[i, :3] for i in face] for face in pyramid_faces]

    alpha_val = (np.sin(frame * 0.1) + 1) / 2 * 0.9 + 0.1
    colors = ['cyan', 'magenta', 'yellow', 'lime']

    poly3d.set_verts(verts)
    poly3d.set_alpha(alpha_val)
    poly3d.set_edgecolor(colors[int((frame / 20) % 4)])
    poly3d.set_facecolor((0.1, 0.5, 0.8))

    return line2d, traj_line, poly3d

ani = animation.FuncAnimation(fig, update, frames=200, interval=50)
plt.show()