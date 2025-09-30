import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_trajectory(ax, pred, target=None, alpha_scale=1.0):
    ax.clear()
    ax.set_title('Action Trajectory', fontsize=14, fontweight='bold')

    if pred.ndim == 3:
        pred = pred[0]

    points = pred.reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]

    n_points = len(points)
    plt.cm.viridis(np.linspace(0, 1, n_points))

    ax.plot(x, y, '-', color='gray', alpha=0.3*alpha_scale, linewidth=2)
    ax.scatter(x, y, c=np.arange(n_points),
               cmap='viridis',
               alpha=0.7*alpha_scale,
               s=50)

    if target is not None:
        if target.ndim == 3:
            target = target[0]
        ax.plot(target[:, 0], target[:, 1], '--',
                color='red', alpha=0.5*alpha_scale,
                linewidth=2, label='Target')
        ax.legend()

    ax.grid(True, linestyle='--', alpha=0.3)


def plot_vector_field(ax, vel, resolution=20, radius=5.0):
    ax.clear()
    ax.set_title('Vector Field', fontsize=14, fontweight='bold')

    if vel.shape[0] == 1 and vel.ndim == 3:
        vel = vel.squeeze(0)

    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            dists = np.linalg.norm(vel[:, :] - point, axis=1)
            nearest = dists < radius
            if np.any(nearest):
                U[i, j], V[i, j] = vel[nearest, :].mean(axis=0)

    magnitude = np.sqrt(U**2 + V**2)
    ax.quiver(X, Y, U, V, magnitude,
              cmap='viridis',
              width=0.003,
              headwidth=4,
              headlength=5,
              headaxislength=4.5,
              alpha=0.7)

    ax.grid(True, linestyle='--', alpha=0.3)


def plot_trajectory_with_vectors(ax, vel, traj):
    ax.clear()
    ax.set_title('Actions with Vector Field', fontsize=14, fontweight='bold')

    if vel.shape[0] == 1 and vel.ndim == 3:
        vel = vel.squeeze(0)
    if traj.shape[0] == 1 and traj.ndim == 3:
        traj = traj.squeeze(0)

    x, y = traj[:, 0], traj[:, 1]
    ax.plot(x, y, '-', color='gray', alpha=0.6, linewidth=2.5)

    for i, point in enumerate(traj):
        vx, vy = vel[i]

        ax.quiver(point[0], point[1], vx, vy,
                  color=plt.cm.viridis(i/len(traj)),
                  scale=10,
                  width=0.008,
                  headwidth=6,
                  headlength=7,
                  headaxislength=5.5,
                  alpha=0.8)

        target_x = point[0] + vx * 0.1
        target_y = point[1] + vy * 0.1
        ax.scatter(target_x, target_y,
                   color=plt.cm.viridis(i/len(traj)),
                   alpha=0.3, s=50)

    ax.scatter(x, y,
               c=np.arange(len(x)),
               cmap='viridis',
               alpha=0.9,
               s=30,
               zorder=5)

    ax.grid(True, linestyle='--', alpha=0.3)

    return ax


def create_denoising_plot(figsize=(15, 5)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Flare Visualization', fontsize=16, y=1.05)
    return fig, (ax1, ax2, ax3)


def plot_denoising_step(axes, traj, vel):
    ax1, ax2, ax3 = axes
    plot_trajectory(ax1, traj)
    plot_trajectory_with_vectors(ax2, vel, traj)
    plot_vector_field(ax3, vel)
    plt.tight_layout()


def create_vector_field_video(traj_history, velocity_history, fps=10):
    n_frames = min(len(traj_history), len(velocity_history))
    fig, axes = create_denoising_plot()

    def update(frame):
        if frame < n_frames:
            traj = traj_history[frame].numpy()
            vel = velocity_history[frame].numpy()
            plot_denoising_step(axes, traj, vel)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=30000/fps)
    plt.close()
    return ani


def create_vector_field_plots(traj_history, vel_history):
    """
    Creates a Matplotlib figure with one row per denoising step,
    and 3 columns: (1) trajectory, (2) trajectory + vectors, (3) vector field.
    Returns the Matplotlib figure (not closed).
    """

    n_steps = len(traj_history)
    fig, axes = plt.subplots(n_steps, 3, figsize=(15, 5*n_steps))
    fig.suptitle("Denoising Steps", fontsize=16)

    if n_steps == 1:
        axes = axes.reshape(1, 3)

    for i in range(n_steps):
        ax1, ax2, ax3 = axes[i]

        traj = traj_history[i]
        vel = vel_history[i]

        plot_trajectory(ax1, traj)
        plot_trajectory_with_vectors(ax2, vel, traj)
        plot_vector_field(ax3, vel)

    plt.tight_layout()
    return fig


def plot_ode_steps(traj_history):
    n_steps = len(traj_history)
    fig, axes = plt.subplots(n_steps, 1, figsize=(4, 4 * n_steps))
    fig.suptitle("Denoising Steps", fontsize=16)

    if n_steps == 1:
        axes = [axes]

    for i, traj in enumerate(traj_history):
        ax = axes[i]
        traj_2d = traj[:, :2] if traj.ndim == 2 else traj.squeeze(0)[:, :2]
        plot_trajectory(ax, traj_2d)
        ax.set_title(f"Step {i}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
