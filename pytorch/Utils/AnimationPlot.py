import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.animation as animation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def animation_plot(animations, parents, interval, save_path=None):
    parents = parents

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    ax.set_xlim3d(-40, 40)
    ax.set_zlim3d(0, 40)
    ax.set_ylim3d(-50, 50)
    ax.set_aspect('auto')

    acolors = list(sorted(colors.cnames.keys()))[::-1]

    lines = [[plt.plot([0, 0], [0, 0], [0, 0], color=acolors[0],
                       lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in
              range(animations.shape[1])]]

    def animate(i):
        changed = []

        for j in range(len(parents)):
            if parents[j] != -1:
                lines[0][j].set_data_3d(
                    [animations[i, j, 0], animations[i, parents[j], 0]],
                    [animations[i, j, 2], animations[i, parents[j], 2]],
                    [animations[i, j, 1], animations[i, parents[j], 1]]
                )

        changed += lines

        return changed

    plt.tight_layout()

    ani = animation.FuncAnimation(
        fig,
        animate,
        len(animations),
        interval=interval
    )

    if save_path is not None:
        ani.save(save_path, writer='pillow', fps=30)

    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.show()


def animation_plot_frame(animations, parents, save_path=None):
    parents = parents.copy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    ax.set_xlim3d(-40, 40)
    ax.set_zlim3d(0, 40)
    ax.set_ylim3d(-50, 50)
    ax.set_aspect('auto')
    plt.tight_layout()

    acolors = list(sorted(colors.cnames.keys()))[::-1]

    # 直接绘制单帧数据
    i = 0  # 只使用第一帧数据
    for j in range(len(parents)):
        if parents[j] != -1:
            # 绘制每个关节点与父节点的连线
            x_pair = [animations[i, j, 0], animations[i, parents[j], 0]]
            y_pair = [animations[i, j, 2], animations[i, parents[j], 2]]
            z_pair = [animations[i, j, 1], animations[i, parents[j], 1]]

            ax.plot(x_pair, y_pair, z_pair,
                    color=acolors[0],
                    lw=2,
                    path_effects=[
                        pe.Stroke(linewidth=3, foreground='black'),
                        pe.Normal()
                    ])

    # 隐藏坐标轴文字
    plt.xticks(color='white')
    plt.yticks(color='white')

    # 保存图片
    if save_path is not None:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')

    # plt.show()

# 使用方法示例：
# animation_plot(single_frame_data, parents, save_path="output.png")