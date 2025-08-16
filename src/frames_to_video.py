import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


def frames_to_video(frames, fps=24):
    fig = plt.figure(
        figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100), dpi=100
    )
    ax = plt.axes()
    ax.set_axis_off()

    if len(frames[0].shape) == 2:  # Grayscale image
        im = ax.imshow(frames[0], cmap="gray")
    else:  # Color image
        im = ax.imshow(frames[0])

    def init():
        if len(frames[0].shape) == 2:
            im.set_data(frames[0], cmap="gray")
        else:
            im.set_data(frames[0])
        return (im,)

    def update(frame):
        if len(frames[frame].shape) == 2:
            im.set_data(frames[frame], cmap="gray")
        else:
            im.set_data(frames[frame])
        return (im,)

    interval = 1000 / fps
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        blit=True,
        interval=interval,
    )
    plt.close()
    return HTML(anim.to_html5_video())
