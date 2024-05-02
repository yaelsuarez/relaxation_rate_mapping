import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from jsonargparse import CLI
from pyvista.plotting.plotter import Plotter
import os
from scipy.optimize import curve_fit
from tqdm import tqdm
import SimpleITK as sitk
from typing import Tuple, List
import pyvistaqt as pvqt
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def read_supervolume(root) -> Tuple[np.ndarray, List[float]]:
    """Read a folder with one folder per echo time. Assumes that the folder names are the echo times

    Args:
        root (str): root dir where all the echo times live

    Returns:
        Tuple[np.ndarray, List[float]]:
            Volume with shape (E, X, Y, Z) where E is the echo time.
            List with the sorted echo times (names of the folders).
    """
    volumes = []
    echo_times = []
    for dir in sorted(
        [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))],
        key=lambda x: float(x),
    ):
        echo_times.append(float(dir))
        dir = os.path.join(root, dir)
        print(dir)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        volumes.append(sitk.GetArrayFromImage(image))
    volumes = np.stack(volumes)
    return volumes, echo_times

def read_supervolume_cg(root) -> Tuple[np.ndarray, List[float]]:
    """Read a folder with one folder per echo time. Assumes that the folder names are the echo times

    Args:
        root (str): root dir where all the echo times live

    Returns:
        Tuple[np.ndarray, List[float]]:
            Volume with shape (E, X, Y, Z) where E is the echo time.
            List with the sorted echo times (names of the folders).
    """
    volumes = []
    echo_times = []
    max_shape = None  # Variable to store the maximum shape

    for dir in sorted(
        [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))],
        key=lambda x: float(x),
    ):
        echo_times.append(float(dir))
        dir = os.path.join(root, dir)
        print(dir)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        volume = sitk.GetArrayFromImage(image)

        # Update the maximum shape
        if max_shape is None or volume.shape > max_shape:
            max_shape = volume.shape

        volumes.append(volume)

    # Pad volumes to match the maximum shape
    padded_volumes = []
    for volume in volumes:
        pad_width = [(0, m - s) for s, m in zip(volume.shape, max_shape)]
        padded_volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
        padded_volumes.append(padded_volume)

    volumes = np.stack(padded_volumes)

    return volumes, echo_times


def fit_r(echo_time, mean_intensity):
    def func(x, I0, rs):
        return I0 * np.exp(-(rs) * x)

    popt, pcov = curve_fit(
        func,
        echo_time,
        mean_intensity,
    )
    return popt


def fit_params_over_echotime(super_volume, echo_times):
    """For each voxel, fit the params over the echo_time

    Args:
        super_volume (ndarray): 4D ndarray with shape (E, X, Y, Z)
        echo_times (List[float]): list of echo_times

    Returns:
        ndarray: The `rs` parameter fitter to each voxel with shape (X, Y, Z)
    """
    s = super_volume.shape
    fitted_volume = np.zeros([s[1], s[2], s[3]])
    pbar = tqdm(
        desc="fitting params", total=s[1] * s[2] * s[3], mininterval=0.5, unit="voxel"
    )
    for x in range(s[1]):
        for y in range(s[2]):
            for z in range(s[3]):
                try:
                    I0, rs = fit_r(echo_times, super_volume[:, x, y, z])
                    fitted_volume[x, y, z] = rs
                except RuntimeError as e:
                    print(x, y, z, e)
                    fitted_volume[x, y, z] = 0
                pbar.update()
    return fitted_volume

def main(
    fit_rs: bool,
    root: str,
    vol_to_fit_file: str,
    opacity_map:List[float],
    clim: List[float],
    colormap: str,
    background: str = "white",
    
):
    rs_vol_path = os.path.join(root, vol_to_fit_file)
    if fit_rs:
        super_volume, echo_times = read_supervolume_cg(root)
        super_volume.shape, echo_times
        # select a smaller volume to make the fitting faster
        s = super_volume.shape
        # xlim = [s[1] // 3, 2 * s[1] // 3]
        # ylim = [s[2] // 3, 2 * s[2] // 3]
        # zlim = [s[3] // 3, 2 * s[3] // 3]
        # super_volume_crop = super_volume[
        #     :, xlim[0] : xlim[1], ylim[0] : ylim[1], zlim[0] : zlim[1]
        # ]
        # fit the params
        # fitted_volume = fit_params_over_echotime(super_volume_crop, echo_times)
        fitted_volume = fit_params_over_echotime(super_volume, echo_times)
        # save volume for fast loading
        np.savez(rs_vol_path, rs=fitted_volume)

    f = np.load(rs_vol_path)
    fitted_volume = f["rs"].astype(np.float32)
    
    cmap = plt.get_cmap(colormap)
    fig, ax = plt.subplots()
    n, bins, patches = plt.hist(fitted_volume[:, :, :].flatten(), bins=100)

    # Define the range of x values you want to apply the colormap to
    x_min = clim[0]
    x_max = clim[1]

    # Create a normalization instance to map the range of x values to [0, 1]
    norm = Normalize(vmin=x_min, vmax=x_max)

    for c, p in zip(bins, patches):
        normalized_value = norm(c)
        # Use the colormap to get the color based on the normalized value
        color = cmap(normalized_value)
        plt.setp(p, 'facecolor', color)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Relaxation rate (s$^{-1}$)')

    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    if background == 'black':
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        cbar.set_label('Relaxation rate (s$^{-1}$)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        cbar.ax.tick_params(colors='white')  # Set color of colorbar tick labels
        cbar.ax.set_ylabel('Relaxation rate (s$^{-1}$)', color='white')  # Set color of colorbar label
        plt.title("", color='white')

    plt.xlabel("Relaxation rate (s$^{-1}$)")
    plt.ylabel("Number of pixels")
    plt.title("")
    #plt.savefig(os.path.join(root,'histogram_rs.svg'))
    plt.show()

    # print(fitted_volume.dtype)
    # fitted_volume = (fitted_volume - np.mean(fitted_volume)) / np.std(fitted_volume)
    # fitted_volume = np.clip(fitted_volume, 0, 0.3)/0.3
    # plt.imshow(fitted_volume[30,:,:])
    # plt.hist(fitted_volume[10,:,:].flatten(), bins=50)
    # plt.show()

    if background != "white":
        pv.global_theme.background = background

    # you can increase the resolution here
    # p = pv.Plotter(window_size=[900, 900])

    def orient(volume):
        # in case you need to flip the image, change the order of the axes
        return np.flip(np.transpose(volume, (2, 1, 0)), axis=[0, 1])

    v = orient(fitted_volume[:, :, :])
    grid = pv.ImageData()
    grid.dimensions = grid.dimensions = np.array(v.shape) + 1
    grid.cell_data["values"] = v.flatten(order="F")  # Flatten the array
    # grid.plot(show_edges=True)
    p = Plotter()

    #   X Bounds:   0.000e+00, 6.400e+01
    #   Y Bounds:   2.407e+01, 3.572e+01
    #   Z Bounds:   0.000e+00, 6.400e+01

    # plane = p.add_mesh_slice(grid, interaction_event=vtk.vtkCommand.InteractionEvent)
    va = p.add_volume(
        volume=v,
        # flip_scalars=True,
        cmap=colormap, # you can change the color map here
        opacity=opacity_map,
        clim=clim,
        mapper="gpu",
        # log_scale=True,
        # blending="composite",
        render=False,  # maybe start w/o rendering and then turn it on once you are happy with the results
        #show_scalar_bar=True,
    )
    
    #p.add_bounding_box()
    scalar_bar = p.add_scalar_bar("Relaxation rate \n")
    # Change the font color of the colorbar labels to red
    scalar_bar.GetLabelTextProperty().SetColor(1,1,1) 
    scalar_bar.GetTitleTextProperty().SetColor(1,1,1) 
    p.add_orientation_widget(va)
    #p.show()
    # print(p.plane_sliced_meshes[0])

    # p.add_mesh_clip_plane(v)

    # plot the volume

    # va = p.add_volume(v,
    #              cmap="jet",
    #              opacity="sigmoid_4", mapper="smart",
    #              opacity_unit_distance=1,
    #              shade=True)

    # f_opa = lambda val: va.GetProperty().SetScalarOpacityUnitDistance(val)
    # f_clim = lambda val: va.GetProperty().SetScalarClim(val)

    # p.add_slider_widget(f_opa, [0, 4], title="Opacity Distance")
    # p.add_slider_widget(f_clim, [0,1], title="C lim", pointa=(0.4, 0.8), pointb=(0.9, 0.8))

    # # you can show what area you are grabbing by putting a second volume with the same shape with 1s in the area that you selected
    # grid = pv.wrap(orient(label_img[:, :chest_start, :])).threshold(0.1)
    # surface = grid.extract_surface().smooth_taubin(n_iter=50, pass_band=0.05)

    # # this is to show the area that you selected
    # p.add_mesh(
    #     surface, show_scalar_bar=False, color="orange", opacity=0.8, style="wireframe"
    # )

    # p.add_axes()

    # p.enable_anti_aliasing()

    # camera_pos = [(-381.74, -46.02, 216.54), (74.8305, 89.2905, 100.0), (0.23, 0.072, 0.97)]
    # p.camera_position = camera_pos
    #p.show(interactive=True)
    resolution = (2560, 1440) 
    screenshot_png = (os.path.join(root, "render.png"))
    p.show(screenshot=screenshot_png, interactive=True, window_size=resolution)  # Save the screenshot first
    ## this is used to create a video as mp4

    # p.camera.zoom(1.5)
    # p.show(auto_close=False)
    # path = p.generate_orbital_path(
    #     n_points=720,
    #     shift=50,
    # )
    # p.open_movie("orbit2.mp4")
    # p.orbit_on_path(path, write_frames=True)
    # p.close()

    # Use the custom plotter with key event handling


if __name__ == "__main__":
    default_config_path = os.path.join(os.path.dirname(__file__), "config_rs.yaml")
    CLI(main, as_positional=False, default_config_files=[default_config_path])