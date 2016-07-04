import numpy as np
import nibabel as nib
from dipy.viz import actor, window
from dipy.tracking import utils
from dipy.reconst.peaks import PeaksAndMetrics
from dipy.core.sphere import Sphere


def load_nifti(fname, return_img=False, return_voxsize=False):
    img = nib.load(fname)
    hdr = img.get_header()
    data = img.get_data()
    vox_size = hdr.get_zooms()[:3]

    ret_val = [data, img.get_affine()]
    if return_voxsize:
        ret_val.append(vox_size)
    if return_img:
        ret_val.append(img)

    return tuple(ret_val)


def save_nifti(fname, data, affine):
    result_img = nib.Nifti1Image(data, affine)
    result_img.to_filename(fname)


def load_peaks(fname, verbose=False):
    """ Load PeaksAndMetrics NPZ file
    """

    pam_dix = np.load(fname)

    pam = PeaksAndMetrics()
    pam.affine = pam_dix['affine']
    pam.peak_dirs = pam_dix['peak_dirs']
    pam.peak_values = pam_dix['peak_values']
    pam.peak_indices = pam_dix['peak_indices']
    pam.shm_coeff = pam_dix['shm_coeff']
    pam.sphere = Sphere(xyz=pam_dix['sphere_vertices'])
    pam.B = pam_dix['B']
    pam.total_weight = pam_dix['total_weight']
    pam.ang_thr = pam_dix['ang_thr']
    pam.gfa = pam_dix['gfa']
    pam.qa = pam_dix['qa']
    pam.odf = pam_dix['odf']

    if verbose:
        print('Affine')
        print(pam.affine)
        print('Dirs Shape')
        print(pam.peak_dirs.shape)
        print('SH Shape')
        print(pam.shm_coeff.shape)
        print('ODF')
        print(pam.odf.shape)
        print('Total weight')
        print(pam.total_weight)
        print('Angular threshold')
        print(pam.ang_thr)
        print('Sphere vertices shape')
        print(pam.sphere.vertices.shape)

    return pam


def save_peaks(fname, pam, compressed=True):
    """ Save NPZ file with all important attributes of object PeaksAndMetrics
    """

    if compressed:
        save_func = np.savez_compressed
    else:
        save_func = np.savez

    save_func(fname,
              affine=pam.affine,
              peak_dirs=pam.peak_dirs,
              peak_values=pam.peak_values,
              peak_indices=pam.peak_indices,
              shm_coeff=pam.shm_coeff,
              sphere_vertices=pam.sphere.vertices,
              B=pam.B,
              total_weight=pam.total_weight,
              ang_thr=pam.ang_thr,
              gfa=pam.gfa,
              qa=pam.qa,
              odf=pam.odf)


def load_trk(fname):
    trkfile = nib.streamlines.load(fname)
    return trkfile.streamlines, trkfile.header


def save_trk(fname, streamlines, hdr=None, affine_to_rasmm=None):
    tractogram = nib.streamlines.Tractogram(streamlines,
                                            affine_to_rasmm=affine_to_rasmm)
    trkfile = nib.streamlines.TrkFile(tractogram, header=hdr)
    nib.streamlines.save(trkfile, fname)


def save_trk_old_style(filename, points, vox_to_ras, shape):
    """A temporary helper function for saving trk files.
    This function will soon be replaced by better trk file support in nibabel.
    """
    voxel_order = nib.orientations.aff2axcodes(vox_to_ras)
    voxel_order = "".join(voxel_order)

    # Compute the vox_to_ras of "trackvis space"
    zooms = np.sqrt((vox_to_ras * vox_to_ras).sum(0))
    vox_to_trk = np.diag(zooms)
    vox_to_trk[3, 3] = 1
    vox_to_trk[:3, 3] = zooms[:3] / 2.

    points = utils.move_streamlines(points,
                                    input_space=vox_to_ras,
                                    output_space=vox_to_trk)

    data = ((p, None, None) for p in points)

    hdr = nib.trackvis.empty_header()
    hdr['dim'] = shape
    hdr['voxel_order'] = voxel_order
    hdr['voxel_size'] = zooms[:3]

    nib.trackvis.write(filename, data, hdr)


def show_two_images(vol1, affine1, vol2, affine2, shift=50):
    """ Show 2 images side by side"""

    renderer = window.Renderer()
    mean, std = vol1[vol1 > 0].mean(), vol1[vol1 > 0].std()
    value_range1 = (mean - 0.5 * std, mean + 1.5 * std)
    mean, std = vol2[vol2 > 0].mean(), vol2[vol2 > 0].std()
    value_range2 = (mean - 0.5 * std, mean + 1.5 * std)

    slice_actor1 = actor.slicer(vol1, affine1, value_range1)
    slice_actor2 = actor.slicer(vol2, affine2, value_range2)

    renderer.add(slice_actor1)
    renderer.add(slice_actor2)

    slice_actor2.SetPosition(slice_actor1.shape[0] + shift, 0, 0)

    window.show(renderer)


def show_mosaic(data, affine, border=70):
    """ Show a simple mosaic of the given image
    """

    renderer = window.Renderer()
    mean, std = data[data > 0].mean(), data[data > 0].std()
    value_range = (mean - 0.5 * std, mean + 1.5 * std)
    slice_actor = actor.slicer(data, affine, value_range)

    renderer.clear()
    renderer.projection('parallel')
    cnt = 0

    X, Y, Z = slice_actor.shape[:3]

    rows = 10
    cols = 15
    border = 70

    for j in range(rows):
        for i in range(cols):
            slice_mosaic = slice_actor.copy()
            slice_mosaic.display(None, None, cnt)
            slice_mosaic.SetPosition(
                (X + border) * i,
                0.5 * cols * (Y + border) - (Y + border) * j,
                0)
            renderer.add(slice_mosaic)
            cnt += 1
            if cnt > Z:
                break
        if cnt > Z:
            break

    renderer.reset_camera()
    renderer.zoom(1.6)

    window.show(renderer, size=(900, 600), reset_camera=False)


def show_bundles(bundles, colors=None, size=(1080, 600),
                 show=False, fname=None):

    ren = window.Renderer()
    ren.background((1., 1, 1))

    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        lines = actor.line(bundle, color, linewidth=1.5)
        ren.add(lines)

    ren.reset_clipping_range()
    ren.reset_camera()

    # if show:
    window.show(ren, size=size, reset_camera=True)

    if fname is not None:
        window.record(ren, n_frames=1, out_path=fname, size=size)
