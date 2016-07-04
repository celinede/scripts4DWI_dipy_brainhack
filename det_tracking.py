from __future__ import division, print_function, absolute_import

import numpy as np
import nibabel as nib
from dipy.reconst.peaks import peaks_from_model, PeaksAndMetrics
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.tracking import utils
from common import load_nifti, save_trk, save_peaks, load_peaks

from ipdb import set_trace

def show_results(streamlines, vol, affine):

    from dipy.viz import actor, window, widget

    shape = data.shape
    world_coords = True

    if not world_coords:
        from dipy.tracking.streamline import transform_streamlines
        streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

    ren = window.Renderer()
    stream_actor = actor.line(streamlines)

    if not world_coords:
        image_actor = actor.slicer(vol, affine=np.eye(4))
    else:
        image_actor = actor.slicer(vol, affine)

    slicer_opacity = .6
    image_actor.opacity(slicer_opacity)

    ren.add(stream_actor)
    ren.add(image_actor)

    show_m = window.ShowManager(ren, size=(1200, 900))
    show_m.initialize()

    def change_slice(obj, event):
        z = int(np.round(obj.get_value()))
        image_actor.display_extent(0, shape[0] - 1,
                                   0, shape[1] - 1, z, z)

    slider = widget.slider(show_m.iren, show_m.ren,
                           callback=change_slice,
                           min_value=0,
                           max_value=shape[2] - 1,
                           value=shape[2] / 2,
                           label="Move slice",
                           right_normalized_pos=(.98, 0.6),
                           size=(120, 0), label_format="%0.lf",
                           color=(1., 1., 1.),
                           selected_color=(0.86, 0.33, 1.))

    global size
    size = ren.GetSize()

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            slider.place(ren)
            size = obj.GetSize()

    show_m.initialize()

    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()

    # ren.zoom(1.5)
    # ren.reset_clipping_range()

    # window.record(ren, out_path='bundles_and_a_slice.png', size=(1200, 900),
    #               reset_camera=False)

    del show_m


def simple_viewer(streamlines, vol, affine):

    from dipy.viz import actor, window

    renderer = window.Renderer()
    renderer.add(actor.line(streamlines))
    renderer.add(actor.slicer(vol, affine))
    window.show(renderer)


dname = '/Users/ghfc/Desktop/Celine/brainhack/data/'

fdwi = dname + '2dseq_conv_32.nii.gz'

data, affine = load_nifti(fdwi)

fbval = dname + 'P64_F01.bval'
fbvec = dname + 'P64_F01.bvec'

bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

gtab = gradient_table(bvals, bvecs, b0_threshold=50)

fmask = dname + 'mask_extern.nii'

mask_vol, mask_affine = load_nifti(fmask)

sh_order = 8
if data.shape[-1] < 15:
    raise ValueError('You need at least 15 unique DWI volumes to '
                     'compute fiber ODFs. You currently have: {0}'
                     ' DWI volumes.'.format(data.shape[-1]))
elif data.shape[-1] < 30:
    sh_order = 6

from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)

response, ratio = auto_response(gtab, data)
response = list(response)

peaks_sphere = get_sphere('symmetric362')

model = ConstrainedSphericalDeconvModel(gtab, response,
                                        sh_order=sh_order)

peaks_csd = peaks_from_model(model=model,
                             data=data,
                             sphere=peaks_sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             mask=mask_vol,
                             return_sh=True,
                             sh_order=sh_order,
                             normalize_peaks=True,
                             parallel=False)
peaks_csd.affine = affine

fpeaks = dname + 'peaks.npz'

save_peaks(fpeaks, peaks_csd)

from dipy.io.trackvis import save_trk
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier,
                                 LocalTracking)

stopping_thr = 0.25

pam = load_peaks(fpeaks)

ffa = dname + 'fa.nii.gz'

fa, fa_affine = load_nifti(ffa)

classifier = ThresholdTissueClassifier(fa,
                                       stopping_thr)

seed_density = 1

seed_mask = fa > 0.4

seeds = utils.seeds_from_mask(
    seed_mask,
    density=seed_density,
    affine=affine)

#if use_sh:
#    detmax_dg = \
#        DeterministicMaximumDirectionGetter.from_shcoeff(
#            pam.shm_coeff,
#            max_angle=30.,
#            sphere=pam.sphere)
#
#    streamlines = \
#        LocalTracking(detmax_dg, classifier, seeds, affine,
#                      step_size=.5)
#
#else:

streamlines = LocalTracking(pam, classifier,
                            seeds, affine=affine, step_size=.5)

# Compute streamlines and store as a list.
streamlines = list(streamlines)

ftractogram = dname + 'tractogram.trk'

save_trk(ftractogram, streamlines, affine)

show_results(streamlines[:1000], fa, fa_affine)
