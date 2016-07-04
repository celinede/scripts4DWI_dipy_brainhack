
# coding: utf-8

# In[ ]:

#get_ipython().magic(u'pylab')

from __future__ import division, print_function, absolute_import

import numpy as np
import nibabel as nib
from dipy.reconst.peaks import peaks_from_model, PeaksAndMetrics
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.tracking import utils
from common import load_nifti, save_trk, save_peaks, load_peaks, save_trk_old_style
from dipy.viz import actor, window, fvtk

#from ipdb import set_trace


# In[ ]:

def show_results(data, streamlines, vol, affine, world_coords=True, opacity=0.6):

    from dipy.viz import actor, window, widget

    shape = data.shape

    if not world_coords:
        from dipy.tracking.streamline import transform_streamlines
        streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

    ren = window.Renderer()
    if streamlines is not None:
        stream_actor = actor.line(streamlines)

    if not world_coords:
        image_actor = actor.slicer(vol, affine=np.eye(4))
    else:
        image_actor = actor.slicer(vol, affine)

    slicer_opacity = opacity #.6
    image_actor.opacity(slicer_opacity)

    if streamlines is not None:
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


# In[ ]:

def simple_viewer(streamlines, vol, affine):

    renderer = window.Renderer()
    renderer.add(actor.line(streamlines))
    renderer.add(actor.slicer(vol, affine))
    window.show(renderer)


# In[ ]:

def show_gradients(gtab):
    
    renderer = window.Renderer()
    renderer.add(fvtk.point(gtab.gradients, (1,0,0), point_radius=100))
    renderer.add(fvtk.point(-gtab.gradients, (1,0,0), point_radius=100))
    
    window.show(renderer)


# In[ ]:

def track(dname, fdwi, fbval, fbvec, fmask=None, seed_density = 1, show=False):
    
    data, affine = load_nifti(fdwi)
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=50)
    
    if fmask is None: 
        from dipy.segment.mask import median_otsu
        b0_mask, mask = median_otsu(data)  # TODO: check parameters to improve the mask
    else: 
        mask, mask_affine = load_nifti(fmask)
        mask = np.squeeze(mask) #fix mask dimensions 
        
    # compute DTI model
    from dipy.reconst.dti import TensorModel
    tenmodel = TensorModel(gtab)#, fit_method='OLS') #, min_signal=5000)
    
    # fit the dti model
    tenfit = tenmodel.fit(data, mask=mask)
    
    # save fa
    ffa = dname + 'tensor_fa.nii.gz'

    fa_img = nib.Nifti1Image(tenfit.fa.astype(np.float32), affine)
    nib.save(fa_img, ffa)
    
    sh_order = 8 #TODO: check what that does
    if data.shape[-1] < 15:
        raise ValueError('You need at least 15 unique DWI volumes to '
                         'compute fiber ODFs. You currently have: {0}'
                         ' DWI volumes.'.format(data.shape[-1]))
    elif data.shape[-1] < 30:
        sh_order = 6
        
    # compute the response equation ?
    from dipy.reconst.csdeconv import auto_response
    response, ratio = auto_response(gtab, data)
    response = list(response)
    
    peaks_sphere = get_sphere('symmetric362')

    #TODO: check what that does
    peaks_csd = peaks_from_model(model=tenmodel,
                                 data=data,
                                 sphere=peaks_sphere,
                                 relative_peak_threshold=.5, #.5
                                 min_separation_angle=25,
                                 mask=mask,
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

    stopping_thr = 0.25   #0.25

    pam = load_peaks(fpeaks)

    #ffa = dname + 'tensor_fa_nomask.nii.gz'
    fa, fa_affine = load_nifti(ffa)

    classifier = ThresholdTissueClassifier(fa,
                                           stopping_thr)
    
    # seeds 
    

    seed_mask = fa > 0.4 #0.4 #TODO: check this parameter

    seeds = utils.seeds_from_mask(
        seed_mask,
        density=seed_density,
        affine=affine)
    
    # tractography, if affine then in world coordinates
    streamlines = LocalTracking(pam, classifier,
                                seeds, affine=affine, step_size=.5)

    # Compute streamlines and store as a list.
    streamlines = list(streamlines)

    ftractogram = dname + 'tractogram.trk'

    #save .trk
    save_trk_old_style(ftractogram, streamlines, affine, fa.shape)

    if show:
        #render
        show_results(data,streamlines, fa, fa_affine)


# In[ ]:

def filterlength(dname, fdwi, ffa, ftrk, thr_length, show=False):
    
    fa_img = nib.load(ffa)
    fa = fa_img.get_data()
    affine = fa_img.get_affine()

    img = nib.load(fdwi)
    data = img.get_data()
    
    from nibabel import trackvis
    streams, hdr = trackvis.read(ftrk)
    streamlines = [s[0] for s in streams]
    
    # threshold on streamline length

    from dipy.tracking.utils import length
    lengths = list(length(streamlines))

    new_streamlines = [ s for s, l in zip(streamlines, lengths) if l > thr_length ] #3.5
    
    # info length streamlines

    print(len(streamlines))
    print(len(new_streamlines))

    print(max(length(streamlines)))
    print(min(length(streamlines)))

    print(max(length(new_streamlines)))
    print(min(length(new_streamlines)))
    
    # show new tracto

    new_streamlines = list(new_streamlines)
    new_lengths = list(length(new_streamlines))

    fnew_tractogram = dname + 'filteredtractogram.trk'
    save_trk_old_style(fnew_tractogram, new_streamlines, affine, fa.shape)

    if show:
        show_results(data, new_streamlines, fa, affine, opacity=0.6)

