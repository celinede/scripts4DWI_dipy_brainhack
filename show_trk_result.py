def show_results(streamlines, vol, affine, world_coords=True, opacity=0.6):

    from dipy.viz import actor, window, widget
    import numpy as np
    shape = vol.shape

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

