def xrfmap(*, xstart, xnumstep, xstepsize, 
            ystart, ynumstep, ystepsize, 
            rois = []
	    #shutter = True, align = False,
            #acqtime, numrois=1, i0map_show=True, itmap_show=False, record_cryo = False,
            #setenergy=None, u_detune=None, echange_waittime=10
            ):

    '''
    input:
         xstart, xnumstep, xstepsize (float)
         ystart, ynumstep, ystepsize (float)
        
    '''

    #define detector used for xrf mapping functions
    xrfdet = [sclr] #currently only the scalar; to-do: save full spectra

    xstop = xstart + xnumstep*xstepsize
    ystop = ystart + ynumstep*ystepsize

    #setup live callbacks:

    livetableitem = [xy_stage.x, xy_stage.y]
    livecallbacks = []
    
    for roi in rois:
        livecallbacks.append(LiveGrid((ynumstep+1, xnumstep+1), roi, xlabel = 'x (mm)', ylabel = 'y (mm)', extent=[xstart, xstop, ystart, ystop]))
        livetableitem.append(roi)

    livecallbacks.append(LiveTable(livetableitem))

    xrfmap_scanplan = outer_product_scan(xrfdet, xy_stage.y, ystart, ystop, ynumstep+1, xy_stage.x, xstart, xstop, xnumstep+1, False)
    xrfmap_scanplan = bp.subs_wrapper(xrfmap_scanplan, livecallbacks)

    scaninfo = yield from xrfmap_scanplan

    return xrfmap_scanplan


