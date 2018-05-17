from bluesky.callbacks.broker import LiveTiffExporter, post_run

xrfmapTiffOutputDir = '/home/xf08bm/DATA2017/Comissioning/20170619/' 
#hard-coded for testing now; need to be set to automatically use SAF, today's date, etc.

sclrDataKeyList = [sclr.cnts.channels.chan01.name,
                   sclr.cnts.channels.chan02.name,
                   sclr.cnts.channels.chan03.name,
                   sclr.cnts.channels.chan04.name,
                   sclr.cnts.channels.chan05.name,
                   sclr.cnts.channels.chan06.name,
                   sclr.cnts.channels.chan07.name,
                   sclr.cnts.channels.chan08.name,
                   sclr.cnts.channels.chan09.name,
                   sclr.cnts.channels.chan10.name,
                   sclr.cnts.channels.chan11.name,
                   sclr.cnts.channels.chan12.name,
                   sclr.cnts.channels.chan13.name,
                   sclr.cnts.channels.chan14.name,
                   sclr.cnts.channels.chan15.name,
                   sclr.cnts.channels.chan16.name,
                   sclr.cnts.channels.chan17.name,
                   sclr.cnts.channels.chan18.name,
                   sclr.cnts.channels.chan19.name,
                   sclr.cnts.channels.chan20.name]



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
    callbackTokenList = []
    
    for roi in rois:
        livecallbacks.append(LiveGrid((ynumstep+1, xnumstep+1), roi,
                                      xlabel='x (mm)',
                                      ylabel='y (mm)',
                                      extent=[xstart, xstop, ystart, ystop]))
        livetableitem.append(roi)


        #setup LiveOutput
        xrfmapOutputTiffTemplate = xrfmapTiffOutputDir+"xrfmap_scan{start[scan_id]}" +roi+ ".tiff"
#        xrfmapTiffexporter = LiveTiffExporter(roi, xrfmapOutputTiffTemplate, db=db)
        xrfmapTiffexporter = RasterMaker(xrfmapOutputTiffTemplate, roi)        
        callbackToken = RE.subscribe('all', post_run(xrfmapTiffexporter, db=db))
        #check to see if this is the right way to have RE here...
        callbackTokenList.append(callbackToken)

    livecallbacks.append(LiveTable(livetableitem))
    
    #setup LiveOutput

    if sclr in xrfdet:
        for sclrDataKey in sclrDataKeyList:
            xrfmapOutputTiffTemplate = xrfmapTiffOutputDir+"xrfmap_scan{start[scan_id]}" + sclrDataKey + ".tiff"
#           xrfmapTiffexporter = LiveTiffExporter(roi, xrfmapOutputTiffTemplate, db=db)  #LiveTiffExporter exports one array from one event, commented out for future reference
            xrfmapTiffexporter = RasterMaker(xrfmapOutputTiffTemplate, sclrDataKey)
            callbackToken = RE.subscribe('all', post_run(xrfmapTiffexporter, db=db))#check to see if this is the right way to have RE here...
            callbackTokenList.append(callbackToken)

    xrfmap_scanplan = outer_product_scan(xrfdet, xy_stage.y, ystart, ystop, ynumstep+1, xy_stage.x, xstart, xstop, xnumstep+1, False)
    xrfmap_scanplan = bp.subs_wrapper(xrfmap_scanplan, livecallbacks)

    scaninfo = yield from xrfmap_scanplan

    for callbackToken in callbackTokenList:
        RE.unsubscribe(callbackToken)

    return xrfmap_scanplan

