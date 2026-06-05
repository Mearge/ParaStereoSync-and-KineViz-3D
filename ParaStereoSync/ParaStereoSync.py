import os, sys
import time
import tkinter as tk
import tkinter.filedialog
import numpy as np
import scipy
import cv2
from multiprocessing import Pool, cpu_count, shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from eccTrackVideo_Multistep_warp_guess import eccTrackVideo
from inputs import input2
from pickTemplates import pickTemplates
from triangulatePoints2 import triangulatePoints2

# Global function for parallel camera processing (camera-level parallelism)
def parallel_eccTrackCamera(args):
    """
    Wrapper function to process an entire camera with all its points in parallel.
    This is the recommended approach for better performance.
    Args: tuple containing (icam, videoFilepath, tmpltFilepath, tmpltFrameId, frameRange, poisList, mTable_shape)
    Returns: tuple containing (icam, result_mTable)
    """
    icam, videoFilepath, tmpltFilepath, tmpltFrameId, frameRange, poisList, mTable_shape = args
    
    # Create a temporary mTable for this camera
    mTable = np.ones(mTable_shape, dtype=np.float64) * np.nan
    
    try:
        eccTrackVideo(videoFilepath=videoFilepath, tmpltFilepath=tmpltFilepath,
                      tmpltFrameId=tmpltFrameId, frameRange=frameRange,
                      tmpltRange=poisList, mTable=mTable, saveFilepath=None)
        return (icam, mTable)
    except Exception as e:
        print(f"# Error in parallel tracking for camera {icam+1}: {str(e)}")
        return (icam, mTable)  # Return empty mTable on error


# Global function for parallel eccTrackVideo processing (legacy, less efficient)
def parallel_eccTrackVideo(args):
    """
    Wrapper function to enable parallel processing of eccTrackVideo
    Args: tuple containing (icam, videoFilepath, tmpltFilepath, tmpltFrameId, frameRange, tmpltRange, mTable_shape)
    Returns: tuple containing (icam, result_mTable)
    """
    icam, videoFilepath, tmpltFilepath, tmpltFrameId, frameRange, tmpltRange, mTable_shape = args
    
    # Create a temporary mTable for this camera
    mTable = np.ones(mTable_shape, dtype=np.float64) * np.nan
    
    try:
        eccTrackVideo(videoFilepath=videoFilepath, tmpltFilepath=tmpltFilepath,
                      tmpltFrameId=tmpltFrameId, frameRange=frameRange,
                      tmpltRange=tmpltRange, mTable=mTable, saveFilepath=None)
        return (icam, mTable)
    except Exception as e:
        print(f"# Error in parallel tracking for camera {icam+1}: {str(e)}")
        return (icam, mTable)  # Return empty mTable on error


# Wrapper function to enable parallel processing of a single point of eccTrackVideo
def parallel_eccTrackPoint(args):
    """
    Process tracking for a single point in one camera.
    Args: tuple containing (icam, iPoi, videoFilepath, tmpltFilepath, tmpltFrameId, frameRange, mTable_shape)
    Returns: tuple containing (icam, iPoi, result_mTable)
    """
    icam, iPoi, videoFilepath, tmpltFilepath, tmpltFrameId, frameRange, mTable_shape = args
    # create temporary mTable shaped like the per-camera buffer eccTrackVideo expects
    mTable = np.ones(mTable_shape, dtype=np.float64) * np.nan
    try:
        # tmpltRange should be a list with single point index
        eccTrackVideo(videoFilepath=videoFilepath, tmpltFilepath=tmpltFilepath,
                      tmpltFrameId=tmpltFrameId, frameRange=frameRange,
                      tmpltRange=[iPoi], mTable=mTable, saveFilepath=None)
        col_start = iPoi * 5
        col_end = col_start + 5
        poi_slice = mTable[:, col_start:col_end].copy()
        return (icam, iPoi, poi_slice)
    except Exception as e:
        print(f"# Error in parallel point tracking for camera {icam+1} point {iPoi+1}: {str(e)}")
        col_start = iPoi * 5
        col_end = col_start + 5
        poi_slice = mTable[:, col_start:col_end].copy()
        return (icam, iPoi, poi_slice)

global win, bigTable, fVideos, fTmplts, fCalibs, videos, nFrames, nPoints, imgInits, tmplts, cmats, dvecs, rvecs, tvecs
global prjErr
bigTable = np.array([], dtype=np.float64)
fVideos = [None, None] 
fTmplts = [None, None]
fCalibs = [None, None]
videos = [None, None]
nFrames = [0, 0]
nPoints = [0, 0]
imgInits = [None, None]
tmplts = [None, None]
cmats = [None, None]
dvecs = [None, None]
rvecs = [None, None]
tvecs = [None, None]

def ParaStereoSync():
    global win
    win = tk.Tk()
    win.title("ParaStereoSync (v.0.1)")
    win.geometry("1400x700")

    # Create a scrollable main area to prevent bottom widgets from being covered
    main_container = tk.Frame(win)
    main_container.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_container, highlightthickness=0)
    v_scroll = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=v_scroll.set)

    v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    content = tk.Frame(canvas)
    content_window = canvas.create_window((0, 0), window=content, anchor="nw")

    def _on_content_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        canvas.itemconfigure(content_window, width=event.width)

    content.bind("<Configure>", _on_content_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    # Frame 1 about working directory
    frame1 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame1.pack(fill=tk.X)
    # Button btWorkDir
    btWorkDir = tk.Button(frame1, width=40, text="Working directory")
    btWorkDir.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txWorkDir
    txWorkDir = tk.Text(frame1, width=100, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txWorkDir.pack(side=tk.LEFT, padx=5, pady=2)

    # Frame 2 about videos
    frame2 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame2.pack(fill=tk.X)
    lbAboutVid = tk.Label(frame2, text="About videos", fg='gray')
    lbAboutVid.pack(side=tk.TOP, anchor=tk.W)
    # Button btVidFile1
    btVidFile1 = tk.Button(frame2, width=30, text="Video file 1")
    btVidFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txVidFile1
    txVidFile1 = tk.Text(frame2, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txVidFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btVidFile2
    btVidFile2 = tk.Button(frame2, width=30, text="Video file 2")
    btVidFile2.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txVidFile2
    txVidFile2 = tk.Text(frame2, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txVidFile2.pack(side=tk.LEFT, padx=5, pady=2)

    # Frame 3 about calibration
    frame3 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame3.pack(fill=tk.X)
    lbAboutCalibration = tk.Label(frame3, text="About calibration", fg='gray')
    lbAboutCalibration.pack(side=tk.TOP, anchor=tk.W)
    # Button btCalibFile1
    btCalibFile1 = tk.Button(frame3, width=30, text="Calibration file 1")
    btCalibFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibFile1
    txCalibFile1 = tk.Text(frame3, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txCalibFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btCalibFile2
    btCalibFile2 = tk.Button(frame3, width=30, text="Calibration file 2")
    btCalibFile2.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibFile2
    txCalibFile2 = tk.Text(frame3, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txCalibFile2.pack(side=tk.LEFT, padx=5, pady=2)
    # Frame 4a about templates frame 
    frame4a = tk.Frame(content, highlightbackground="gray", highlightthickness=0)
    frame4a.pack(fill=tk.X)
    # Label template frame ID (the frame that defines template images)
    lbAboutTmplts = tk.Label(frame4a, text="About tracking points (templates)", fg='gray')
    lbAboutTmplts.pack(side=tk.TOP, anchor=tk.W)
    # Text txTmpltFrameId1 and txTmpltFrameId2
    lbTmpltFrameId1 = tk.Label(frame4a, text="Frame of templates of camera 1 (1 for the 1st frame)", fg='gray')
    lbTmpltFrameId1.pack(side=tk.LEFT, padx=5, pady=2)
    txTmpltFrameId1 = tk.Text(frame4a, width=40, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTmpltFrameId1.pack(side=tk.LEFT, padx=5, pady=2)
    lbTmpltFrameId2 = tk.Label(frame4a, text="Frame of templates of camera 2 (1 for the 1st frame)", fg='gray')
    lbTmpltFrameId2.pack(side=tk.LEFT, padx=5, pady=2)
    txTmpltFrameId2 = tk.Text(frame4a, width=40, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTmpltFrameId2.pack(side=tk.LEFT, padx=5, pady=2)
    # Frame 4 about templates
    frame4 = tk.Frame(content, highlightbackground="gray", highlightthickness=0)
    frame4.pack(fill=tk.X)
    # Button btTmpltFile1
    btTmpltFile1 = tk.Button(frame4, width=30, text="Templates file 1")
    btTmpltFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btNewTF1
    btNewTF1 = tk.Button(frame4, width=10, text="New TF1")
    btNewTF1.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibText1
    txTmpltFile1 = tk.Text(frame4, width=38, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTmpltFile1.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btCalibFile1
    btTmpltFile2 = tk.Button(frame4, width=30, text="Templates file 2")
    btTmpltFile2.pack(side=tk.LEFT, padx=5, pady=2)
    # Button btNewTF2
    btNewTF2 = tk.Button(frame4, width=10, text="New TF2")
    btNewTF2.pack(side=tk.LEFT, padx=5, pady=2)
    # Text txCalibFile2
    txTmpltFile2 = tk.Text(frame4, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTmpltFile2.pack(side=tk.LEFT, padx=5, pady=2)


    # Frame 5 about tracking
    frame5 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame5.pack(fill=tk.X, pady=2)
    lbAboutTracking = tk.Label(frame5, text="About tracking", fg='gray')
    lbAboutTracking.grid(row=0, column=0, sticky="W")
    # Label lbTrackCamRange
    lbTrackCamRange = tk.Label(frame5, width=30, text="Range of tracking cameras")
    lbTrackCamRange.grid(row=1, column=0, padx=5, pady=2)
    # Text txTrackCamRange
    txTrackCamRange = tk.Text(frame5, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTrackCamRange.grid(row=1, column=1, padx=5, pady=2)
    # Label lbTrackPointRange
    lbTrackPointRange = tk.Label(frame5, width=30, text="Range of tracking points")
    lbTrackPointRange.grid(row=2, column=0, padx=5, pady=2)
    # Text txTrackPointRange
    txTrackPointRange = tk.Text(frame5, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTrackPointRange.grid(row=2, column=1, padx=5, pady=2)
    # Label lbTrackFrameRange
    lbTrackFrameRange = tk.Label(frame5, width=30, text="Range of tracking frames")
    lbTrackFrameRange.grid(row=3, column=0, padx=5, pady=2)
    # Text txTrackFrameRange
    txTrackFrameRange = tk.Text(frame5, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTrackFrameRange.grid(row=3, column=1, padx=5, pady=2)
    # Button btImageTrack
    btImageTrack = tk.Button(frame5, width=30, text="Image Track")
    btImageTrack.grid(row=3, column=2, padx=5, pady=2)
    
    # Checkbox for parallel processing
    parallelVar = tk.BooleanVar(value=True)  # Default to enabled
    cbParallel = tk.Checkbutton(frame5, text="Parallel Processing", variable=parallelVar)
    cbParallel.grid(row=4, column=0, padx=5, pady=2, sticky="W")
    
    # Progress label
    lbProgress = tk.Label(frame5, text="Ready", fg='green')
    lbProgress.grid(row=4, column=1, padx=5, pady=2, sticky="W")

    # Frame 6 about synchronization analysis
    frame6 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame6.pack(fill=tk.X, pady=2)
    lbAboutSync = tk.Label(frame6, text="About synchronization analysis", fg='gray')
    lbAboutSync.grid(row=0, column=0, sticky="W")
    # Label lbPossibleLag
    lbPossibleLag = tk.Label(frame6, width=30, text="Range of possible lag")
    lbPossibleLag.grid(row=1, column=0, padx=5, pady=2)
    # Text txPossibleLag
    txPossibleLag = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txPossibleLag.grid(row=1, column=1, padx=5, pady=2)
    # Label lbSyncPoiRange
    lbSyncPoiRange = tk.Label(frame6, width=30, text="Range of points for sync analysis")
    lbSyncPoiRange.grid(row=2, column=0, padx=5, pady=2)
    # Text txSyncPoiRange
    txSyncPoiRange = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSyncPoiRange.grid(row=2, column=1, padx=5, pady=2)
    # Label lbSyncFrameRange
    lbSyncFrameRange = tk.Label(frame6, width=30, text="Range of frames for sync analysis")
    lbSyncFrameRange.grid(row=3, column=0, padx=5, pady=2)
    # Text txSyncFrameRange
    txSyncFrameRange = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSyncFrameRange.grid(row=3, column=1, padx=5, pady=2)
    # Button btFindTimeLags
    btFindTimeLags = tk.Button(frame6, width=30, text="Find best time lags (each point)")
    btFindTimeLags.grid(row=1, column=2, padx=5, pady=2)
    # Button btLags
    lbLags = tk.Label(frame6, width=30, text="Time lags (each point)")
    lbLags.grid(row=2, column=2, padx=5, pady=2)
    # Text txLags
    txLags = tk.Text(frame6, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txLags.grid(row=2, column=3, padx=5, pady=2)

    # Frame 7 about triangulation
    frame7 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame7.pack(fill=tk.X, pady=2)
    lbAboutTriang = tk.Label(frame7, text="About triangulation", fg='gray')
    lbAboutTriang.grid(row=0, column=0, sticky="W")
    # Label lbTriangPoiRange
    lbTriangPoiRange = tk.Label(frame7, width=30, text="Range of triangulation points")
    lbTriangPoiRange.grid(row=1, column=0, padx=5, pady=2)
    # Text txTriangPoiRange
    txTriangPoiRange = tk.Text(frame7, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTriangPoiRange.grid(row=1, column=1, padx=5, pady=2)
    # Label lbTriangFrameRange
    lbTriangFrameRange = tk.Label(frame7, width=30, text="Range of triangulation frames")
    lbTriangFrameRange.grid(row=2, column=0, padx=5, pady=2)
    # Text txTriangFrameRange
    txTriangFrameRange = tk.Text(frame7, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txTriangFrameRange.grid(row=2, column=1, padx=5, pady=2)
    # Button Triangulation
    btTriang = tk.Button(frame7, width=30, text="Triangulation")
    btTriang.grid(row=1, column=2, padx=5, pady=2)
    
    # Frame 8 about load and save 
    frame8 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame8.pack(fill=tk.X, pady=2)
    lbAboutSave = tk.Label(frame8, text="About loading/saving files", fg='gray')
    lbAboutSave.grid(row=0, column=0, sticky="W")
    # Button btLoadConfig
    btLoadConfig = tk.Button(frame8, width=30, text="Load config")
    btLoadConfig.grid(row=1, column=0, padx=5, pady=2)
    # Text txLoadConfig
    txLoadConfig = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txLoadConfig.grid(row=1, column=1, padx=5, pady=2)
    txLoadConfig.delete('0.0', 'end')
    txLoadConfig.insert('0.0', 'Stereosync_config.xml')
    # Button btLoadData
    btLoadData = tk.Button(frame8, width=30, text="Load data")
    btLoadData.grid(row=2, column=0, padx=5, pady=2)
    # Text txLoadData
    txLoadData = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txLoadData.grid(row=2, column=1, padx=5, pady=2)
    txLoadData.delete('0.0', 'end')
    txLoadData.insert('0.0', 'Stereosync_data.csv')
    # Button btSaveConfig
    btSaveConfig = tk.Button(frame8, width=30, text="Save config")
    btSaveConfig.grid(row=1, column=2, padx=5, pady=2)
    # Text txSaveConfig
    txSaveConfig = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSaveConfig.grid(row=1, column=3, padx=5, pady=2)
    txSaveConfig.delete('0.0', 'end')
    txSaveConfig.insert('0.0', 'Stereosync_config.xml')
    # Button btSaveData
    btSaveData = tk.Button(frame8, width=30, text="Save data")
    btSaveData.grid(row=2, column=2, padx=5, pady=2)
    # Text txSaveConfig
    txSaveData = tk.Text(frame8, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txSaveData.grid(row=2, column=3, padx=5, pady=2)
    txSaveData.delete('0.0', 'end')
    txSaveData.insert('0.0', 'Stereosync_data.csv')

    # Frame 9 about visualization
    frame9 = tk.Frame(content, highlightbackground="gray", highlightthickness=1)
    frame9.pack(fill=tk.X, pady=2)
    lbAboutPlot = tk.Label(frame9, text="About plotting", fg='gray')
    lbAboutPlot.grid(row=0, column=0, sticky="W")
    # Button btPlot
    btPlot = tk.Button(frame9, width=30, text="Plot")
    btPlot.grid(row=1, column=0, padx=5, pady=2)
    # Text txPlot
    txPlot = tk.Text(frame9, width=50, height=1, undo=True, 
                        autoseparators=True, maxundo=-1)
    txPlot.grid(row=1, column=1, padx=5, pady=2)
    txPlot.delete('0.0', 'end')
    txPlot.insert('0.0', '1 402 1 403 1 404')
     

    # bind event of btWorkDir
    def event_btWorkDir(event):
        tmpwin = tk.Tk()
        tmpwin.lift()    
        workDir = tk.filedialog.askdirectory(title="Select the working directory")
        tmpwin.destroy()
        # quit if user cancels the file dialog
        if type(workDir) == type('') and len(workDir) == 0:
            print("You cancelled the directory dialog.")
            return
        # display workDir to txWorkDir 
        txWorkDir.delete("1.0", "end")
        txWorkDir.insert("1.0", workDir)
        # load config
        loadConfig()
        # load data 
        loadData()
    btWorkDir.bind("<ButtonRelease-1>", event_btWorkDir)

    ####################################
    # bind event of btVidFile1, btVidFile2, btCalibFile1, btCalibFile2, btTmpltFile1, btTmpltFile2
    # as they have similar behaviors, we use a general function event_btFileDialogAndShow()
    ####################################
    def event_btFileDialogAndShow(event, title, display):
        tmpwin = tk.Tk()
        tmpwin.lift()
        initialdir = txWorkDir.get("1.0", "end").strip()
        if os.path.isdir(initialdir) == False:
            initialdir = "c:/"
        vfile = tk.filedialog.askopenfilename(title=title,
                                              initialdir=initialdir)
        tmpwin.destroy()
        # quit if user cancels the file dialog
        if type(vfile) == type('') and len(vfile) == 0:
            print("You cancelled the file dialog.")
            return
        # display file to display widget 
        if os.path.dirname(vfile) == txWorkDir.get("1.0", "end").strip():
            display.delete("1.0", "end")
            display.insert("1.0", os.path.basename(vfile))
        else:
            display.delete("1.0", "end")
            display.insert("1.0", vfile)
            
    # When user clicks "Video file 1" or "Video file 2"
    btVidFile1.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the video file of video 1",
                      display=txVidFile1))
    btVidFile2.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the video file of video 2",
                      display=txVidFile2))
    # When user clicks "Calibration file 1" or "Calibration file 2"
    btCalibFile1.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the calibration file of video 1",
                      display=txCalibFile1))
    btCalibFile2.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the calibration file of video 2",
                      display=txCalibFile2))
    # When user clicks "Templates file 1" or "Templates file 2"
    btTmpltFile1.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the templates file of video 1",
                      display=txTmpltFile1))
    btTmpltFile2.bind("<ButtonRelease-1>", lambda event:
                    event_btFileDialogAndShow(event,
                      title="Select the templates file of video 2",
                      display=txTmpltFile2))
    # When user clicks "New TF1" or "New TF2"
    def event_btNewTFx(event):
        tmpwin = tk.Tk()
        tmpwin.lift()
        tk.messagebox.showinfo("Information", "Define templates by mouse. Switch to terminal to interact with this program.")
        if event.widget == btNewTF1:
            icam = 0
            txTmpltFilex = txTmpltFile1
            txVidFilex = txVidFile1
            txTmpltFrameIdx = txTmpltFrameId1
        if event.widget == btNewTF2:
            icam = 1
            txTmpltFilex = txTmpltFile2
            txVidFilex = txVidFile2
            txTmpltFrameIdx = txTmpltFrameId2
        print("# You are about to define camera %d templates by mouse now." % (icam+1))
        # Which frame index is used to define images of templates (tmpltFrameId)
        # Normally you would select tmpltFrameId as 0, meaning the very first frame is used to define templates.
        # However some video slightly vibrates at beginning because finger pressing the "start" button on camera.
        # In that case, the templates is defined at an image frame after the vibration dissipates. 
        print("# Which frame do you want to define templates? (1 for the very beginning frame. -1 or 0 to exit.):")
        tmpltFrameId_tmp = int(input2())
        if tmpltFrameId_tmp < 1:
            return
        tmpltFrameId = tmpltFrameId_tmp
        txTmpltFrameIdx.delete('0.0', 'end')
        txTmpltFrameIdx.insert('0.0', '%d' % tmpltFrameId_tmp)
        print("# Enter number of POIs for video %d: (-1 to exit)" % (icam+1))
        npPick = int(input2())
        workDir = txWorkDir.get('0.0', 'end').strip()
        try:
            vx = cv2.VideoCapture(os.path.join(workDir, txVidFilex.get('0.0', 'end').strip()))
            if vx.isOpened():
                for i in range(tmpltFrameId):
                    ret, imgInits[icam] = vx.read()
                imgInits[icam] = cv2.cvtColor(imgInits[icam], cv2.COLOR_BGR2GRAY)
            else:
                print("# Failed to open video file %s." % fVideos[icam])
                return
        except:
            print("# Failed to open video file %s." % fVideos[icam])    
            return
        vx.release()
        # Set new template file name
        fTmpltsBase = 'new_templates_%d.csv' % (icam+1)
        fTmplts[icam] = os.path.join(workDir, fTmpltsBase)
        txTmpltFilex.delete('0.0', 'end')
        txTmpltFilex.insert('0.0', fTmpltsBase)
        fSplitExt = os.path.splitext(fTmplts[icam])
        fTmpltImg = fSplitExt[0] + '_pickedTemplates' + '.JPG'
        results = pickTemplates(
                    img = imgInits[icam], 
                    nPoints=npPick,
                    savefile=fTmplts[icam], 
                    saveImgfile=fTmpltImg)
        print("# Completed defining templates for camera %d." % (icam+1))
    # end of def event_btNewTFx():
    btNewTF1.bind("<ButtonRelease-1>", event_btNewTFx)
    btNewTF2.bind("<ButtonRelease-1>", event_btNewTFx)
    
    # From colon-separate integer range to list of integer 
    # Range is in Matlab style, not Python style.
    # That is, "2:5" mean 2 3 4 5,　not 2 3 4. 
    # For example, colonRangeToIntList("2:5  10  15  17:20") would be [2, 3, 4, 5, 10, 15, 17, 18, 19, 20]
    def colonRangeToIntList(theStr):
        sequences = []
        for part in theStr.split():
            try:
                # Try converting the part to an integer (single number)
                num = int(part)
                sequences.append(num)
            except ValueError:
                # If conversion fails, assume it's a colon-separated range
                start, end = map(int, part.split(":"))
                sequences.extend(range(start, end+1))
        return sequences

    # bind event of btImageTrack ("Image Track")
    def event_btImageTrack(event):
        global bigTable, nPoints
        
        # Update progress
        lbProgress.config(text="Starting tracking...", fg='orange')
        win.update()
        tracking_start_time = time.time()
        
        # Determine optimal worker count (less is more for I/O-bound tasks)
        optimal_workers = min(4, cpu_count())  # Use 4 workers max to reduce I/O contention
        
        # get cameras list (E.g., camsList = [1, 2])
        camsText = txTrackCamRange.get("0.0", "end").strip()
        camsList_1base = colonRangeToIntList(camsText)
        camsList = [num - 1 for num in camsList_1base] # from 1-base to 0-base
        # get frame range (E.g., frameRange = [1, 4898]). Frames must be continuous integers.
        # Only start and end need to be assigned. 
        framesText = txTrackFrameRange.get("0.0", "end").strip()
        framesList_1base = colonRangeToIntList(framesText)
        framesRange = np.array([min(framesList_1base)-1, max(framesList_1base)-1], dtype=int).flatten()
        # get points list (E.g., poiList = [1, 2, 3, ... ])
        poisText = txTrackPointRange.get("0.0", "end").strip()
        poisList_1base = colonRangeToIntList(poisText)
        poisList = [num - 1 for num in poisList_1base] # from 1-base to 0-base
        # get template frame id
        tmpltFrameIds = [0, 0]
        try:
            tmpltFrameIds[0] = int(txTmpltFrameId1.get('0.0', 'end').strip()) - 1
        except:
            tmpltFrameIds[0] = 0
            txTmpltFrameId1.delete('0.0', 'end')
            txTmpltFrameId1.insert('0.0', '1')
        try:
            tmpltFrameIds[1] = int(txTmpltFrameId2.get('0.0', 'end').strip()) - 1
        except:
            tmpltFrameIds[1] = 0
            txTmpltFrameId2.delete('0.0', 'end')
            txTmpltFrameId2.insert('0.0', '1')
        # get templates (tmplts[icam][iPoi, 0:6] defines the template of point ipoi of camera icam)
        fTmplts = [None, None]
        # get file names of videos and templates
        workDir = txWorkDir.get('0.0', 'end').strip()
        fVideos[0] = os.path.join(workDir, txVidFile1.get('0.0', 'end').strip())
        fVideos[1] = os.path.join(workDir, txVidFile2.get('0.0', 'end').strip())
        fTmplts[0] = os.path.join(workDir, txTmpltFile1.get('0.0', 'end').strip())
        fTmplts[1] = os.path.join(workDir, txTmpltFile2.get('0.0', 'end').strip())
        # get nPoints - try without skipping rows first, then with skiprows if it fails
        try:
            tmplts[0] = np.loadtxt(fTmplts[0], delimiter=',')
        except:
            tmplts[0] = np.loadtxt(fTmplts[0], delimiter=',', skiprows=1)
        try:
            tmplts[1] = np.loadtxt(fTmplts[1], delimiter=',')
        except:
            tmplts[1] = np.loadtxt(fTmplts[1], delimiter=',', skiprows=1)
        nPoints[0] = tmplts[0].shape[0]
        nPoints[1] = tmplts[1].shape[0]
     
        # create video objects
        for icam in range(2):
            videos[icam] = cv2.VideoCapture(fVideos[icam])
            if not videos[icam].isOpened(): 
                print("# Error opening video file %d" % icam)
            nFrames[icam] = round(videos[icam].get(cv2.CAP_PROP_FRAME_COUNT))
        nFrameMax = np.max(nFrames)
        # Allocate memory for bigTable (324 points with gaps)
        # Structure: 1 + 1620 (cam0) + 10 (gap) + 1620 (cam1) + 10 (gap) + 2268 (triangulation) = 5529
        if bigTable.shape[0] != nFrameMax or bigTable.shape[1] != 5529:
            bigTable = np.ones((nFrameMax, 5529), dtype=np.float64) * np.nan
            bigTable[:,0] = np.linspace(1, nFrameMax, nFrameMax)
        
        # Decide tracking mode
        use_parallel = parallelVar.get()
        if use_parallel:
            print(f"# Starting camera-level parallel tracking using {optimal_workers} workers (optimized for I/O)...")
            print(f"# This parallelizes cameras simultaneously to maximize throughput")
        else:
            print(f"# Starting sequential tracking for {len(camsList)} cameras...")

        points = list(poisList)
        if len(points) == 0:
            print(f"# No points to process.")
            lbProgress.config(text="No points to process", fg='red')
            win.update()
            return

        # NEW APPROACH: Parallelize cameras (much better for I/O-bound tasks)
        if use_parallel:
            # Process both cameras in parallel
            futures = {}
            with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                for icam in camsList:
                    videoFilepath = fVideos[icam]
                    tmpltFilepath = fTmplts[icam]
                    tmpltFrameId = tmpltFrameIds[icam]
                    frameRange = framesRange
                    mTable_shape = (nFrameMax, 1620)  # 324 points * 5 columns per point
                    
                    args = (icam, videoFilepath, tmpltFilepath, tmpltFrameId, frameRange, points, mTable_shape)
                    futures[executor.submit(parallel_eccTrackCamera, args)] = icam
                    print(f"# Submitted camera {icam+1} for parallel processing...")
                    lbProgress.config(text=f"Submitted camera {icam+1}...", fg='orange')
                    win.update()

                # Collect results as they complete
                for future in as_completed(futures):
                    icam = futures[future]
                    try:
                        icam_result, mTable_result = future.result()
                        # Copy results to bigTable (cam0: 1-1620, cam1: 1631-3250)
                        cam_offset = 0 if icam_result == 0 else 1630
                        bigTable[:, (1+cam_offset):(1620+cam_offset+1)] = mTable_result[:, 0:1620]
                        print(f"# Completed tracking for camera {icam_result+1}")
                        lbProgress.config(text=f"Completed camera {icam_result+1}", fg='blue')
                        win.update()
                    except Exception as e:
                        print(f"# Error processing camera {icam+1}: {str(e)}")
                        lbProgress.config(text=f"Error in camera {icam+1}", fg='red')
                        win.update()
        else:
            # Sequential processing (original approach)
            for icam in camsList:
                videoFilepath = fVideos[icam]
                tmpltFilepath = fTmplts[icam]
                tmpltFrameId = tmpltFrameIds[icam]
                frameRange = framesRange

                print(f"# Processing camera {icam+1}...")
                lbProgress.config(text=f"Processing camera {icam+1}...", fg='orange')
                win.update()

                # Camera offsets: cam0=0, cam1=1630 (includes 10 col gap)
                cam_offset = 0 if icam == 0 else 1630
                mTable = bigTable[:, (1+cam_offset):(1620+cam_offset+1)]  # 324 points * 5 columns
                try:
                    eccTrackVideo(videoFilepath=videoFilepath, tmpltFilepath=tmpltFilepath,
                                  tmpltFrameId=tmpltFrameId, frameRange=frameRange,
                                  tmpltRange=points, mTable=mTable, saveFilepath=None)
                    print(f"# Completed tracking for camera {icam+1}")
                    lbProgress.config(text=f"Completed camera {icam+1}", fg='blue')
                    win.update()
                except Exception as e:
                    print(f"# Error processing camera {icam+1}: {str(e)}")
                    lbProgress.config(text=f"Error in camera {icam+1}", fg='red')
                    win.update()
        
        print("# All tracking completed!")
        lbProgress.config(text="All tracking completed!", fg='green')
        win.update()
        # save bigTable to file
        event_btSaveData(event)
        total_seconds = time.time() - tracking_start_time
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"# Total tracking time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} (hh:mm:ss)")
        pass
    # end of event_btImageTrack()
    # bind button "Image Track" to event_btImageTrack
    btImageTrack.bind("<ButtonRelease-1>", event_btImageTrack)

    # bind event of button "Save data" to event_btSaveData   
    def event_btSaveData(event):
        global bigTable
        workDir = txWorkDir.get('0.0', 'end').strip()
        fBigTable = os.path.join(workDir, txSaveData.get('0.0', 'end').strip())
        np.savetxt(fBigTable, bigTable, delimiter=",")
    btSaveData.bind("<ButtonRelease-1>", event_btSaveData)

    # bind event of "Find best time lags"
    def event_btFindTimeLags(event):
        global bigTable, nFrames, cmats, dvecs, rvecs, tvecs
        global prjErr
        print("# event_btFindTimeLags().")
        # get nPoints
        workDir = txWorkDir.get('0.0', 'end').strip()
        fTmplts[0] = os.path.join(workDir, txTmpltFile1.get('0.0', 'end').strip())
        fTmplts[1] = os.path.join(workDir, txTmpltFile2.get('0.0', 'end').strip())
        try:
            tmplts[0] = np.loadtxt(fTmplts[0], delimiter=',')
        except:
            tmplts[0] = np.loadtxt(fTmplts[0], delimiter=',', skiprows=1)
        try:
            tmplts[1] = np.loadtxt(fTmplts[0], delimiter=',')
        except:
            tmplts[1] = np.loadtxt(fTmplts[0], delimiter=',', skiprows=1)
        nPoints[0] = tmplts[0].shape[0]
        nPoints[1] = tmplts[1].shape[0]
        # Range of possible lags. t_lag_trials needs to be continuous integer list.
        tLagTrialsText = txPossibleLag.get('0.0', 'end').strip()
        t_lag_trials = np.array(colonRangeToIntList(tLagTrialsText), dtype=int)
        t_lag_bounds = [np.min(t_lag_trials), np.max(t_lag_trials)]
        # Range of points for sync analysis
        poiRangeForSyncTxt = txSyncPoiRange.get('0.0', 'end').strip()
        poiRangeForSync = np.array(colonRangeToIntList(poiRangeForSyncTxt), dtype=int)
        # Range of sync analysis frames. t_interest_range needs to be continuous integer list.
        t_interest_text = txSyncFrameRange.get('0.0', 'end').strip()
        t_interest = np.array(colonRangeToIntList(t_interest_text), dtype=int)
        t_interest -= 1 # convert from 1-based index to 0-based index. Frame 1 for user is frame 0 for numpy.
        t_interest_range = [np.min(t_interest), np.max(t_interest)]
        # Clear Time lags (each point)
        txLags.delete('0.0', 'end')
        # check 
        # t_interest[0] - t_lag_bounds[1] should be >= 0.0 
        # That is, right cam may have a time lag up to t_lag_bounds[1], 
        # and if t_interest[0] is too small, right cam would miss 
        # the beginning part of the video time of interests.
        t_interest_range_forcedToChange = False
        t_interest_range_original = t_interest_range.copy()
        if t_interest_range[0] - t_lag_bounds[1] < 0.0:
            t_interest_range[0] = int(t_lag_bounds[1] + 1) 
            t_interest_range_forcedToChange = True
        # t_v1_valid[1] - t_lag_bounds[0] should be < (nFrames[1]-1)
        # That is, right cam may have a minimum t_lag_bounds[0],
        # and if nFrames[1]-1 is not long enough, right cam would miss 
        # the ending part of the video time of interests.
        if t_interest_range[1] - t_lag_bounds[0] > (nFrames[1]-1):
            t_interest_range[1] = int(nFrames[1]-1 + t_lag_bounds[0]) 
            t_interest_range_forcedToChange = True
        if t_interest_range_forcedToChange:
            print("# Warning: t_lag_bounds is changed from (%d,%d) to (%d,%d)" %
                (t_interest_range_original[0], t_interest_range_original[1], 
                t_interest_range[0],          t_interest_range[1]))
        # t_interest[icam]: time tags of frames of interests 
        #   For example, if you are only interested in the triangulation of 
        #   frames 300 to 3900 in cam 0, t_interest = [300, 301, ..., 3900]
        #   and t_v2_interest = [330, 331, 332., ...] (if t_lag_trial is 30)
        nFrameInterest = round(t_interest_range[1] - 
                            t_interest_range[0])
        t_interest = np.linspace(t_interest_range[0], 
                                t_interest_range[1],
                                nFrameInterest+1)
        #
        imgPoints1_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        imgPoints2_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        prjErr = np.zeros((min(nPoints), t_lag_trials.size), dtype=float)
        ticMeasures = np.zeros(100)
        for ilag in range(t_lag_trials.size):
            tic = time.time()
            t_lag = t_lag_trials[ilag]
            # t_v1 and t_v2: based on the t_tag, set time tags of every frame   
            # t_v1[] would be [0., 1., 2., ...]
            # t_v2[] would be [30., 31., 32., ...] if video 2 lags 30 seconds (turned on 30s later)    
            t_v1 = np.linspace(0, nFrames[0] - 1, round(nFrames[0]))
            t_v2 = np.linspace(0, nFrames[1] - 1, round(nFrames[1])) + t_lag
            ticMeasures[0] += time.time() - tic
            for iPoi in range(min(nPoints)):
                # print info
                # image points of this point in cam 0
                tic = time.time()
                icam = 0
                cam_offset = 0  # Camera 0 offset
                xi = bigTable[0:t_v1.size, iPoi*5+cam_offset+1]
                yi = bigTable[0:t_v1.size, iPoi*5+cam_offset+2]
                imgPoints1_xyi_thisPoi_interest[:,0] = xi[
                    round(t_interest[0]):round(t_interest[-1])+1].copy()
                imgPoints1_xyi_thisPoi_interest[:,1] = yi[
                    round(t_interest[0]):round(t_interest[-1])+1].copy()
                ticMeasures[1] += time.time() - tic    
                # image points of this point in cam 1

#--------------------------------------------------------------------------------------------------------
                 # Extract the slice and ensure it matches the target size
                start_idx = int(round(t_interest[0]))
                end_idx = int(round(t_interest[-1])) + 1
                xi_slice = xi[start_idx:end_idx].copy()
                yi_slice = yi[start_idx:end_idx].copy()

                # Ensure the slice size matches the target array size
                target_size = imgPoints1_xyi_thisPoi_interest.shape[0]
                if xi_slice.size != target_size:
                    print(f"# Warning: Size mismatch for point {iPoi+1}. Expected {target_size}, got {xi_slice.size}")
                    # Resize to match - either truncate or pad with last values
                    if xi_slice.size > target_size:
                        xi_slice = xi_slice[:target_size]
                        yi_slice = yi_slice[:target_size]
                    elif xi_slice.size < target_size:
                        # Pad with the last available values
                        if xi_slice.size > 0:
                            xi_padded = np.full(target_size, xi_slice[-1])
                            yi_padded = np.full(target_size, yi_slice[-1])
                            xi_padded[:xi_slice.size] = xi_slice
                            yi_padded[:yi_slice.size] = yi_slice
                            xi_slice = xi_padded
                            yi_slice = yi_padded
                        else:
                            # If no data available, fill with NaN
                            xi_slice = np.full(target_size, np.nan)
                            yi_slice = np.full(target_size, np.nan)
                
                imgPoints1_xyi_thisPoi_interest[:,0] = xi_slice
                imgPoints1_xyi_thisPoi_interest[:,1] = yi_slice
                ticMeasures[1] += time.time() - tic    

#--------------------------------------------------------------------------------------------------------
 # create interpolation objects for cam 2 (v2)
                tic = time.time()                
                icam = 1
                cam_offset = 1630  # Camera 1 offset (1620 + 10 gap)
                xi = bigTable[0:t_v2.size, iPoi*5+cam_offset+1]
                yi = bigTable[0:t_v2.size, iPoi*5+cam_offset+2]

                # in case xi has nan, fill all nan in xi with its previous value
                if np.isnan(xi[0]):
                    xi[0] = 0.0
                    print("# Warning: Cam 2 Point %d starts from nan. " % (iPoi))
                for iiFrame in range(xi.size):
                    if np.isnan(xi[iiFrame]):
                        xi[iiFrame] = xi[iiFrame - 1]
                if np.isnan(yi[0]):
                    yi[0] = 0.0
                    print("# Warning: Cam 2 Point %d starts from nan. " % (iPoi))
                for iiFrame in range(yi.size):
                    if np.isnan(yi[iiFrame]):
                        yi[iiFrame] = yi[iiFrame - 1]
                ticMeasures[2] += time.time() - tic    

                tic = time.time()
                intpv2xi = scipy.interpolate.interp1d(t_v2, xi, kind='cubic')
                intpv2yi = scipy.interpolate.interp1d(t_v2, yi, kind='cubic')
                imgPoints2_xyi_thisPoi_interest[:,0] = intpv2xi(t_interest)
                imgPoints2_xyi_thisPoi_interest[:,1] = intpv2yi(t_interest)
                ticMeasures[3] += time.time() - tic    
                # plot for debug
                debugPlotSync = False
                if debugPlotSync and ilag == 0 and iPoi == 0:
                    import matplotlib.pyplot as plt
                    plt.plot(t_interest, imgPoints1_xyi_thisPoi_interest[:,0] - imgPoints1_xyi_thisPoi_interest[:,0][0], '.-', markersize=3, label='Cam 1')
                    plt.plot(t_interest, imgPoints2_xyi_thisPoi_interest[:,0] - imgPoints2_xyi_thisPoi_interest[:,0][0], '.-', markersize=3, label='Cam 2')
                    plt.legend()
                    plt.show()
                # Triangulation
                tic = time.time()
                objPoints, objPoints1, objPoints2, \
                    prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
                    triangulatePoints2(cmats[0], dvecs[0], rvecs[0], tvecs[0], 
                                    cmats[1], dvecs[1], rvecs[1], tvecs[1], 
                                    imgPoints1_xyi_thisPoi_interest, 
                                    imgPoints2_xyi_thisPoi_interest)
                ticMeasures[4] += time.time() - tic    
                # debug
                debugPlotPrjErrors = False
                if debugPlotPrjErrors and ilag == 0 and iPoi == 0:
                    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
                    fig.suptitle('Projection errors (in pixel)')
                    axes[0].plot(t_interest, prjErrors1[:,0], '.-', markersize=3, label='c1x'); axes[0].grid('True')
                    axes[1].plot(t_interest, prjErrors1[:,1], '.-', markersize=3, label='c1y'); axes[1].grid('True')
                    axes[2].plot(t_interest, prjErrors2[:,0], '.-', markersize=3, label='c2x'); axes[2].grid('True')
                    axes[3].plot(t_interest, prjErrors2[:,1], '.-', markersize=3, label='c2y'); axes[3].grid('True')
                # norm of error
                tic = time.time()
#                err = (np.linalg.norm(prjErrors1, axis=1).mean()+np.linalg.norm(prjErrors2, axis=1).mean()) * .5
#                err = np.linalg.norm(np.concatenate((prjErrors1.flatten(),prjErrors2.flatten()))) / (prjErrors1.size + prjErrors2.size)**.5
                errvec = np.concatenate((prjErrors1.flatten(),prjErrors2.flatten()))
                mse = np.mean(np.square(errvec))
                prjErr[iPoi, ilag] = mse
                ticMeasures[5] += time.time() - tic    
                # 
                # print info
#                print("# If time lag is %.1f frames, average projection error of point %d is %.3f" % (t_lag, iPoi+1, err) )
            # end of for iPoi in range(min(nPoints))
            avgErr = np.sum(prjErr[:,ilag]) / prjErr.shape[0]
            print("# If time lag is %.1f frames, the mean-square-error (MSE) of projection error of all points is %.3f" % (t_lag, avgErr) )
        # end of for ilag in range(t_lag_trials.size)
        #
        t_lags = np.zeros(min(nPoints), dtype=float)
        for iPoi in range(min(nPoints)):
            # find the t_lag which results in minimum prjErr[iPoi, :]
            ilagmin = np.argmin(prjErr[iPoi,:])
            t2 = t_lag_trials[ilagmin]
            t1 = t2 - 1
            t3 = t2 + 1
            e1 = prjErr[iPoi, ilagmin - 1]
            e2 = prjErr[iPoi, ilagmin]
            e3 = prjErr[iPoi, ilagmin + 1]
            # shift time axis so that t2_ = 0, t2_ = t2 - t2
            tmat = np.array([1, -1, 1, 0, 0, 1, 1, 1, 1], dtype=float).reshape(3,3)
            evec = np.array([e1, e2, e3], dtype=float).reshape(3, 1)
            coef = (np.linalg.inv(tmat) @ evec).flatten()
            t_lags[iPoi] = -.5 * coef[1] / coef[0] + t2
        # end of for iPoi in range(min(nPoints))
        # display t_lags to 
        txtmp = '\t'.join(map(str, t_lags))
        txLags.delete('0.0', 'end')
        txLags.insert('0.0', txtmp)
        pass
    # end of def event_btFindTimeLags(event):
    btFindTimeLags.bind("<ButtonRelease-1>", event_btFindTimeLags)

    # bind event of buttonn "Triangulation"
    def event_btTriang(event):
        global nFrames, bigTable
        # based on best time lags, do triangulation
        print('# event_btTriang')
        # get time lags (time lag of each point) from txLags
        tLagsText = txLags.get('0.0', 'end').strip()
        tLagsText = tLagsText.replace(',', ' ').replace('\t',' ').replace('\n',' ').replace('\r',' ').replace(';',' ')
        tLagsText = tLagsText.replace('(', ' ').replace(')',' ').replace('[',' ').replace(']',' ')
        t_lags = np.fromstring(tLagsText, sep=' ', dtype=float)
        # Range of triangulation frames. 
        t_interest_text = txTriangFrameRange.get('0.0', 'end').strip()
        t_interest = np.array(colonRangeToIntList(t_interest_text), dtype=int)
        t_interest -= 1 # convert to 0-based index
        # Range of triangulation points. 
        poisTriangulation_text = txTriangPoiRange.get('0.0', 'end').strip()
        poisTriangulation = np.array(colonRangeToIntList(poisTriangulation_text), dtype=int)
        poisTriangulation -= 1 # convert to 0-based index
        # triangulation is done point by point.
        imgPoints1_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        imgPoints2_xyi_thisPoi_interest = np.zeros((t_interest.size, 2), dtype=np.float64)
        # run loop (each point)
        for iPoi in poisTriangulation:
            # get time lag of this point, as each point has different time lag
            t_lag = t_lags[iPoi]
            t_v1 = np.linspace(0, nFrames[0] - 1, round(nFrames[0]))
            t_v2 = np.linspace(0, nFrames[1] - 1, round(nFrames[1])) + t_lag
            # image points of this point in cam 0
            icam = 0
            cam_offset = 0  # Camera 0 offset
            xi = bigTable[:, iPoi*5+cam_offset+1]
            yi = bigTable[:, iPoi*5+cam_offset+2]
            imgPoints1_xyi_thisPoi_interest[:,0] = xi[round(t_interest[0]):(round(t_interest[-1])+1)].copy()
            imgPoints1_xyi_thisPoi_interest[:,1] = yi[round(t_interest[0]):(round(t_interest[-1])+1)].copy()
            # image points of this point in cam 1
            # create interpolation objects for cam 2 (v2)
            icam = 1
            cam_offset = 1630  # Camera 1 offset (1620 + 10 gap)
#            xi = bigTable[:, iPoi*5+cam_offset+1]
#            yi = bigTable[:, iPoi*5+cam_offset+2]
            xi = bigTable[:t_v2.shape[0], iPoi*5+cam_offset+1]
            yi = bigTable[:t_v2.shape[0], iPoi*5+cam_offset+2]
            # in case xi has nan, fill all nan in xi with its previous value
            if np.isnan(xi[0]):
                xi[0] = 0.0
                print("# Warning: Cam 2 Point %d starts from nan. " % (iPoi))
            for iiFrame in range(xi.size):
                if np.isnan(xi[iiFrame]):
                    xi[iiFrame] = xi[iiFrame - 1]
            if np.isnan(yi[0]):
                yi[0] = 0.0
                print("# Warning: Cam 2 Point %d starts from nan. " % (iPoi))
            for iiFrame in range(yi.size):
                if np.isnan(yi[iiFrame]):
                    yi[iiFrame] = yi[iiFrame - 1]
            intpv2xi = scipy.interpolate.interp1d(t_v2, xi, kind='cubic')
            intpv2yi = scipy.interpolate.interp1d(t_v2, yi, kind='cubic')
            imgPoints2_xyi_thisPoi_interest[:,0] = intpv2xi(t_interest)
            imgPoints2_xyi_thisPoi_interest[:,1] = intpv2yi(t_interest)
            # Triangulation
            objPoints, objPoints1, objPoints2, \
                prjPoints1, prjPoints2, prjErrors1, prjErrors2 = \
                triangulatePoints2(cmats[0], dvecs[0], rvecs[0], tvecs[0], 
                                cmats[1], dvecs[1], rvecs[1], tvecs[1], 
                                imgPoints1_xyi_thisPoi_interest, 
                                imgPoints2_xyi_thisPoi_interest)
            # debug 
            tryDebugToFindStrangeData = True
            if tryDebugToFindStrangeData == True:
                if abs(objPoints[0,0]) > 1e7 or abs(objPoints[1,0]) > 1e7:
                    print('Got you.')
            # write to bigTable (triangulation starts at 3261)
            bigTable[t_interest, 3261 + iPoi * 7] = objPoints[:,0]
            bigTable[t_interest, 3262 + iPoi * 7] = objPoints[:,1]
            bigTable[t_interest, 3263 + iPoi * 7] = objPoints[:,2]
            bigTable[t_interest, 3264 + iPoi * 7] = prjErrors1[:,0]
            bigTable[t_interest, 3265 + iPoi * 7] = prjErrors1[:,1]
            bigTable[t_interest, 3266 + iPoi * 7] = prjErrors2[:,0]
            bigTable[t_interest, 3267 + iPoi * 7] = prjErrors2[:,1]
        # end of for iPoi in range(min(nPoints))
        pass
    # end of def event_btTriang(event)
    btTriang.bind('<ButtonRelease-1>', event_btTriang)

    # bind event of "Save config"
    def event_btSaveConfig(event):
        workDir = txWorkDir.get("1.0", "end").strip()
        confFilebase = 'Stereosync_config.xml'
        confFilename = os.path.join(workDir, confFilebase)
        # Create a FileStorage object for writing the XML file
        fs = cv2.FileStorage(confFilename, cv2.FILE_STORAGE_WRITE_BASE64)
        # Save config to file
        #   working directory
        workDir = txWorkDir.get('1.0', 'end').strip()
        fs.write('working_directory', workDir)
        #   video file 1
        vidFile1 = txVidFile1.get('1.0', 'end').strip()
        fs.write('video_file_1', vidFile1)
        #   video file 2
        vidFile2 = txVidFile2.get('1.0', 'end').strip()
        fs.write('video_file_2', vidFile2)
        #   calibration file 1
        calibFile1 = txCalibFile1.get('1.0', 'end').strip()
        fs.write('calibration_file_1', calibFile1)
        #   calibration file 2
        calibFile2 = txCalibFile2.get('1.0', 'end').strip()
        fs.write('calibration_file_2', calibFile2)
        #   template frame ID 1
        tmpltFrameId1 = txTmpltFrameId1.get('0.0', 'end').strip()
        fs.write('template_frame_1', tmpltFrameId1)
        #   template frame ID 2
        tmpltFrameId2 = txTmpltFrameId2.get('0.0', 'end').strip()
        fs.write('template_frame_2', tmpltFrameId2)
        #   templates file 1
        tmpltFile1 = txTmpltFile1.get('1.0', 'end').strip()
        fs.write('templates_file_1', tmpltFile1)
        #   templates file 2
        tmpltFile2 = txTmpltFile2.get('1.0', 'end').strip()
        fs.write('templates_file_2', tmpltFile2)
        #   range of tracking cameras
        trackCams = txTrackCamRange.get('1.0', 'end').strip()
        fs.write('range_of_tracking_cameras', trackCams)
        #   range of tracking points
        trackPois = txTrackPointRange.get('1.0', 'end').strip()
        fs.write('range_of_tracking_points', trackPois)
        #   range of tracking frames
        trackFrames = txTrackFrameRange.get('1.0', 'end').strip()
        fs.write('range_of_tracking_frames', trackFrames)
        #   range of possible lags
        possibleLags = txPossibleLag.get('1.0', 'end').strip()
        fs.write('range_of_possible_lags', possibleLags)
        #   range of points for sync analysis
        syncPois = txSyncPoiRange.get('1.0', 'end').strip()
        fs.write('range_of_points_for_sync_analysis', syncPois)
        #   range of frames for sync analysis
        syncFrames = txSyncFrameRange.get('1.0', 'end').strip()
        fs.write('range_of_frames_for_sync_analysis', syncFrames)
        #   range of points for triangulation
        triangPois = txTriangPoiRange.get('1.0', 'end').strip()
        fs.write('range_of_points_for_triangulation', triangPois)
        #   range of frames for triangulation
        triangFrames = txTriangFrameRange.get('1.0', 'end').strip()
        fs.write('range_of_frames_for_triangulation', triangFrames)
        # release
        fs.release()
        
        #   time lags - save to separate text file due to string length limitations
        timeLags = txLags.get('1.0', 'end').strip()
        timeLagsFile = os.path.join(workDir, 'time_lags_each_point.txt')
        try:
            with open(timeLagsFile, 'w') as f:
                f.write(timeLags)
            print("# Config file saved.")
            print("# Time lags saved to: %s" % timeLagsFile)
        except Exception as e:
            print("# Config file saved (without time lags).")
            print("# Warning: Could not save time lags: %s" % str(e))
    #
    # Make <Tab> key switching to next widget, not adding tab in Text
    #
    def tabkey_focus_next(event):
        """Handles Tab key press and switches focus to the next text widget."""
        widget = event.widget  # Get the widget that triggered the event (the current text widget)
        widget.tk_focusNext().focus_set()  # Get the next widget in the focus order and set focus on it
        return "break"  # Prevent further default tab handling by Tkinter
    
    txWorkDir.bind("<Tab>", tabkey_focus_next)
    txVidFile1.bind("<Tab>", tabkey_focus_next)
    txVidFile2.bind("<Tab>", tabkey_focus_next)
    txCalibFile1.bind("<Tab>", tabkey_focus_next)
    txCalibFile2.bind("<Tab>", tabkey_focus_next)
    txTmpltFile1.bind("<Tab>", tabkey_focus_next)
    txTmpltFile2.bind("<Tab>", tabkey_focus_next)
    txTrackCamRange.bind("<Tab>", tabkey_focus_next)
    txTrackFrameRange.bind("<Tab>", tabkey_focus_next)
    txTrackPointRange.bind("<Tab>", tabkey_focus_next)
    cbParallel.bind("<Tab>", tabkey_focus_next)
    txSyncFrameRange.bind("<Tab>", tabkey_focus_next)
    txSyncPoiRange.bind("<Tab>", tabkey_focus_next)
    txSaveConfig.bind("<Tab>", tabkey_focus_next)
    txSaveData.bind("<Tab>", tabkey_focus_next)
    txPossibleLag.bind("<Tab>", tabkey_focus_next)
    txTriangFrameRange.bind("<Tab>", tabkey_focus_next)
    txTriangPoiRange.bind("<Tab>", tabkey_focus_next)
    txLags.bind("<Tab>", tabkey_focus_next)

    # function of loading config
    def loadConfig(event=None):
        global fTmplts, tmplts, nPoints
        # Create a FileStorage object for reading the XML file
        workDir = txWorkDir.get("1.0", "end").strip()
        confFilebase = txLoadConfig.get('0.0', 'end').strip()
        confFilename = os.path.join(workDir, confFilebase)
        if os.path.exists(confFilename) == False:
            print("# Error: Cannot file config file: %s" % confFilename)
            return
        fs = cv2.FileStorage(confFilename, cv2.FILE_STORAGE_READ)
        # Load config to file
        #   video file 1
        vidFile1 = fs.getNode('video_file_1').string()
        txVidFile1.delete('1.0', 'end')
        txVidFile1.insert('1.0', vidFile1)
        #   video file 2
        vidFile2 = fs.getNode('video_file_2').string()
        txVidFile2.delete('1.0', 'end')
        txVidFile2.insert('1.0', vidFile2)
        #   calibration file 1
        calibFile1 = fs.getNode('calibration_file_1').string()
        txCalibFile1.delete('1.0', 'end')
        txCalibFile1.insert('1.0', calibFile1)
        #   calibration file 2
        calibFile2 = fs.getNode('calibration_file_2').string()
        txCalibFile2.delete('1.0', 'end')
        txCalibFile2.insert('1.0', calibFile2)
        #   template frame ID 1
        tmpltFrameId1 = fs.getNode('template_frame_1').string()
        txTmpltFrameId1.delete('1.0', 'end')
        txTmpltFrameId1.insert('1.0', tmpltFrameId1)
        #   template frame ID 2
        tmpltFrameId2 = fs.getNode('template_frame_2').string()
        txTmpltFrameId2.delete('1.0', 'end')
        txTmpltFrameId2.insert('1.0', tmpltFrameId2)
        #   templates file 1
        tmpltFile1 = fs.getNode('templates_file_1').string()
        txTmpltFile1.delete('1.0', 'end')
        txTmpltFile1.insert('1.0', tmpltFile1)
        #   templates file 2
        tmpltFile2 = fs.getNode('templates_file_2').string()
        txTmpltFile2.delete('1.0', 'end')
        txTmpltFile2.insert('1.0', tmpltFile2)
        #   range of tracking cameras
        trackCams = fs.getNode('range_of_tracking_cameras').string()
        txTrackCamRange.delete('1.0', 'end')
        txTrackCamRange.insert('1.0', trackCams)
        #   range of tracking points
        trackPois = fs.getNode('range_of_tracking_points').string()
        txTrackPointRange.delete('1.0', 'end')
        txTrackPointRange.insert('1.0', trackPois)
        #   range of tracking frames
        trackFrames = fs.getNode('range_of_tracking_frames').string()
        txTrackFrameRange.delete('1.0', 'end')
        txTrackFrameRange.insert('1.0', trackFrames)
        #   range of possible lags
        possibleLags = fs.getNode('range_of_possible_lags').string()
        txPossibleLag.delete('1.0', 'end')
        txPossibleLag.insert('1.0', possibleLags)
        #   range of points for sync analysis
        syncPois = fs.getNode('range_of_points_for_sync_analysis').string()
        txSyncPoiRange.delete('1.0', 'end')
        txSyncPoiRange.insert('1.0', syncPois)
        #   range of frames for sync analysis
        syncFrames = fs.getNode('range_of_frames_for_sync_analysis').string()
        txSyncFrameRange.delete('1.0', 'end')
        txSyncFrameRange.insert('1.0', syncFrames)
        #   range of points for triangulation
        triangPois = fs.getNode('range_of_points_for_triangulation').string()
        txTriangPoiRange.delete('1.0', 'end')
        txTriangPoiRange.insert('1.0', triangPois)
        #   range of frames for triangulation
        triangFrames = fs.getNode('range_of_frames_for_triangulation').string()
        txTriangFrameRange.delete('1.0', 'end')
        txTriangFrameRange.insert('1.0', triangFrames)
        # release
        fs.release()
        
        #   time lags - load from separate text file
        timeLagsFile = os.path.join(workDir, 'time_lags_each_point.txt')
        txLags.delete('1.0', 'end')
        if os.path.exists(timeLagsFile):
            try:
                with open(timeLagsFile, 'r') as f:
                    timeLags = f.read().strip()
                txLags.insert('1.0', timeLags)
                print("# Time lags loaded from: %s" % timeLagsFile)
            except Exception as e:
                print("# Warning: Could not load time lags: %s" % str(e))
        else:
            # Try loading from old XML format (backward compatibility)
            try:
                fs2 = cv2.FileStorage(confFilename, cv2.FILE_STORAGE_READ)
                if fs2.isOpened():
                    timeLags = fs2.getNode('time_lags_each_point').string()
                    if timeLags:
                        txLags.insert('1.0', timeLags)
                    fs2.release()
            except:
                pass
        # set nFrames
        fVideos[0] = os.path.join(workDir, txVidFile1.get('0.0', 'end').strip())
        fVideos[1] = os.path.join(workDir, txVidFile2.get('0.0', 'end').strip())
        for icam in range(2):
            vx = cv2.VideoCapture(fVideos[icam])
            if vx.isOpened(): 
                nFrames[icam] = round(vx.get(cv2.CAP_PROP_FRAME_COUNT))
                print('# Video %d has %d frames.' % (icam+1, nFrames[icam]))
                try:
                    ret, imgInits[icam] = vx.read()
                    imgInits[icam] = cv2.cvtColor(imgInits[icam], cv2.COLOR_BGR2GRAY)
                except:
                    print("# Error: Cannot read image from %s" % fVideos[icam]) 
            else:
                print("# Error opening video file %d:%s" % (icam, fVideos[icam]))
        # load camera parameters: cmats, dvecs, rvecs, tvecs
        fCalibs=[None, None]
        fCalibs[0] = os.path.join(workDir, txCalibFile1.get('0.0', 'end').strip())
        fCalibs[1] = os.path.join(workDir, txCalibFile2.get('0.0', 'end').strip())
        for icam in range(2):
            # load calibration parameters
            try:
                camParam = np.loadtxt(fCalibs[icam], delimiter=',')
            except:
                try:
                    camParam = np.loadtxt(fCalibs[icam], delimiter=',', skiprows=1)
                except:
                    print(f"# Failed to load camera calibration file {fCalibs[icam]}")
                    continue
            rvecs[icam] = camParam[2:5]
            tvecs[icam] = camParam[5:8]
            cmats[icam] = camParam[8:17].reshape(3,3)
            dvecs[icam] = camParam[17:]
        # load templates
        fTmplts = [None, None]
        nPoints = [0, 0]
        fTmplts[0] = os.path.join(workDir, txTmpltFile1.get('0.0', 'end').strip())
        fTmplts[1] = os.path.join(workDir, txTmpltFile2.get('0.0', 'end').strip())

#        nPoints = 
        print("# Config file loaded.")
    # end of def loadConfig(event=None):
        
    # bind event of btLoadConfig
    btLoadConfig.bind("<ButtonRelease-1>", loadConfig)

    # function of load data
    def loadData(event=None):
        global bigTable
        workDir = txWorkDir.get("1.0", "end").strip()
        dataFilebase = txLoadData.get('0.0', 'end').strip()
        dataFilename = os.path.join(workDir, dataFilebase)
        try:
            bigTable = np.loadtxt(dataFilename, delimiter=',')
        except:
            bigTable = np.loadtxt(dataFilename, delimiter=',', skiprows=1)
        print("# Loaded data size: (%d, %d)" % (bigTable.shape[0], bigTable.shape[1]))
    # end of def loadData(event=None)

    # bind event of btLoadData and SaveConfig
    btLoadData.bind('<ButtonRelease-1>', loadData)
    btSaveConfig.bind("<ButtonRelease-1>", event_btSaveConfig)
    
    # bind event of Plot
    def plotData(event=None):
        # get column indices of x (xcol) and y (ycol)
        try:
            import matplotlib.pyplot as plt
            global bigTable
            theStr = txPlot.get('0.0', 'end').strip()
            pindex = np.fromstring(theStr, sep=' ', dtype=np.int32).flatten()
            nLine = pindex.size // 2
            fig, ax = plt.subplots()
            lines = []
            for iline in range(nLine):
                xdata = bigTable[:,pindex[iline*2]-1]
                ydata = bigTable[:,pindex[iline*2+1]-1]
                valid_indices = np.where(~np.isnan(xdata) & ~np.isnan(ydata))[0]
                xdatp = xdata[valid_indices]
                ydatp = ydata[valid_indices]
                ydatp = ydata[valid_indices] - ydatp[0]
                theLine, = ax.plot(xdatp, ydatp,
                           label="Line %d" % (iline+1), linewidth=1, 
                           marker='o', markersize=2)
            ax.set(xlabel="X-axis", ylabel="Y-axis", title="Data plot")
            ax.legend()
            ax.grid(True)
            plt.show()
        except:
            print("# Error. Cannot parse plot argument %s" % theStr)
    btPlot.bind("<ButtonRelease-1>", plotData)


    win.mainloop()







if __name__ == '__main__':
    ParaStereoSync()

