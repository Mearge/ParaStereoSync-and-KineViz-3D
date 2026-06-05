import cv2 as cv


def imshow2(winname, img, winmax=(1600, 900), interp=cv.INTER_CUBIC):
    """
    This function is an enhanced imshow that allows user to pan and zoom
    the image by mouse dragging and wheel rolling. 
    Pressing CTRL or ALT displays coordinates on the console screen and image
    itself, respectively.

    Parameters
    ----------
    winname : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.
    winmax : TYPE, optional
        DESCRIPTION. The default is (1600, 900).
    interp : TYPE, optional
        DESCRIPTION. The default is cv.INTER_LINEAR.

    Returns
    -------
    None.

    """
    # define x0, y0, x1, y1 as the region of image to display in imdhow window
    # x0, y0, x1, y1 are integers
    # x0 has to be between [0, x1)
    # x1 has to be between (x0, img.shape[1] - 1] 
    # y0 has to be between [0, y1)
    # y1 has to be between (y0, img.shape[0] - 1]
    
    # (x0, y0) is the upper-left pixel in the region to display    
    x0, x1, y0, y1 = 0, img.shape[1] - 1, 0, img.shape[0] - 1
    # x0_ is the previous status of x0. So are x1_, y0_, and y1_
    x0_, x1_, y0_, y1_ = -1, -1, -1, -1
    # mx is mouse cursor in window coord. mxi is that in image coord.
    mx, my, mxi, myi = 0, 0, 0, 0
    # mx_ is the previous status of mx
    mx_, my_, mxi_, myi_ = -1, -1, -1, -1
    # mbl, mbm, mbr are mouse-button status (left, middle, right buttons). 
    # 0 is released, 1 is being pressed
    mbl, mbm, mbr = 0, 0, 0
    # mbl_ is the previous status of mbl. So are mbm_ and mbr_ 
    mbl_, mbm_, mbr_ = -1, -1, -1    
    # scale is the current magnified scale of the image, which will be calculated later
    scale = 1.
    # wheel zoom factor
    wf = 1.2  
    max_scale = 30.0
    # about saved status right before dragging
    mx_before_dragging, my_before_dragging = 0, 0
    x0_before_dragging, y0_before_dragging = 0, 0
    x1_before_dragging, y1_before_dragging = 0, 0
    # mflags
    mflags, mflags_ = 0, 0
    
    def on_mouse(event, x, y, flags, params):
        nonlocal scale, winmax
        nonlocal mx, my, mxi, myi, x0, x1, y0, y1 
        nonlocal mbl, mbm, mbr
        nonlocal mx_, my_, mxi_, myi_, x0_, x1_, y0_, y1_ 
        nonlocal mbl_, mbm_, mbr_
        nonlocal mx_before_dragging, my_before_dragging
        nonlocal x0_before_dragging, y0_before_dragging
        nonlocal x1_before_dragging, y1_before_dragging
        nonlocal mflags, mflags_

#        print('\b'*100, end='')
#        print('x:%d y:%d     '%(x,y), end='')
        # update current status and previous status
        # mouse cursor location (window coordinate and image coordinate)
        mx_, my_, mxi_, myi_ = mx, my, mxi, myi
        mbl_, mbm_, mbr_ = mbl, mbm, mbr
        mx, my = x, y
        mxi = int(x0 + x / scale + 0.5)
        myi = int(y0 + y / scale + 0.5)
        mflags_ = mflags
        mflags = flags
        # update mbl, mbm, mbr, mbl_, mbm_, and mbr_
        if flags & cv.EVENT_FLAG_LBUTTON != 0:
            mbl = 1
        else:
            mbl = 0
        if flags & cv.EVENT_FLAG_MBUTTON != 0:
            mbm = 1
        else:
            mbm = 0
        if flags & cv.EVENT_FLAG_RBUTTON != 0:
            mbr = 1
        else:
            mbr = 0
        # # if mouse left button is just clicked
        if mbl_ == 0 and mbl != 0:
            mx_before_dragging = mx
            my_before_dragging = my
            x0_before_dragging = x0
            y0_before_dragging = y0
            x1_before_dragging = x1
            y1_before_dragging = y1
        # if mouse left button is pressed and dragging
        if mbl_ == 1 and mbl == 1:
            dx = (-mx + mx_before_dragging) / scale
            dy = (-my + my_before_dragging) / scale
            x0 = int(x0_before_dragging + dx + 0.5)
            y0 = int(y0_before_dragging + dy + 0.5)
            x1 = int(x1_before_dragging + dx + 0.5)
            y1 = int(y1_before_dragging + dy + 0.5)
            # print("dragging: %d %d %d" % (mxi, mxi_before_dragging, dx))
            if x0 >= x1:
                x0 = x1 - 1
            if x0 < 0:
                x0_trial = 0
                x1_trial = x0_trial + (x1 - x0)
                x0 = x0_trial
                if x1_trial < img.shape[1]:
                    x1 = x1_trial
            if x1 >= img.shape[1]:
                x1_trial = (img.shape[1] - 1)
                x0_trial =  x1_trial - (x1 - x0)
                x1 = x1_trial
                if x0_trial >= 0:
                    x0 = x0_trial
            if y0 >= y1:
                y0 = y1 - 1
            if y0 < 0:
                y0_trial = 0
                y1_trial = y0_trial + (y1 - y0)
                y0 = y0_trial
                if y1_trial < img.shape[0]:
                    y1 = y1_trial                
            if y1 >= img.shape[0]:
                y1_trial = (img.shape[0] - 1)
                y0_trial =  y1_trial - (y1 - y0)
                y1 = y1_trial
                if y0_trial >= 0:
                    y0 = y0_trial
        if event == cv.EVENT_MOUSEMOVE:
            mx, my = x, y
            mxi = x0 + x / scale 
            myi = y0 + y / scale
        if event == cv.EVENT_MOUSEWHEEL:
            # mouse wheel up, zoom in 
            if flags > 0 and scale * wf < max_scale:
                mxi = x0 + x / scale 
                x0 = int(mxi - (mxi - x0) / wf + 0.5)
                x1 = int(mxi + (x1 - mxi) / wf + 0.5)
                myi = y0 + y / scale
                y0 = int(myi - (myi - y0) / wf + 0.5)
                y1 = int(myi + (y1 - myi) / wf + 0.5)
                if x0 >= x1:
                    x0 = x1 - 1
                if x0 < 0:
                    x0_trial = 0
                    x1_trial = x0_trial + (x1 - x0)
                    x0 = x0_trial
                    if x1_trial < img.shape[1]:
                        x1 = x1_trial
                if x1 >= img.shape[1]:
                    x1_trial = (img.shape[1] - 1)
                    x0_trial =  x1_trial - (x1 - x0)
                    x1 = x1_trial
                    if x0_trial >= 0:
                        x0 = x0_trial
                if y0 >= y1:
                    y0 = y1 - 1
                if y0 < 0:
                    y0_trial = 0
                    y1_trial = y0_trial + (y1 - y0)
                    y0 = y0_trial
                    if y1_trial < img.shape[0]:
                        y1 = y1_trial                
                if y1 >= img.shape[0]:
                    y1_trial = (img.shape[0] - 1)
                    y0_trial =  y1_trial - (y1 - y0)
                    y1 = y1_trial
                    if y0_trial >= 0:
                        y0 = y0_trial
            elif flags < 0:
                # mouse wheel down 
                mxi = x0 + x / scale 
                x0 = int(mxi - (mxi - x0) * wf + 0.5)
                x1 = int(mxi + (x1 - mxi) * wf + 0.5)
                myi = y0 + y / scale
                y0 = int(myi - (myi - y0) * wf + 0.5)
                y1 = int(myi + (y1 - myi) * wf + 0.5)
                if x0 >= x1:
                    x0 = x1 - 1
                if x0 < 0:
                    x0_trial = 0
                    x1_trial = x0_trial + (x1 - x0)
                    x0 = x0_trial
                    if x1_trial < img.shape[1]:
                        x1 = x1_trial
                if x1 >= img.shape[1]:
                    x1_trial = (img.shape[1] - 1)
                    x0_trial =  x1_trial - (x1 - x0)
                    x1 = x1_trial
                    if x0_trial >= 0:
                        x0 = x0_trial
                if y0 >= y1:
                    y0 = y1 - 1
                if y0 < 0:
                    y0_trial = 0
                    y1_trial = y0_trial + (y1 - y0)
                    y0 = y0_trial
                    if y1_trial < img.shape[0]:
                        y1 = y1_trial                
                if y1 >= img.shape[0]:
                    y1_trial = (img.shape[0] - 1)
                    y0_trial =  y1_trial - (y1 - y0)
                    y1 = y1_trial
                    if y0_trial >= 0:
                        y0 = y0_trial
        return

    # create image window    
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, on_mouse)
    # the main look of imshow2
    while True:
        # update (zoom and resize) image
        if x0 != x0_ or y0 != y0_ or x1 != x1_ or y1 != y1_:
            # print("## %d %d %d %d" % (x0, x1, y0, y1))
            # generate new image
            scalex = winmax[0] / (x1 - x0)
            scaley = winmax[1] / (y1 - y0)
            scale = min(scalex, scaley)
            scaled_width  = int((x1 - x0) * scale + 0.5)
            scaled_height = int((y1 - y0) * scale + 0.5)
            imgScaled = cv.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                  dsize=(scaled_width, scaled_height), 
                                  interpolation=interp)
            # update 
            x0_, y0_, x1_, y1_ = x0, y0, x1, y1
        if (mxi != mxi_ or myi != myi_) and mflags & cv.EVENT_FLAG_CTRLKEY != 0:
            print("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bX:%.1f Y:%.1f Scale:%.1f " % (mxi, myi, scale), end='')            
        if (mxi != mxi_ or myi != myi_) and mflags & cv.EVENT_FLAG_ALTKEY != 0:
            showStr = "X:%.1f Y:%.1f Scale:%.1f " % (mxi, myi, scale)
            font = cv.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            color = (0, 255, 0)
            thickness = 2
            fontScale = 0.8
            imgScaled = cv.resize(img[y0:(y1 + 1), x0:(x1 + 1)], 
                                  dsize=(scaled_width, scaled_height), 
                                  interpolation=interp)
            cv.putText(imgScaled, showStr, org, font, fontScale, color, thickness)
        cv.imshow(winname, imgScaled)
        ikey = cv.waitKey(30)
        if ikey == ord("q") or ikey == ord("Q") or ikey == 27:
            break
    try:
        cv.destroyWindow(winname)
    except:
        return


def test_imshow2():
    image_path = r'c:/temp/example02.jpg'
    img = cv.imread(image_path)
    imshow2("TEST IMSHOW2", img, interp=cv.INTER_NEAREST)

# test_imshow2()







 

