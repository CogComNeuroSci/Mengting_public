from psychopy import visual, event, core, gui, sound, tools, monitors 
import os, time, sys
import numpy as np
from psychopy.visual import GratingStim
from math import fabs 
import exp_utils_stinfo as exputils
import pylink  # module for tracking
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy  # for calibration and validation

#####
# Set up a variable to run the script on a computer not connected to the tracker
# We will use this variable in a series of if-else statements everytime there would be a line of code calling the tracker
# We will change it to 'False' when running the experiment in the lab

dummy_mode = False
######

####################################################
# Observer's info
####################################################

info = {"obs": "", "session":""}
infoDlg = gui.DlgFromDict(dictionary = info, title = 'Participant information')
obs = int(info['obs'])
session = int(info['session'])  # 1 or 2 or 3
block = 1  # must be 1

last_pause_trial_n = 0

training = True
staircase = False

timeAndDate = time.strftime('%d_%m_%y_%H_%M_%S', time.localtime())
task_out, outfolder, et_outfolder = exputils.init_log(obs, session, timeAndDate, training, staircase)

edf_file = '%s.EDF' % (obs)
if not os.path.exists(et_outfolder):
    os.makedirs(et_outfolder)
local_edf = os.path.join(et_outfolder, edf_file) # we call this at the end to transfer .EDF from ET PC to STIM PC

####################################################
# Create the Psychopy window#
####################################################

myMon = monitors.Monitor('testMonitor')
myMon.setDistance(93)
myMon.setWidth(53)
myWin = visual.Window(fullscr=True, screen=0, monitor=myMon, allowGUI=True)
myWin.setMouseVisible(False)

####################################################
# Eye-tracking set-up
####################################################

# Define functions for beeping during trials

def beep_away(part_trial):
    et_tracker = pylink.getEYELINK()
    sample = et_tracker.getNewestSample()
    
    if sample is None:
        # no sample data
        g_x = -32768
        g_y = -32768
    else:
        if eye_used == 1 and sample.isRightSample():
            g_x, g_y = sample.getRightEye().getGaze()
            print(g_x)
        elif eye_used == 0 and sample.isLeftSample():
            g_x, g_y = sample.getLeftEye().getGaze()
        f_x = g_x - 960
        print(f_x)
        if float(x_right) < g_x or g_x < float(x_left) and not g_x == -32768.: # this was changed, now it can be anywhere in the y-axis # avoid beep while blink
            beep.play()
                #et_tracker.sendMessage('!V TRIAL SEQUENCE_FIX %s' % part_trial) # doing this leads to a loss of messages (because it's sent every X ms, and without a pumpDelay, samples are lost in the buffer
        else:
            beep.stop()

# Define functions for defining ROIs

def roi_points(stim_name, stim_size, x_displacement, y_displacement, units, monitor):
    '''
    Returns coordinates to draw areas of interest (shape rectangle)
    Drawing ROIs as rectangles requires the top, bottom, left and right coordinates
    The tracker coordinate space has 0,0 at the top, left corner
    We need thus to translate from PsychoPy to that coordinate system
    '''
    
    if units == 'deg':
        x_pos = tools.monitorunittools.deg2pix(stim_name.pos[0], monitor)
        y_pos = tools.monitorunittools.deg2pix(stim_name.pos[1], monitor)
        width = tools.monitorunittools.deg2pix(stim_size/2, monitor)
        height = tools.monitorunittools.deg2pix(stim_size/2, monitor)
    else:
        x_pos = stim_name.pos[0]
        y_pos = stim_name.pos[1]
        width = stim_size
        height = stim_size
    
    # PsychoPy coordinates
    
    left = x_pos - width
    right = x_pos + width
    top =  y_pos - height
    bottom = y_pos + height
    
    # Translate to EyeLink coordinate system
    
    left = x_displacement + left
    right = x_displacement + right
    top = y_displacement + top
    bottom = y_displacement + bottom
    
    return left, right, top, bottom
    
# Define functions for catching errors

def skip_trial():
    """Ends recording """

    et_tracker = pylink.getEYELINK()
    # Stop recording
    if et_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        et_tracker.stopRecording()
    # Clean the screen
    myWin.flip()
    # send a message to mark trial end
    et_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
    return pylink.TRIAL_ERROR


def abort_exp():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    et_tracker = pylink.getEYELINK()

    if et_tracker.isConnected():
        error = et_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            skip_trial()

            # Put tracker in Offline mode
        et_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        et_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        et_tracker.closeDataFile()
        try:
            et_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        et_tracker.close()

    # close the PsychoPy window
    myWin.close()
    core.quit()
    sys.exit()


# Start of the experiment
# 1. Open the connection to the ET PC

if dummy_mode:
    et_tracker = pylink.EyeLink(None)
    et_version = 0  # set version to 0, in case running in Dummy mode
else:
    try:  # at the end of the eye-tracking section (before we move onto ioHub, there is a brief discussion on this try-except statements)
        et_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        dlg = gui.Dlg("Dummy Mode?")
        dlg.addText("Couldn't connect to tracker at 100.1.1.1 -- continue in Dummy Mode?")
        # dlg.addField('File Name:', edf_fname)
        # show dialog and wait for OK or Cancel
        ok_data = dlg.show()
        if dlg.OK:  # if ok_data is not None
            # print('EDF data filename: {}'.format(ok_data[0]))
            dummy_mode = True
            et_tracker = pylink.EyeLink(None)
        else:
            print('user cancelled')
            core.quit()
            sys.exit()

# 2. Open the .EDF file

if not dummy_mode:
    try:
        et_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('ERROR:', err)
        # close the link if we have one open
        if et_tracker.isConnected():
            et_tracker.close()
        core.quit()
        sys.exit()

# Add preamble (optional)

preamble_text = 'Behavioral Oscillations and Cognitive Control'  # REMEMBER TO ADD SOMETHING MEANINGFUL
et_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# 3. Configure the tracker

if not dummy_mode:
    et_tracker.setOfflineMode()
    pylink.pumpDelay(100)

    et_tracker.sendCommand("sample_rate 1000")
    et_tracker.sendCommand("recording_parse_type = GAZE")
    et_tracker.sendCommand("select_parser_configuration 0")
    et_tracker.sendCommand("calibration_type = HV13")
    et_tracker.sendCommand("screen_pixel_coords = 0 0 %d %d" % (1920 - 1, 1080 - 1))  # this needs to be modified to the Display PC screen size you are using
    et_tracker.sendMessage("DISPLAY_COORDS 0 0 %d %d" % (1920 - 1, 1080 - 1))  # this needs to be modified to the Display PC screen size you are using

    # get the tracker version to see what data can be stored

    vstr = et_tracker.getTrackerVersionString()
    et_version = int(vstr.split()[-1].split('.')[0])

    # events to store

    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

    # samples to store

    if et_version > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    et_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    et_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    et_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    et_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

####################################################
# Create Psychopy visual objects
####################################################

# Beep practice trials components
# Beep sound that plays when gaze is away from the dot
beep = sound.Sound(value = 'A', octave=4, secs = 1.00, volume=1.0, hamming=True, stereo = True)  # 800 Hz tone
beep.setVolume(1)
# Create fix dot
fix_dot = visual.Circle(myWin, size = (20,20), units = 'pix', fillColor = 'green')
# Create ROI
# As per Senoussi et al. (2022), 1.5 degrees of visual angle
roi_area = visual.Circle(myWin, units = 'deg', fillColor = 'white', size = 1.5) #kept 1.5 in beep trials
roi_area_exp = visual.Circle(myWin, units = 'deg', fillColor = 'white', size = 2) #increased from 1.5 to 2 in the task
print(roi_area.pos)

# Create values for beep_away
# Log changes 31/01/2025
# Now the coordinates for beep are 1.5 (which is the size allowed in Mehdi et al., and the one used for ROI central fixation, so it makes sense to also train participants with that space)
# Note that I am only drawing on the x-axis; else, beeping is also triggered by blinks, which we don't want 
width_screen = tools.monitorunittools.deg2pix(1.5/2, myMon)
# displacement
x_right = 960 + width_screen
x_left = 960 - width_screen

fixcross_size = .15
fixh = visual.Line(myWin, units='deg', start=(-fixcross_size, 0), end=(fixcross_size, 0), lineColor=(-1, -1, -1),
                   lineWidth=2)
fixv = visual.Line(myWin, units='deg', start=(0, -fixcross_size), end=(0, fixcross_size), lineColor=(-1, -1, -1),
                   lineWidth=2)

# load instructions
lett_size = .75
lett_pos = .75

# instruction letters with Unicode arrows
stim_lett_textures = np.array([
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, lett_pos), text='L', alignText='center',
                    anchorHoriz='center', color='black'),
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, lett_pos), text='R', alignText='center',
                    anchorHoriz='center', color='black'),
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, lett_pos), text='←', alignText='center',
                    anchorHoriz='center', color='black'),
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, lett_pos), text='→', alignText='center',
                    anchorHoriz='center', color='black')
])

hand_lett_textures = np.array([
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, -lett_pos), text='L', alignText='center',
                    anchorHoriz='center', color='black'),
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, -lett_pos), text='R', alignText='center',
                    anchorHoriz='center', color='black'),
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, -lett_pos), text='←', alignText='center',
                    anchorHoriz='center', color='black'),
    visual.TextStim(myWin, units='deg', height=lett_size, pos=(0, -lett_pos), text='→', alignText='center',
                    anchorHoriz='center', color='black')
])

# grating parameters and create grating objects
stimSize = 5
contrast = .2
oris = [0, 0]
dist_to_center = 7.5
spatFreq = 2
myStim = []
myStim.append(GratingStim(myWin, tex='sin', mask='raisedCos', sf=spatFreq, size=[stimSize, stimSize],
                          units='deg', pos=(-dist_to_center, 0.0), interpolate=False, ori=oris[0], contrast=contrast))
myStim.append(GratingStim(myWin, tex='sin', mask='raisedCos', sf=spatFreq, size=[stimSize, stimSize],
                          units='deg', pos=(dist_to_center, 0.0), interpolate=False, ori=oris[1], contrast=contrast))

tiltLvl = 7

# Additional: Retrieve coordinates for ROIs
# The initial code used to send a trigger every time the beep sounded
# While this may seem ideal, it is redundant (i.e., just one trigger would suffice per time window)
# We also lose samples because of how beep_away() works (lack of pumpDelay + sending a trigger every X ms)
# An alternative, given that this is just to control for participants' eye movements
# Is to draw ROIs and then explore them in pre-processing
# This also eases the burden for the main_task
# For ROIs, I am using the position of the gratings BUT THIS MAY NEED TO BE ADAPTED TO YOUR NEEDS
# I am also drawing them at the beginning to prevent losing samples during the task

Lleft, Lright, Ltop, Lbottom = roi_points(myStim[0], stimSize, 960., 540., 'deg', myMon)
Rleft, Rright, Rtop, Rbottom = roi_points(myStim[1], stimSize, 960., 540., 'deg', myMon)
Cleft, Cright, Ctop,  Cbottom = roi_points(roi_area_exp, 2, 960., 540., 'deg', myMon) #increased from 1.5 to 2

####################################################


####################################################
# Timing
####################################################
ref_rate = 1 / 60.
preinstdur = 0.5
instdur = .2
ISIdur = np.linspace(1.2, 1.7, 11)  # 20Hz
stimdur = .05
ITIdur = np.linspace(0.5, 1.5, 5)
max_respTime = 1
feedback_time = .2

timer = core.CountdownTimer()

########################################################################
# Create trial structure (instructions, ISIs, and stim combination)
########################################################################

#for each participant,four cues on different sides are same in three sessions
trials_info_all, trial_info_blocks, random_instructions = exputils.make_mystinfo_design(obs=obs, session=session, n_instructions=12, n_ISIs=ISIdur.shape[0],
                                                                   n_stimCombi=4, session_1_n_blocks=18,session_2_n_blocks=26, trials_per_block=25)
trials_info = trials_info_all.astype(np.uint8)
n_trials = trials_info.shape[0]
print(f"random_instructions: {random_instructions}")
# Create key bindings
key_asso = {
    'ccwtilt_keys': ['2', '9'],
    'cwtilt_keys': ['3', '0'],
    'left_keys': ['2', '3'],
    'right_keys': ['9', '0']
}

valid_response_keys = np.hstack([key_asso['left_keys'], key_asso['right_keys'], 'c'])
all_possible_keys = np.hstack([valid_response_keys, 'space'])

####################################################
# Show instructions for the experiment
####################################################
myWin.clearBuffer()
slide_n = 0
while slide_n < 9:
    suffix = ''
    instr_protocol = visual.ImageStim(myWin, './material/introduction_slides_training/slide_%i' % (slide_n + 1) + suffix + '.tif')
    instr_protocol.draw()
    myWin.flip()
    k = event.waitKeys(keyList=['right', 'left', 'space'])
    if k[0] == 'left':
        slide_n -= 1
    elif k[0] == 'right' or k[0] == 'space':
        slide_n += 1
    if slide_n < 0: slide_n = 0

myWin.flip()

##################################
# Calibration and validation
#################################

if not dummy_mode:
    genv = EyeLinkCoreGraphicsPsychoPy(et_tracker, myWin)  # we are using openGraphicsEx(), cf. manual openGraphics versus this.
    pylink.openGraphicsEx(genv)
    try:
        et_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        et_tracker.exitCalibration()

##############################################################
# Display an instruction before starting the beep trials
##############################################################

instruction_text = ("The beep phase is about to begin. \n\nWhen you're ready, please press the SPACE to start.")
instruction_stim = visual.TextStim(myWin, text=instruction_text, units='pix', pos=(0, 0), height=30, color='black')
instruction_stim.draw()
myWin.flip()
event.waitKeys(keyList=['space'])

########################################
# The practice with beep and dot
########################################
# Changed the size of the shown circle & gaze circle
# Now shown circle is 1.5 degrees size
# Gaze circle is slightly bigger too

if not dummy_mode:
    # Start recording
    et_tracker.sendMessage('TRIALID 1_BEEP') # just for better segmentation
    # put tracker in idle/offline mode before recording
    et_tracker.setOfflineMode()
    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    if not dummy_mode:
        try:
            et_tracker.startRecording(1, 1, 1, 1)
        except RuntimeError as error:
            print("ERROR:", error)
            skip_trial()
     # determine which eye(s) is/are available
    # 0- left, 1-right, 2-binocular
    eye_used = et_tracker.eyeAvailable()
    if eye_used == 1:
        et_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        et_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("Error in getting the eye information!")
    
    # Draw fixation cross
    fixh.draw()
    fixv.draw()
    fix_dot.draw()
    roi_area.draw()
    duration = 60  # 1 minute
    timer = core.CountdownTimer(duration)
    myWin.flip()
    
    new_sample = None
    old_sample = None
    
    while timer.getTime() > 0:

        # gaze-contingent part
        new_sample = et_tracker.getNewestSample()
        check_event =new_sample.getRightEye().getGaze()
        # we will only allow 1.5 deg around the fixation cross to be fixated 
        # that mean that from -100, 100 in the x-axis and -100, 100 in the y-axis
        # however note that the coordinate system of the tracker starts at 0,0 top, left corner
        # therefore, the area that we allow is 860, 1060 in the x-axis, 440, 640 in the y-axis
         
        if new_sample is not None:
            if old_sample is not None:
                if new_sample.getTime() != old_sample.getTime():
                    # check if the new sample has data for the eye
                    # currently being tracked; if so, we retrieve the current
                    # gaze position and PPD (how many pixels correspond to 1
                    # deg of visual angle, at the current gaze position)
                    if eye_used == 1 and new_sample.isRightSample():
                        g_x, g_y = new_sample.getRightEye().getGaze()
                    if eye_used == 0 and new_sample.isLeftSample():
                        g_x, g_y = new_sample.getLeftEye().getGaze()
                    f_x = g_x - 960.
                    if g_y < 540.:
                        f_y = fabs(g_y - 540.)
                    else:
                        f_y = -g_y + 540.
                    fix_dot.pos = (f_x, f_y)
                    roi_area.draw()
                    fixh.draw()
                    fixv.draw()
                    fix_dot.draw()
                    myWin.flip()
                    if float(x_right) < g_x or g_x < float(x_left) and not g_x == -32768.: # this was changed, now it can be anywhere in the y-axis # avoid beep while blink
                        beep.play()
                    else:
                        beep.stop()

        old_sample = new_sample
        
        error = et_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            et_tracker.sendMessage('tracker_disconnected')
            skip_trial()
            break
        for keycode, modifier in event.getKeys(modifiers=True):
            if keycode == 'enter': # for skipping a trial
                et_tracker.sendMessage('trial_skipped')
                skip_trial()
                # End the while loop
                break
            if keycode == "c" and (modifier['ctrl'] is True): # for terminating experiment
                et_tracker.sendMessage('experiment_aborted')
                abort_exp()

##############################################################
# MESSAGE HERE TO COMMUNICATE THAT THE TASK IS ABOUT TO START
##############################################################
stxt = ("The task is about to begin. \n\nWhen you're ready, please press the SPACE to start.")
task_start_msg = visual.TextStim(myWin, units='pix', height=30, pos=(0, 0), text=stxt,
                                 alignText='center', anchorHoriz='center', color='black')
task_start_msg.draw()
myWin.flip()
event.waitKeys(keyList=['space'])

####################################################
# The experiment
####################################################

timings = np.zeros(shape=(200, 7))
resp_times = np.zeros(shape=200)
corr_resps = np.zeros(shape=200, dtype=np.uint8)
ISI_trial_n = np.zeros(shape=200)

if staircase:
    # reversals holds trial numbers when a reversal in the staircase occured
    reversals = np.zeros(200)
    tiltChanges = np.zeros(200)
    lastTiltChangeSign = 0
    lastRespondedTrial = 0
    minTiltLvl = .5
    maxTiltLvl = 30
    tiltStep = 3
    minTiltStep = .1
    tiltHistory = []
    n_trials = 100

txt = ''
trial_n = 0
block_n = 0 ## THIS IS NEW
n_trial_pause = 25  # 25 trials in each block

if training:
    ISIdur = np.linspace(1.7, 2.21, 11)
    stimdur = .1
    max_respTime = 1.5
    if session == 1:
        n_trials = 100  # max_training_trials:100
    else:
        n_trials = 25  # max_training_trials:25

# put tracker in idle/offline mode before recording
et_tracker.setOfflineMode()

# Start recording, at the beginning of a new run
# arguments: sample_to_file, events_to_file, sample_over_link,
# event_over_link (1-yes, 0-no)
try:
    et_tracker.startRecording(1, 1, 1, 1)
except RuntimeError as error:
    print("ERROR:", error)
    abort_exp()

# Allocate some time for the tracker to cache some samples
pylink.pumpDelay(100)

while txt != 'Bye' and trial_n < n_trials:
    break_trial = False
    trial_num=trial_n+1
    et_tracker.sendMessage('TRIALID %d' % trial_num)
    print(trial_num)
    et_tracker.sendCommand("record_status_message %d" % trial_num)
    et_tracker.sendMessage("!V IAREA RECTANGLE 1 %d %d %d %d LEFT" % (Lleft, Ltop, Lright, Lbottom))
    et_tracker.sendMessage("!V IAREA RECTANGLE 2 %d %d %d %d RIGHT" % (Rleft, Rtop, Rright, Rbottom))
    et_tracker.sendMessage("!V IAREA RECTANGLE 3 %d %d %d %d CENTER" % (Cleft, Ctop, Cright, Cbottom))
    while not break_trial:
        ####### ITI #######
        # Change log: 31/01/2025
        # Mengting's code now also records the ITI :)
        event.Mouse(visible=False)
        fixh.lineColor = [1, 1, 1]
        fixv.lineColor = [1, 1, 1]
        fixh.draw()
        fixv.draw()
        selected_iti = np.random.choice(ITIdur)
        timings[trial_n, 0] = selected_iti
        timings[trial_n, 1] = myWin.flip()
        et_tracker.sendMessage('ITI') # messages need to be sent _immediately after_ win.flip()
        timer.reset(selected_iti - (ref_rate / 2.))
        myWin.clearBuffer()
        while timer.getTime() > 0: 
            beep_away("ITI") # Question is: Given Mehdi's email, maybe in this part we shouldn't do the beep? (" But yes we can take it out during the ITI, that's when I usually tell my participants to blink and move their eyes if needed.")

        ####### trial baseline (pre-instruction) #######
        fixh.lineColor = [-1, -1, -1]
        fixv.lineColor = [-1, -1, -1]
        fixh.draw()
        fixv.draw()
        timings[trial_n, 2] = myWin.flip()
        et_tracker.sendMessage('preinstruction_window')
        timer.reset(preinstdur - (ref_rate / 2.))
        while timer.getTime() > 0:
            beep_away("baseline")
        myWin.clearBuffer()
                        
        ####### Instruction #######
        # choose which letters to display for hand to use and target stimulus
        # instructions (stim-hand):
        # 0 = L-L,1 = ←-←,2 = R-R,3 = →-→,4 = L-←,5 = ←-L,6 = R-→,7 = →-R,
        # 8 = L-R,9 = L-→,10 = ←-R,11 = ←-→,12 = R-L,13 = R-←,14 = →-L,15 = →-←
        # Define left and right stimulus/hand
        left_stim_tex = [stim_lett_textures[0], stim_lett_textures[2]]  # L, ←
        right_stim_tex = [stim_lett_textures[1], stim_lett_textures[3]]  # R, →
        left_hand_tex = [hand_lett_textures[0], hand_lett_textures[2]]  # L, ←
        right_hand_tex = [hand_lett_textures[1], hand_lett_textures[3]]  # R, →

        # stim-hand combinations based on correct annotation
        combinations = [
            # same signs
            (left_stim_tex[0], left_hand_tex[0]),  # L-L
            (left_stim_tex[1], left_hand_tex[1]),  # ←-←
            (right_stim_tex[0], right_hand_tex[0]),  # R-R
            (right_stim_tex[1], right_hand_tex[1]),  # →-→
            # same side, different signs
            (left_stim_tex[0], left_hand_tex[1]),  # L-←
            (left_stim_tex[1], left_hand_tex[0]),  # ←-L
            (right_stim_tex[0], right_hand_tex[1]),  # R-→
            (right_stim_tex[1], right_hand_tex[0]),  # →-R
            # different sides (L-R)
            (left_stim_tex[0], right_hand_tex[0]),  # L-R
            (left_stim_tex[0], right_hand_tex[1]),  # L-→
            (left_stim_tex[1], right_hand_tex[0]),  # ←-R
            (left_stim_tex[1], right_hand_tex[1]),  # ←-→
            # different sides (R-L)
            (right_stim_tex[0], left_hand_tex[0]),  # R-L
            (right_stim_tex[0], left_hand_tex[1]),  # R-←
            (right_stim_tex[1], left_hand_tex[0]),  # →-L
            (right_stim_tex[1], left_hand_tex[1]),  # →-←
        ]

        # Selecting specific combination based on trial index
        stim_comb = combinations[int(trials_info[trial_n, 0])]
        stim_tex = stim_comb[0]  # Selected stimulus
        hand_tex = stim_comb[1]  # Selected hand

        # draw instructions
        hand_tex.draw()
        stim_tex.draw()
        fixh.draw()
        fixv.draw()
        
        # LETTERS SHOWN
        while timer.getTime() > 0: 
            beep_away("instructions")
        timings[trial_n, 3] = myWin.flip()
        et_tracker.sendMessage('instructions')
        et_tracker.sendMessage('!V TRIAL_VAR instruction_type %s' % trials_info[trial_n, 0])
        timer.reset(instdur - (ref_rate / 2.))
        myWin.clearBuffer()

        ####### ISI #######
        fixh.draw()
        fixv.draw()
        ISI_trial_n[trial_n] = ISIdur[trials_info[trial_n, 1]]

        while timer.getTime() > 0: 
            beep_away("ISI")
        timings[trial_n, 4] = myWin.flip()
        et_tracker.sendMessage('ISI')
        et_tracker.sendMessage('!V TRIAL_VAR ISI %s' % trials_info[trial_n, 1])
        timer.reset(ISI_trial_n[trial_n] - (ref_rate / 2.))
        myWin.clearBuffer()

        ####### Grating #######
        # 0 is vertical, 90 is horizontal
        if trials_info[trial_n, 2] == 0:
            myStim[0].ori, myStim[1].ori = [0 + tiltLvl, 0 + tiltLvl]
        elif trials_info[trial_n, 2] == 1:
            myStim[0].ori, myStim[1].ori = [0 - tiltLvl, 0 + tiltLvl]
        elif trials_info[trial_n, 2] == 2:
            myStim[0].ori, myStim[1].ori = [0 + tiltLvl, 0 - tiltLvl]
        elif trials_info[trial_n, 2] == 3:
            myStim[0].ori, myStim[1].ori = [0 - tiltLvl, 0 - tiltLvl]

        myStim[0].draw()
        myStim[1].draw()
        fixh.draw()
        fixv.draw()

        while timer.getTime() > 0: 
            beep_away("gratings")

        timings[trial_n, 5] = myWin.flip()
        et_tracker.sendMessage('gratings')
        et_tracker.sendMessage('!V TRIAL_VAR grating %s' % trials_info[trial_n, 2])
        temp_resp_t = time.time()
        timer.reset(stimdur - (ref_rate / 2.))
        myWin.clearBuffer()

        fixh.lineColor = [-1, -1, 1]
        fixv.lineColor = [-1, -1, 1]
        fixh.draw()
        fixv.draw()

        while timer.getTime() > 0: 
            beep_away("response_window")
        timings[trial_n, 6] = myWin.flip()
        et_tracker.sendMessage('response_window')

        ####### Response processing #######

        k = event.waitKeys(maxWait=max_respTime - (time.time() - temp_resp_t), keyList=valid_response_keys,
                           timeStamped=True, modifiers=True)
        resp_times2 = time.time() - temp_resp_t # of response times recorded in the system
        target_time = timings[trial_n, 5]

        if k is None:
            # k = [['none', -1.]]
            resp_key, resp_times[trial_n] = 'none', -1.
            correct = 0

            wtxt = 'Too late! Do you need a break? press SPACE to continue..'
            waitText = visual.TextStim(myWin, units='pix', height=30,
                                       pos=(0, 0), text=wtxt, alignText='center', anchorHoriz='center', color='black')
            waitText.draw()
            myWin.flip()
            event.waitKeys(keyList=['space'])

            # add the trial to the end of the list
            trials_info = np.concatenate([trials_info, trials_info[trial_n, :][np.newaxis]], axis=0)
            n_trials += 1

            if not dummy_mode:
                et_tracker.sendMessage('!V TRIAL_VAR resp_key %s' % resp_key)
                et_tracker.sendMessage('!V TRIAL_VAR RT_FLIP %s' % resp_times[trial_n])
                et_tracker.sendMessage('!V TRIAL_VAR RT_SYSTEM %s' % resp_times2)
                et_tracker.sendMessage('!V TRIAL_VAR correct %s' % correct)
                # send command
                et_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)  # mark the end of the trial

        elif k[0][0] == 'c' and k[0][1].get('ctrl', False): # DO PROPER CLOSURE OF THE EXPERIMENT
            txt = 'Bye'
            print("Ctrl+C detected, exiting experiment.")
            if not dummy_mode:
                et_tracker.sendMessage('!V CLEAR 128 128 128')
                pylink.pumpDelay(100)
                et_tracker.stopRecording()
                et_tracker.sendMessage('experiment_aborted')
                et_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
                et_tracker.setOfflineMode()
                # Clear the Host PC screen and wait for 500 ms
                et_tracker.sendCommand('clear_screen 0')
                pylink.msecDelay(500)
                # Close the edf data file on the Host
                et_tracker.closeDataFile()
                try:
                    et_tracker.receiveDataFile(edf_file, local_edf)
                except RuntimeError as error:
                    print('ERROR:', error)
        
                # Close the link to the tracker.
                et_tracker.close()
            myWin.close()
            core.quit()
            sys.exit()

        else:
            # if observer responded, check correctness
            resp_key = k[0][0]
            resp_times[trial_n] = k[0][2] - target_time
            [correct, correct_side] = exputils.compute_resp_correctness_discr2(key=resp_key,
                                                                               trial_info=trials_info[trial_n, :],
                                                                               key_asso=key_asso)
            et_tracker.sendMessage('RESP_KEY_PRESS')
            et_tracker.sendMessage('!V TRIAL_VAR correct %s' % correct)
            et_tracker.sendMessage('!V TRIAL_VAR resp_key %s' % resp_key)
            et_tracker.sendMessage('!V TRIAL_VAR RT_FLIP %s' % resp_times[trial_n])
            et_tracker.sendMessage('!V TRIAL_VAR RT_SYSTEM %s' % resp_times2)

            feedback_text = 'Correct!' if correct else 'Incorrect!'
            feedback_color = 'green' if correct else 'red'

            # Create and display feedback text in the center of the screen
            feedback = visual.TextStim(myWin, units='pix', height=30,
                                       pos=(0, 0), text=feedback_text, alignText='center',
                                       anchorHoriz='center', color=feedback_color)
            feedback.draw()
            myWin.flip()
            et_tracker.sendMessage('feedback')
            timer.reset(feedback_time - (ref_rate / 2.))
            while timer.getTime() > 0: continue

            # mark the end of the trial FOR THE TRACKER
            if not dummy_mode:
                et_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)  # mark the end of the trial

        # write in log file
        task_out.write('%s\t%i\t%i\t%i\t%i\t%i\t%s\t%.5f\t%.5f\t%i\n' % (obs, block, trial_n+1,
                                                                             trials_info[trial_n, 0],
                                                                             trials_info[trial_n, 1],
                                                                             trials_info[trial_n, 2], resp_key,
                                                                             resp_times[trial_n], resp_times2, correct))

        corr_resps[trial_n] = correct
        
        # Pause every n_trial_pause trials
        if trial_n >= (last_pause_trial_n + n_trial_pause):
            # Here we should add an if-else statement for no drift in the last round
            # Check with Mengting on Monday what's the flow to add it
            et_tracker.sendMessage('trial_break_start')
            pylink.pumpDelay(100)  # give some time to catch everything
            et_tracker.stopRecording()  # stop recording
            last_pause_trial_n = trial_n
            # Calculate reaction time for the last chunk of trials
            rt_last_chunk = resp_times[(trial_n - n_trial_pause):trial_n]
            rt_last_hits = rt_last_chunk[rt_last_chunk > 0]
            if rt_last_hits.size > 0:
               median_rt = np.median(rt_last_hits) * 1000  
            else:
               median_rt = float('nan')

            # Calculate accuracy for the last chunk of trials
            corr_last_chunk = corr_resps[(trial_n - n_trial_pause):trial_n]

            if corr_last_chunk.size > 0 and n_trial_pause > 0:
                accuracy_last_chunk = corr_last_chunk.sum() / n_trial_pause * 100
            else:
                accuracy_last_chunk = float('nan')

            print("RT last chunk:", median_rt)
            print("Correct responses last chunk:", accuracy_last_chunk)

            if np.isnan(accuracy_last_chunk):
                accuracy_last_chunk = 0.0
            if np.isnan(median_rt):
                median_rt = 0.0

            wtxt = ('A little break!\n\n'
                    'In the last %i trials:\n'
                    'Your accuracy was: %.2f %%\n'
                    'Your reaction time was: %.2f ms\n\n'
                    'Press SPACE to continue..') % (n_trial_pause, accuracy_last_chunk, median_rt)

            waitText = visual.TextStim(myWin, units='pix', height=30,
                                       pos=(0, 0), text=wtxt, alignText='center', anchorHoriz='center', color='black')
            waitText.draw()
            myWin.flip()
            kk = event.waitKeys(keyList=['space'])
            et_tracker.sendMessage('trial_break_end')

            if trial_n < n_trials:
                while not dummy_mode:
                    if (not et_tracker.isConnected()) or et_tracker.breakPressed():
                        abort_exp()
                    try:
                        error = et_tracker.doDriftCorrect(int(1920 / 2.0),
                                                          int(1080 / 2.0), 1, 1)
                        # break following a success drift-check
                        if error is not pylink.ESC_KEY:
                            break
                    except:
                        pass
            
            # put tracker in idle/offline mode before recording
            et_tracker.setOfflineMode()
            
            # Start recording, at the beginning of a new run
            # arguments: sample_to_file, events_to_file, sample_over_link,
            # event_over_link (1-yes, 0-no)
            try:
                et_tracker.startRecording(1, 1, 1, 1)
            except RuntimeError as error:
                print("ERROR:", error)
                abort_exp()
            
            # Allocate some time for the tracker to cache some samples
            pylink.pumpDelay(100)
            block_n += 1

        break_trial = True 

        # Staircase
        if staircase:
            if (trial_n > 1):
                [tiltLvl, tiltStep, tiltChanges, lastTiltChangeSign, reversals] = exputils.do_staircase(tiltLvl,
                                                                                                        tiltStep,
                                                                                                        corr_resps,
                                                                                                        correct_side,
                                                                                                        trial_n,
                                                                                                        tiltChanges,
                                                                                                        lastTiltChangeSign,
                                                                                                        reversals,
                                                                                                        minTiltLvl,
                                                                                                        maxTiltLvl,
                                                                                                        minTiltStep,
                                                                                                        lastRespondedTrial)
                lastRespondedTrial = trial_n
                tiltHistory.append(tiltLvl)
        trial_n += 1

# close the log file
task_out.close()

# We need to close the data file, transfer it from ET PC to STIM PC and then close the connection between both PCs (plus exist PsychoPy)

if not dummy_mode:
    et_tracker.setOfflineMode()
    pylink.pumpDelay(100)
    et_tracker.closeDataFile()  # close the file
    pylink.pumpDelay(500)
    et_tracker.receiveDataFile(edf_file, local_edf)  # transfer the file
    et_tracker.close()  # close the link

# save parameters
if training:
    outfolder = './beh_results/obs_%s/session_%s/training/' % (int(obs), int(session))
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    training_param_file = '%straining_parameters_obs%s_session%s_date%s.npz' % (
        outfolder, int(obs), int(session), str(timeAndDate))
    np.savez(training_param_file, timings=timings, resp_times=resp_times, stimSize=stimSize,
             contrast=contrast, oris=oris, dist_to_center=dist_to_center, spatFreq=spatFreq,
             ref_rate=ref_rate, preinstdur=preinstdur, instdur=instdur, ISIdur=ISIdur,
             stimdur=stimdur, ITIdur=ITIdur, max_respTime=max_respTime)

if staircase:
    outfolder = './beh_results/obs_%s/session_%s/staircase/' % (int(obs), int(session))
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    output_stairc_file = '%sstaircase_parameters_obs%s.npz' % (outfolder, int(obs))
    np.savez(output_stairc_file, tiltLvl=tiltLvl, tiltStep=tiltStep, tiltChanges=tiltChanges,
             reversals=reversals, tiltHistory=np.array(tiltHistory))

    # tiltLvl_file = '%sstaircase_tiltLvl_obs%s.npy' % (outfolder, obs)
    # tiltStep_file = '%sstaircase_tiltStep_obs%s.npy' % (outfolder, obs)
    # tiltChanges_file = '%sstaircase_tiltChanges_obs%s.npy' % (outfolder, obs)
    # reversals_file = '%sstaircase_reversals_obs%s.npy' % (outfolder, obs)
    # tiltHistory_file = '%sstaircase_tiltHistory_obs%s.npy' % (outfolder, obs)
    #
    # tiltHistory = np.array(tiltHistory)
    # np.save(tiltLvl_file, tiltLvl)
    # np.save(tiltStep_file, tiltStep)
    # np.save(tiltChanges_file, tiltChanges)
    # np.save(reversals_file, reversals)
    # np.save(tiltHistory_file, tiltHistory)

# End of experiment
txt = 'FINISHED ! Thanks !'
endText = visual.TextStim(myWin, units='pix', height=30,
                          pos=(0, 0), text=txt, alignText='center', anchorHoriz='center', color='black')
endText.draw()
myWin.flip()
time.sleep(3)

print("Experiment finished.")

# disp.close()
myWin.close()
core.quit()