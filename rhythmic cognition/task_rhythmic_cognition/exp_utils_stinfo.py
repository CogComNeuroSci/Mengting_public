import os
import numpy as np

###################
# Log stimulation #
###################

def init_log(obs, session, timeAndDate, training, staircase):
    """Initialize log files with observer ID, group, block, and timestamp."""

    # Base directory with session level
    base_outfolder = './beh_results/obs_%s/session_%s/' % (obs, session)
    et_base_outfolder = './et_results/obs_%s/session_%s/' % (obs, session)

    if training:
        outfolder = base_outfolder + 'training/'
        et_outfolder = et_base_outfolder + 'training/'
        output_task_file = '%straining_obs%s_session%s_date%s.txt' % (
            outfolder, obs, session, timeAndDate)
    elif staircase:
        outfolder = base_outfolder + 'staircase/'
        et_outfolder = et_base_outfolder + 'staircase/'
        output_task_file = '%sstaircase_obs%s_session%s_date%s.txt' % (
            outfolder, obs, session, timeAndDate)
    else:
        outfolder = base_outfolder + 'main_task/'
        et_outfolder = et_base_outfolder + 'main_task/'
        output_task_file = '%stask_obs%s_session%s_date%s.txt' % (
            outfolder, obs, session, timeAndDate)

    os.makedirs(outfolder, exist_ok=True)
    os.makedirs(et_outfolder, exist_ok=True)

    try:
        task_out = open(output_task_file, 'a')
    except IOError as e:
        print(f"Failed to open file {output_task_file}: {e}")
        raise

    # Write a header line describing the data fields
    task_out.write(
        'observer\tblock\ttrial\tinstr_type\tISI\tstimcombi\tresponse\tresptime\tresptime2\trespcorrect\n')

    return task_out, outfolder, et_outfolder

#################################
# Create the paradigm structure #
#################################
def make_mystinfo_design(obs, session, n_instructions, n_ISIs, n_stimCombi, session_1_n_blocks, session_2_n_blocks, trials_per_block):
    trials_info_folder = f'./beh_results/obs_{obs}/trials_info'
    if not os.path.exists(trials_info_folder):
        os.makedirs(trials_info_folder)

    session_trials = {}
    random_instructions = None

    # Load existing session data if available
    for i in range(1, 3):
        filename = f"obs_{obs}_session_{i}_trialinfo.npz"
        filepath = os.path.join(trials_info_folder, filename)
        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True)
            session_trials[i] = (data['trials_info'], data['trials_info_blocks'])
            if 'random_instructions' in data and random_instructions is None:
                random_instructions = data['random_instructions']

    # Generate new session data if needed
    if session == 1 and not all(k in session_trials for k in [1, 2]):
        print("Generating trials for session 1 and 2...")

        # Ensure random_instructions is generated only once
        if random_instructions is None:
            random_instructions_LR = np.random.choice(np.arange(8, 12), size=2, replace=False)
            random_instructions_RL = np.random.choice(np.arange(12, 16), size=2, replace=False)
            random_instructions = np.concatenate([random_instructions_LR, random_instructions_RL])

        trials_info_1 = make_trials(n_instructions, n_ISIs, n_stimCombi, random_instructions)
        trials_info_2 = make_trials(n_instructions, n_ISIs, n_stimCombi, random_instructions)

        print(f"Session 1: trials_info_1 shape = {trials_info_1.shape}")
        print(f"Session 2: trials_info_2 shape = {trials_info_2.shape}")

        # Reallocate last 100 trials of session 1
        trials_info_1_main = trials_info_1[:-100]  # Keep only first 568 trials
        trials_info_1_extra_2 = trials_info_1[-100:]  # Extra trials for session 3

        session_trials[1] = (trials_info_1_main, [])
        session_trials[2] = (np.vstack((trials_info_2, trials_info_1_extra_2)), [])

        # distribute trials into blocks
        def distribute_blocks(trials_info, n_blocks):
            trials_info_blocks = []
            trial_n = 0
            for _ in range(n_blocks - 1):
                trials_info_blocks.append(trials_info[trial_n:trial_n + trials_per_block])
                trial_n += trials_per_block
            trials_info_blocks.append(trials_info[trial_n:])  # Last block may contain fewer trials
            return trials_info_blocks

        for i in range(1, 3):
            # assign blocks for all sessions
            session_trials[i] = (session_trials[i][0], distribute_blocks(session_trials[i][0], eval(f"session_{i}_n_blocks")))
            # check block information
            print(f"Session {i}: total trials = {sum(len(b) for b in session_trials[i][1])}, total blocks = {len(session_trials[i][1])}")
            # Save the data
            filename = f"obs_{obs}_session_{i}_trialinfo.npz"
            filepath = os.path.join(trials_info_folder, filename)
            np.savez(filepath, trials_info=session_trials[i][0], trials_info_blocks=np.array(session_trials[i][1], dtype=object),
                     random_instructions=random_instructions)
        print("Session 1, 2 trials saved.")

    if session not in session_trials:
        raise KeyError(f"Session {session} data is missing!")

    return session_trials[session][0], session_trials[session][1], random_instructions


def make_trials(n_instructions, n_ISIs, n_stimCombi, random_instructions):
    ISI_inds = np.arange(n_ISIs)
    n_trials = n_instructions * n_ISIs * n_stimCombi
    trials_info = np.zeros(shape=[n_trials, 3])
    trial_n = 0

    for instr_n in np.arange(8):
        for delay in ISI_inds:
            for stimCombi_n in np.arange(n_stimCombi):
                trials_info[trial_n, :] = [instr_n, delay, stimCombi_n]
                trial_n += 1

    for instr_n in random_instructions:
        for delay in ISI_inds:
            for stimCombi_n in np.arange(n_stimCombi):
                trials_info[trial_n, :] = [instr_n, delay, stimCombi_n]
                trial_n += 1

    np.random.shuffle(trials_info)

    new_arr = trials_info
    verif_arr = new_arr[:-1, 0] == new_arr[1:, 0]
    print('\tVerifying instructions order..')

    while sum(verif_arr):
        bad_inst_ind = np.where(verif_arr)[0][0] + 1
        non_bad_inst_inds = new_arr[:, 0] != new_arr[bad_inst_ind, 0]
        ind_to_place_bad_inst = np.where(non_bad_inst_inds[1:].astype(int) + non_bad_inst_inds[:-1].astype(int) > 1)[0][
                                    0] + 1
        bad_inst_info = new_arr[bad_inst_ind, :]
        mask = np.ones(n_trials).astype(bool)
        mask[bad_inst_ind] = False
        new_arr = new_arr[mask, :]
        new_arr = np.concatenate(
            [new_arr[:ind_to_place_bad_inst, :], bad_inst_info[np.newaxis, :], new_arr[ind_to_place_bad_inst:, :]],
            axis=0)
        verif_arr = new_arr[:-1, 0] == new_arr[1:, 0]
    print('\tAll done! (sum(verif_arr=%i)' % sum(new_arr[:-1, 0] == new_arr[1:, 0]))

    trials_info = new_arr

    return trials_info


###########################################################
# Computes whether the observer gave the correct response #
###########################################################

def compute_resp_correctness_discr2(key, trial_info, key_asso):
    correct = False
    # Extract trial information

    # trial_info[2]: two grating (0-3)
    # 0= cw-cw, 1=ccw-cw, 2=cw-ccw,3=ccw-ccw

    # trial_info[0]: instructions (0-15) (stim-hand)
    # instructions (stim-hand):
    # 0 = L-L,1 = ←-←,2 = R-R,3 = →-→,4 = L-←,5 = ←-L,6 = R-→,7 = →-R,
    # 8 = L-R,9 = L-→,10 = ←-R,11 = ←-→,12 = R-L,13 = R-←,14 = →-L,15 = →-←

    # case in which the stim are cw-cw (r-r)
    if trial_info[2] == 0:
        if trial_info[0] in [0, 1, 4, 5] and key in key_asso['left_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (l-l)
        if trial_info[0] in [12, 13, 14, 15] and key in key_asso['left_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (r-l)
        if trial_info[0] in [8, 9, 10, 11] and key in key_asso['right_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (l-r)
        if trial_info[0] in [2, 3, 6, 7] and key in key_asso['right_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (r-r)

    # case in which the stim are ccw-cw (l-r)
    if trial_info[2] == 1:
        if trial_info[0] in [0, 1, 4, 5] and key in key_asso['left_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (l-l)
        if trial_info[0] in [12, 13, 14, 15] and key in key_asso['left_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (r-l)
        if trial_info[0] in [8, 9, 10, 11] and key in key_asso['right_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (l-r)
        if trial_info[0] in [2, 3, 6, 7] and key in key_asso['right_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (r-r)

    # case in which the stim are cw-ccw (r-l)
    if trial_info[2] == 2:
        if trial_info[0] in [0, 1, 4, 5] and key in key_asso['left_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (l-l)
        if trial_info[0] in [12, 13, 14, 15] and key in key_asso['left_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (r-l)
        if trial_info[0] in [8, 9, 10, 11] and key in key_asso['right_keys'] and key in key_asso[
            'cwtilt_keys']: correct = True;  # (l-r)
        if trial_info[0] in [2, 3, 6, 7] and key in key_asso['right_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (r-r)

    # case in which the stim are ccw-ccw (l-l)
    if trial_info[2] == 3:
        if trial_info[0] in [0, 1, 4, 5] and key in key_asso['left_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (l-l)
        if trial_info[0] in [12, 13, 14, 15] and key in key_asso['left_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (r-l)
        if trial_info[0] in [8, 9, 10, 11] and key in key_asso['right_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (l-r)
        if trial_info[0] in [2, 3, 6, 7] and key in key_asso['right_keys'] and key in key_asso[
            'ccwtilt_keys']: correct = True;  # (r-r)

    correct_side = (trial_info[0] in [0, 1, 4, 5, 12, 13, 14, 15] and key in key_asso['left_keys']) or (
                trial_info[0] in [2, 3, 6, 7, 8, 9, 10, 11] and key in key_asso['right_keys'])

    return correct, correct_side


##################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Do the staircase procedure using the Levitt rule and returns
# variables that might have change 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def do_staircase(tiltLvl, tiltStep, corr_resps, correct_side, trial_n, tiltChanges,
                 lastTiltChangeSign, reversals, minTiltLvl, maxTiltLvl, minTiltStep, lastRespondedTrial):
    tilt = tiltLvl

    if not sum(reversals):
        # 1st reversal do simple one-up-one-down
        if corr_resps[trial_n]:
            tiltSign = -1
        else:
            tiltSign = 1

        tilt += tiltSign * tiltStep
        if (tilt > minTiltLvl) & (tilt < maxTiltLvl):
            tiltLvl = tilt
            tiltChanges[trial_n] = tiltSign
            if tiltChanges[trial_n] == -lastTiltChangeSign:
                reversals[trial_n] = 1
                tiltStep = tiltStep / 2.

            lastTiltChangeSign = tiltSign

    elif correct_side:
        tiltSign = 0
        # not 1st reversal: one-up-two-down
        if not corr_resps[trial_n]:
            tiltSign = 1  # one-up -> increase tiltLvl
        elif corr_resps[trial_n] and corr_resps[lastRespondedTrial]:
            tiltSign = -1  # two-down -> decrease tiltLvl

        tilt += tiltSign * tiltStep

        # stores the new tilt if it doesn't exceed the tilt limits
        if (tilt > minTiltLvl) & (tilt < maxTiltLvl):
            print('\ntiltLvl changed %i\n' % tiltSign)
            tiltLvl = tilt
            tiltChanges[trial_n] = tiltSign

            # if there is a reversal in this feature's tiltChange
            if tiltSign == -lastTiltChangeSign:
                reversals[trial_n] = 1
                # takes care of the Levitt rule
                if np.mod(len(np.argwhere(reversals != 0)), 2):
                    # if the reversal number is odd divide tiltStep by 2
                    newTiltStep = tiltStep / 2.

                    # if no minimum for tilt step
                    # tiltStep = newTiltStep;

                    # if minimum tilt step
                    if newTiltStep >= minTiltStep:
                        tiltStep = newTiltStep

            if tiltSign: lastTiltChangeSign = tiltSign

    return tiltLvl, tiltStep, tiltChanges, lastTiltChangeSign, reversals
