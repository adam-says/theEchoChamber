# Biomimetic Closed-Loop LFP

## Summary:
    This project aims to implement a closed-loop reservoir computing algorithm to prevent epileptic seizure-like activity in rodent brain slices.
    The algorithm will be implemented in Python and will use the NI USB-6343 board for digital interface.
    On the biological side, brain slices (hippocampal-cortical) will be incubated with 4-AP to increase excitability. Seizure-like activity will be triggered by cutting the Schaffer collaterals, disrupting the hipp-ctx loop.
    This will be monited using extracellular LFP recordings: one recording electrode in the CA3 region, and a second one in the entorhinal cortex.
    A stimulation electrode will be placed adjecent to the cortex, after the cut.
    
## Software-side plan:
    1. Connect to the NI board
    2. Set up the board for simultaneous recording and stimulation (two AI channels/one AO channel, FS = 20000 Hz)
    3. Create a visual monitor of the data stream, using Python-based websockets (so we have a web interface to monitor the experiment, where we can see the recorded activity, and the eventual stimulation in realtime)
        3.1. This visual interface should offer the possibility of start/stop acquisition, start/stop gap-free recording, and the choice between control recording (no stimulation), and the closed-loop recording (stimulation is applied - stimulation defined by the RC algorithm)
    4. Implement a buffer system to store incoming data and feed it into Angel's control algorithm - should be in realtime
    5. Channel the RC algorithm output to the AO channel to stimulate the brain slice
    6. Be sure that the data is saved to disk in a format that can be used to analyze the experiment
    