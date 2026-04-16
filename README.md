# theEchoChamber

This project aims to implement a closed-loop reservoir computing algorithm to prevent epileptic seizure-like activity in rodent brain slices.
The algorithm will be implemented in Python and will use the NI USB-6343 board for digital interface.
On the biological side, brain slices (hippocampal-cortical) will be incubated with 4-AP to increase excitability. Seizure-like activity will be triggered by cutting the Schaffer collaterals, disrupting the hipp-ctx loop.
This will be monited using extracellular LFP recordings: one recording electrode in the CA3 region, and a second one in the entorhinal cortex.
A stimulation electrode will be placed adjecent to the cortex, after the cut.

## Contributors
[Adam Armada-Moreira](https://github.com/adam-says)

[Angel Canal-Alonso](https://github.com/AngelCanal)

[Alessio Di Clemente](https://github.com/alediclemente)

[Laura Monni](https://github.com/LauraMonni1)

[Michele Giugliano](https://github.com/mgiugliano)

## Usage
If connected to the NI board
```
python closed_loop.py
```

If in testing mode (no device connected)
```
python closed_loop.py --mock
```
