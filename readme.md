## MtMs - demonstration on the sinusoidal task ##

This repository contains the refactored MtMs model used in the prediction part of the M6 forecasting competition.
All core functions can be found in the folder `src/MtMs`. 
File `src/main.R` contains a demonstration of how the MtMs model can be used in practice.
For more details about the model, please read the accompanying PDF.

To illustrate how it could be used to solve meta-learning and multi-task problems, we apply it to the sinusoidal regression task, which is commonly used to benchmark different meta-learning approaches.
To replicate this demonstration:

1) Install dependencies in `requirements.txt`
2) Run the experiment via either:
   - If you are a [DVC](https://dvc.org/ "https://dvc.org/") user, simply run `dvc exp run` in the console or run the `run.bat` file.
   - If not, you can also run directly the `src/main.R`.

After the experiment is finished, you can find the outputs located in the folder `outputs`.
To modify experiment settings, you can change the contents of the `par.yaml` file.

