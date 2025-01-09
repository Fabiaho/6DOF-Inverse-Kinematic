

# Model Usage and Training Instructions

## Model Download Links

Download the models and `run_args` from Hugging Face and save them in the `runs/` directory.

- **Model without orientation:** [IK_no_orientation_best_model](https://huggingface.co/hange/IK_no_orientation_best_model)
- **Model with orientation:** [IK8](https://huggingface.co/hange/IK8)

## Setup and Installation

To set up the environment, install the required packages using `requirements.txt`. A CUDA-compatible setup is not mandatory, but it is highly recommended for better performance.

Run the following command to create and activate the virtual environment:

```bash
# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Training Script

Use the following command to start training with an example set of parameters:

```bash
python train.py --chain-path assets/UR5/urdf/ur5_robot.urdf --num-joints 6
```
- `--chain-path`: Specify the URDF file path for your robotic arm.  
- `--num-joints`: Define the number of joints (e.g., 6 for a 6-DOF robotic arm).

Adjust the parameters as needed for your specific use case.

## Data Location and Sample Dataset

- **Full Dataset:** The complete dataset can be found in the `root/project files/` directory.  
- **Sample Dataset:** A small sample dataset for testing purposes is pre-configured as the default option in the code.

## Notes on Model Variants

- The `model_no_orientation` directory is incomplete for training and only contains files that differ from the full model with orientation (`model_training/`).  
- Therefore exchange files for model training.

## Results

### Model Performance Summary

#### Model (with Orientation)  
- **Accuracy:** 52%  
- **RMSE:** 0.637  
- **Parameters:** ~33 Million  

#### Model (without Orientation)  
- **Accuracy:** 65%  
- **RMSE:** 0.100  
- **Parameters:** ~33 Million  

### Training Details
- **Training Duration:** Approximately 4–6 hours per train run (using RTX 3060 GPU)  
- **Hyperparameter Tuning:**  
  Multiple models were trained with varying hyperparameters, such as:  
  - Number of epochs  
  - Early stopping criteria  
  - Batch size  
  - Number of Gaussians  
  - Embedding dimensions  
  - Hidden size of the joint network  

The best-performing models' configurations can be found on Hugging Face in the `run_args.json` file of the respective model.  
