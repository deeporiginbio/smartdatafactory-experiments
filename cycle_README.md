# README

## Running the Training Cycle

To run the training cycle, ensure that the following parameters are properly configured in the configuration file:

### Configuration Instructions:
1. **Seed Configuration**:
   - Set the **seed** to ensure reproducibility.

2. **Buffer Settings**:
   - Define the following parameters:
     - `buffer_size`: Size of the buffer.
     - `seed_size`: Initial size of the buffer.
     - `buffer_step_size`: Increment size for the buffer at each step.

3. **Training Data Path**:
   - Provide the **path** to the processed training data. The data should be in `.npz` file format.

4. **Use Model Predictions or Random Selection**:
   - Indicate whether to use model prediction scores or select randomly using the `use_buffer_predictions` parameter.

5. **Step Configuration**:
   - Set the **step_num** to define the number of steps in the cycle.

6. **Training Parameters**:
   - Specify the following:
     - `max_epoch`: Maximum number of epochs to train each model.
     - `gpu_device`: GPU device to be used for training.

7. **Model Configurations**:
   - Provide **config files** for each model.  
   - Each model's configuration file should include:
     - The model type and name.
     - Basic configurations required for the model.

### Running the Cycle:

To execute the training cycle, use the following command:

Use [cycle experiment config file](force_field_models/sdf_experiments/test_configurations/cycle_readme_example.yaml)
```bash
python -m force_field_models.train.cycle -tc cycle_readme_example.yaml
```
