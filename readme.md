## Overview
check image_classifier_project.ipynb/-.html to get insight about project scope and details about model training

## Using command line 
  
### training a model with default parameters
here, default pre-settings will be used to build, train a model and save the corresponding checkpoint. 
> - pre-trained model architecture: vgg11
> - learning rate: 0.003
> - epochs: 20
> - 2 fully connected layer of size: 256, 128

execute following command to run in default mode:
```
python train.py path_dataset_folder
```

### specifying parameters to train model
following parameters can specified wrt model training:
> - arch (architecture pre-trained model)
```
python train.py path_dataset_folder --arch provide_architecture
```
> - learning_rate
```
python train.py path_dataset_folder --learning_rate provide_learning_rate
```
> - epochs
```
python train.py path_dataset_folder --epochs number_epochs
```
> - fully connected layer architecture
```
python train.py path_dataset_folder --hidden_units number_first number_second 
```
> - map tensors to GPU (default is CPU)
```
python train.py path_dataset_folder --gpu
```

optionally, checkpoints can be saved in specified directory. however, always provide directory_name in combination with file_name to save checkpoint in specified folder.
error will be thrown if no such path structure is provided with corresponding argument.
```
python train.py path_dataset_folder --save_directory save_direcory/file_name
```

### reconstructing trained model from checkpoint and making predictions
similar to model training, making predictions can either be executed in default mode or with optional parameters.

#### default mode
file_paths specifying checkpoint and input image need to be provided.
```
python predict.py image_path checkpoint_path
```

#### optional parameters
> - map tensors to GPU (default is CPU). only valid if OS supports CUDA. if used without support, an error will be thrown.
```
python predict.py image_path checkpoint_path --gpu
```
> - arch (architecture pre-trained model). should be provided if architecture of pre-trained model is known and no information is provided in checkpoint.
otherwise the default trained model checkpoint (checkpoint_iteration_7.pth) is loaded and concatenated with the correct pre-trained architecture (vgg11). 
```
python predict.py image_path checkpoint_path --arch provide_architecture
```
> - top_k (number of predicted (best) class indices shown)
```
python predict.py image_path checkpoint_path --top_k number_classes
```
> - category_names (number of predicted (best) class names shown). default mapping dict is cat_to_name.json
```
python predict.py image_path checkpoint_path --category_names path_mapping_dict
```
