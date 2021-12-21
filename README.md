# Backdoored Image Detector

Through this project we intend to create a backdoored image detector which otherwise would be detected as a valid class by the badnet.

## Project Members
Shashank Shekhar 

Sai Kiran

Gayatri Kalindi

## Folder Structure

├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h5
    
    ├── Repaired Models
        └── Sunglasses.pickle
        └── multi_eye.pickle
        └── multi_lipstick.pickle
        └── multi_sun.pickle
        └── ano1.pickle
        └── ano2.pickle
        
    
├── architecture.py
└── eval.py // this is the evaluation script
