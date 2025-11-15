**Download**

Download the experimental `Data_lists_cleaned` from [Data_lists_cleaned repo](https://huggingface.co/datasets/Iceswordlab/EDR3.6B-63F) and extract them into the `Data_lists_cleaned/` directory to obtain `Families.json`, `TRAIN.json`, `TEST.json` and `TEST_forUnknownAttack.json`.

Download the `Meta_data_cleaned` from [Meta_data_cleaned repo](https://huggingface.co/datasets/Iceswordlab/EDR3.6B-63F) and extract it into the `Meta_data_cleaned/` directory.

**File descriptions:**  
   - `Data_lists_cleaned/` — Directory containing all data lists needed for both the main experiments and the ablation study.
    - `Families.json` — List of families used in the experiments.
    - `TRAIN.json` — List of training data samples.  
    - `TEST.json` — List of test data samples.  
    - `TEST_forUnknownAttack.json` — List of an additional independent test set containing 9,000 normal samples and 10,000 unknown attack samples.
   - `Meta_data_cleaned/` — Directory containing all the raw data required for both the main experiments and the ablation study. 
   
**Preprocessing (recommended):**
- To facilitate experiments, we recommend first running:  
     ```bash
     python get_npz_convenient.py
     ```  
- This script reads the raw data (text) in `Meta_data_cleaned/` and converts each sample into a `.pkl.gz` file with the same name.  
- The generated `.pkl.gz` files are also saved in the `Meta_data_cleaned/` directory.