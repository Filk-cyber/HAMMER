### data
Please download the data/directory data from [the link](https://drive.google.com/file/d/1evfYVpMtS4-GUUz8yWObsYS-d_V3dzz5/view?usp=drive_link).
### Preprocessing
After downloading the dataset, run the following command to create the development and test sets used in our experiments:
```python
python preprocessing.py \
    --dataset hotpotqa \
    --raw_data_folder data/hotpotqa/raw_data \
    --save_data_folder data/hotpotqa 
```
Among them, `--raw_data_folder` specifies the folder containing the raw data, and `--save_data_folder` specifies the folder where development and testing data will be saved.
### step 1
Run the followiing command to generate KGs:
```python
python generate_knowledge_triples.py \
    --dataset hotpotqa \
    --input_data_file data/hotpotqa/test.json \
    --save_data_file data/hotpotqa/test_with_kgs.json
```
### step 2:

