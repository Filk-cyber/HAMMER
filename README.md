### data
Please download the data/directory data from [the link](https://drive.google.com/file/d/1evfYVpMtS4-GUUz8yWObsYS-d_V3dzz5/view?usp=drive_link).
### Preprocessing：
```python
python preprocessing.py \
    --dataset hotpotqa \
    --raw_data_folder data/hotpotqa/raw_data \
    --save_data_folder data/hotpotqa 
···

### step 1:
Run the followiing command to generate KGs:
```python
python generate_knowledge_triples.py \
    --dataset hotpotqa \
    --input_data_file data/hotpotqa/test.json \
    --save_data_file data/hotpotqa/test_with_kgs.json
```
### step 2:

