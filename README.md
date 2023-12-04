# LM CLRS
GitHub for the CLRS tasks in language form.
TODO:
Clean the data i.e. the tricks I used in checker
    Check topo sort, bridges
Add the bible to here

## Reference

## How to use
1. Install Requirements
    ```SHELL
    pip install -r requirements.txt
    ```

2. Generate data
    ```SHELL
    python data_generator.py --task <TASK> --part <train or test> --num_samples <number of generated samples>
    ```

3. (Optional) Create txt files of inputs
    This only works for 2d array inputs.
    ```SHELL
    python inputs_to_text_files.py --task <TASK> --dp <number of decimal places to round data to>
    ```

4. Generate prompts
    We give an optional framework to generate prompts automatically
    ```SHELL
    python prompt_generator.py --task <TASK> --part <train or test> --num_samples <number of generated samples>`
    ```

    You can then call `prompt_gen(task, partition, max_samples)` to generate prompts at test time.

### Origional GNN CLRS
https://github.com/google-deepmind/clrs
