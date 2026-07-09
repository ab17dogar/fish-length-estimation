from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import random
import os
from matplotlib import pyplot as plt

def plot_frequency(all_train_samples):
    '''
    Plot the frequency of each group in all the different splits as a bar plot
    args:
        all_train_samples:
    '''
    # Count frequencies of numbers from 1 to 25
    frequencies = {i: all_train_samples.count(i) for i in range(1, 26)}

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(frequencies.keys(), frequencies.values(), color='teal')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Frequency of Numbers in All train_split Samples')
    plt.xticks(range(1, 26))  # Ensure x-axis shows numbers from 1 to 25
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('figure.png')

def get_template_yaml(yaml_path:str, yaml:YAML) -> CommentedMap:
    '''
    Loads the yaml file from path and hardcodes the validation sets as empty.
    args:
        yaml_path: The path to the yaml template file.
        YAML: ruamel.yaml.YAML object
    returns:
        template_yaml: CommentedMap, similar to dict
    '''
    with open(yaml_path, 'r') as file:
        template_yaml = yaml.load(file)

    template_yaml['val_split'] = []
    template_yaml['val_dataset'] = []
    return template_yaml


if __name__ == '__main__':
    '''
    Script to generate different split configurations training, for different amounts of training groups.
    '''
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--min', type=int, default=1, help='Minimum number of training groups')
    ap.add_argument('--max', type=int, default=20, help='Maximum number of training groups')
    ap.add_argument('--folds', type=int, help='Number of folds')
    ap.add_argument('--yaml', type=str, help='Path to the yaml template')
    ap.add_argument('--seed', type=int, required=False, default=35, help='Random seed')
    ap.add_argument('--test_set', choices=['random', 'stratified'], type=str, default='stratified', help='Test set. If stratified, then thest set is groups 10, 14, 20, 21, 22. Otherwise random' )
    args = ap.parse_args()

    random.seed(args.seed)
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes and formatting
    yaml.default_flow_style = None
    template_yaml = get_template_yaml(args.yaml, yaml)
    samples = []

    for n in range(args.min, args.max + 1):
        dir_name = f"{n}"  # Directory name based on n
        os.makedirs(dir_name, exist_ok=True)  # Create directory if it doesn't exist

        print(f"\nGenerating {args.folds} sets of {n} different numbers and saving to '{dir_name}' directory.")
        for x in range(1, args.folds + 1):
            template_yaml['output_dir'] = f'../output/n_groups_{n}_split_{x}'

            if args.test_set == 'stratified':
                train_split = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 23, 24, 25], n)  # Generate n unique random numbers
                template_yaml['train_split'] = train_split
                template_yaml['test_split'] = [10, 14, 20, 21, 22]
            else:
                train_split = random.sample(range(1,26), n)
                template_yaml['train_split'] = train_split
                template_yaml['test_split'] = [group for group in range(1,26) if group not in train_split]

            # Save samples to study frequency
            samples.extend(train_split)
            # Create file path for the YAML file
            file_name = os.path.join(dir_name, f"split_{x}.yaml")
            
            # Save the modified YAML
            with open(file_name, 'w') as output_file:
                yaml.dump(template_yaml, output_file)
            
            print(f"Saved: {file_name}")

    plot_frequency(samples)


