import subprocess

n = [1,2]
trans = ['augment']
datasets = ['mixe', 'random_prompts_finetunedter']

for dataset in datasets:
    if dataset == 'random_prompts_finetuned':
        e = 25
    else:
        e = 5
    for unf in [False, True]:
        for i in n:
            for t in trans:
                subprocess.run(['python',
                                '/users/eleves-a/2022/matthieu.bruant/cheese-classification/train.py',
                                f'model.instance.layers={i}',
                                f"dataset/train_transform={t}",
                                f"model.instance.unfreeze_last_layer={unf}",
                                f"experiment_name=whichdata_{dataset}_{t}_{i}_unf_{unf}",
                                f"dataset_name={dataset}",
                                f"epochs={e}",
                                ])