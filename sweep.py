import subprocess

n = [1,2,3]
trans = ['simple', 'augment']

# Launch subprocess
for unf in [False, True]:
    for i in n:
        for t in trans:
            subprocess.run(['python',
                            '/users/eleves-a/2022/matthieu.bruant/cheese-classification/train.py',
                            f'model.instance.layers={i}',
                            f"dataset/train_transform={t}",
                            f"model.instance.unfreeze_last_layer={unf}",
                            f"experiment_name={t}_{i}_unf_{unf}",
                            ])