from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1="/workspace/logs/openai-2025-10-02-22-55-51-324873/images_209000",
    input2="/workspace/consistency_models/datasets/cifar10/test",
    fid=True,
    isc=True,
    batch_size=64,
    device="cuda:0",
    samples_find_deep=True,
)

print("FID:", metrics["frechet_inception_distance"])
print("IS mean:", metrics["inception_score_mean"])
print("IS std:", metrics["inception_score_std"])
