import torch_fidelity
fid_statistics_file = 'assets/fid_stats/imagenet256_guided_diffusion.npz'
save_folder = 'examples/MaskGIT+ReCAP/output/'
metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=True,
        )
fid = metrics_dict['frechet_inception_distance']
inception_score = metrics_dict['inception_score_mean']
print(f'FID: {fid}, Inception Score: {inception_score}')