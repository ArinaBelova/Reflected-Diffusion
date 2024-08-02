from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torch
import wandb 
import utils
import os
import dataset
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_generated_dataset(path):
    # TODO: add transforms that may be needed in the future!
    dataset = ImageFolder(root=path, transform=transforms.ToTensor())
    length_dataset = len(dataset)
    return DataLoader(dataset, pin_memory=True), length_dataset

def run(cfg, work_dir):
    # TODO: for now we accept that our eval is only on 1 CUDA device
    value = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    wandb.init(
            project="reflected_diffusion",
        )
  
    # Get the datasets 
    gen_loader, gen_length = get_generated_dataset(cfg.generated_data_path)
    train_loader, test_loader = dataset.get_dataset(cfg, distributed=False, eval=True)
    
    if cfg.eval_methods.IS:
        logger.info('compute IS ...')
        inception = InceptionScore(normalize=True).to(device)

        for data, _ in gen_loader:
            data = data.to(device)
            # data = rgb_transform(data)
            inception.update(data)

        logger.info(f'IS data max is: {torch.max(data)}')
        logger.info(f'IS data min is: {torch.min(data)}')
        logger.info(f'IS data shape is: {data.shape}')

        is_mean, is_std = inception.compute()

        value['IS mean'] = is_mean
        value['IS std'] = is_std


        wandb.log({'IS mean': value['IS mean']})
        wandb.log({'IS std': value['IS std']})

        logger.info(value)
        logger.info('done computing IS.')

    if cfg.eval_methods.FID:
        fid_loder = test_loader if cfg.fid_on_test else train_loader
        # Subselect the length of gen_loader datasamples
        logger.info(f"FID loader length {len(fid_loder)}")
        ex_image = next(iter(fid_loder))
        logger.info(f"Truth Image type is {ex_image[0].dtype}")
        logger.info(f"Truth Image max value is {torch.max(ex_image[0])}")
        logger.info(f"Truth Image min value is {torch.min(ex_image[0])}")

        gen_ex_image = next(iter(gen_loader))
        logger.info(f"Gen Image type is {gen_ex_image[0].dtype}")
        logger.info(f"Gen Image max value is {torch.max(gen_ex_image[0])}")
        logger.info(f"Gen Image min value is {torch.min(gen_ex_image[0])}")


        # normalise=True as we have image in float32 type in [0,1]
        fid = FrechetInceptionDistance(feature=cfg.feature, normalize=True).to(device)

        logger.info(f"Loading {cfg.data} dataset data into the Inception module")
        
        # Uncomment if want to use a subset of dataloader
        # ground_truth_loader_iter = iter(fid_loder)
        # for _ in range(gen_length):
        #     data, _ = next(ground_truth_loader_iter)
        #     fid.update(data, real=True)

        for data, _ in fid_loder:
            data = data.to(device)
            # data = rgb_transform(inverse_scaler(data))
            fid.update(data, real=True)

        logger.info(f"Loading our generated {cfg.data} data into the Inception module")
        logger.info(f"Length of our generated dataset is {len(gen_loader)}")
        for data, _ in gen_loader:
            data = data.to(device)
            # data = rgb_transform(data)
            fid.update(data, real=False)

        logger.info('Computing FID ...')
        fid_score = fid.compute().detach().cpu().numpy()

        logger.info(f'FID gen data max, is: {torch.max(data)}')
        logger.info(f'FID gen data min is: {torch.min(data)}')
        logger.info(f'FID gen data shape is: {data.shape}')
        logger.info(f'FID gen data dtype is: {data.dtype}')

        value['FID'] = fid_score.item()

        wandb.log({'FID': value['FID']})

        logger.info(value)
        logger.info('done computing FID.')


from run_eval import run
@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(work_dir)

    run(cfg, work_dir)


if __name__ == "__main__":
    main()