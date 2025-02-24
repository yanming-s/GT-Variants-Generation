import hydra
import logging


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg):
    if cfg.dataset.name == "zinc":
        from data.zinc.process_zinc import ZincDataset
        dataset = ZincDataset(cfg)
        train_smiles = dataset.get_train_smiles()
    
    elif cfg.dataset.name == "perov5":
        pass
    
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not supported.")
    
    logging.info(f"Dataset: {type(dataset).__name__}")


if __name__ == "__main__":
    main()
