from finetune import NeutDataModule

data_module = NeutDataModule()
val_dataloader = data_module.val_dataloader()

images, labels = next(iter(val_dataloader))
print(f'Images have shape {images.shape}, labels have shape {labels.shape}')