from .threedpw import ThreeDPWDataset, ThreeDPWValDataset, ThreeDPWTestDataset
from torch.utils.data import DataLoader
from .Hi4D import Hi4DDataset, Hi4DTestDataset, Hi4DValDataset

def find_dataset_using_name(name):
    mapping = {
        "ThreeDPW": ThreeDPWDataset,
        "ThreeDPWVal": ThreeDPWValDataset,
        "ThreeDPWTest": ThreeDPWTestDataset,
        "Hi4D": Hi4DDataset,
        "Hi4DVal": Hi4DValDataset,
        "Hi4DTest": Hi4DTestDataset,
    }
    cls = mapping.get(name, None)
    if cls is None:
        raise ValueError(f"Fail to find dataset {name}") 
    return cls


def create_dataset(opt):
    dataset_cls = find_dataset_using_name(opt.dataset)
    dataset = dataset_cls(opt)
    if opt.worker == 0:
        return DataLoader(
            dataset,
            batch_size=opt.batch_size,
            drop_last=opt.drop_last,
            shuffle=opt.shuffle,
            num_workers=opt.worker,
            pin_memory=True,
            persistent_workers=False,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=opt.batch_size,
            drop_last=opt.drop_last,
            shuffle=opt.shuffle,
            num_workers=opt.worker,
            pin_memory=True,
            # both False because of memory issue
            persistent_workers=False,
        )

