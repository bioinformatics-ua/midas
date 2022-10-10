from midas.utils import check_if_lib_is_installed

__version__="0.0.5"

# expose DataLoader class
if check_if_lib_is_installed("tensorflow"):
    from midas.dataloaders import DataLoader