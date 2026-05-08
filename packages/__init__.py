import os as _os
PACKAGE_DICT = {
    "EinsumNet":"simple_einet",
    "RatSPN":"RatSPN",
    "MultiBench":"MultiBench",
    "Radiate":"radiate_sdk",
    "probmetrics":"probmetrics",
    "AVRobustBench":"AVROBUSTBENCH",
    # local torchfsdd: actual module sits at packages/torchfsdd/lib/torchfsdd/
    # so we add packages/torchfsdd/lib (not packages/torchfsdd) to sys.path
    "torchfsdd": _os.path.join("torchfsdd", "lib"),
}