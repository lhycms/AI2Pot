import os
from dpdata import LabeledSystem, MultiSystems
from ai2pot.utils import MsShifter


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe", "OUTCAR")
ReNbSTe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSTe", "OUTCAR")

ms = MultiSystems()
ls1 = LabeledSystem(ReNbSSe_OUTCAR_PATH)
ls2 = LabeledSystem(ReNbSTe_OUTCAR_PATH)
ms.append(ls1)
ms.append(ls2)
ms_shiter = MsShifter(ms)
print(ms_shiter)
