# Copyright (C) 2025 Hanyu Liu
#
# This file is part of AI2Pot.
#
# AI2Pot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI2Pot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.

import lightning as L


class DescriptorNormCallback(L.Callback):
    def __init__(self):
        super(DescriptorNormCallback, self).__init__()


    def on_fit_start(self, 
                     trainer: L.Trainer, 
                     pl_module: L.LightningModule):
        'Calculate max-min for descriptors like NEP'
        pass


    def on_train_epoch_start(self,
                             trainer: L.Trainer,
                             pl_module: L.LightningModule):
        'Calculate running mean/std/(max-min)'
        pass
    