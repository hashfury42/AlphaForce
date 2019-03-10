#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")