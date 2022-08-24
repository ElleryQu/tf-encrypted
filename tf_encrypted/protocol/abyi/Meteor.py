# flake8: noqa
# pylint: disable=all
"""
Implementation of the ABYI framework.
"""
from __future__ import absolute_import

import abc
from asyncio import protocols
import sys
import random
from functools import partial, reduce
from math import ceil
from math import log2
from math import log
from turtle import filling
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from xmlrpc.client import Boolean

import numpy as np
import tensorflow as tf

from ...config import get_config
from ...operations import secure_random as crypto
from ...player import Player
from ...tensor import fixed64
from ...tensor import fixed64_ni
from ...tensor.boolfactory import bool_factory
from ...tensor.factory import AbstractConstant
from ...tensor.factory import AbstractFactory
from ...tensor.factory import AbstractTensor
from ...tensor.fixed import FixedpointConfig
from ...tensor.fixed import _validate_fixedpoint_config
from ...tensor.helpers import inverse
from ...tensor.native import native_factory
from ..protocol import Protocol
from ..protocol import memoize
from ..protocol import nodes
from ...utils import wrap_in_variables

from .abyi import ABYI, ABYIPrivateTensor, ABYIPublicTensor, ABYITensor, ARITHMETIC, BOOLEAN, TFEInputter, TF_NATIVE_TYPES, i64_factory, b_factory

TFEInputter = Callable[[], Union[List[tf.Tensor], tf.Tensor]]
TF_NATIVE_TYPES = [tf.bool, tf.int8, tf.int16, tf.int32, tf.int64]


class Meteor(ABYI):
    """Meteor framework."""
    def __init__(
        self,
        server_0=None,
        server_1=None,
        server_2=None,
        use_noninteractive_truncation=True,
    ):
        super(Meteor, self).__init__(self, server_0, server_1, server_2, use_noninteractive_truncation)


class MeteorTensor(ABYITensor):
    pass


class MeteorPublicTensor(ABYIPublicTensor):
    pass


class MeteorPrivateTensor(MeteorTensor):
    
    dispatch_id = "private"

    def __init__(self, prot, shares, is_scaled, share_type):
        assert len(shares) == 3 & len(shares[0]) == 2, "Not valid shares.{}(3),{}(2)".format(len(shares), len(shares[0]))

        Delta, delta = shares
        for D, d in zip(Delta, delta):
            assert isinstance(D, ABYIPublicTensor) \
                and all(
                    isinstance(s, ABYIPrivateTensor) for s in d
                ) , "Not valid shares."
            assert D.shape == shares[0][0].shape, "Shares have different shapes. Delta shape error"
            assert all(
                (s.shape == shares[0][0].shape) for s in d
            ), "Shares have different shapes. delta shape error"

        super(MeteorPrivateTensor, self).__init__(prot, is_scaled, share_type)
        self.shares = shares
    
    def __repr__(self) -> str:
        return "MeteorPrivateTensor(shape={}, share_type={})".format(
            self.shape, self.share_type
        )
    
    def reveal(self) -> MeteorPublicTensor:  
        return self.prot.reveal(self)


class MeteorPrivateVariable(MeteorPrivateTensor):
    """
  This class essentially represents a private value, however it additionally
  records the fact that the backing tensor was declared as a variable in
  order to allow treating it as a variable itself.
  """

    def __init__(self, prot, shares, is_scaled, share_type):
        super(MeteorPrivateVariable, self).__init__(prot, shares, is_scaled, share_type)
        self.shares = shares

        initializer_list = []
        Delta, delta = shares
        for D, d in zip(Delta, delta):
            initializer_list += D.initializer
            initializer_list += [s.initializer for s in d]
        self.initializer = tf.group(
            *initializer_list
        )

    def __repr__(self) -> str:
        return "MeteorPrivateVariable(shape={}, share_type={})".format(
            self.shape, self.share_type
        )
