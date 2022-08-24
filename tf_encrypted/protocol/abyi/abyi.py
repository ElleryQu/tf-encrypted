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

from .Meteor import Meteor

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

TFEInputter = Callable[[], Union[List[tf.Tensor], tf.Tensor]]
TF_NATIVE_TYPES = [tf.bool, tf.int8, tf.int16, tf.int32, tf.int64]

_THISMODULE = sys.modules[__name__]

# ===== Share types =====
ARITHMETIC = 0
BOOLEAN = 1

# ===== Factory =====
i64_factory = native_factory(tf.int64)
i16_factory = native_factory(tf.int16)
b_factory = bool_factory()

# ===== Aided Protocol =====
aprot = Meteor()


class ABYI(Protocol):
    """ABYI framework."""

    def __init__(
        self,
        server_0=None,
        server_1=None,
        server_2=None,
        use_noninteractive_truncation=True,
    ):
        self._initializers = list()
        config = get_config()
        self.servers = [None, None, None]
        self.servers[0] = config.get_player(server_0 if server_0 else "server0")
        self.servers[1] = config.get_player(server_1 if server_1 else "server1")
        self.servers[2] = config.get_player(server_2 if server_2 else "server2")

        int_factory = i64_factory

        if use_noninteractive_truncation:
            fixedpoint_config = fixed64_ni
        else:
            fixedpoint_config = fixed64

        self.fixedpoint_config = fixedpoint_config
        self.int_factory = int_factory
        self.bool_factory = b_factory

        self.pairwise_keys, self.pairwise_nonces = self.setup_pairwise_randomness()
        self.b2a_keys_1, self.b2a_keys_2, self.b2a_nonce = self.setup_b2a_generator()

    @property
    def nbits(self):
        return self.int_factory.nbits

    def setup_pairwise_randomness(self):
        """
        Initial setup for pairwise randomness: Every two parties hold a shared key.
        """

        keys = [[None, None], [None, None], [None, None]]

        if crypto.supports_seeded_randomness():
            with tf.device(self.servers[0].device_name):
                seed_0 = crypto.secure_seed()
            with tf.device(self.servers[1].device_name):
                seed_1 = crypto.secure_seed()
            with tf.device(self.servers[2].device_name):
                seed_2 = crypto.secure_seed()
        else:
            # Shape and Type are kept consistent with the 'secure_seed' version
            with tf.device(self.servers[0].device_name):
                seed_0 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[1].device_name):
                seed_1 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[2].device_name):
                seed_2 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )

        # Replicated keys
        # NOTE: The following `with` contexts do NOT have any impact for the Python-only operations.
        #       We use them here only for indicating "which server has which seed".
        #       In other words, `keys[0][1] = seed_1` only stores the TF graph node `seed_1` in the
        #       Python list `keys`, but does NOT actually "send" `seed_1` to server 0, which only happens
        #       when a future TF operation on server 0 uses `keys[0][1]`.
        # The same NOTE applies to other places where we use Python list to store TF graph nodes in the
        # `with` context.
        with tf.device(self.servers[0].device_name):
            keys[0][0] = seed_0
            keys[0][1] = seed_1
        with tf.device(self.servers[1].device_name):
            keys[1][0] = seed_1
            keys[1][1] = seed_2
        with tf.device(self.servers[2].device_name):
            keys[2][0] = seed_2
            keys[2][1] = seed_0

        # nonces[0] for server 0 and 1, nonces[1] for server 1 and 2, nonces[2] for server 2 and 0
        nonces = np.array([0, 0, 0], dtype=np.int)

        return keys, nonces

    def setup_b2a_generator(self):
        """
    Initial setup for generating shares during the conversion
    from boolean sharing to arithmetic sharing
    """

        # Type 1: Server 0 and 1 hold three keys, while server 2 holds two
        b2a_keys_1 = [[None, None, None], [None, None, None], [None, None, None]]

        if crypto.supports_seeded_randomness():
            with tf.device(self.servers[0].device_name):
                seed_0 = crypto.secure_seed()
            with tf.device(self.servers[1].device_name):
                seed_1 = crypto.secure_seed()
            with tf.device(self.servers[2].device_name):
                seed_2 = crypto.secure_seed()
        else:
            # Shape and Type are kept consistent with the 'secure_seed' version
            with tf.device(self.servers[0].device_name):
                seed_0 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[1].device_name):
                seed_1 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[2].device_name):
                seed_2 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )

        with tf.device(self.servers[0].device_name):
            b2a_keys_1[0][0] = seed_0
            b2a_keys_1[0][1] = seed_1
            b2a_keys_1[0][2] = seed_2
        with tf.device(self.servers[1].device_name):
            b2a_keys_1[1][0] = seed_0
            b2a_keys_1[1][1] = seed_1
            b2a_keys_1[1][2] = seed_2
        with tf.device(self.servers[2].device_name):
            b2a_keys_1[2][0] = seed_0
            b2a_keys_1[2][2] = seed_2

        # Type 2: Server 1 and 2 hold three keys, while server 0 holds two
        b2a_keys_2 = [[None, None, None], [None, None, None], [None, None, None]]

        if crypto.supports_seeded_randomness():
            with tf.device(self.servers[0].device_name):
                seed_0 = crypto.secure_seed()
            with tf.device(self.servers[1].device_name):
                seed_1 = crypto.secure_seed()
            with tf.device(self.servers[2].device_name):
                seed_2 = crypto.secure_seed()
        else:
            # Shape and Type are kept consistent with the 'secure_seed' version
            with tf.device(self.servers[0].device_name):
                seed_0 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[1].device_name):
                seed_1 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[2].device_name):
                seed_2 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )

        with tf.device(self.servers[0].device_name):
            b2a_keys_2[0][0] = seed_0
            b2a_keys_2[0][1] = seed_1
        with tf.device(self.servers[1].device_name):
            b2a_keys_2[1][0] = seed_0
            b2a_keys_2[1][1] = seed_1
            b2a_keys_2[1][2] = seed_2
        with tf.device(self.servers[2].device_name):
            b2a_keys_2[2][0] = seed_0
            b2a_keys_2[2][1] = seed_1
            b2a_keys_2[2][2] = seed_2

        b2a_nonce = 0
        return b2a_keys_1, b2a_keys_2, b2a_nonce

    def define_constant(
        self,
        value: Union[np.ndarray, int, float],
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a constant to use in computation.

    .. code-block:: python

        x = prot.define_constant(np.array([1,2,3,4]), apply_scaling=False)

    :See: tf.constant

    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which tensor type to represent this value with.
    """
        assert isinstance(value, (np.ndarray, int, float))

        if isinstance(value, (int, float)):
            value = np.array([value])

        factory = factory or self.int_factory

        value = self._encode(value, apply_scaling)
        with tf.name_scope("constant{}".format("-" + name if name else "")):
            with tf.device(self.servers[0].device_name):
                x_on_0 = factory.constant(value)

            with tf.device(self.servers[1].device_name):
                x_on_1 = factory.constant(value)

            with tf.device(self.servers[2].device_name):
                x_on_2 = factory.constant(value)

        return ABYIConstant(self, [x_on_0, x_on_1, x_on_2], apply_scaling, share_type)

    def define_public_placeholder(
        self,
        shape,
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """Define a `public` placeholder to use in computation. This will be known
    to both parties.

    .. code-block:: python

        x = prot.define_public_placeholder(shape=(1024, 1024))

    :See: tf.placeholder

    :param List[int] shape: The shape of the placeholder.
    :param bool apply_scaling: Whether or not to scale the value.
    :param int share_type: ARITHMETIC or BOOLEAN.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which tensor type to represent this value
        with.
    """

        factory = factory or self.int_factory
        suffix = "-" + name if name else ""

        with tf.name_scope("public-placeholder{}".format(suffix)):

            with tf.device(self.servers[0].device_name):
                x_on_0 = factory.placeholder(shape)

            with tf.device(self.servers[1].device_name):
                x_on_1 = factory.placeholder(shape)

            with tf.device(self.servers[2].device_name):
                x_on_2 = factory.placeholder(shape)

        return ABYIPublicPlaceholder(self, (x_on_0, x_on_1, x_on_2), apply_scaling, share_type)

    def define_private_placeholder(
        self,
        shape,
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """Define a `private` placeholder to use in computation. This will only be
    known by the party that defines it.

    .. code-block:: python

        x = prot.define_private_placeholder(shape=(1024, 1024))

    :See: tf.placeholder

    :param List[int] shape: The shape of the placeholder.
    :param bool apply_scaling: Whether or not to scale the value.
    :param int share_type: ARITHMETIC or BOOLEAN.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which tensor type to represent this value
        with.
    """

        factory = factory or self.int_factory

        suffix = "-" + name if name else ""
        with tf.name_scope("private-placeholder{}".format(suffix)):

            with tf.device(self.servers[0].device_name):
                x0 = factory.placeholder(shape)

            with tf.device(self.servers[1].device_name):
                x1 = factory.placeholder(shape)

            with tf.device(self.servers[2].device_name):
                x2 = factory.placeholder(shape)

        return ABYIPrivatePlaceholder(self, ((x0, x1), (x1, x2), (x2, x0)), apply_scaling, share_type)

    def define_public_variable(
        self,
        initial_value,
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """Define a public variable.

    This is like defining a variable in tensorflow except it creates one that
    can be used by the protocol.

    For most cases, you can think of this as the same as the one from
    TensorFlow and you don't generally need to consider the difference.

    For those curious, under the hood, the major difference is that this
    function will pin your data to a specific device which will be used to
    optimize the graph later on.

    :see: tf.Variable

    :param Union[np.ndarray,tf.Tensor,PondPublicTensor] initial_value: The
        initial value.
    :param bool apply_scaling: Whether or not to scale the value.
    :param int share_type: ARITHMETIC or BOOLEAN.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which tensor type to represent this value
        with.
    """
        assert isinstance(
            initial_value, (np.ndarray, tf.Tensor, ABYIPublicTensor)
        ), type(initial_value)

        factory = factory or self.int_factory

        with tf.name_scope("public-var{}".format("-" + name if name else "")):

            if isinstance(initial_value, np.ndarray):
                v = self._encode(initial_value, apply_scaling)
                x0, x1, x2 = v, v, v

            elif isinstance(initial_value, tf.Tensor):
                inttype = factory.native_type
                # eq modify
                # v = self._encode(initial_value, apply_scaling, tf_int_type=inttype)
                v = self._encode(initial_value, apply_scaling)
                x0, x1, x2 = v, v, v

            elif isinstance(initial_value, ABYIPublicTensor):
                x0, x1, x2 = initial_value.unwrapped

            else:
                raise TypeError(
                    ("Don't know how to turn {} into a " "public variable").format(
                        type(initial_value)
                    )
                )

            with tf.device(self.servers[0].device_name):
                x_on_0 = factory.variable(x0)

            with tf.device(self.servers[1].device_name):
                x_on_1 = factory.variable(x1)

            with tf.device(self.servers[2].device_name):
                x_on_2 = factory.variable(x2)

        x = ABYIPublicVariable(self, (x_on_0, x_on_1, x_on_2), apply_scaling, share_type)
        return x

    def define_private_variable(
        self,
        initial_value,
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a private variable.

    This will take the passed value and construct shares that will be split up
    between those involved in the computation.

    For example, in a three party replicated sharing, this will split the value into
    three shares and transfer two shares to each party in a secure manner.

    :see tf.Variable

    :param Union[np.ndarray,tf.Tensor,ABYIPublicTensor] initial_value: The initial value.
    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which tensor type to represent this value with.
    """
        init_val_types = (np.ndarray, tf.Tensor, ABYIPrivateTensor)
        assert isinstance(initial_value, init_val_types), type(initial_value)

        factory = factory or self.int_factory
        suffix = "-" + name if name else ""

        with tf.name_scope("private-var{}".format(suffix)):

            if isinstance(initial_value, np.ndarray):
                initial_value = self._encode(initial_value, apply_scaling)
                v = factory.tensor(initial_value)
                shares = self._share(v, share_type=share_type)

            elif isinstance(initial_value, tf.Tensor):
                initial_value = self._encode(initial_value, apply_scaling)
                v = factory.tensor(initial_value)
                shares = self._share(v, share_type=share_type)

            elif isinstance(initial_value, ABYIPrivateTensor):
                shares = initial_value.unwrapped

            else:
                raise TypeError(
                    ("Don't know how to turn {} " "into private variable").format(
                        type(initial_value)
                    )
                )

            # The backing factory for the shares might have changed after the sharing step
            factory = shares[0][0].factory
            x = [[None, None], [None, None], [None, None]]
            with tf.device(self.servers[0].device_name):
                x[0][0] = factory.variable(shares[0][0])
                x[0][1] = factory.variable(shares[0][1])

            with tf.device(self.servers[1].device_name):
                x[1][0] = factory.variable(shares[1][0])
                x[1][1] = factory.variable(shares[1][1])

            with tf.device(self.servers[2].device_name):
                x[2][0] = factory.variable(shares[2][0])
                x[2][1] = factory.variable(shares[2][1])

        x = ABYIPrivateVariable(self, x, apply_scaling, share_type)
        return x

    def define_local_computation(
        self,
        player,
        computation_fn,
        arguments=None,
        apply_scaling=True,
        share_type=ARITHMETIC,
        name=None,
        factory=None,
    ):
        """
    Define a local computation that happens on plaintext tensors.

    :param player: Who performs the computation and gets to see the values in plaintext.
    :param apply_scaling: Whether or not to scale the outputs.
    :param name: Optional name to give to this node in the graph.
    :param factory: Backing tensor type to use for outputs.
    """

        factory = factory or self.int_factory

        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        def share_output(v: tf.Tensor):
            assert (
                v.shape.is_fully_defined()
            ), "Shape of return value '{}' on '{}' not fully defined".format(
                name if name else "", player.name,
            )

            v = self._encode(v, apply_scaling)
            w = factory.tensor(v)
            x = self._share_and_wrap(w, apply_scaling, share_type, player)

            return x

        def reconstruct_input(x, player):

            if isinstance(x, tf.Tensor):
                return x

            if isinstance(x, ABYIPublicTensor):
                w, _ = x.unwrapped
                v = self._decode(w, x.is_scaled)
                return v

            if isinstance(x, ABYIPrivateTensor):
                shares = x.unwrapped
                w = self._reconstruct(shares, player, share_type)
                v = self._decode(w, x.is_scaled)
                return v

            raise TypeError(
                ("Don't know how to process input argument " "of type {}").format(
                    type(x)
                )
            )

        with tf.name_scope(name if name else "local-computation"):

            with tf.device(player.device_name):
                if arguments is None:
                    inputs = []
                else:
                    if not isinstance(arguments, (list, tuple)):
                        arguments = [arguments]

                    inputs = [reconstruct_input(x, player) for x in arguments]

                outputs = computation_fn(*inputs)

                if isinstance(outputs, tf.Operation):
                    return outputs

                if isinstance(outputs, tf.Tensor):
                    return share_output(outputs)

                if isinstance(outputs, (list, tuple)):
                    return [share_output(output) for output in outputs]

                raise TypeError(
                    "Don't know how to handle results of "
                    "type {}".format(type(outputs))
                )

    def define_private_input(
        self,
        player,
        inputter_fn,
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a private input.

    This represents a `private` input owned by the specified player into the graph.

    :param Union[str,Player] player: Which player owns this input.
    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which backing type to use for this input
        (e.g. `int100` or `int64`).
    """
        suffix = "-" + name if name else ""

        return self.define_local_computation(
            player=player,
            computation_fn=inputter_fn,
            arguments=[],
            apply_scaling=apply_scaling,
            share_type=share_type,
            name="private-input{}".format(suffix),
            factory=factory,
        )

    def define_public_input(
        self,
        player: Union[str, Player],
        inputter_fn: TFEInputter,
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a public input.

    This represents a `public` input owned by the specified player into the
    graph.

    :param Union[str,Player] player: Which player owns this input.
    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    """
        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        factory = factory or self.int_factory
        suffix = "-" + name if name else ""

        def helper(v: tf.Tensor) -> "ABYIPublicTensor":
            assert (
                v.shape.is_fully_defined()
            ), "Shape of input '{}' on '{}' is not fully defined".format(
                name if name else "", player.name,
            )
            v = self._encode(v, apply_scaling)
            w = factory.tensor(v)
            return ABYIPublicTensor(self, [w, w, w], apply_scaling, share_type)

        with tf.name_scope("public-input{}".format(suffix)):

            with tf.device(player.device_name):

                inputs = inputter_fn()

                if isinstance(inputs, tf.Tensor):
                    # single input -> single output
                    v = inputs
                    return helper(v)

                if isinstance(inputs, (list, tuple)):
                    # multiple inputs -> multiple outputs
                    return [helper(v) for v in inputs]

                raise TypeError(
                    ("Don't know how to handle inputs of type {}").format(type(inputs))
                )

    def define_public_tensor(
        self,
        tensor: tf.Tensor,
        apply_scaling: bool = True,
        share_type=ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Convert a tf.Tensor to an ABYIPublicTensor.
    """
        assert isinstance(tensor, tf.Tensor)
        assert (
            tensor.shape.is_fully_defined()
        ), "Shape of input '{}' is not fully defined".format(name if name else "")

        factory = factory or self.int_factory

        with tf.name_scope("public-tensor"):
            tensor = self._encode(tensor, apply_scaling)
            w = factory.tensor(tensor)
            return ABYIPublicTensor(self, [w, w, w], apply_scaling, share_type)

    def define_output(
        self, player, arguments, outputter_fn, name=None,
    ):
        """
    Define an output for this graph.

    :param player: Which player this output will be sent to.
    """

        def result_wrapper(*args):
            op = outputter_fn(*args)
            # wrap in tf.group to prevent sending back any tensors (which might hence
            # be leaked)
            return tf.group(op)

        return self.define_local_computation(
            player=player,
            computation_fn=result_wrapper,
            arguments=arguments,
            name="output{}".format("-" + name if name else ""),
        )

    def _encode(
        self,
        rationals: Union[tf.Tensor, np.ndarray],
        apply_scaling: bool,
        factory=None,
    ) -> Union[tf.Tensor, np.ndarray]:
        """
    Encode tensor of rational numbers into tensor of ring elements. Output is
    of same type as input to allow function to be used for constants.
    """

        with tf.name_scope("encode"):

            # we first scale as needed
            if apply_scaling:
                scaled = rationals * self.fixedpoint_config.scaling_factor
            else:
                scaled = rationals

            # and then we round to integers

            if isinstance(scaled, np.ndarray):
                integers = scaled.astype(np.int64).astype(object)

            elif isinstance(scaled, tf.Tensor):
                factory = factory or self.int_factory
                tf_native_type = factory.native_type
                assert tf_native_type in TF_NATIVE_TYPES
                integers = tf.cast(scaled, dtype=tf_native_type)

            else:
                raise TypeError("Don't know how to encode {}".format(type(rationals)))

            assert type(rationals) == type(integers)
            return integers

    @memoize
    def _decode(self, elements: AbstractTensor, is_scaled: bool) -> tf.Tensor:
        """Decode tensor of ring elements into tensor of rational numbers."""

        with tf.name_scope("decode"):
            scaled = elements.to_native()
            if not is_scaled:
                return scaled
            return scaled / self.fixedpoint_config.scaling_factor

    def _share(self, secret: AbstractTensor, share_type: str, player=None):
        """Secret-share an AbstractTensor.

    Args:
      secret: `AbstractTensor`, the tensor to share.

    Returns:
      A pair of `AbstractTensor`, the shares.
    """

        with tf.name_scope("share"):
            if share_type == ARITHMETIC or share_type == BOOLEAN:
                share0 = secret.factory.sample_uniform(secret.shape)
                share1 = secret.factory.sample_uniform(secret.shape)
                if share_type == ARITHMETIC:
                    share2 = secret - share0 - share1
                elif share_type == BOOLEAN:
                    share2 = secret ^ share0 ^ share1
                # Replicated sharing
                shares = ((share0, share1), (share1, share2), (share2, share0))
                return shares

            else:
                raise NotImplementedError("Unknown share type.")

    def _share_and_wrap(
        self, secret: AbstractTensor, is_scaled: bool, share_type: str, player=None,
    ) -> "ABYIPrivateTensor":
        shares = self._share(secret, share_type, player)
        return ABYIPrivateTensor(self, shares, is_scaled, share_type)

    def _reconstruct(self, shares, player, share_type):
        """
    Reconstruct the plaintext value at a specified player.
    The shares might locate at three different players, so we need the 'player' argument
    in order to optimally use two local shares and one (probably) remote share to
    minimize communication.

    :param shares:
    :param player: Where to reconstruct
    :return:
    """

        def helper(s0, s1, s2):
            if share_type == ARITHMETIC:
                return s0 + s1 + s2
            elif share_type == BOOLEAN:
                return s0 ^ s1 ^ s2
            else:
                raise NotImplementedError(
                    "Only arithmetic and boolean sharings are supported."
                )

        with tf.name_scope("reconstruct"):
            if share_type == ARITHMETIC or share_type == BOOLEAN:
                if player == self.servers[0]:
                    return helper(shares[0][0], shares[0][1], shares[2][0])
                elif player == self.servers[1]:
                    return helper(shares[1][0], shares[1][1], shares[0][0])
                elif player == self.servers[2]:
                    return helper(shares[2][0], shares[2][1], shares[1][0])
                else:
                    # The player is not any of the three ABYI servers, so
                    # we just let each server give one share to this player
                    # in order to have a fair communication cost for each server
                    return helper(shares[0][0], shares[1][0], shares[2][0])

            else:
                raise NotImplementedError("Unknown share type.")

    def _gen_zero_sharing(self, shape, share_type=ARITHMETIC, factory=None):
        def helper(f0, f1):
            if share_type == ARITHMETIC:
                return f0 - f1
            elif share_type == BOOLEAN:
                return f0 ^ f1
            else:
                raise NotImplementedError(
                    "Only arithmetic and boolean sharings are supported."
                )

        factory = factory or self.int_factory
        with tf.name_scope("zero-sharing"):
            with tf.device(self.servers[0].device_name):
                f00 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][0] + self.pairwise_nonces[2]
                )  # yapf: disable
                f01 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][1] + self.pairwise_nonces[0]
                )  # yapf: disable
                a0 = helper(f00, f01)
            with tf.device(self.servers[1].device_name):
                f10 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][0] + self.pairwise_nonces[0]
                )  # yapf: disable
                f11 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][1] + self.pairwise_nonces[1]
                )  # yapf: disable
                a1 = helper(f10, f11)
            with tf.device(self.servers[2].device_name):
                f20 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][0] + self.pairwise_nonces[1]
                )  # yapf: disable
                f21 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][1] + self.pairwise_nonces[2]
                )  # yapf: disable
                a2 = helper(f20, f21)

        self.pairwise_nonces = self.pairwise_nonces + 1
        return a0, a1, a2

    def _gen_random_sharing(self, shape, share_type=ARITHMETIC, factory=None):

        r = [[None] * 2 for _ in range(3)]
        factory = factory or self.int_factory
        with tf.name_scope("random-sharing"):
            with tf.device(self.servers[0].device_name):
                r[0][0] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][0] + self.pairwise_nonces[2]
                )  # yapf: disable
                r[0][1] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][1] + self.pairwise_nonces[0]
                )  # yapf: disable
            with tf.device(self.servers[1].device_name):
                r[1][0] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][0] + self.pairwise_nonces[0]
                )  # yapf: disable
                r[1][1] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][1] + self.pairwise_nonces[1]
                )  # yapf: disable
            with tf.device(self.servers[2].device_name):
                r[2][0] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][0] + self.pairwise_nonces[1]
                )  # yapf: disable
                r[2][1] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][1] + self.pairwise_nonces[2]
                )  # yapf: disable

        self.pairwise_nonces = self.pairwise_nonces + 1

        return ABYIPrivateTensor(self, r, True, share_type)

    def _gen_b2a_sharing(self, shape, b2a_keys):
        shares = [[None, None], [None, None], [None, None]]
        with tf.device(self.servers[0].device_name):
            shares[0][0] = self.int_factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[0][0] + self.b2a_nonce
            )  # yapf: disable
            shares[0][1] = self.int_factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[0][1] + self.b2a_nonce
            )  # yapf: disable
            x_on_0 = None
            if b2a_keys[0][2] is not None:
                share_2 = self.int_factory.sample_seeded_uniform(
                    shape=shape, seed=b2a_keys[0][2] + self.b2a_nonce
                )  # yapf: disable
                x_on_0 = shares[0][0] ^ shares[0][1] ^ share_2

        with tf.device(self.servers[1].device_name):
            shares[1][0] = self.int_factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[1][1] + self.b2a_nonce
            )  # yapf: disable
            shares[1][1] = self.int_factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[1][2] + self.b2a_nonce
            )  # yapf: disable
            x_on_1 = None
            if b2a_keys[1][0] is not None:
                share_0 = self.int_factory.sample_seeded_uniform(
                    shape=shape, seed=b2a_keys[1][0] + self.b2a_nonce
                )  # yapf: disable
                x_on_1 = share_0 ^ shares[1][0] ^ shares[1][1]

        with tf.device(self.servers[2].device_name):
            shares[2][0] = self.int_factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[2][2] + self.b2a_nonce
            )  # yapf: disable
            shares[2][1] = self.int_factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[2][0] + self.b2a_nonce
            )  # yapf: disable
            x_on_2 = None
            if b2a_keys[2][1] is not None:
                share_1 = self.int_factory.sample_seeded_uniform(
                    shape=shape, seed=b2a_keys[2][1] + self.b2a_nonce
                )  # yapf: disable
                x_on_2 = share_1 ^ shares[2][0] ^ shares[2][1]

        self.b2a_nonce = self.b2a_nonce + 1
        return x_on_0, x_on_1, x_on_2, shares

    def _ot(
        self,
        sender,
        receiver,
        helper,
        m0,
        m1,
        c_on_receiver,
        c_on_helper,
        key_on_sender,
        key_on_helper,
        nonce,
    ):
        """
    Three-party OT protocol.

    'm0' and 'm1' are the two messages located on the sender.
    'c_on_receiver' and 'c_on_helper' should be the same choice bit, located on receiver and helper respectively.
    'key_on_sender' and 'key_on_helper' should be the same key, located on sender and helper respectively.
    'nonce' is a non-repeating ID for this call of the OT protocol.
    """
        assert m0.shape == m1.shape, "m0 shape {}, m1 shape {}".format(
            m0.shape, m1.shape
        )
        assert m0.factory == self.int_factory
        assert m1.factory == self.int_factory
        assert c_on_receiver.factory == self.bool_factory
        assert c_on_helper.factory == self.bool_factory

        with tf.name_scope("OT"):
            int_factory = self.int_factory
            with tf.device(sender.device_name):
                w_on_sender = int_factory.sample_seeded_uniform(
                    shape=[2] + m0.shape.as_list(), seed=key_on_sender + nonce
                )
                masked_m0 = m0 ^ w_on_sender[0]
                masked_m1 = m1 ^ w_on_sender[1]
            with tf.device(helper.device_name):
                w_on_helper = int_factory.sample_seeded_uniform(
                    shape=[2] + m0.shape.as_list(), seed=key_on_helper + nonce
                )
                w_c = int_factory.where(
                    c_on_helper.value, w_on_helper[1], w_on_helper[0], v2=False
                )
            with tf.device(receiver.device_name):
                masked_m_c = int_factory.where(
                    c_on_receiver.value, masked_m1, masked_m0, v2=False
                )
                m_c = masked_m_c ^ w_c

        return m_c

    @memoize
    def assign(self, variable: "ABYIPrivateVariable", value) -> tf.Operation:
        """See tf.assign."""
        assert isinstance(variable, ABYIPrivateVariable), type(variable)
        assert isinstance(value, ABYIPrivateTensor), type(value)
        assert (
            variable.is_scaled == value.is_scaled
        ), "Scaling must match: {}, {}".format(variable.is_scaled, value.is_scaled,)

        var_shares = variable.unwrapped
        val_shares = value.unwrapped

        with tf.name_scope("assign"):

            # Having this control_dependencies is important in order to avoid that
            # computationally-dependent shares are updated in different pace
            # (e.g., share0 is computed from share1, and we need to make sure that
            # share1 is NOT already updated).
            with tf.control_dependencies(
                [
                    val_shares[0][0].value,
                    val_shares[0][1].value,
                    val_shares[1][0].value,
                    val_shares[1][1].value,
                    val_shares[2][0].value,
                    val_shares[2][1].value,
                ]
            ):

                with tf.device(self.servers[0].device_name):
                    op00 = var_shares[0][0].assign_from_same(val_shares[0][0])
                    op01 = var_shares[0][1].assign_from_same(val_shares[0][1])

                with tf.device(self.servers[1].device_name):
                    op10 = var_shares[1][0].assign_from_same(val_shares[1][0])
                    op11 = var_shares[1][1].assign_from_same(val_shares[1][1])

                with tf.device(self.servers[2].device_name):
                    op20 = var_shares[2][0].assign_from_same(val_shares[2][0])
                    op21 = var_shares[2][1].assign_from_same(val_shares[2][1])

                op = tf.group(op00, op01, op10, op11, op20, op21)

        return op

    @memoize
    def add(self, x, y):
        """
    Adds two tensors `x` and `y`.

    :param ABYITensor x: The first operand.
    :param ABYITensor y: The second operand.
    """
        x, y = self.lift(x, y)
        return self.dispatch("add", x, y)

    def lift(self, x, y=None, share_type=ARITHMETIC):
        """
    Convenience method for working with mixed typed tensors in programs:
    combining any of the ABYI objects together with e.g. ints and floats
    will automatically lift the latter into ABYI objects.

    Lifting will guarantee the two outputs are both scaled or unscaled if at
    least one of them is lifted from int or float.
    """

        if y is None:

            if isinstance(x, (np.ndarray, int, float)):
                return self.define_constant(x, share_type=share_type)

            if isinstance(x, tf.Tensor):
                return self.define_public_tensor(x, share_type=share_type)

            if isinstance(x, ABYITensor):
                return x

            raise TypeError("Don't know how to lift {}".format(type(x)))

        if isinstance(x, (np.ndarray, int, float)):

            if isinstance(y, (np.ndarray, int, float)):
                x = self.define_constant(x, share_type=share_type)
                y = self.define_constant(y, share_type=share_type)
                return x, y

            if isinstance(y, tf.Tensor):
                x = self.define_constant(x, share_type=share_type)
                y = self.define_public_tensor(y, share_type=share_type)
                return x, y

            if isinstance(y, ABYITensor):
                x = self.define_constant(
                    x,
                    apply_scaling=y.is_scaled,
                    share_type=share_type,
                    factory=y.backing_dtype,
                )
                return x, y

            raise TypeError(
                ("Don't know how to lift " "{}, {}").format(type(x), type(y))
            )

        if isinstance(x, tf.Tensor):

            if isinstance(y, (np.ndarray, int, float)):
                x = self.define_public_tensor(x, share_type=share_type)
                y = self.define_constant(y, share_type=share_type)
                return x, y

            if isinstance(y, tf.Tensor):
                x = self.define_public_tensor(x, share_type=share_type)
                y = self.define_public_tensor(y, share_type=share_type)
                return x, y

            if isinstance(y, ABYITensor):
                x = self.define_public_tensor(
                    x,
                    apply_scaling=y.is_scaled,
                    share_type=share_type,
                    factory=y.backing_dtype,
                )
                return x, y

            raise TypeError(
                ("Don't know how to lift " "{}, {}").format(type(x), type(y))
            )

        if isinstance(x, ABYITensor):

            if isinstance(y, (np.ndarray, int, float)):
                y = self.define_constant(
                    y,
                    apply_scaling=x.is_scaled,
                    share_type=share_type,
                    factory=x.backing_dtype,
                )
                return x, y

            if isinstance(y, tf.Tensor):
                y = self.define_public_tensor(
                    y,
                    apply_scaling=x.is_scaled,
                    share_type=share_type,
                    factory=x.backing_dtype,
                )
                return x, y

            if isinstance(y, ABYITensor):
                return x, y

        raise TypeError(("Don't know how to lift " "{}, {}").format(type(x), type(y)))

    @memoize
    def add_n(self, tensors):
        # TODO(Morten) we could optimize by doing lazy reductions, potentially
        #              segmenting as needed
        return reduce(lambda x, y: x + y, tensors)

    @memoize
    def sub(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("sub", x, y)

    @memoize
    def negative(self, x):
        x = self.lift(x)
        return self.dispatch("negative", x)

    @memoize
    def mul(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("mul", x, y)

    @memoize
    def mul_trunc2(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("mul_trunc2", x, y)

    @memoize
    def div(self, x, y):
        """
    Performs a true division of `x` by `y` where `y` is public.

    No flooring is performing if `y` is an integer type as it is implicitly
    treated as a float.
    """

        assert isinstance(x, ABYITensor)

        if isinstance(y, float):
            y_inverse = 1.0 / y
        elif isinstance(y, int):
            y_inverse = 1.0 / float(y)
        elif isinstance(y, ABYIPublicTensor):
            y_inverse = 1.0 / y.decode()
        else:
            raise TypeError("Don't know how to divide by type {}".format(type(y)))

        return self.mul(x, y_inverse)

    @memoize
    def pow(self, x, p):
        x = self.lift(x)
        return self.dispatch("pow", x, p)

    @memoize
    def matmul(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("matmul", x, y)

    def gather_bit(self, x, even):
        assert x.share_type is BOOLEAN
        return self.dispatch("gather_bit", x, even)

    def xor_indices(self, x):
        assert x.share_type is BOOLEAN
        return self.dispatch("xor_indices", x)

    @memoize
    def transpose(self, x, perm=None):
        x = self.lift(x)
        return self.dispatch("transpose", x, perm)

    def indexer(self, x: "ABYITensor", slc) -> "ABYITensor":
        return self.dispatch("indexer", x, slc)

    def reshape(self, x: "ABYITensor", axe) -> "ABYITensor":
        return self.dispatch("reshape", x, axe)

    @memoize
    def concat(self, xs, axis):
        if all(isinstance(x, ABYIPublicTensor) for x in xs):
            return _concat_public(self, xs, axis=axis)

        if all(isinstance(x, ABYIPrivateTensor) for x in xs):
            return _concat_private(self, xs, axis=axis)

        raise TypeError("Don't know how to do a concat {}".format(type(xs)))

    @memoize
    def reduce_sum(self, x, axis=None, keepdims=False):
        x = self.lift(x)
        return self.dispatch("reduce_sum", x, axis=axis, keepdims=keepdims)

    @memoize
    def truncate(self, x: "ABYITensor"):
        return self.dispatch("truncate", x)

    @memoize
    def reveal(self, x):
        return self.dispatch("reveal", x)

    @memoize
    def B_xor(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_xor", x, y)

    @memoize
    def B_and(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_and", x, y)

    @memoize
    def B_or(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_or", x, y)

    @memoize
    def B_not(self, x):
        x = self.lift(x, share_type=BOOLEAN)
        return self.dispatch("B_not", x)

    # eq modify
    @memoize
    def B_ppa(self, x, y, n_bits=None, topology="sklansky"):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_ppa", x, y, n_bits, topology)

    @memoize
    def B_add(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_add", x, y)

    @memoize
    def B_sub(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_sub", x, y)

    @memoize
    def lshift(self, x, steps):
        return self.dispatch("lshift", x, steps)

    @memoize
    def rshift(self, x, steps):
        return self.dispatch("rshift", x, steps)

    @memoize
    def logical_rshift(self, x, steps):
        return self.dispatch("logical_rshift", x, steps)

    @memoize
    def A2B(self, x, nbits=None):
        return self.dispatch("A2B", x, nbits)

    @memoize
    def B2A(self, x, nbits=None):
        return self.dispatch("B2A", x, nbits)

    @memoize
    def mul_AB(self, x, y):
        """
    Callers should make sure y is boolean sharing whose backing TF native type is `tf.bool`.
    There is no automatic lifting for boolean sharing in the mixed-protocol multiplication.
    """
        x = self.lift(x)
        return self.dispatch("mul_AB", x, y)

    @memoize
    def bit_extract(self, x, i):
        if x.share_type == BOOLEAN or x.share_type == ARITHMETIC:
            return self.dispatch("bit_extract", x, i)
        else:
            raise ValueError("unsupported share type: {}".format(x.share_type))

    @memoize
    def msb(self, x):
        return self.bit_extract(x, self.nbits - 1)

    @memoize
    def polynomial(self, x, coeffs):
        x = self.lift(x)
        return self.dispatch("polynomial", x, coeffs)

    @memoize
    def polynomial_piecewise(self, x, c, coeffs):
        return self.dispatch("polynomial_piecewise", x, c, coeffs)

    @memoize
    def sigmoid(self, x, approx_type="piecewise_linear"):
        return self.dispatch("sigmoid", x, approx_type)

    @memoize
    def gather(self, x, indices, axis):
        raise NotImplementedError("Unsupported share type: {}".format(x.share_type))

    @memoize
    def stack(self, xs, axis):
        raise TypeError("Don't know how to do a stack {}".format(type(xs)))

    def write(self, x, filename_prefix):
        if not isinstance(x, ABYIPrivateTensor):
            raise TypeError("Only support writing ABYIPrivateTensor to disk.")
        return self.dispatch("write", x, filename_prefix)

    def read(self, filename_prefix, batch_size, n_columns):
        return self.dispatch("read", filename_prefix, batch_size, n_columns)

    def iterate(
        self,
        tensor: "ABYIPrivateTensor",
        batch_size: int,
        repeat=True,
        shuffle=True,
        seed: int = None,
    ):
        if not isinstance(tensor, ABYIPrivateTensor):
            raise TypeError("Only support iterating ABYIPrivateTensor.")
        return self.dispatch("iterate", tensor, batch_size, repeat, shuffle, seed)

    def blinded_shuffle(self, tensor: "ABYIPrivateTensor"):
        """
    Shuffle the rows of the given tenosr privately.
    After the shuffle, none of the share holder could know the exact shuffle order.
    """
        if not isinstance(tensor, ABYIPrivateTensor):
            raise TypeError(
                (
                    "Only support blindly shuffle ABYIPrivateTensor. "
                    "For public tensor, use the shuffle() method"
                )
            )
        return self.dispatch("blinded_shuffle", tensor)

    def dispatch(self, base_name, *args, container=None, **kwargs):
        """
    Finds the correct protocol logicto perform based on the dispatch_id
    attribute of the input tensors in args.
    """
        suffix = "_".join(
            [arg.dispatch_id for arg in args if hasattr(arg, "dispatch_id")]
        )
        func_name = "_{}_{}".format(base_name, suffix)

        if container is None:
            container = _THISMODULE

        func = getattr(container, func_name, None)
        if func is not None:
            return func(self, *args, **kwargs)  # pylint: disable=not-callable
        raise TypeError(
            ("Don't know how to {}: {}").format(base_name, [type(arg) for arg in args])
        )

    def cache(self, xs):
        """
    Wraps all input tensors, including private and masked, in variables so
    that computation and masking of these can be reused between different
    runs.

    For private predictions this may be used to avoid re-masking model
    weights between each run, thereby saving on communication.
    For private training this may be used to avoid re-masked the traning
    data between each iteration, again saving on communication.
    """

        if isinstance(xs, (list, tuple)):
            # apply recursively
            # [cache(x) = [op, x]]; updaters = iter(op), cached = iter(x).
            updaters, cached = zip(*[self.cache(x) for x in xs])
            return tf.group(*updaters), cached

        # base case
        node_key = ("cache", xs)
        cached = nodes.get(node_key, None)

        if cached is not None:
            return cached

        dispatch = {
            ABYIPublicTensor: _cache_public,
            ABYIPrivateTensor: _cache_private
        }
        func = dispatch.get(_type(xs), None)
        if func is None:
            raise TypeError("Don't know how to cache {}".format(type(xs)))

        # updater: for share in x in xs, define a variable v_share_x = share.
        # cached: [v_share_x], v_share_x is initialized to 0. 
        updater, cached = func(self, xs)
        nodes[node_key] = cached

        return updater, cached

    @memoize
    def relu(self, x: "ABYITensor", **kwargs):  # pylint: disable=unused-argument
        """A Chebyshev polynomial approximation of the ReLU function."""
        assert isinstance(x, ABYITensor), type(x)
        return self.dispatch("relu", x)

    @memoize
    def A2Bi(self, x, nbits=None):
        return self.dispatch("A2Bi", x, nbits)
    
    @memoize
    def B_ppai(self, x, y, n_bits=None):
        '''
        Compute PPA circuit: [[x]]^B + [[y]]^B.
        '''
        try:
            Delta1, delta1 = x
            Delta2, delta2 = y
            assert isinstance(Delta1, ABYIPublicTensor) & isinstance(Delta2, ABYIPublicTensor), \
                "Input isn't a valid mask sharing. Type(x[0]) or Type(y[0]) isn't ABYIConstant."
            assert isinstance(delta1, ABYIPrivateTensor) & isinstance(delta2, ABYIPrivateTensor), \
                "Input isn't a valid mask sharing. Type(x[1]) or Type(y[1]) isn't ABYIPrivateTensor."
        except:
            raise Exception("Input x {}, y {} isn't a valid mask sharing. (ABYIConstant, ABYIPrivateTensor)".format(x,y))
        return _B_ppai_private_private(self, x, y, n_bits)


#
# Classes representing the base values in the ABYI protocol.
#


class ABYITensor(abc.ABC):
    """
  This class functions mostly as a convenient way of exposing operations
  directly on the various tensor objects, ie allowing one to write `x + y`
  instead of `prot.add(x, y)`. Since this functionality is shared among all
  tensors we put it in this superclass.

  This class should never be instantiated on its own.
  Instead you should use your chosen protocols factory methods::

      x = prot.define_private_input(tf.constant(np.array([1,2,3,4])))
      y = prot.define_public_input(tf.constant(np.array([4,5,6,7])))

      z = x + y

      with config.Session() as sess:
          answer = z.reveal().eval(sess)

          print(answer) # => [5, 7, 9, 11]
    """

    def __init__(self, prot, is_scaled, share_type):
        self.prot = prot
        self.is_scaled = is_scaled
        self.share_type = share_type

    @property
    @abc.abstractmethod
    def shape(self) -> List[int]:
        """
    :rtype: List[int]
    :returns: The shape of this tensor.
    """
        pass

    @property
    @abc.abstractmethod
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        pass

    def add(self, other):
        """
    Add `other` to this ABYITensor.  This can be another tensor with the same
    backing or a primitive.

    This function returns a new ABYITensor and does not modify this one.

    :param ABYITensor other: a or primitive (e.g. a float)
    :return: A new ABYITensor with `other` added.
    :rtype: ABYITensor
    """
        if self.share_type == ARITHMETIC:
            return self.prot.add(self, other)
        else:
            raise ValueError(
                "unsupported share type for add: {}".format(self.share_type)
            )

    def __add__(self, other):
        """
    See :meth:`~tf_encrypted.protocol.ABYI.ABYITensor.add`
    """
        return self.add(other)

    def __radd__(self, other):
        return self + other

    def reduce_sum(self, axis=None, keepdims=False):
        """
    Like :meth:`tensorflow.reduce_sum`

    :param int axis:  The axis to reduce along
    :param bool keepdims: If true, retains reduced dimensions with length 1.
    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.prot.reduce_sum(self, axis, keepdims)

    def sum(self, axis=None, keepdims=False):
        """
    See :meth:`ABYITensor.reduce_sum`
    """
        return self.reduce_sum(axis, keepdims)

    def sub(self, other):
        """
    Subtract `other` from this tensor.

    :param ABYITensor other: to subtract
    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        if self.share_type == ARITHMETIC:
            return self.prot.sub(self, other)
        else:
            raise ValueError(
                "unsupported share type for sub: {}".format(self.share_type)
            )

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        if self.share_type == ARITHMETIC:
            return self.prot.sub(other, self)
        else:
            raise ValueError(
                "unsupported share type for sub: {}".format(self.share_type)
            )

    def mul(self, other):
        """
    Multiply this tensor with `other`

    :param ABYITensor other: to multiply
    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.prot.mul(self, other)

    def __mul__(self, other):
        return self.prot.mul(self, other)

    def __rmul__(self, other):
        return self.prot.mul(other, self)

    def __truediv__(self, other):
        return self.prot.div(self, other)

    def __mod__(self, other):
        return self.prot.mod(self, other)

    def __pow__(self, p):
        return self.prot.pow(self, p)

    def square(self):
        """
    Square this tensor.

    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.prot.square(self)

    def matmul(self, other):
        """
    MatMul this tensor with `other`.  This will perform matrix multiplication,
    rather than elementwise like
    :meth:`~tf_encrypted.protocol.ABYI.ABYITensor.mul`

    :param ABYITensor other: to mul
    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.prot.matmul(self, other)

    def dot(self, other):
        """
    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.matmul(other)

    def __getitem__(self, slc):
        return self.prot.indexer(self, slc)

    def transpose(self, perm=None):
        """
    Transpose this tensor.

    See :meth:`tensorflow.transpose`

    :param List[int]: A permutation of the dimensions of this tensor.

    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.prot.transpose(self, perm)

    def truncate(self):
        """
    Truncate this tensor.

    `TODO`

    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.prot.truncate(self)

    def expand_dims(self, axis=None):
        """
    :See: tf.expand_dims

    :return: A new ABYITensor
    :rtype: ABYITensor
    """
        return self.prot.expand_dims(self, axis=axis)

    def reshape(self, shape: List[int]) -> "ABYITensor":
        """
    :See: tf.reshape

    :param List[int] shape: The new shape of the tensor.
    :rtype: ABYITensor
    :returns: A new tensor with the contents of this tensor, but with the new
        specified shape.
    """
        return self.prot.reshape(self, shape)

    def __neg__(self):
        return self.prot.negative(self)

    def negative(self) -> "ABYITensor":
        """
    :See: tf.negative

    :rtype: ABYITensor
    :returns: A new tensor with numerical negative value element-wise computed.
    """
        return self.prot.negative(self)

    def reduce_max(self, axis: int) -> "ABYITensor":
        """
    :See: tf.reduce_max

    :param int axis: The axis to take the max along
    :rtype: ABYITensor
    :returns: A new ABYI tensor with the max value from each axis.
    """
        return self.prot.reduce_max(self, axis)

    def bitwise_xor(self, other):
        if self.share_type == BOOLEAN:
            return self.prot.B_xor(self, other)
        else:
            raise ValueError(
                "Unsupported share type for xor: {}".format(self.share_type)
            )

    def __xor__(self, other):
        return self.bitwise_xor(other)

    def bitwise_and(self, other):
        if self.share_type == BOOLEAN:
            return self.prot.B_and(self, other)
        else:
            raise ValueError(
                "unsupported share type for and: {}".format(self.share_type)
            )

    def __and__(self, other):
        return self.bitwise_and(other)

    def bitwise_or(self, other):
        if self.share_type == BOOLEAN:
            return self.prot.B_or(self, other)
        else:
            raise ValueError(
                "unsupported share type for and: {}".format(self.share_type)
            )

    def __or__(self, other):
        return self.bitwise_or(other)

    def invert(self):
        if self.share_type == BOOLEAN:
            return self.prot.B_not(self)
        else:
            raise ValueError(
                "unsupported share type for and: {}".format(self.share_type)
            )

    def __invert__(self):
        return self.invert()

    def __lshift__(self, steps):
        return self.prot.lshift(self, steps)

    def lshift(self, steps):
        return self.prot.lshift(self, steps)

    def __rshift__(self, steps):
        return self.prot.rshift(self, steps)

    def rshift(self, steps):
        return self.prot.rshift(self, steps)

    def arith_rshift(self, steps):
        return self.rshift(steps)

    def logical_rshift(self, steps):
        return self.prot.logical_rshift(self, steps)

    def write(self, filename_prefix):
        return self.prot.write(self, filename_prefix)


class ABYIPublicTensor(ABYITensor):
    """
  This class represents a public tensor, known by at least by the three servers
  but potentially known by more. Although there is only a single value we
  replicate it on both servers to avoid sending it from one to the other
  in the operations where it's needed by both (eg multiplication).
  """

    dispatch_id = "public"

    def __init__(
        self, prot: ABYI, values: List[AbstractTensor], is_scaled: bool, share_type
    ) -> None:
        assert all(isinstance(v, AbstractTensor) for v in values)
        assert all((v.shape == values[0].shape) for v in values)

        super(ABYIPublicTensor, self).__init__(prot, is_scaled, share_type)
        self.values = values

    def __repr__(self) -> str:
        return "ABYIPublicTensor(shape={}, share_type={})".format(
            self.shape, self.share_type
        )

    @property
    def shape(self) -> List[int]:
        return self.values[0].shape

    @property
    def backing_dtype(self):
        return self.values[0].factory

    @property
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        """
    Unwrap the tensor.

    This will return the value for each of the parties that collectively own
    the tensor.

    In most cases, this will be the same value on each device.

    .. code-block:: python

        x_0, y_0, z_0 = tensor.unwrapped
        # x_0 == 10 with the value pinned to player_0's device.
        # y_0 == 10 with the value pinned to player_1's device.
        # z_0 == 10 with the value pinned to player_2's device.

    In most cases you will want to work on this data on the specified device.

    .. code-block:: python

        x_0, y_0, z_0= tensor.unwrapped

        with tf.device(prot.player_0.device_name):
            # act on x_0

        with tf.device(prot.player_1.device_name):
            # act on y_0

        with tf.device(prot.player_2.device_name):
            # act on z_0

    In most cases you will not need to use this method.  All funtions
    will hide this functionality for you (e.g. `add`, `mul`, etc).
    """
        return self.values

    def decode(self) -> Union[np.ndarray, tf.Tensor]:
        return self.prot._decode(
            self.values[0], self.is_scaled
        )  # pylint: disable=protected-access

    def to_native(self):
        return self.decode()


class ABYIConstant(ABYIPublicTensor):
    """
  This class essentially represents a public value, however it additionally
  records the fact that the underlying value was declared as a constant.
  """

    def __init__(self, prot, constants, is_scaled, share_type):
        assert all(isinstance(c, AbstractConstant) for c in constants)
        assert all((c.shape == constants[0].shape) for c in constants)

        super(ABYIConstant, self).__init__(prot, constants, is_scaled, share_type)
        self.constants = constants

    def __repr__(self) -> str:
        return "ABYIConstant(shape={}, share_type={})".format(
            self.shape, self.share_type
        )


class ABYIPrivateTensor(ABYITensor):
    """
  This class represents a private value that may be unknown to everyone.
  """

    dispatch_id = "private"

    def __init__(self, prot, shares, is_scaled, share_type):
        assert len(shares) == 3
        assert all(
            (ss.shape == shares[0][0].shape) for s in shares for ss in s
        ), "Shares have different shapes."

        super(ABYIPrivateTensor, self).__init__(prot, is_scaled, share_type)
        self.shares = shares

    def __repr__(self) -> str:
        return "ABYIPrivateTensor(shape={}, share_type={})".format(
            self.shape, self.share_type
        )

    @property
    def shape(self) -> List[int]:
        return self.shares[0][0].shape

    @property
    def backing_dtype(self):
        return self.shares[0][0].factory

    @property
    def unwrapped(self):
        return self.shares

    def reveal(self) -> ABYIPublicTensor:
        return self.prot.reveal(self)


class ABYIPublicPlaceholder(ABYIPublicTensor):
    """
  This class essentially represents a public value, however it additionally
  records the fact that the backing tensor was declared as a placeholder in
  order to allow treating it as a placeholder itself.
  """

    def __init__(self, prot, shares, is_scaled, share_type):
        super(ABYIPublicPlaceholder, self).__init__(prot, shares, is_scaled, share_type)
        self.shares = shares

    def __repr__(self) -> str:
        return "ABYIPublicPlaceholder(shape={}, share_type={})".format(
            self.shape, self.share_type
        )

    def feed(self, value):
        """
    Feed `value` to placeholder
    """
        enc = self.prot._encode(value, self.is_scaled)
        feed0 = dict()
        for share_i in self.shares:
            feed0 |= share_i.feed(enc)
        return {**feed0}


class ABYIPrivatePlaceholder(ABYIPrivateTensor):

    def __init__(self, prot, shares, is_scaled, share_type):
        super(ABYIPrivatePlaceholder, self).__init__(prot, shares, is_scaled, share_type)
        self.shares = shares

    def __repr__(self) -> str:
        return "ABYIPrivatePlaceholder(shape={}, share_type={})".format(
            self.shape, self.share_type
        )

    def feed(self, value):
        """
    Feed `value` to placeholder
    """
        assert isinstance(value, np.ndarray), type(value)
        enc = self.prot._encode(value, self.is_scaled)
        assert isinstance(enc, np.ndarray)

        # TODO(Ellery)
        #
        # Because what have been writen in tf_encrypted.protocol.pond.PondPrivatePlaceholder,
        # which says they want to keep feeding op been done outside the TF graph,
        # we will use the same way to construct shares.
        if self.share_type == ARITHMETIC or self.share_type == BOOLEAN:
            shape = self.shape
            minval = self.backing_dtype.min
            maxval = self.backing_dtype.max
            # TODO(Morten) not using secure randomness here; reconsider after TF2
            x0 = np.array(
                [random.randrange(minval, maxval) for _ in range(np.product(shape))]
            ).reshape(shape)
            x1 = np.array(
                [random.randrange(minval, maxval) for _ in range(np.product(shape))]
            ).reshape(shape)

            if self.share_type == ARITHMETIC:
                x2 = enc - x0 - x1
            elif self.share_type == BOOLEAN:
                x2 = enc ^ x0 ^ x1
        else:
            raise NotImplementedError("Unknown share type.")
        
        feed0 = dict()
        feeded_shares = ((x0, x1), (x1, x2), (x2, x0))
        for share_i, feeded_i in zip(self.shares, feeded_shares):
            for x_i, f_i in zip(share_i, feeded_i):
                feed0 |= x_i.feed(f_i)
        return {**feed0}


class ABYIPublicVariable(ABYIPublicTensor):
    """
  This class essentially represents a public value, however it additionally
  records the fact that the backing tensor was declared as a variable in
  order to allow treating it as a variable itself.
  """
    def __init__(self, prot, shares, is_scaled, share_type):
        super(ABYIPublicVariable, self).__init__(prot, shares, is_scaled, share_type)
        self.shares = shares
        # eq modify
        # self.initializer = tf.group(
        #     *[var.initializer for share in shares for var in share]
        # )
        self.initializer = tf.group(
            *[share.initializer for share in shares]
        )

    def __repr__(self) -> str:
        return "ABYIPublicVariable(shape={}, share_type={})".format(
            self.shape, self.share_type
        )


class ABYIPrivateVariable(ABYIPrivateTensor):
    """
  This class essentially represents a private value, however it additionally
  records the fact that the backing tensor was declared as a variable in
  order to allow treating it as a variable itself.
  """

    def __init__(self, prot, shares, is_scaled, share_type):
        super(ABYIPrivateVariable, self).__init__(prot, shares, is_scaled, share_type)
        self.shares = shares
        self.initializer = tf.group(
            *[var.initializer for share in shares for var in share]
        )

    def __repr__(self) -> str:
        return "ABYIPrivateVariable(shape={}, share_type={})".format(
            self.shape, self.share_type
        )


class ABYICachedPublicTensor(ABYIPublicTensor):
    """A PondPublicTensor that has been cached for reuse."""

    def __init__(self, prot, shares, is_scaled, updater, share_type=ARITHMETIC):
        for share in shares:
            assert isinstance(share, AbstractTensor), "Cached var not tensor."
        assert isinstance(updater, tf.Operation), type(updater)

        super(ABYICachedPublicTensor, self).__init__(
            prot, shares, is_scaled, share_type
        )
        self.updater = updater

    def __repr__(self) -> str:
        return "ABYICachedPublicTensor(shape={})".format(self.shape)


class ABYICachedPrivateTensor(ABYIPrivateTensor):
    """A PondPrivateTensor that has been cached for reuse."""

    def __init__(self, prot, shares, is_scaled, updater, share_type=ARITHMETIC):
        for share in shares:
            for s in share:
                assert isinstance(s, AbstractTensor), "Cached var not tensor."
        assert isinstance(updater, tf.Operation), type(updater)

        super(ABYICachedPrivateTensor, self).__init__(prot, shares, is_scaled, share_type)
        self.updater = updater

    def __repr__(self) -> str:
        return "ABYICachedPrivateTensor(shape={})".format(self.shape)


#
# reveal helpers
#


def _reveal_private(prot, x):
    assert isinstance(x, ABYIPrivateTensor), type(x)

    with tf.name_scope("reveal"):

        shares = x.unwrapped

        with tf.device(prot.servers[0].device_name):
            z_on_0 = prot._reconstruct(shares, prot.servers[0], x.share_type)

        with tf.device(prot.servers[1].device_name):
            z_on_1 = prot._reconstruct(shares, prot.servers[1], x.share_type)

        with tf.device(prot.servers[2].device_name):
            z_on_2 = prot._reconstruct(shares, prot.servers[2], x.share_type)

    return ABYIPublicTensor(prot, [z_on_0, z_on_1, z_on_2], x.is_scaled, x.share_type)


#
# add helpers
#


def _add_private_private(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("add"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x.shares[i][0] + y.shares[i][0]
                z[i][1] = x.shares[i][1] + y.shares[i][1]

    return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


def _add_private_public(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, (
        "Cannot mix different encodings: " "{} {}"
    ).format(x.is_scaled, y.is_scaled)

    shares = x.unwrapped
    y_on_0, _, y_on_2 = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("add"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] + y_on_0
            z[0][1] = shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0]
            z[1][1] = shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0]
            z[2][1] = shares[2][1] + y_on_2
    return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


def _add_public_private(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)
    assert x.is_scaled == y.is_scaled, (
        "Cannot mix different encodings: " "{} {}"
    ).format(x.is_scaled, y.is_scaled)

    x_on_0, _, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("add"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] + x_on_0
            z[0][1] = shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0]
            z[1][1] = shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0]
            z[2][1] = shares[2][1] + x_on_2

    return ABYIPrivateTensor(prot, z, x.is_scaled, y.share_type)


def _add_public_public(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot add tensors with different scales"

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    z = [None] * 3
    with tf.name_scope("add"):
        for i in range(3):
            z[i] = x_shares[i] + y_shares[i]

    return ABYIPublicTensor(prot, z, x.is_scaled, x.share_type)


#
# sub helpers
#


def _sub_private_private(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)
    assert x.is_scaled == y.is_scaled

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("sub"):
        x_shares = x.unwrapped
        y_shares = y.unwrapped
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0] - y_shares[i][0]
                z[i][1] = x_shares[i][1] - y_shares[i][1]

    return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


def _sub_private_public(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)
    assert x.is_scaled == y.is_scaled

    shares = x.unwrapped
    y_on_0, _, y_on_2 = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("sub"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] - y_on_0
            z[0][1] = shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0]
            z[1][1] = shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0]
            z[2][1] = shares[2][1] - y_on_2

    return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


def _sub_public_private(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    x_on_0, _, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("sub"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_on_0 - shares[0][0]
            z[0][1] = -shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = -shares[1][0]
            z[1][1] = -shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = -shares[2][0]
            z[2][1] = x_on_2 - shares[2][1]

    return ABYIPrivateTensor(prot, z, x.is_scaled, y.share_type)


#
# negative helpers
#


def _negative_private(prot, x):
    assert isinstance(x, ABYIPrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("negative"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = -x_shares[i][0]
                z[i][1] = -x_shares[i][1]

        z = ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _negative_public(prot, x):
    assert isinstance(x, ABYIPublicTensor), type(x)

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("negative"):
        with tf.device(prot.servers[0].device_name):
            x_on_0_neg = -x_on_0
        with tf.device(prot.servers[1].device_name):
            x_on_1_neg = -x_on_1
        with tf.device(prot.servers[2].device_name):
            x_on_2_neg = -x_on_2
        x_neg = ABYIPublicTensor(
            prot, [x_on_0_neg, x_on_1_neg, x_on_2_neg], x.is_scaled, x.share_type
        )
    return x_neg


#
# mul helpers
#


def _mul_public_private(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("mul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] * x_on_0
            z[0][1] = shares[0][1] * x_on_0

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0] * x_on_1
            z[1][1] = shares[1][1] * x_on_1

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0] * x_on_2
            z[2][1] = shares[2][1] * x_on_2

        z = ABYIPrivateTensor(prot, z, x.is_scaled or y.is_scaled, y.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_private_public(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)

    shares = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("mul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] * y_on_0
            z[0][1] = shares[0][1] * y_on_0

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0] * y_on_1
            z[1][1] = shares[1][1] * y_on_1

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0] * y_on_2
            z[2][1] = shares[2][1] * y_on_2

        z = ABYIPrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_private_private(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("mul"):
        a0, a1, a2 = prot._gen_zero_sharing(x.shape)
        with tf.device(prot.servers[0].device_name):
            z0 = (
                x_shares[0][0] * y_shares[0][0]
                + x_shares[0][0] * y_shares[0][1]
                + x_shares[0][1] * y_shares[0][0]
                + a0
            )

        with tf.device(prot.servers[1].device_name):
            z1 = (
                x_shares[1][0] * y_shares[1][0]
                + x_shares[1][0] * y_shares[1][1]
                + x_shares[1][1] * y_shares[1][0]
                + a1
            )

        with tf.device(prot.servers[2].device_name):
            z2 = (
                x_shares[2][0] * y_shares[2][0]
                + x_shares[2][0] * y_shares[2][1]
                + x_shares[2][1] * y_shares[2][0]
                + a2
            )
        # Re-sharing
        with tf.device(prot.servers[0].device_name):
            z[0][0] = z0
            z[0][1] = z1
        with tf.device(prot.servers[1].device_name):
            z[1][0] = z1
            z[1][1] = z2
        with tf.device(prot.servers[2].device_name):
            z[2][0] = z2
            z[2][1] = z0

        z = ABYIPrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_public_public(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)

    x_on_i = x.unwrapped
    y_on_i = y.unwrapped

    z = [None, None, None]
    with tf.name_scope("mul"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i] = x_on_i[i] * y_on_i[i]

        z = ABYIPublicTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z  # unrealize
        return z


def _mul_trunc2_private_private(prot, x, y):
    """
  Multiplication with the Trunc2 protocol in the ABYI paper.
  This is more efficient (in terms of communication rounds)
  than `mul` in the onlline phase only when pre-computation
  is left out of consideration.
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    # If there will not be any truncation, then just call the simple multiplication protocol.
    if not (x.is_scaled and y.is_scaled):
        return _mul_private_private(prot, x, y)

    x_shares = x.unwrapped
    y_shares = y.unwrapped
    shape = x_shares[0][0].shape
    amount = prot.fixedpoint_config.precision_fractional

    with tf.name_scope("mul_trunc2"):
        # Step 1: Generate a Random Truncation Pair
        # If TF is smart enough, this part is supposed to be pre-computation.
        r = prot._gen_random_sharing(shape, share_type=BOOLEAN)
        r_trunc = r.arith_rshift(amount)
        r = prot.B2A(r)
        r_trunc = prot.B2A(r_trunc)

        # Step 2: Compute 3-out-of-3 sharing of (x*y - r)
        a0, a1, a2 = prot._gen_zero_sharing(x.shape)
        with tf.device(prot.servers[0].device_name):
            z0 = (
                x_shares[0][0] * y_shares[0][0]
                + x_shares[0][0] * y_shares[0][1]
                + x_shares[0][1] * y_shares[0][0]
                + a0
                - r.shares[0][0]
            )

        with tf.device(prot.servers[1].device_name):
            z1 = (
                x_shares[1][0] * y_shares[1][0]
                + x_shares[1][0] * y_shares[1][1]
                + x_shares[1][1] * y_shares[1][0]
                + a1
                - r.shares[1][0]
            )

        with tf.device(prot.servers[2].device_name):
            z2 = (
                x_shares[2][0] * y_shares[2][0]
                + x_shares[2][0] * y_shares[2][1]
                + x_shares[2][1] * y_shares[2][0]
                + a2
                - r.shares[2][0]
            )

        # Step 3: Reveal (x*y - r) / 2^d
        # xy_minus_r = z0 + z1 + z2
        # xy_minus_r_trunc = xy_minus_r.right_shift(amount)
        # z = ABYIPublicTensor(prot, [xy_minus_r_trunc, xy_minus_r_trunc, xy_minus_r_trunc], True, ARITHMETIC)
        xy_minus_r_trunc = [None] * 3
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                xy_minus_r_trunc[i] = z0 + z1 + z2
                xy_minus_r_trunc[i] = xy_minus_r_trunc[i].right_shift(amount)
        z = ABYIPublicTensor(prot, xy_minus_r_trunc, True, ARITHMETIC)

        # Step 4: Final addition
        z = z + r_trunc

        return z


def _matmul_public_private(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("matmul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_on_0.matmul(shares[0][0])
            z[0][1] = x_on_0.matmul(shares[0][1])

        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_on_1.matmul(shares[1][0])
            z[1][1] = x_on_1.matmul(shares[1][1])

        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_on_2.matmul(shares[2][0])
            z[2][1] = x_on_2.matmul(shares[2][1])

        z = ABYIPrivateTensor(prot, z, x.is_scaled or y.is_scaled, y.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _matmul_private_public(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)

    shares = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("matmul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0].matmul(y_on_0)
            z[0][1] = shares[0][1].matmul(y_on_0)

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0].matmul(y_on_1)
            z[1][1] = shares[1][1].matmul(y_on_1)

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0].matmul(y_on_2)
            z[2][1] = shares[2][1].matmul(y_on_2)

        z = ABYIPrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _matmul_private_private(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    # Tensorflow supports matmul for more than 2 dimensions,
    # with the inner-most 2 dimensions specifying the 2-D matrix multiplication
    result_shape = tf.TensorShape((*x.shape[:-1], y.shape[-1]))

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("matmul"):
        a0, a1, a2 = prot._gen_zero_sharing(result_shape)

        with tf.device(prot.servers[0].device_name):
            z0 = (
                x_shares[0][0].matmul(y_shares[0][0])
                + x_shares[0][0].matmul(y_shares[0][1])
                + x_shares[0][1].matmul(y_shares[0][0])
                + a0
            )

        with tf.device(prot.servers[1].device_name):
            z1 = (
                x_shares[1][0].matmul(y_shares[1][0])
                + x_shares[1][0].matmul(y_shares[1][1])
                + x_shares[1][1].matmul(y_shares[1][0])
                + a1
            )

        with tf.device(prot.servers[2].device_name):
            z2 = (
                x_shares[2][0].matmul(y_shares[2][0])
                + x_shares[2][0].matmul(y_shares[2][1])
                + x_shares[2][1].matmul(y_shares[2][0])
                + a2
            )
        # Re-sharing
        with tf.device(prot.servers[0].device_name):
            z[0][0] = z0
            z[0][1] = z1
        with tf.device(prot.servers[1].device_name):
            z[1][0] = z1
            z[1][1] = z2
        with tf.device(prot.servers[2].device_name):
            z[2][0] = z2
            z[2][1] = z0

        z = ABYIPrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _truncate_private(prot: ABYI, x: ABYIPrivateTensor) -> ABYIPrivateTensor:
    assert isinstance(x, ABYIPrivateTensor)

    if prot.fixedpoint_config.use_noninteractive_truncation:
        return _truncate_private_noninteractive(prot, x)

    return _truncate_private_interactive(prot, x)


def _truncate_private_noninteractive(
    prot: ABYI, x: ABYIPrivateTensor,
) -> ABYIPrivateTensor:
    assert isinstance(x, ABYIPrivateTensor), type(x)

    base = prot.fixedpoint_config.scaling_base
    amount = prot.fixedpoint_config.precision_fractional
    shares = x.unwrapped

    y = [[None, None], [None, None], [None, None]]
    with tf.name_scope("truncate"):

        # First step: compute new shares

        with tf.device(prot.servers[2].device_name):
            r_on_2 = prot.int_factory.sample_seeded_uniform(
                shares[2][0].shape, prot.pairwise_keys[2][0] + prot.pairwise_nonces[1]
            )

        with tf.device(prot.servers[0].device_name):
            y0 = shares[0][0].truncate(amount, base)

        with tf.device(prot.servers[1].device_name):
            r_on_1 = prot.int_factory.sample_seeded_uniform(
                shares[1][0].shape, prot.pairwise_keys[1][1] + prot.pairwise_nonces[1]
            )
            t = shares[1][0] + shares[1][1]
            # tmp = 0 - (0 - t).truncate(amount, base)
            tmp = t.truncate(amount, base)
            y1 = tmp - r_on_1

        prot.pairwise_nonces[1] = prot.pairwise_nonces[1] + 1

        # Second step: replicate shares

        with tf.device(prot.servers[0].device_name):
            y[0][0] = y0
            y[0][1] = y1
        with tf.device(prot.servers[1].device_name):
            y[1][0] = y1
            y[1][1] = r_on_1
        with tf.device(prot.servers[2].device_name):
            y[2][0] = r_on_2
            y[2][1] = y0

    return ABYIPrivateTensor(prot, y, x.is_scaled, x.share_type)


def _truncate_private_interactive(
    prot: ABYI, a: ABYIPrivateTensor
) -> ABYIPrivateTensor:
    """
  See protocol TruncPr (3.1) in
    "Secure Computation With Fixed-Point Numbers" by Octavian Catrina and Amitabh
    Saxena, FC'10.

  We call it "interactive" to keep consistent with the 2pc setting,
  but in fact, our protocol uses only one round communication, exactly the same as
  that in the "non-interactive" one.
  """
    assert isinstance(a, ABYIPrivateTensor), type(a)

    with tf.name_scope("truncate-i"):
        scaling_factor = prot.fixedpoint_config.scaling_factor
        scaling_factor_inverse = inverse(
            prot.fixedpoint_config.scaling_factor, prot.int_factory.modulus
        )

        # we first rotate `a` to make sure reconstructed values fall into
        # a non-negative interval `[0, 2B)` for some bound B; this uses an
        # assumption that the values originally lie in `[-B, B)`, and will
        # leak private information otherwise

        # 'a + bound' will automatically lift 'bound' by another scaling factor,
        # so we should first divide bound by the scaling factor if we want to
        # use this convenient '+' operation.
        bound = prot.fixedpoint_config.bound_double_precision
        b = a + (bound / scaling_factor)

        # next step is for servers to add a statistical mask to `b`, reveal
        # it to server1 and server2, and compute the lower part
        trunc_gap = prot.fixedpoint_config.truncation_gap
        mask_bitlength = ceil(log2(bound)) + 2 + trunc_gap

        b_shares = b.unwrapped
        a_shares = a.unwrapped
        shape = a.shape

        # NOTE: The following algorithm has an assumption to ensure the correctness:
        # c = a + bound + r0 + r1  SHOULD be positively smaller than
        # the max int64 number 2^{63} - 1. This is necessary to ensure the correctness of
        # the modulo operation 'c % scaling_factor'.
        # As a simple example, consider a 4-bit number '1111', when we think of it as a signed
        # number, it is '-1', and '-1 % 3 = 2'. But when we think of it as an unsigned number,
        # then '15 % 3 = 0'. AND the following works only if c is a positive number that is within
        # 63-bit, because 64-bit becomes a negative number.
        # Therefore, 'mask_bitlength' is better <= 61 if we use int64 as the underlying type, because
        # r0 is 61-bit, r1 is 61-bit, bound is much smaller, and (assuming) a is much smaller than bound.

        d = [[None] * 2 for _ in range(3)]
        with tf.device(prot.servers[0].device_name):
            r0_on_0 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[0][0] + prot.pairwise_nonces[2],
                mask_bitlength,
            )
            r1_on_0 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[0][1] + prot.pairwise_nonces[0],
                mask_bitlength,
            )
            c0_on_0 = b_shares[0][0] + r0_on_0
            c1_on_0 = b_shares[0][1] + r1_on_0

            r0_lower_on_0 = r0_on_0 % scaling_factor
            r1_lower_on_0 = r1_on_0 % scaling_factor

            a_lower0_on_0 = -r0_lower_on_0
            a_lower1_on_0 = -r1_lower_on_0

            d[0][0] = (a_shares[0][0] - a_lower0_on_0) * scaling_factor_inverse
            d[0][1] = (a_shares[0][1] - a_lower1_on_0) * scaling_factor_inverse

        with tf.device(prot.servers[1].device_name):
            r1_on_1 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[1][0] + prot.pairwise_nonces[0],
                mask_bitlength,
            )
            c1_on_1 = b_shares[1][0] + r1_on_1
            c2_on_1 = b_shares[1][1]

            # server0 sends c0 to server1, revealing c to server1
            c_on_1 = c0_on_0 + c1_on_1 + c2_on_1

            r1_lower_on_1 = r1_on_1 % scaling_factor

            a_lower1_on_1 = -r1_lower_on_1
            a_lower2_on_1 = c_on_1 % scaling_factor

            d[1][0] = (a_shares[1][0] - a_lower1_on_1) * scaling_factor_inverse
            d[1][1] = (a_shares[1][1] - a_lower2_on_1) * scaling_factor_inverse

        with tf.device(prot.servers[2].device_name):
            r0_on_2 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[2][1] + prot.pairwise_nonces[2],
                mask_bitlength,
            )
            c0_on_2 = b_shares[2][1] + r0_on_2
            c2_on_2 = b_shares[2][0]

            # server1 sends c1 to server2, revealing c to server2
            c_on_2 = c0_on_2 + c1_on_1 + c2_on_2

            r0_lower_on_2 = r0_on_2 % scaling_factor

            a_lower0_on_2 = -r0_lower_on_2
            a_lower2_on_2 = c_on_2 % scaling_factor

            d[2][0] = (a_shares[2][0] - a_lower2_on_2) * scaling_factor_inverse
            d[2][1] = (a_shares[2][1] - a_lower0_on_2) * scaling_factor_inverse

        prot.pairwise_nonces[0] += 1
        prot.pairwise_nonces[2] += 1

    return ABYIPrivateTensor(prot, d, a.is_scaled, a.share_type)


def _B_xor_private_private(prot: ABYI, x: ABYIPrivateTensor, y: ABYIPrivateTensor):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("b_xor"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = x.shares[0][0] ^ y.shares[0][0]
            z[0][1] = x.shares[0][1] ^ y.shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = x.shares[1][0] ^ y.shares[1][0]
            z[1][1] = x.shares[1][1] ^ y.shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = x.shares[2][0] ^ y.shares[2][0]
            z[2][1] = x.shares[2][1] ^ y.shares[2][1]

    return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


def _B_xor_private_public(prot: ABYI, x: ABYIPrivateTensor, y: ABYIPublicTensor):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("b_xor"):
        y_on_0, y_on_1, y_on_2 = y.unwrapped
        with tf.device(prot.servers[0].device_name):
            z[0][0] = x.shares[0][0] ^ y_on_0
            z[0][1] = x.shares[0][1] ^ y_on_0

        with tf.device(prot.servers[1].device_name):
            z[1][0] = x.shares[1][0] ^ y_on_1
            z[1][1] = x.shares[1][1] ^ y_on_1

        with tf.device(prot.servers[2].device_name):
            z[2][0] = x.shares[2][0] ^ y_on_2
            z[2][1] = x.shares[2][1] ^ y_on_2

    return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


def _B_xor_public_private(prot: ABYI, x: ABYIPublicTensor, y: ABYIPrivateTensor):
    return _B_xor_private_public(prot, y, x)


def _B_xor_public_public(prot: ABYI, x: ABYIPublicTensor, y: ABYIPublicTensor):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    z = [None, None, None]
    with tf.name_scope("b_xor"):
        x_on_0, x_on_1, x_on_2 = x.unwrapped
        y_on_0, y_on_1, y_on_2 = y.unwrapped
        with tf.device(prot.servers[0].device_name):
            z[0] = x_on_0 ^ y_on_0

        with tf.device(prot.servers[1].device_name):
            z[1] = x_on_1 ^ y_on_1

        with tf.device(prot.servers[2].device_name):
            z[2] = x_on_2 ^ y_on_2

    return ABYIPublicTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)


def _B_and_private_private(prot: ABYI, x: ABYIPrivateTensor, y: ABYIPrivateTensor):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("b_and"):
        a0, a1, a2 = prot._gen_zero_sharing(
            x.shape, share_type=BOOLEAN, factory=x.backing_dtype
        )

        with tf.device(prot.servers[0].device_name):
            tmp0 = x_shares[0][0] & y_shares[0][0]
            tmp1 = x_shares[0][0] & y_shares[0][1]
            tmp2 = x_shares[0][1] & y_shares[0][0]
            z0 = tmp0 ^ tmp1 ^ tmp2 ^ a0

        with tf.device(prot.servers[1].device_name):
            tmp0 = x_shares[1][0] & y_shares[1][0]
            tmp1 = x_shares[1][0] & y_shares[1][1]
            tmp2 = x_shares[1][1] & y_shares[1][0]
            z1 = tmp0 ^ tmp1 ^ tmp2 ^ a1

        with tf.device(prot.servers[2].device_name):
            tmp0 = x_shares[2][0] & y_shares[2][0]
            tmp1 = x_shares[2][0] & y_shares[2][1]
            tmp2 = x_shares[2][1] & y_shares[2][0]
            z2 = tmp0 ^ tmp1 ^ tmp2 ^ a2

        # Re-sharing
        with tf.device(prot.servers[0].device_name):
            z[0][0] = z0
            z[0][1] = z1
        with tf.device(prot.servers[1].device_name):
            z[1][0] = z1
            z[1][1] = z2
        with tf.device(prot.servers[2].device_name):
            z[2][0] = z2
            z[2][1] = z0

        z = ABYIPrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        return z


def _B_and_private_public(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(x)
    assert x.backing_dtype == y.backing_dtype

    x_shares = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("B_and"):
        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_shares[0][0] & y_on_0
            z[0][1] = x_shares[0][1] & y_on_0
        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_shares[1][0] & y_on_1
            z[1][1] = x_shares[1][1] & y_on_1
        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_shares[2][0] & y_on_2
            z[2][1] = x_shares[2][1] & y_on_2

    z = ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _B_and_public_private(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("B_and"):
        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_on_0 & y_shares[0][0]
            z[0][1] = x_on_0 & y_shares[0][1]
        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_on_1 & y_shares[1][0]
            z[1][1] = x_on_1 & y_shares[1][1]
        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_on_2 & y_shares[2][0]
            z[2][1] = x_on_2 & y_shares[2][1]

    z = ABYIPrivateTensor(prot, z, y.is_scaled, y.share_type)
    return z


def _B_and_public_public(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPublicTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [None, None, None]
    with tf.name_scope("B_and"):
        with tf.device(prot.servers[0].device_name):
            z[0] = x_on_0 & y_on_0
        with tf.device(prot.servers[1].device_name):
            z[1] = x_on_1 & y_on_1
        with tf.device(prot.servers[2].device_name):
            z[2] = x_on_2 & y_on_2

    z = ABYIPublicTensor(prot, z, x.is_scaled or y.is_scaled, y.share_type)
    return z


def _B_or_private_private(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    with tf.name_scope("B_or"):
        z = (x ^ y) ^ (x & y)

    return z


def _B_not_private(prot, x):
    assert isinstance(x, ABYIPrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("B_not"):
        with tf.device(prot.servers[0].device_name):
            # We use the `~` operator instead of XORing a constant, because we want it to work for both
            # the int_factory and the bool_factory
            z[0][0] = ~x_shares[0][0]
            z[0][1] = x_shares[0][1]
        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_shares[1][0]
            z[1][1] = x_shares[1][1]
        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_shares[2][0]
            z[2][1] = ~x_shares[2][1]
        z = ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _lshift_private(prot, x, steps):
    """
  Left shift.
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("lshift"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0] << steps
                z[i][1] = x_shares[i][1] << steps

        z = ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)

    return z


def _lshift_public(prot, x, steps):
    """
  Left shift.
  """
    assert isinstance(x, ABYIPublicTensor), type(x)

    x_on_i = x.unwrapped

    z = [None, None, None]
    with tf.name_scope("lshift"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i] = x_on_i[i] << steps

        z = ABYIPublicTensor(prot, z, x.is_scaled, x.share_type)

    return z


def _rshift_private(prot, x, steps):
    """
  Arithmetic right shift.
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("rshift"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0] >> steps
                z[i][1] = x_shares[i][1] >> steps

        z = ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)

    return z


def _logical_rshift_private(prot, x, steps):
    """
  Logical right shift.
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("logical-rshift"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0].logical_rshift(steps)
                z[i][1] = x_shares[i][1].logical_rshift(steps)

        z = ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)

    return z


def _B_add_private_private(prot, x, y):
    raise NotImplementedError(
        "Addition with boolean sharing is not implemented, and not recommended."
    )


def _B_sub_private_private(prot, x, y):
    raise NotImplementedError(
        "Sbustraction with boolean sharing is not implemented, and not recommended."
    )


def _B_ppa_private_private(prot, x, y, n_bits, topology="kogge_stone"):
    """
  Parallel prefix adder (PPA). This adder can be used for addition of boolean sharings.

  `n_bits` can be passed as an optimization to constrain the computation for least significant
  `n_bits` bits.

  AND Depth: log(k)
  Total gates: klog(k)
  """

    if topology == "kogge_stone":
        return _B_ppa_kogge_stone_private_private(prot, x, y, n_bits)
    elif topology == "sklansky":
        return _B_ppa_sklansky_private_private(prot, x, y, n_bits)
    else:
        raise NotImplementedError("Unknown adder topology.")


def _B_ppa_sklansky_private_private(prot, x, y, n_bits):
    """
  Parallel prefix adder (PPA), using the Sklansky adder topology.
  """

    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    if x.backing_dtype.native_type != tf.int64:
        raise NotImplementedError(
            "Native type {} not supported".format(x.backing_dtype.native_type)
        )

    with tf.name_scope("B_ppa"):
        keep_masks = [
            0x5555555555555555,     # 55 55: 01010101 01010101   8B(int64)
            0x3333333333333333,     # 33 33: 00110011 00110011
            0x0F0F0F0F0F0F0F0F,     # 0F 0F: 00001111 00001111
            0x00FF00FF00FF00FF,     # 00 FF: 00000000 11111111
            0x0000FFFF0000FFFF,     # 
            0x00000000FFFFFFFF,
        ]  # yapf: disable
        copy_masks = [
            0x5555555555555555,     # 55: 01010101
            0x2222222222222222,     # 22: 00100010
            0x0808080808080808,     # 08: 00001000
            0x0080008000800080,     # 80: 01000000
            0x0000800000008000,
            0x0000000080000000,
        ]  # yapf: disable

        G = x & y                   # G_j
        P = x ^ y                   # P_j

        k = prot.nbits
        if n_bits is not None:
            k = n_bits
        # eq test 
        for i in range(ceil(log2(k))):      # 6
        # for i in range(1): 
            c_mask = prot.define_constant(
                np.ones(x.shape, dtype=np.object) * copy_masks[i],
                apply_scaling=False,
                share_type=BOOLEAN,
            )
            k_mask = prot.define_constant(
                np.ones(x.shape, dtype=np.object) * keep_masks[i],
                apply_scaling=False,
                share_type=BOOLEAN,
            )
            # Copy the selected bit to 2^i positions:
            # For example, when i=2, the 4-th bit is copied to the (5, 6, 7, 8)-th bits
            # eq test
            # with tf.control_dependencies([tf.print(k_mask.unwrapped[0].bits().value[0][0], summarize=-1)]):
            G1 = (G & c_mask) << 1      # G_{j-1}
            P1 = (P & c_mask) << 1      # P_{j-1}
            for j in range(i):
                G1 = (G1 << (2 ** j)) ^ G1
                P1 = (P1 << (2 ** j)) ^ P1
            """
      Two-round impl. using algo. that assume using OR gate is free, but in fact,
      here using OR gate cost one round.
      The PPA operator 'o' is defined as:
      (G, P) o (G1, P1) = (G + P*G1, P*P1), where '+' is OR, '*' is AND
      """
            # G1 and P1 are 0 for those positions that we do not copy the selected bit to.
            # Hence for those positions, the result is: (G, P) = (G, P) o (0, 0) = (G, 0).
            # In order to keep (G, P) for these positions so that they can be used in the future,
            # we need to let (G1, P1) = (G, P) for these positions, because (G, P) o (G, P) = (G, P)
            #
            # G1 = G1 ^ (G & k_mask)
            # P1 = P1 ^ (P & k_mask)
            #
            # G = G | (P & G1)
            # P = P & P1
            """
      One-round impl. by modifying the PPA operator 'o' as:
      (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
      This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
      the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
      at the same time.
      And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
      This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
      if we want to keep G and P bits.
      """
            # Option 1: Using (G, P) o (0, P) = (G, P)
            # P1 = P1 ^ (P & k_mask)
            # Option 2: Using (G, P) o (0, 1) = (G, P)
            P1 = P1 ^ k_mask

            G = G ^ (P & G1)
            P = P & P1

        # G stores the carry-in to the next position
        C = G << 1
        P = x ^ y
        z = C ^ P

    return z


def _B_ppa_kogge_stone_private_private(prot, x, y, n_bits):
    """
  Parallel prefix adder (PPA), using the Kogge-Stone adder topology.
  """

    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)

    if x.backing_dtype.native_type != tf.int64:
        raise NotImplementedError(
            "Native type {} not supported".format(x.backing_dtype.native_type)
        )

    with tf.name_scope("B_ppa"):
        keep_masks = []
        for i in range(ceil(log2(prot.nbits))):
            keep_masks.append((1 << (2 ** i)) - 1)
        """
    For example, if prot.nbits = 64, then keep_masks is:
    keep_masks = [0x0000000000000001, 0x0000000000000003, 0x000000000000000f,
                  0x00000000000000ff, 0x000000000000ffff, 0x00000000ffffffff]
    """

        G = x & y
        P = x ^ y
        k = prot.nbits if n_bits is None else n_bits
        for i in range(ceil(log2(k))):
            k_mask = prot.define_constant(
                np.ones(x.shape, dtype=np.object) * keep_masks[i],
                apply_scaling=False,
                share_type=BOOLEAN,
            )

            G1 = G << (2 ** i)
            P1 = P << (2 ** i)
            """
      One-round impl. by modifying the PPA operator 'o' as:
      (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
      This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
      the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
      at the same time.
      And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
      This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
      if we want to keep G and P bits.
      """
            # Option 1: Using (G, P) o (0, P) = (G, P)
            # P1 = P1 ^ (P & k_mask)
            # Option 2: Using (G, P) o (0, 1) = (G, P)
            P1 = P1 ^ k_mask

            G = G ^ (P & G1)
            P = P & P1

        # G stores the carry-in to the next position
        C = G << 1
        P = x ^ y
        z = C ^ P
    return z


def _A2B_private(prot, x, nbits):
    """
  Bit decomposition: Convert an arithmetic sharing to a boolean sharing.
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.share_type == ARITHMETIC

    x_shares = x.unwrapped
    zero = prot.define_constant(
        np.zeros(x.shape, dtype=np.int64), apply_scaling=False, share_type=BOOLEAN
    )
    zero_on_0, zero_on_1, zero_on_2 = zero.unwrapped
    a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

    operand1 = [[None, None], [None, None], [None, None]]
    operand2 = [[None, None], [None, None], [None, None]]
    with tf.name_scope("A2B"):
        # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
        # We need to reshare it into two operands that will be fed into an addition circuit:
        # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
        # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2
        with tf.device(prot.servers[0].device_name):
            x0_plus_x1 = x_shares[0][0] + x_shares[0][1]
            operand1[0][0] = x0_plus_x1 ^ a0
            operand1[0][1] = a1

            operand2[0][0] = zero_on_0
            operand2[0][1] = zero_on_0

        with tf.device(prot.servers[1].device_name):
            operand1[1][0] = a1
            operand1[1][1] = a2

            operand2[1][0] = zero_on_1
            operand2[1][1] = x_shares[1][1]

        with tf.device(prot.servers[2].device_name):
            operand1[2][0] = a2
            operand1[2][1] = operand1[0][0]

            operand2[2][0] = x_shares[2][0]
            operand2[2][1] = zero_on_2

        operand1 = ABYIPrivateTensor(prot, operand1, x.is_scaled, BOOLEAN)
        operand2 = ABYIPrivateTensor(prot, operand2, x.is_scaled, BOOLEAN)

        # Step 2: Parallel prefix adder that requires log(k) rounds of communication
        result = prot.B_ppa(operand1, operand2, nbits)

    return result


def _bit_extract_private(prot, x, i):
    """
  Bit extraction: Extracts the `i`-th bit of an arithmetic sharing or boolean sharing
  to a single-bit boolean sharing.
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.backing_dtype == prot.int_factory

    with tf.name_scope("bit_extract"):
        if x.share_type == ARITHMETIC:
            with tf.name_scope("A2B_partial"):
                x_shares = x.unwrapped
                zero = prot.define_constant(
                    np.zeros(x.shape, dtype=np.int64),
                    apply_scaling=False,
                    share_type=BOOLEAN,
                )
                zero_on_0, zero_on_1, zero_on_2 = zero.unwrapped
                a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

                operand1 = [[None, None], [None, None], [None, None]]
                operand2 = [[None, None], [None, None], [None, None]]
                # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
                # We need to reshare it into two operands that will be fed into an addition circuit:
                # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
                # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2
                with tf.device(prot.servers[0].device_name):
                    x0_plus_x1 = x_shares[0][0] + x_shares[0][1]
                    operand1[0][0] = x0_plus_x1 ^ a0
                    operand1[0][1] = a1

                    operand2[0][0] = zero_on_0
                    operand2[0][1] = zero_on_0

                with tf.device(prot.servers[1].device_name):
                    operand1[1][0] = a1
                    operand1[1][1] = a2

                    operand2[1][0] = zero_on_1
                    operand2[1][1] = x_shares[1][1]

                with tf.device(prot.servers[2].device_name):
                    operand1[2][0] = a2
                    operand1[2][1] = operand1[0][0]

                    operand2[2][0] = x_shares[2][0]
                    operand2[2][1] = zero_on_2

                operand1 = ABYIPrivateTensor(prot, operand1, x.is_scaled, BOOLEAN)
                operand2 = ABYIPrivateTensor(prot, operand2, x.is_scaled, BOOLEAN)

                # Step 2: Parallel prefix adder that requires log(i+1) rounds of communication
                x = prot.B_ppa(operand1, operand2, i + 1)

        # Take out the i-th bit
        #
        # NOTE: Don't use x = x & 0x1. Even though we support automatic lifting of 0x1
        # to an ABYITensor, but it also includes automatic scaling to make the two operands have
        # the same scale, which is not what want here.
        #
        mask = prot.define_constant(
            np.array([0x1 << i]), apply_scaling=False, share_type=BOOLEAN
        )
        x = x & mask

        x_shares = x.unwrapped
        result = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                result[i][0] = x_shares[i][0].cast(prot.bool_factory)
                result[i][1] = x_shares[i][1].cast(prot.bool_factory)
        result = ABYIPrivateTensor(prot, result, False, BOOLEAN)

    return result


def _B2A_private(prot, x, nbits):
    """
  Bit composition: Convert a boolean sharing to an arithmetic sharing.
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.share_type == BOOLEAN

    # In semi-honest, the following two calls can be further optimized because we don't
    # need the boolean shares of x1 and x2. We only need their original values on intended servers.
    x1_on_0, x1_on_1, x1_on_2, x1_shares = prot._gen_b2a_sharing(
        x.shape, prot.b2a_keys_1
    )
    assert x1_on_2 is None
    x2_on_0, x2_on_1, x2_on_2, x2_shares = prot._gen_b2a_sharing(
        x.shape, prot.b2a_keys_2
    )
    assert x2_on_0 is None

    a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

    with tf.name_scope("B2A"):
        # Server 1 reshares (-x1-x2) as private input
        neg_x1_neg_x2 = [[None, None], [None, None], [None, None]]
        with tf.device(prot.servers[1].device_name):
            value = -x1_on_1 - x2_on_1
            neg_x1_neg_x2[1][0] = value ^ a1
            neg_x1_neg_x2[1][1] = a2
        with tf.device(prot.servers[0].device_name):
            neg_x1_neg_x2[0][0] = a0
            neg_x1_neg_x2[0][1] = neg_x1_neg_x2[1][0]
        with tf.device(prot.servers[2].device_name):
            neg_x1_neg_x2[2][0] = a2
            neg_x1_neg_x2[2][1] = a0
        neg_x1_neg_x2 = ABYIPrivateTensor(prot, neg_x1_neg_x2, x.is_scaled, BOOLEAN)

        # Compute x0 = x + (-x1-x2) using the parallel prefix adder
        x0 = prot.B_ppa(x, neg_x1_neg_x2, nbits)

        # Reveal x0 to server 0 and 2
        with tf.device(prot.servers[0].device_name):
            x0_on_0 = prot._reconstruct(x0.unwrapped, prot.servers[0], BOOLEAN)
        with tf.device(prot.servers[2].device_name):
            x0_on_2 = prot._reconstruct(x0.unwrapped, prot.servers[2], BOOLEAN)

        # Construct the arithmetic sharing
        result = [[None, None], [None, None], [None, None]]
        with tf.device(prot.servers[0].device_name):
            result[0][0] = x0_on_0
            result[0][1] = x1_on_0
        with tf.device(prot.servers[1].device_name):
            result[1][0] = x1_on_1
            result[1][1] = x2_on_1
        with tf.device(prot.servers[2].device_name):
            result[2][0] = x2_on_2
            result[2][1] = x0_on_2
        result = ABYIPrivateTensor(prot, result, x.is_scaled, ARITHMETIC)

    return result


def _mul_AB_public_private(prot, x, y):
    assert isinstance(x, ABYIPublicTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(x)
    assert x.share_type == ARITHMETIC
    assert y.share_type == BOOLEAN

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("mul_AB"):
        z = __mul_AB_routine(prot, x_on_2, y, 2)
        z = ABYIPrivateTensor(prot, z, x.is_scaled, ARITHMETIC)

    return z


def _mul_AB_private_private(prot, x, y):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert isinstance(y, ABYIPrivateTensor), type(y)
    assert x.share_type == ARITHMETIC
    assert y.share_type == BOOLEAN

    x_shares = x.unwrapped

    with tf.name_scope("mul_AB"):
        with tf.name_scope("term0"):
            w = __mul_AB_routine(prot, x_shares[0][0], y, 0)
            w = ABYIPrivateTensor(prot, w, x.is_scaled, ARITHMETIC)

        with tf.name_scope("term1"):
            with tf.device(prot.servers[1].device_name):
                a = x_shares[1][0] + x_shares[1][1]
            z = __mul_AB_routine(prot, a, y, 1)
            z = ABYIPrivateTensor(prot, z, x.is_scaled, ARITHMETIC)
        z = w + z

    return z


def __mul_AB_routine(prot, a, b, sender_idx):
    """
    A sub routine for multiplying a value 'a' (located at servers[sender_idx]) with a boolean sharing 'b'.
    """
    assert isinstance(a, AbstractTensor), type(a)
    assert isinstance(b, ABYIPrivateTensor), type(b)

    with tf.name_scope("__mul_AB_routine"):
        b_shares = b.unwrapped
        s = [None, None, None]
        s[0], s[1], s[2] = prot._gen_zero_sharing(a.shape, ARITHMETIC)

        z = [[None, None], [None, None], [None, None]]
        idx0 = sender_idx
        idx1 = (sender_idx + 1) % 3
        idx2 = (sender_idx + 2) % 3
        with tf.device(prot.servers[idx0].device_name):
            z[idx0][0] = s[idx2]
            z[idx0][1] = s[idx1]
            tmp = (b_shares[idx0][0] ^ b_shares[idx0][1]).cast(a.factory) * a
            m0 = tmp + s[idx0]
            m1 = -tmp + a + s[idx0]

        with tf.device(prot.servers[idx1].device_name):
            z[idx1][0] = s[idx1]
            z[idx1][1] = prot._ot(
                prot.servers[idx0],
                prot.servers[idx1],
                prot.servers[idx2],
                m0,
                m1,
                b_shares[idx1][1],
                b_shares[idx2][0],
                prot.pairwise_keys[idx0][0],
                prot.pairwise_keys[idx2][1],
                prot.pairwise_nonces[idx2],
            )
            prot.pairwise_nonces[idx2] = prot.pairwise_nonces[idx2] + 1

        with tf.device(prot.servers[idx2].device_name):
            z[idx2][0] = prot._ot(
                prot.servers[idx0],
                prot.servers[idx2],
                prot.servers[idx1],
                m0,
                m1,
                b_shares[idx2][0],
                b_shares[idx1][1],
                prot.pairwise_keys[idx0][1],
                prot.pairwise_keys[idx1][0],
                prot.pairwise_nonces[idx0],
            )
            z[idx2][1] = s[idx2]
            prot.pairwise_nonces[idx0] = prot.pairwise_nonces[idx0] + 1

    return z


def _pow_private(prot, x, p):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.share_type == ARITHMETIC
    assert p >= 1, "Exponent should be >= 0"

    # NOTE: pow should be able to use the `memoir` memoization

    with tf.name_scope("pow"):
        result = 1
        tmp = x
        while p > 0:
            bit = p & 0x1
            if bit > 0:
                result = result * tmp
            p >>= 1
            if p > 0:
                tmp = tmp * tmp
    return result


def _polynomial_private(prot, x, coeffs):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.share_type == ARITHMETIC

    with tf.name_scope("polynomial"):
        result = prot.define_constant(np.zeros(x.shape), apply_scaling=x.is_scaled)
        for i in range(len(coeffs)):
            if i == 0:
                result = result + coeffs[i]
            elif coeffs[i] == 0:
                continue
            elif (coeffs[i] - int(coeffs[i])) == 0:
                # Optimization when coefficient is integer: mulitplication can be performed
                # locally without interactive truncation
                tmp = prot.define_constant(np.array([coeffs[i]]), apply_scaling=False)
                tmp = tmp * (x ** i)
                result = result + tmp
            else:
                tmp = coeffs[i] * (x ** i)
                result = result + tmp
    return result


def _polynomial_piecewise_private(prot, x, c, coeffs):
    """
  :param prot:
  :param x:
  :param c: A list of splitting points between pieces
  :param coeffs: Two-dimensional list: 1st dimension is the polynomial index, 2nd dimension is the coefficient index
  :return:
  """
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert len(c) + 1 == len(coeffs), "# of pieces do not match # of polynomials"

    with tf.name_scope("polynomial_piecewise"):
        # Compute the selection bit for each polynomial
        with tf.name_scope("polynomial-selection-bit"):
            msbs = [None] * len(c)
            for i in range(len(c)):
                msbs[i] = prot.msb(x - c[i])
            b = [None] * len(coeffs)
            b[0] = msbs[0]
            for i in range(len(c) - 1):
                b[i + 1] = ~msbs[i] & msbs[i + 1]
            b[len(c)] = ~msbs[len(c) - 1]

        # Compute the piecewise combination result
        result = 0
        for i in range(len(coeffs)):
            fi = prot.polynomial(x, coeffs[i])
            result = result + prot.mul_AB(fi, b[i])
    return result


def _sigmoid_private(prot, x, approx_type):
    assert isinstance(x, ABYIPrivateTensor), type(x)

    with tf.name_scope("sigmoid"):
        if approx_type == "piecewise_linear":
            c = (-2.5, 2.5)
            coeffs = ((1e-4,), (0.50, 0.17), (1 - 1e-4,))
        else:
            raise NotImplementedError(
                "Only support piecewise linear approximation of sigmoid."
            )

        result = prot.polynomial_piecewise(x, c, coeffs)
    return result


#
# transpose helpers
#


def _transpose_private(prot, x, perm=None):
    assert isinstance(x, ABYIPrivateTensor)

    x_shares = x.unwrapped

    with tf.name_scope("transpose"):
        z = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0].transpose(perm=perm)
                z[i][1] = x_shares[i][1].transpose(perm=perm)

        return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


def _transpose_public(prot, x, perm=None):
    assert isinstance(x, ABYIPublicTensor)

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("transpose"):

        with tf.device(prot.servers[0].device_name):
            x_on_0_t = x_on_0.transpose(perm=perm)

        with tf.device(prot.servers[1].device_name):
            x_on_1_t = x_on_1.transpose(perm=perm)

        with tf.device(prot.servers[2].device_name):
            x_on_2_t = x_on_2.transpose(perm=perm)

        return ABYIPublicTensor(
            prot, [x_on_0_t, x_on_1_t, x_on_2_t], x.is_scaled, x.share_type
        )


#
# reduce_sum helpers
#


def _reduce_sum_public(prot, x, axis=None, keepdims=False):

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("reduce_sum"):

        with tf.device(prot.servers[0].device_name):
            y_on_0 = x_on_0.reduce_sum(axis, keepdims)

        with tf.device(prot.servers[1].device_name):
            y_on_1 = x_on_1.reduce_sum(axis, keepdims)

        with tf.device(prot.servers[2].device_name):
            y_on_2 = x_on_2.reduce_sum(axis, keepdims)

    return ABYIPublicTensor(prot, [y_on_0, y_on_1, y_on_2], x.is_scaled, x.share_type)


def _reduce_sum_private(prot, x, axis=None, keepdims=False):

    x_shares = x.unwrapped

    with tf.name_scope("reduce_sum"):
        z = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0].reduce_sum(axis, keepdims)
                z[i][1] = x_shares[i][1].reduce_sum(axis, keepdims)
    return ABYIPrivateTensor(prot, z, x.is_scaled, x.share_type)


#
# concat helpers
#


def _concat_public(prot, xs, axis):
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    xs_on_0, xs_on_1, xs_on_2 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("concat"):

        with tf.device(prot.servers[0].device_name):
            x_on_0_concat = factory.concat(xs_on_0, axis=axis)

        with tf.device(prot.servers[1].device_name):
            x_on_1_concat = factory.concat(xs_on_1, axis=axis)

        with tf.device(prot.servers[2].device_name):
            x_on_2_concat = factory.concat(xs_on_2, axis=axis)

        return ABYIPublicTensor(
            prot,
            [x_on_0_concat, x_on_1_concat, x_on_2_concat],
            is_scaled,
            xs[0].share_type,
        )


def _concat_private(prot, xs, axis):
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    share_type = xs[0].share_type

    xs_shares = [x.unwrapped for x in xs]
    z = [[None, None], [None, None], [None, None]]
    for i in range(3):
        z[i][0] = [x_shares[i][0] for x_shares in xs_shares]
        z[i][1] = [x_shares[i][1] for x_shares in xs_shares]

    with tf.name_scope("concat"):

        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = factory.concat(z[i][0], axis=axis)
                z[i][1] = factory.concat(z[i][1], axis=axis)

        return ABYIPrivateTensor(prot, z, is_scaled, share_type)


def _write_private(prot, x, filename_prefix):
    assert isinstance(x, ABYIPrivateTensor), type(x)

    def encode(feature_row):
        # Converting a row to a string seems to be the only way of writing out
        # the dataset in a distributed way
        feature = tf.strings.reduce_join(
            tf.dtypes.as_string(tf.reshape(feature_row, [-1])), separator=","
        )
        return feature

    x_shares = x.unwrapped
    ops = []
    for i in range(3):
        with tf.device(prot.servers[i].device_name):
            for j in range(2):
                data = tf.data.Dataset.from_tensor_slices(x_shares[i][j].value).map(
                    encode
                )
                writer = tf.data.experimental.TFRecordWriter(
                    "{}_share{}{}".format(filename_prefix, i, j)
                )
                ops.append(writer.write(data))

    return tf.group(*ops)


def _read_(prot, filename_prefix, batch_size, n_columns):

    row_shape = [n_columns]

    def decode(line):
        fields = tf.string_split([line], ",").values
        fields = tf.strings.to_number(fields, tf.int64)
        fields = tf.reshape(fields, row_shape)
        return fields

    batch = [[None] * 2 for _ in range(3)]
    for i in range(3):
        with tf.device(prot.servers[i].device_name):
            for j in range(2):
                data = (
                    tf.data.TFRecordDataset(
                        ["{}_share{}{}".format(filename_prefix, i, j)]
                    )
                    .map(decode)
                    .repeat()
                    .batch(batch_size=batch_size)
                )
                it = data.make_one_shot_iterator()
                batch[i][j] = it.get_next()
                batch[i][j] = tf.reshape(batch[i][j], [batch_size] + row_shape)
                batch[i][j] = prot.int_factory.tensor(batch[i][j])

    return ABYIPrivateTensor(prot, batch, True, ARITHMETIC)


def _iterate_private(
    prot,
    tensor: "ABYIPrivateTensor",
    batch_size: int,
    repeat=True,
    shuffle=True,
    seed: int = None,
):

    assert isinstance(tensor, ABYIPrivateTensor)
    shares = tensor.unwrapped
    iterators = [[None] * 2 for _ in range(3)]
    results = [[None] * 2 for _ in range(3)]

    if seed is None:
        seed = np.random.randint(1, 1 << 32)  # this seed is publicly known.
    batch_size = max(1, batch_size)

    def helper(idx):
        with tf.device(prot.servers[idx].device_name):
            out_shape = shares[idx][0].value.shape.as_list()
            out_shape[0] = batch_size
            for i in range(2):
                dataset = tf.data.Dataset.from_tensor_slices(shares[idx][i].value)

                if repeat:
                    dataset = dataset.repeat()

                if shuffle:
                    dataset = dataset.shuffle(buffer_size=512, seed=seed)

                dataset = dataset.batch(batch_size)

                # NOTE: initializable_iterator needs to run initializer.
                iterators[idx][i] = tf.compat.v1.data.make_initializable_iterator(
                    dataset
                )
                batch = iterators[idx][i].get_next()
                # Wrap the tf.tensor as a dense tensor (no extra encoding is needed)
                results[idx][i] = prot.int_factory.tensor(tf.reshape(batch, out_shape))

            prot._initializers.append(
                tf.group(iterators[idx][0].initializer, iterators[idx][1].initializer)
            )

    for idx in range(3):
        helper(idx)

    # Synchronize the reading of all 6 dataset iterators
    with tf.control_dependencies(
        [share.value for result in results for share in result]
    ):
        for i in range(3):
            results[i][0] = results[i][0].identity()
            results[i][1] = results[i][1].identity()

    return ABYIPrivateTensor(prot, results, tensor.is_scaled, tensor.share_type)


def _indexer_private(prot: ABYI, tensor: ABYIPrivateTensor, slc) -> "ABYIPrivateTensor":
    shares = tensor.unwrapped
    results = [[None] * 2 for _ in range(3)]
    with tf.name_scope("index"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                results[i][0] = shares[i][0][slc]
                results[i][1] = shares[i][1][slc]
    return ABYIPrivateTensor(prot, results, tensor.is_scaled, tensor.share_type)


def _reshape_private(prot: ABYI, tensor: ABYIPrivateTensor, axe):
    shares = tensor.unwrapped
    results = [[None] * 2 for _ in range(3)]
    with tf.name_scope("reshape"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                results[i][0] = shares[i][0].reshape(axe)
                results[i][1] = shares[i][1].reshape(axe)
    return ABYIPrivateTensor(prot, results, tensor.is_scaled, tensor.share_type)


def _relu_private(prot: ABYI, x: ABYIPrivateTensor):
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.share_type == ARITHMETIC, "ReLU only for ARITHMETIC sharing, getting BOOLEN sharing."
    with tf.name_scope("relu"):
        result = prot.polynomial_piecewise(x, (0,), ((0,), (0, 1)))
    return result


def _cache_public(prot, x):
    assert isinstance(x, ABYIPublicTensor), type(x)

    shares = x.unwrapped

    with tf.name_scope("cache"):

        with tf.device(prot.servers[0].device_name):
            updater0, [x_on_0_cached] = wrap_in_variables(shares[0])

        with tf.device(prot.servers[1].device_name):
            updater1, [x_on_1_cached] = wrap_in_variables(shares[1])

        with tf.device(prot.servers[2].device_name):
            updater2, [x_on_2_cached] = wrap_in_variables(shares[2])

        combined_updater = tf.group(updater0, updater1, updater2)

    return (
        combined_updater,
        ABYICachedPublicTensor(
            prot, (x_on_0_cached, x_on_1_cached, x_on_2_cached), x.is_scaled, combined_updater,
        ),
    )


def _cache_private(prot, x):
    assert isinstance(x, ABYIPrivateTensor), type(x)

    shares = x.unwrapped

    with tf.name_scope("cache"):
        
        updaters, x_cached = [], []
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                # XD: updater0 = assign, x0_cached = variable(0).
                updater0, [x0_cached] = wrap_in_variables(shares[i][0])
                updater1, [x1_cached] = wrap_in_variables(shares[i][1])
                updaters.append(updater0)
                updaters.append(updater1)
                x_cached.append((x0_cached, x1_cached))

        combined_updater = tf.group(updaters)

    return (
        combined_updater,
        ABYICachedPrivateTensor(
            prot, x_cached, x.is_scaled, combined_updater,
        ),
    )


def _type(x):
    """Helper to check and return PondTensor types."""

    if isinstance(x, ABYIPublicTensor):
        return ABYIPublicTensor

    if isinstance(x, ABYIPrivateTensor):
        return ABYIPrivateTensor

    return type(x)


def _A2Bi_private(prot, x, nbits):
    """
    Bit decomposition: Convert an arithmetic sharing to a boolean sharing.
    Import mask-shared: (\Delta, <\delta>). \Delta is a public value, and \delta is a mask
    shared in 23-RSS.
    """
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.share_type == ARITHMETIC

    x_shares = x.unwrapped
    zero = prot.define_constant(
        np.zeros(x.shape, dtype=np.int64), apply_scaling=False, share_type=BOOLEAN
    )

    # used to construct the 23-RSS shares of x0+x1.
    # a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)
    delta1 = prot._gen_random_sharing(x.shape, share_type=BOOLEAN)
    delta1_shares = delta1.unwrapped
    z1 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)
    # used to construct the 23-RSS shares of x2.
    z2 = zero.unwrapped
    delta2 = prot._gen_random_sharing(x.shape, share_type=BOOLEAN)


    x01, Delta1 = [[None, None], [None, None], [None, None]], None       # x0+x1
    y, Delta2 = [[None, None], [None, None], [None, None]], None            # x2
    with tf.name_scope("A2Bi"):
        # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
        # We need to reshare it into two 23-RSS operands which will be used to generate mask sharing:
        # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
        # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2

        with tf.device(prot.servers[0].device_name):
            x0_plus_x1 = x_shares[0][0] + x_shares[0][1]
            x0 = x0_plus_x1 ^ delta1_shares[0][0] ^ z1[0]

            y[0][0] = z2[0]
            y[0][1] = z2[0] 

        with tf.device(prot.servers[1].device_name):
            x1 = delta1_shares[1][0] ^ z1[1]

            y[1][0] = z2[1]
            y[1][1] = x_shares[1][1] 

        with tf.device(prot.servers[2].device_name):
            x2 = delta1_shares[2][0] ^ z1[2]

            y[2][0] = x_shares[2][0] 
            y[2][1] = z2[2]

        with tf.name_scope("Build_x_ms"):
            with tf.device(prot.servers[0].device_name):
                x01[0][0] = x0
                x01[0][1] = x1
            
            with tf.device(prot.servers[1].device_name):
                x01[1][0] = x1
                x01[1][1] = x2
            
            with tf.device(prot.servers[2].device_name):
                x01[2][0] = x2
                x01[2][1] = x0

            x01 = ABYIPrivateTensor(prot, x01, x.is_scaled, BOOLEAN)    # x0+x1 in 23 RSS
            y = ABYIPrivateTensor(prot, y, x.is_scaled, BOOLEAN)        # x2 in 23 RSS
            y = y ^ delta2

            Delta1, Delta2 = x01.reveal(), y.reveal()
            x01, y = [Delta1, delta1], [Delta2, delta2]

        result = prot.B_ppai(x01, y, nbits)

    return ABYIPrivateTensor(prot, result, x.is_scaled, BOOLEAN)


def _B_ppai_private_private(prot, x, y, n_bits):
    """
    Parallel prefix adder (PPA). This adder can be used for addition of boolean sharings.

    `n_bits` can be passed as an optimization to constrain the computation for least significant
    `n_bits` bits.

    AND Depth: log(k)
    Total gates: klog(k)
    """

    # if topology == "kogge_stone":
    #     return _B_ppa_kogge_stone_private_private(prot, x, y, n_bits)
    # elif topology == "sklansky":
    #     return _B_ppa_sklansky_private_private(prot, x, y, n_bits)
    # else:
    #     raise NotImplementedError("Unknown adder topology.")
    return _B_ppai_sklansky_private_private(prot, x, y, n_bits)


def _B_ppai_sklansky_private_private(prot, x, y, n_bits):
    
    assert isinstance(x[0], ABYIPublicTensor) & isinstance(x[1], ABYIPrivateTensor), x
    assert isinstance(y[0], ABYIPublicTensor) & isinstance(y[1], ABYIPrivateTensor), y
    assert x[0].is_scaled == x[1].is_scaled == y[0].is_scaled == y[1].is_scaled, "is_scaled not aligned."

    if x[0].backing_dtype.native_type != tf.int64:
        raise NotImplementedError(
            "Native type {} not supported".format(x.backing_dtype.native_type)
        )
    # eq Debug
    shape = x[0].shape
    scaled = x[0].is_scaled
    
    '''
    ---------------------------------- Support Functions. -------------------------------------
    '''
    # Remember: ms(mask sharing):   (ABYIConstant, ABYIPrivateTensor) 
    #           rss(23 RSS):        ABYIPrivateTensor
    #           ss(33 SS):          (DenseTensor, DenseTensor, DenseTensor)
    def ms_xor(x: list, y: list) -> list:
        '''
        Compute [[z]] = [[x]] ^ [[y]].
        '''
        with tf.name_scope("ms_xor"):
            z0, z1 = x[0] ^ y[0], x[1] ^ y[1]
            return z0, z1

    def ms_and(x: list, y: list) -> ABYIPrivateTensor:
        '''
        Compute <z> = [[x]] & [[y]].
        '''
        with tf.name_scope("ms_and"):
            # x*y = (Delta0-delta0) * (Delta1-delta1)
            z = x[1] & y[1]
            z = x[0] & y[0] ^ x[0] & y[1] ^ x[1] & y[0] ^ z
            return z
    
    def ms_and_to_ms(x: list, y: list) -> list:
        '''
        Compute [[z]] = [[x]] & [[y]].
        '''
        with tf.name_scope("ms_and_to_ms"):
            z = ms_and(x, y)
            return rss_upshare(z)
    
    def ms_and_constant(x: list, c: ABYIPublicTensor) -> list:
        '''
        Compute [[z]] = [[x]] * c.
        '''
        with tf.name_scope("ms_and_constant"):
            Delta, delta = x
            return c & Delta, c & delta
    
    def ms_lshift(x: list, k: int) -> list:
        '''
        Compute [[x']] = [[x]] << k.
        '''
        with tf.name_scope("ms_lshift"):
            Delta, delta = x
            return Delta << k, delta << k
    
    def ms_downshare(x: list) -> ABYIPrivateTensor:
        '''
        Translate [[x]] to <x>. Not safe.
        '''
        with tf.name_scope("ms_downshare"):
            Delta, delta = x
            x = Delta ^ delta
            return x
    
    def rss_and(x: ABYIPrivateTensor, y: ABYIPrivateTensor) -> ABYIPrivateTensor:
        '''
        Compute [z] = <x> & <y>.
        '''
        with tf.name_scope("ms_and"):
            x_shares, y_shares = x.unwrapped, y.unwrapped

            z = [None, None, None]

            a = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = (
                        x_shares[i][0] & y_shares[i][0]
                        ^ x_shares[i][0] & y_shares[i][1]
                        ^ x_shares[i][1] & y_shares[i][0]
                        ^ a[i]
                    )
            
            return z
    
    def rss_upshare(x):
        '''
        Translate <x> to [[x]].
        '''
        with tf.name_scope("rss_upshare"):
            delta = prot._gen_random_sharing(shape, share_type=BOOLEAN)
            D = delta ^ x
            Delta = D.reveal()
            return Delta, delta
    
    def rss_downshare(x):
        '''
        Translate <x> to [x]. Not safe.
        '''
        with tf.name_scope("rss_downshare"):
            z, x_shares = [None, None, None], x.unwrapped
            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = x_shares[i][0]
            
            return z

    def ss_xor(x, y):
        '''
        Compute [z] = [x] ^ [y].
        '''
        z = [None, None, None]

        with tf.name_scope("ss_xor"):
            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = x[i] ^ y[i]
        return z
    
    def ss_upshare(x):
        '''
        Translate [x] to [[x]].
        '''
        with tf.name_scope("ss_upshare"):
            delta = prot._gen_random_sharing(x[0].shape, share_type=BOOLEAN)
            d = delta.unwrapped
            z, Delta = [None, None, None], [None, None, None]
            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = d[i][0] ^ x[i] 

            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    Delta[i] = z[0] ^ z[1] ^ z[2]
            
            Delta = ABYIPublicTensor(prot, Delta, scaled, BOOLEAN)
            return Delta, delta
    
    '''
    --------------------------------- Body --------------------------------- 
    '''
    with tf.name_scope("B_ppai"):
        # Recall: 
        #   classical PPA using 23 rss: 4 Merge, 2 Mul per Merge.                               online:     24bit.
        # 
        #   4And Solution:
        #         G[1,4] 
        #       = G[3,4] ^ P[3,4] & G[1,2] 
        #       = ( G[4,4] ^ P[4,4] & G[3,3] ) ^ ( P[4,4] & P[3,3] & ( G[2,2] ^ P[2,2] & G[1,1] ) )
        #       1 And4 6 bit, 1 And 3 bit, P[1,4] need 1 And4 6 bit (sum to 6*2+3=15bit). 
        #         
        #         G[1,3]
        #       = G[3,3] ^ P[3,3] & G[1,2]
        #       1 And4 6 bit, P[1,3] need 1 And4 6 bit (sum to 12bit).
        #
        #         G[1,2] 
        #       = G[2,2] ^ P[2,2] & G[1,1]
        #       1 And 3 bit, P[1,2] need 1 And 3 bit (sum to 6bit).                             offline:    24bit.
        #                                                                                       online:     33bit.
        # 
        # Optimization: mul in ms: G[1,2], P[3,4], P[1,2];                                      offline:    9bit.
        #               mul in rss: A = P[3,4] & G[1,2], G[3,4], G[1,3], P[1,3], P[1,4].
        #               G12ms = rss_upshare(G12rss), G13ms = ss_upshare(G13ss), 
        #               G14ms = ss_upshare(Ass) ^ G34rss);                                      online:     15bit.
        #               P12ms = rss_upshare(P12rss), P13ms = ss_upshare(P13ss),
        #               P14ms = ss_upshare(P14ss).                                              online:     15bit.
        # Practice: G11, G12, G13, G14 both in ss, so does P. ss->ms.                           online:     48bit.

        # In practice, we use s_mask(signal) to indicate the G/P signal in position[i+1,i+1], 
        # k_mask(keep) to indicate the bits used in the computation.
        # For example, to compute G[1,2] = G[2,2] ^ P[2,2] & G[1,1] in the second layer, we will 
        # use s_mask[0][1] = 0x0008000800080008 to select the G signal G11, use 
        # k_mask[1][1] = 0x00F000F000F000F0 to select the G/P bits in position[2, 2].
        
        s_mask = [None for i in range(3)]    # G/P[i+1, i+1], layer k.
        k_mask = [None for i in range(4)]
        s_mask[0] = [
            0x1111111111111111,
            0x0008000800080008,
            0x0000000000008000,
        ]                                   # to select G/P[1,1]. (G11ms = G & G11_masks)
        s_mask[1] = [
            0x2222222222222222,
            0x0080008000800080,
            0x0000000080000000,
        ]                                   # to select G/P[2,2].  
        s_mask[2] = [
            0x4444444444444444,
            0x0800080008000800,
            0x0000800000000000,
        ]                                   # to select G/P[3,3]. 
        # s_mask[3] = [
        #     0x8888888888888888,
        #     0x8000800080008000,
        #     0x8000000000000000,
        # ]                                 # to select G/P[4,4]. Useless.
        k_mask[0] = [
            0x1111111111111111,
            0x000F000F000F000F,
            0x000000000000FFFF,             
        ]                                   # to select the bits in [1,1].
        k_mask[1] = [
            0x2222222222222222,
            0x00F000F000F000F0,
            0x00000000FFFF0000,
        ]                                   # to select the bits in [2,2].
        k_mask[2] = [
            0x4444444444444444,
            0x0F000F000F000F00,
            0x0000FFFF00000000,
        ]                                   # to select the bits in [3,3].
        '''
        NOTICE: k_mask's msb is 1, which is the signal bit of int64. When transform 1 uint64 value a 
        whos msb is 1, it will be described as a two's complement of value b. However, for every bit 
        op of int64(a), the actual digit is b(true form). So we need to translate k_mask to its two's
        complement.
        '''
        k_mask[3] = [
            -0x7777777777777778,            # stand for 0x8888888888888888,   
                                            # 0xF777777777777778 doesn't work for some reason..
            -0x0FFF0FFF0FFF1000,            # stand for 0xF000F000F000F000,
            -0x0001000000000000,            # stand for 0xFFFF000000000000.
        ]                                   # to select the bits in [4,4].

        def wrap_mask(mask: list) -> ABYIPublicTensor:
            '''
            Wrap mask list to a ABYIConstant.
            '''
            mask =  prot.define_constant(
                np.ones(shape, dtype=np.int64) * np.array(mask).astype(np.int64),
                apply_scaling=False,
                share_type=BOOLEAN,
            )
            return mask

        def filling_mask(mask, i) -> ABYIPublicTensor:
            '''
            Spread the signal. 
            For example, to compute P[2,2] & G[1,1], we should put G11 at the beginning of [2,2]. 
            Then, we need to filling the rest of [2,2] with signal G11.
            '''
            if isinstance(mask, list) or isinstance(mask, tuple):
                mask = [*mask]                                      # sometimes mask is a tuple, which should be list.
                for j in range(len(mask)):
                    mask[j] = filling_mask(mask[j], i)
            else:
                mask = mask << 1
                for j in range(2*i):
                    mask = (mask << (2 ** j)) ^ mask
            return mask
        
        def debug_ss_reveal(x: list) -> list:
            x_rss = ms_downshare(ss_upshare(x))
            x = x_rss.reveal().unwrapped[0].bits().value[0][0]
            return x

        G = ms_and_to_ms(x, y)                        
        P = ms_xor(x, y)                  

        k = prot.nbits
        if n_bits is not None:
            k = n_bits
        for i in range(ceil(log(k, 4))):      # i = 0, 1, 2. Indicate layer i.
        # for i in range(1):            # eq DEBUG
            # Compute G[1,1] and P[1,1].
            k_mask11 = wrap_mask(k_mask[0][i])

            G11_bits_ms = ms_and_constant(G, k_mask11)
            G11_bits_ss = rss_downshare(ms_downshare(G11_bits_ms))

            P11_bits_ms = ms_and_constant(P, k_mask11)
            P11_bits_ss = rss_downshare(ms_downshare(P11_bits_ms))
            
            # Compute G[1,2]. G12rss = (G11ms << 1) & P22ms ^ G22rss.
            s_mask11 = wrap_mask(s_mask[0][i])
            k_mask22 = wrap_mask(k_mask[1][i])

            G11_sig_ms = ms_and_constant(G, s_mask11)
            G11_sig_ms_sp = filling_mask(G11_sig_ms, i)

            P22_bits_ms = ms_and_constant(P, k_mask22)

            G22_bits_ms = ms_and_constant(G, k_mask22) 

            G12_bits_rss = ms_downshare(G22_bits_ms) ^ ms_and(G11_sig_ms_sp, P22_bits_ms)
            G12_bits_ss = rss_downshare(G12_bits_rss)

            # Compute P[1,2]. P12rss = P11ms & P22ms.
            P11_sig_ms = ms_and_constant(P, s_mask11)   # P11_mask = G11_mask
            P11_sig_ms_sp = filling_mask(P11_sig_ms, i)
            P12_bits_rss = ms_and(P11_sig_ms_sp, P22_bits_ms)
            P12_bits_ss = rss_downshare(P12_bits_rss)

            # Compute G[1,3]. G13ss = (G12rss << 1) & P33rss ^ G33ss.
            s_mask22 = wrap_mask(s_mask[1][i])
            k_mask33 = wrap_mask(k_mask[2][i])

            G12_sig_rss = G12_bits_rss & s_mask22           # & denote for rss_and_constant, which is free.
            G12_sig_rss_sp = filling_mask(G12_sig_rss, i)

            P33_bits_ms = ms_and_constant(P, k_mask33)
            P33_bits_rss = ms_downshare(P33_bits_ms)

            G33_bits_ms = ms_and_constant(G, k_mask33)
            # G33_bits_rss = ms_downshare(G33_bits_ms)
            G33_bits_ss = rss_downshare(ms_downshare(G33_bits_ms))

            G13_bits_ss = ss_xor(rss_and(G12_sig_rss_sp, P33_bits_rss), G33_bits_ss)

            # Compute P[1,3]. P13ss = P12rss & P33rss.
            P12_sig_rss = P12_bits_rss & s_mask22
            P12_sig_rss_sp = filling_mask(P12_sig_rss, i)

            P13_bits_ss = rss_and(P12_sig_rss_sp, P33_bits_rss)

            # Compute G[1,4]. G14ss = G44ss ^ P44rss & (G33rss << 1) ^ (P44ms & (P33ms << 1)) & (G12rss << 2).
            # A denote for P34 & G12.
            s_mask33 = wrap_mask(s_mask[2][i])
            k_mask44 = wrap_mask(k_mask[3][i])

            P33_sig_ms = ms_and_constant(P, s_mask33)
            P33_sig_ms_sp = filling_mask(P33_sig_ms, i)

            P44_bits_ms = ms_and_constant(P, k_mask44)      # eq :Error. Type int64. Fixed.

            P34_bits_rss = ms_and(P33_sig_ms_sp, P44_bits_ms)

            #  G12_sig_rss_sp2, for example: 0000 0000 1000 0000 -> 0000 1000 0000 0000 -> 1111 0000 0000 0000.
            G12_sig_rss_sp2 = filling_mask(G12_sig_rss << (2 ** (2 * i)), i)    

            A_bits_ss = rss_and(P34_bits_rss, G12_sig_rss_sp2)

            G33_sig_ms = ms_and_constant(G, s_mask33)
            G33_sig_rss = ms_downshare(G33_sig_ms)
            G33_sig_rss_sp = filling_mask(G33_sig_rss, i)

            P44_bits_rss = ms_downshare(P44_bits_ms)

            G44_bits_ms = ms_and_constant(G, k_mask44)
            # G44_bits_rss = ms_downshare(G44_bits_ms)
            G44_bits_ss = rss_downshare(ms_downshare(G44_bits_ms))

            G34_bits_ss = ss_xor(G44_bits_ss, rss_and(G33_sig_rss_sp, P44_bits_rss))

            G14_bits_ss = ss_xor(G34_bits_ss, A_bits_ss)

            # Compute P[1,4]. P14ss = P12rss & P34rss.
            P12_sig_rss_sp2 = filling_mask(P12_sig_rss << (2 ** (2 * i)), i)    

            P14_bits_ss = rss_and(P12_sig_rss_sp2, P34_bits_rss)

            # Compute the final G and P. 
            Gprime = ss_xor(ss_xor(G11_bits_ss, G12_bits_ss), ss_xor(G13_bits_ss, G14_bits_ss))
            Pprime = ss_xor(ss_xor(P11_bits_ss, P12_bits_ss), ss_xor(P13_bits_ss, P14_bits_ss))
            # # Debug.
            # op = [
            #     rss_downshare(ms_downshare(G)), rss_downshare(ms_downshare(P)),
            #     G11_bits_ss, G12_bits_ss, G13_bits_ss, G14_bits_ss,
            #     P11_bits_ss, P12_bits_ss, P13_bits_ss, P14_bits_ss,
            #     Gprime, Pprime
            # ]
            # op[0] = tf.print("signal {}".format(0), debug_ss_reveal(op[0]), summarize=-1)
            # for j in range(1, len(op)):
            #     with tf.control_dependencies([op[j-1]]):
            #         op[j] = tf.print("signal {}".format(j), debug_ss_reveal(op[j]), summarize=-1)
            # with tf.control_dependencies(op):
            #     op_pr = tf.print('+'*80)
            # with tf.control_dependencies([op_pr]):
            G, P = ss_upshare(Gprime), ss_upshare(Pprime)

        # G stores the carry-in to the next position
        C = ms_lshift(G, 1)
        P = ms_xor(x, y)
        z = ms_xor(C, P)

    return ms_downshare(z).unwrapped

def _msbi_private(prot, x):
    '''
    Extract msb.
    '''
    assert isinstance(x, ABYIPrivateTensor), type(x)
    assert x.share_type == ARITHMETIC

    x_shares = x.unwrapped
    zero = prot.define_constant(
        np.zeros(x.shape, dtype=np.int64), apply_scaling=False, share_type=BOOLEAN
    )

    # used to construct the 23-RSS shares of x0+x1.
    # a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)
    delta1 = prot._gen_random_sharing(x.shape, share_type=BOOLEAN)
    delta1_shares = delta1.unwrapped
    z1 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)
    # used to construct the 23-RSS shares of x2.
    z2 = zero.unwrapped
    delta2 = prot._gen_random_sharing(x.shape, share_type=BOOLEAN)


    x01, Delta1 = [[None, None], [None, None], [None, None]], None       # x0+x1
    y, Delta2 = [[None, None], [None, None], [None, None]], None            # x2
    with tf.name_scope("A2Bi"):
        # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
        # We need to reshare it into two 23-RSS operands which will be used to generate mask sharing:
        # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
        # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2

        with tf.device(prot.servers[0].device_name):
            x0_plus_x1 = x_shares[0][0] + x_shares[0][1]
            x0 = x0_plus_x1 ^ delta1_shares[0][0] ^ z1[0]

            y[0][0] = z2[0]
            y[0][1] = z2[0] 

        with tf.device(prot.servers[1].device_name):
            x1 = delta1_shares[1][0] ^ z1[1]

            y[1][0] = z2[1]
            y[1][1] = x_shares[1][1] 

        with tf.device(prot.servers[2].device_name):
            x2 = delta1_shares[2][0] ^ z1[2]

            y[2][0] = x_shares[2][0] 
            y[2][1] = z2[2]

        with tf.name_scope("Build_x_ms"):
            with tf.device(prot.servers[0].device_name):
                x01[0][0] = x0
                x01[0][1] = x1
            
            with tf.device(prot.servers[1].device_name):
                x01[1][0] = x1
                x01[1][1] = x2
            
            with tf.device(prot.servers[2].device_name):
                x01[2][0] = x2
                x01[2][1] = x0

            x01 = ABYIPrivateTensor(prot, x01, x.is_scaled, BOOLEAN)    # x0+x1 in 23 RSS
            y = ABYIPrivateTensor(prot, y, x.is_scaled, BOOLEAN)        # x2 in 23 RSS
            y = y ^ delta2

            Delta1, Delta2 = x01.reveal(), y.reveal()
            x01, y = [Delta1, delta1], [Delta2, delta2]

        result = _msb_ppai(prot, x01, y)

    return ABYIPrivateTensor(prot, result, x.is_scaled, BOOLEAN)

def _msb_ppai(prot, x, y):
    '''
    PPA optimized for MSB.
    '''
    assert isinstance(x[0], ABYIPublicTensor) & isinstance(x[1], ABYIPrivateTensor), x
    assert isinstance(y[0], ABYIPublicTensor) & isinstance(y[1], ABYIPrivateTensor), y
    assert x[0].is_scaled == x[1].is_scaled == y[0].is_scaled == y[1].is_scaled, "is_scaled not aligned."

    if x[0].backing_dtype.native_type != tf.int64:
        raise NotImplementedError(
            "Native type {} not supported".format(x.backing_dtype.native_type)
        )
    # eq Debug
    shape = x[0].shape
    scaled = x[0].is_scaled
    
    '''
    ---------------------------------- Support Functions. -------------------------------------
    '''
    # Remember: ms(mask sharing):   (ABYIConstant, ABYIPrivateTensor) 
    #           rss(23 RSS):        ABYIPrivateTensor
    #           ss(33 SS):          (DenseTensor, DenseTensor, DenseTensor)
    def ms_xor(x: list, y: list) -> list:
        '''
        Compute [[z]] = [[x]] ^ [[y]].
        '''
        with tf.name_scope("ms_xor"):
            z0, z1 = x[0] ^ y[0], x[1] ^ y[1]
            return z0, z1

    def ms_and(x: list, y: list) -> ABYIPrivateTensor:
        '''
        Compute <z> = [[x]] & [[y]].
        '''
        with tf.name_scope("ms_and"):
            # x*y = (Delta0-delta0) * (Delta1-delta1)
            z = x[1] & y[1]
            z = x[0] & y[0] ^ x[0] & y[1] ^ x[1] & y[0] ^ z
            return z
    
    def ms_and_to_ms(x: list, y: list) -> list:
        '''
        Compute [[z]] = [[x]] & [[y]].
        '''
        with tf.name_scope("ms_and_to_ms"):
            z = ms_and(x, y)
            return rss_upshare(z)
    
    def ms_and_constant(x: list, c: ABYIPublicTensor) -> list:
        '''
        Compute [[z]] = [[x]] * c.
        '''
        with tf.name_scope("ms_and_constant"):
            Delta, delta = x
            return c & Delta, c & delta
    
    def ms_lshift(x: list, k: int) -> list:
        '''
        Compute [[x']] = [[x]] << k.
        '''
        with tf.name_scope("ms_lshift"):
            Delta, delta = x
            return Delta << k, delta << k
    
    def ms_downshare(x: list) -> ABYIPrivateTensor:
        '''
        Translate [[x]] to <x>. Not safe.
        '''
        with tf.name_scope("ms_downshare"):
            Delta, delta = x
            x = Delta ^ delta
            return x
    
    def rss_and(x: ABYIPrivateTensor, y: ABYIPrivateTensor) -> ABYIPrivateTensor:
        '''
        Compute [z] = <x> & <y>.
        '''
        with tf.name_scope("ms_and"):
            x_shares, y_shares = x.unwrapped, y.unwrapped

            z = [None, None, None]

            a = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = (
                        x_shares[i][0] & y_shares[i][0]
                        ^ x_shares[i][0] & y_shares[i][1]
                        ^ x_shares[i][1] & y_shares[i][0]
                        ^ a[i]
                    )
            
            return z
    
    def rss_upshare(x):
        '''
        Translate <x> to [[x]].
        '''
        with tf.name_scope("rss_upshare"):
            delta = prot._gen_random_sharing(shape, share_type=BOOLEAN)
            D = delta ^ x
            Delta = D.reveal()
            return Delta, delta
    
    def rss_downshare(x):
        '''
        Translate <x> to [x]. Not safe.
        '''
        with tf.name_scope("rss_downshare"):
            z, x_shares = [None, None, None], x.unwrapped
            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = x_shares[i][0]
            
            return z

    def ss_xor(x, y):
        '''
        Compute [z] = [x] ^ [y].
        '''
        z = [None, None, None]

        with tf.name_scope("ss_xor"):
            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = x[i] ^ y[i]
        return z
    
    def ss_upshare(x):
        '''
        Translate [x] to [[x]].
        '''
        with tf.name_scope("ss_upshare"):
            delta = prot._gen_random_sharing(x[0].shape, share_type=BOOLEAN)
            d = delta.unwrapped
            z, Delta = [None, None, None], [None, None, None]
            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    z[i] = d[i][0] ^ x[i] 

            for i in range(3):
                with tf.device(prot.servers[i].device_name):
                    Delta[i] = z[0] ^ z[1] ^ z[2]
            
            Delta = ABYIPublicTensor(prot, Delta, scaled, BOOLEAN)
            return Delta, delta
    
    def wrap_mask(mask: list) -> ABYIPublicTensor:
        '''
        Wrap mask list to a ABYIConstant.
        '''
        mask =  prot.define_constant(
            np.ones(shape, dtype=np.int64) * np.array(mask).astype(np.int64),
            apply_scaling=False,
            share_type=BOOLEAN,
        )
        return mask

    def filling_mask(mask, i) -> ABYIPublicTensor:
        '''
        Spread the signal. 
        For example, to compute P[2,2] & G[1,1], we should put G11 at the beginning of [2,2]. 
        Then, we need to filling the rest of [2,2] with signal G11.
        '''
        if isinstance(mask, list) or isinstance(mask, tuple):
            mask = [*mask]                                      # sometimes mask is a tuple, which should be list.
            for j in range(len(mask)):
                mask[j] = filling_mask(mask[j], i)
        else:
            mask = mask << 1
            for j in range(2*i):
                mask = (mask << (2 ** j)) ^ mask
        return mask
    
    def debug_ss_reveal(x: list) -> list:
        x_rss = ms_downshare(ss_upshare(x))
        x = x_rss.reveal().unwrapped[0].bits().value[0][0]
        return x
    
    def ppa(x, y) -> list:
        with tf.name_scope("msb_ppa"):
            s_mask = ...


def _bit_extract_bool_public(prot, x):
    # Take out the i-th bit of a BOOLEAN sharing tensor.
    assert isinstance(x, ABYIPublicTensor)
    assert x.share_type == BOOLEAN, x.share_type
    
    mask = prot.define_constant(
        np.array([0x1 << i]), apply_scaling=False, share_type=BOOLEAN
    )
    x = x & mask

    x_on_i = x.unwrapped
    bit_on_i = [None]
    for i in range(3):
        with tf.device(prot.servers[i].device_name):
            bit_on_i[i] = x_on_i[i].cast(prot.bool_factory)
    result = ABYIPublicTensor(prot, bit_on_i, False, BOOLEAN)

    return result


def _bit_extract_bool_private(prot, x):
    # Take out the i-th bit of a BOOLEAN sharing tensor.
    assert isinstance(x, ABYIPrivateTensor)
    assert x.share_type == BOOLEAN, x.share_type
    
    mask = prot.define_constant(
        np.array([0x1 << i]), apply_scaling=False, share_type=BOOLEAN
    )
    x = x & mask

    x_shares = x.unwrapped
    result = [[None, None], [None, None], [None, None]]
    for i in range(3):
        with tf.device(prot.servers[i].device_name):
            result[i][0] = x_shares[i][0].cast(prot.bool_factory)
            result[i][1] = x_shares[i][1].cast(prot.bool_factory)
    result = ABYIPrivateTensor(prot, result, False, BOOLEAN)

    return result