import os
import tempfile
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted import session
from tf_encrypted.protocol.abyi import ABYI
from tf_encrypted.protocol.abyi import ARITHMETIC
from tf_encrypted.protocol.abyi import BOOLEAN
from tf_encrypted.protocol.abyi import ABYIPrivateTensor, ABYIPublicTensor

class TestABYI(unittest.TestCase):
    def test_a2b_private(self):
        tf.reset_default_graph()

        prot = ABYI()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ARITHMETIC
        )

        z = tfe.A2B(x)
        assert z.share_type == BOOLEAN

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z.reveal().unwrapped[0].bits())
            print(result)
            # np.testing.assert_allclose(
            #     result, np.array([[1, 2, 3], [4, 5, 6]]), rtol=0.0, atol=0.01
            # )

    def test_a2bi_private(self):
        tf.reset_default_graph()

        prot = ABYI()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=ARITHMETIC
        )

        truth = x
        truth_bits = x.reveal().unwrapped[0].bits()
        res = tfe.A2Bi(x)
        res_bits = res.reveal().unwrapped[0].bits()

        assert res.share_type == BOOLEAN

        with tfe.Session() as sess:
            sess.run(tfe.global_variables_initializer())
            t = sess.run(truth.reveal())
            r = sess.run(res.reveal())
            pr(t, r, "A2Bi", False)
            # t = sess.run(truth_bits)
            # r = sess.run(res_bits)
            # pr(t, r, "A2Bi [0][0] bits", False)
    
    def test_ppa_private_private(self):
        tf.reset_default_graph()

        prot = ABYI()
        tfe.set_protocol(prot)

        x1 = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=BOOLEAN
        )
        y = tfe.define_private_variable(
            tf.constant([[7, 8, 9], [10, 11, 12]]), share_type=BOOLEAN
        )

        # Parallel prefix adder. It is simply an adder for boolean sharing.
        z1 = tfe.B_ppa(x, y, topology="sklansky")
        z2 = tfe.B_ppa(x, y, topology="kogge_stone")

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = sess.run(z1.reveal())
            np.testing.assert_allclose(
                result, np.array([[8, 10, 12], [14, 16, 18]]), rtol=0.0, atol=0.01
            )

            result = sess.run(z2.reveal())
            np.testing.assert_allclose(
                result, np.array([[8, 10, 12], [14, 16, 18]]), rtol=0.0, atol=0.01
            )
    
    def test_ppai_private_private(self):
        tf.reset_default_graph()

        prot = ABYI()
        tfe.set_protocol(prot)

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), share_type=BOOLEAN
        )
        y = tfe.define_private_variable(
            tf.constant([[7, 8, 9], [10, 11, 12]]), share_type=BOOLEAN
        )

        # Parallel prefix adder. It is simply an adder for boolean sharing.
        z1 = tfe.B_ppai(x, y)

        with tfe.Session() as sess:
            # initialize variables
            sess.run(tfe.global_variables_initializer())
            # reveal result
            result = z1.reveal()
            sess.run(tf.print(result))
    
    def test_native_ppai_sklansky(self):
        from math import log
        from random import randint

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
        k_mask[3] = [
            0x8888888888888888,
            0xF000F000F000F000,
            0xFFFF000000000000,
        ]                     

        def filling_mask(mask, i):     
            mask = mask << 1
            for j in range(2*i):
                mask = (mask << (2 ** j)) ^ mask
            return mask

        n = 10
        while n > 0:
            n = n - 1
            x = randint(1, 2 ** 31)
            y = randint(1, 2 ** 31)
        # x, y = 2**30, 2**20 + 1
            G = x & y
            P = x ^ y
            k = 64
            for i in range(int(log(k, 4))):
                # Compute G[1,1] and P[1,1].
                G11 = G & k_mask[0][i]
                P11 = P & k_mask[0][i]

                # Compute G[1,2] and P[1,2].
                G22 = G & k_mask[1][i]
                G11_sp = filling_mask(G & s_mask[0][i], i)
                P22 = P & k_mask[1][i]
                G12 = G22 ^ G11_sp & P22
                P11_sp = filling_mask(P & s_mask[0][i], i)
                P12 = P11_sp & P22

                # Compute G[1,3] and P[1,3].
                G33 = G & k_mask[2][i]
                P33 = P & k_mask[2][i]
                G13 = G33 ^ P33 & filling_mask(G12 & s_mask[1][i], i)
                P13 = P33 & filling_mask(P12 & s_mask[1][i], i)

                # Compute G[1,4] and P[1,4].
                G44 = G & k_mask[3][i]
                P44 = P & k_mask[3][i]
                G34 = G44 ^ P44 & filling_mask(G33 & s_mask[2][i], i)
                P34 = P44 & filling_mask(P33 & s_mask[2][i], i)
                G14 = G34 ^ P34 & filling_mask((G12 & s_mask[1][i]) << 2 ** (2 * i), i)
                P14 = P34 & filling_mask((P12 & s_mask[1][i]) << 2 ** (2 * i), i)
                
                G = G11 ^ G12 ^ G13 ^ G14
                P = P11 ^ P12 ^ P13 ^ P14

            # G stores the carry-in to the next position
            C = G << 1
            P = x ^ y
            z = C ^ P

            truth = x + y

            print('x: {}\ty: {}\n{}\t---truth: {}\tresult: {}'.format(x, y, truth==z, truth, z))
    
    def test_share_conversion(self):
        tf.reset_default_graph()

        prot = ABYI()
        tfe.set_protocol(prot)

        scaled = True

        x = tfe.define_private_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), apply_scaling=scaled, share_type=BOOLEAN
        )
        y = tfe.define_private_variable(
            tf.constant([[11, 21, 31], [41, 51, 61]]), apply_scaling=scaled, share_type=BOOLEAN
        )
        c = tfe.define_public_variable(
            tf.constant([[1, 2, 3], [4, 5, 6]]), apply_scaling=scaled, share_type=BOOLEAN
        )

        shape = x.shape

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
                # a0, a1, a2 = prot._gen_zero_sharing(
                #     x.shape, share_type=BOOLEAN, factory=x.backing_dtype
                # )

                # with tf.device(prot.servers[0].device_name):
                #     tmp0 = x_shares[0][0] & y_shares[0][0]
                #     tmp1 = x_shares[0][0] & y_shares[0][1]
                #     tmp2 = x_shares[0][1] & y_shares[0][0]
                #     z0 = tmp0 ^ tmp1 ^ tmp2 ^ a[0]

                # with tf.device(prot.servers[1].device_name):
                #     tmp0 = x_shares[1][0] & y_shares[1][0]
                #     tmp1 = x_shares[1][0] & y_shares[1][1]
                #     tmp2 = x_shares[1][1] & y_shares[1][0]
                #     z1 = tmp0 ^ tmp1 ^ tmp2 ^ a[1]

                # with tf.device(prot.servers[2].device_name):
                #     tmp0 = x_shares[2][0] & y_shares[2][0]
                #     tmp1 = x_shares[2][0] & y_shares[2][1]
                #     tmp2 = x_shares[2][1] & y_shares[2][0]
                #     z2 = tmp0 ^ tmp1 ^ tmp2 ^ a[2]
                
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
        
        # Test 1: share conversion
        truth = x.reveal()
        r1 = ms_downshare(rss_upshare(x))
        r2 = ms_downshare(ss_upshare(rss_downshare(x)))
        op1, op2 = r1.reveal(), r2.reveal()
        with tfe.Session() as sess:
            print("="*100, '\nTest 1: share conversion\n')
            sess.run(tfe.global_variables_initializer())
            # reveal result
            t = sess.run(truth)
            a1 = sess.run(op1)
            a2 = sess.run(op2)
            pr(t, a1, "Share Conversion 1")
            pr(t, a2, "Share Conversion 2")
        
        # Test 2: op in ms
        op1, op2 = rss_upshare(x), rss_upshare(y)
        # xor
        t_xor = x ^ y
        truth_xor = t_xor.reveal()
        res_xor = ms_downshare(ms_xor(op1, op2)).reveal()
        # and
        t_and = x & y
        truth_and = t_and.reveal()
        res_and = ms_and(op1, op2).reveal()
        # and constant
        t_andc = x & c
        truth_andc = t_andc.reveal()
        res_andc = ms_downshare(ms_and_constant(op1, c)).reveal()
        # lshift
        t_ls = x << 1
        truth_ls = t_ls.reveal()
        res_ls = ms_downshare(ms_lshift(op1, 1)).reveal()
        with tfe.Session() as sess:
            print("="*100, '\nTest 2: share conversion')
            sess.run(tfe.global_variables_initializer())
            t = sess.run(truth_xor)
            r = sess.run(res_xor)
            pr(t, r, "Xor")
            t = sess.run(truth_and)
            r = sess.run(res_and)
            pr(t, r, "And")
            t = sess.run(truth_andc)
            r = sess.run(res_andc)
            pr(t, r, "And Constant")
            t = sess.run(truth_ls)
            r = sess.run(res_ls)
            pr(t, r, "Lshift")
        
        # Test 3: op in rss
        # and
        t_and = x & y
        truth_and = t_and.reveal()
        res_and = ms_downshare(ss_upshare(rss_and(x, y))).reveal()
        with tfe.Session() as sess:
            print("="*100, '\nTest 3: op in rss')
            sess.run(tfe.global_variables_initializer())
            t = sess.run(truth_and)
            r = sess.run(res_and)
            pr(t, r, "And")
        
        # Test 4: op in ss
        op1, op2 = rss_downshare(x), rss_downshare(y)
        # xor
        t_xor = x ^ y
        truth_xor = t_xor.reveal()
        res_xor = ms_downshare(ss_upshare(ss_xor(op1, op2))).reveal()
        with tfe.Session() as sess:
            print("="*100, '\nTest 4: op in ss')
            sess.run(tfe.global_variables_initializer())
            t = sess.run(truth_xor)
            r = sess.run(res_xor)
            pr(t, r, "Xor")

def pr(t, r, summary, test_equal=True):
    print("-"*40, ' {}'.format(summary))
    print("Truth:\n {}\nResult:\n {}".format(t,r))
    if test_equal:
        print("\nis equal?\t{}\n".format((t==r).all()))

if __name__ == "__main__":
    """
    Run these tests with:
    python ppai_test.py
    """
    t = TestABYI()
    t.test_a2bi_private()
    # t.test_a2bi_private()