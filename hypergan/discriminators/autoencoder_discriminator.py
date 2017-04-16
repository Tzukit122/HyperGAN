import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import os
import hypergan

import hypergan.discriminators.minibatch_discriminator as minibatch

def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return a-b

def vae_distance(x, rx):
    print("VAE GAN LOSS")
    x = tf.reshape(x, [(int)(x.get_shape()[0]), -1])
    rx = tf.reshape(rx, [(int)(rx.get_shape()[0]), -1])
    return tf.reduce_sum(-x * tf.log(rx + 1e-8) - (1.0 - x) * tf.log(1.0 - rx + 1e-8), axis=1)

def vae_distance_shifted(x, rx):

    x = (x + 1.) / 2.
    rx = (rx + 1.) / 2.
    print("VAE GAN LOSS")
    x = tf.reshape(x, [(int)(x.get_shape()[0]), -1])
    rx = tf.reshape(rx, [(int)(rx.get_shape()[0]), -1])
    return tf.reduce_sum(-x * tf.log(rx + 1e-8) - (1.0 - x) * tf.log(1.0 - rx + 1e-8), axis=1)


def standard_block(config, net, depth, prefix='d_'):
   batch_norm = config['layer_regularizer']
   activation = config['activation']
   filter_size_w = 2
   filter_size_h = 2
   filter = [1,filter_size_w,filter_size_h,1]
   stride = [1,filter_size_w,filter_size_h,1]

   net = conv2d(net, depth, name=prefix+'_layer', k_w=3, k_h=3, d_h=1, d_w=1, regularizer=None,gain=config.orthogonal_initializer_gain)
   net = tf.nn.avg_pool(net, ksize=filter, strides=stride, padding='SAME')
   print('[discriminator] layer', net)
   return net



def config(
        activation=lrelu,
        block=standard_block,
        depth_increase=2,
        final_activation=None,
        first_conv_size=16,
        first_strided_conv_size=64,
        distance=l1_distance,
        layer_regularizer=layer_norm_1,
        layers=5,
        resize=None,
        noise=None,
        layer_filter=None,
        progressive_enhancement=True,
        orthogonal_initializer_gain=1.0,
        fc_layers=0,
        fc_layer_size=1024,
        extra_layers=4,
        extra_layers_reduction=2,
        strided=False,
        foundation='additive',
        create=None,
        minibatch=False,
        batch_norm_momentum=[0.001],
        batch_norm_epsilon=[0.0001]
        ):
    selector = hc.Selector()
    selector.set("activation", [lrelu])#prelu("d_")])
    selector.set("block", block) 
    selector.set("depth_increase", depth_increase)# Size increase of D's features on each layer
    selector.set("final_activation", final_activation)
    selector.set("first_conv_size", first_conv_size)
    selector.set("first_strided_conv_size", first_conv_size)
    selector.set('foundation', foundation)
    selector.set("layers", layers) #Layers in D
    if create is None:
        selector.set('create', discriminator)
    else:
        selector.set('create', create)

    selector.set('fc_layer_size', fc_layer_size)
    selector.set('fc_layers', fc_layers)
    selector.set('extra_layers', extra_layers)
    selector.set('extra_layers_reduction', extra_layers_reduction)
    selector.set('layer_filter', layer_filter) #add information to D
    selector.set('layer_regularizer', layer_regularizer) # Size of fully connected layers
    selector.set('orthogonal_initializer_gain', orthogonal_initializer_gain)
    selector.set('noise', noise) #add noise to input
    selector.set('progressive_enhancement', progressive_enhancement)
    selector.set('resize', resize)
    selector.set('strided', strided) #TODO: true does not work
    selector.set('distance', distance) #TODO: true does not work
    selector.set('minibatch', minibatch)

    selector.set('batch_norm_momentum', batch_norm_momentum)
    selector.set('batch_norm_epsilon', batch_norm_epsilon)
    return selector.random_config()

#TODO: arguments telescope, root_config/config confusing
def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    h = hypergan.discriminators.pyramid_discriminator.discriminator(gan, config, x, g, xs, gs, prefix)
    with tf.variable_scope("autoencoder", reuse=False):
        generator = hc.Config(hc.lookup_functions(gan.config.generator))
        if 'experiment' in config:
            s = [int(x) for x in h.get_shape()]
            ha  = tf.slice(h, [0,0], [s[0],s[1]//2])
            hb  = tf.slice(h, [0,s[1]//2], [s[0],s[1]//2])
            h = hb

        s = [int(x) for x in h.get_shape()]
        hx  = tf.slice(h, [0,0], [s[0]//2,-1])
        hg  = tf.slice(h, [s[0]//2,0], [s[0]//2,-1])

        if 'vae' in config:

            def vae(h):
                s = [int(x) for x in h.get_shape()]
                mean = tf.slice(h, [0,0], [s[0],s[1]//2])
                second_half = tf.slice(h, [0,s[1]//2], [s[0],s[1]//2])
                stddev = tf.sqrt(tf.exp( second_half ))
                epsilon = tf.random_normal([s[0], s[1]//2])
                return mean, stddev, epsilon, mean + epsilon * stddev
            
            hxm, hxs, hxe, hx = vae(hx)
            hgm, hgs, hge ,hg = vae(hg)
            print("VAE ENABLED")
 

        if config.minibatch:
            mini = h
            #mini = tf.slice(net, [0, 300], [-1, 100])
            mini = minibatch.get_minibatch_features(mini, int(mini.get_shape()[0]), gan.config.dtype, prefix, 25, 100)
        else:
            mini = []

        rx = generator.create(generator, gan, hx, prefix=prefix)[-1]
    with tf.variable_scope("autoencoder", reuse=True):
        rg = generator.create(generator, gan, hg, prefix=prefix)[-1]

    gan.graph.dx = rx
    gan.graph.dg = rg


    #if 'vae' in config:
    #    def vae_loss(m, s, e):
    #        print("VAE", m, s, e)
    #        return tf.reduce_sum(0.5 * (tf.square(m)+tf.square(s)-2*tf.log(s+e)-1.0), axis=1)
    #    dist_x = vae_loss(hxm, hxs, hxe)+l1_distance(x,rx)#+ vae_distance(x, rx)
    #    dist_g = vae_loss(hgm, hgs, hge)+l1_distance(g,rg)# + vae_distance(g, rg)
    #    error = tf.concat([dist_x, dist_g], axis=0)
    #else:
    #    error = tf.concat([config.distance(x, rx), config.distance(g,rg)], axis=0)
    error = tf.concat([config.distance(x, rx), config.distance(g,rg)], axis=0)
    error = tf.reshape(error, [gan.config.batch_size*2, -1])
    error = tf.concat([error]+mini, axis=1)

    return error


