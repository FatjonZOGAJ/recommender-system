from lib.models import autoencoder, fm, fm_rel, kernel_net, nmf, svd

models = {'svd': svd,
          'als': nmf,
          'kernel_net': kernel_net,
          'autoencoder': autoencoder,
          'fm': fm,
          'fm_rel': fm_rel
          }
